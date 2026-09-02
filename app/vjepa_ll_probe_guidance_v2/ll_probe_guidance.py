import os
import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from decord import VideoReader, cpu
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False
    import cv2

logger = logging.getLogger(__name__)

# Default ultrasound video FPS
DEFAULT_AVG_FPS = 15.0

# Default standardization stats from probe guidance experiments
DEFAULT_ACTION_MEAN = np.array(
    [-2.7906366e-05, 5.4028669e-06, 6.0837847e-05, -1.7707590e-03, 2.4102912e-03, -3.7514704e-04],
    dtype=np.float32,
)
DEFAULT_ACTION_STD = np.array(
    [0.00498742, 0.00453256, 0.00675807, 0.55955255, 0.43809873, 0.73082167],
    dtype=np.float32,
)

DEFAULT_STATE_MEAN = np.array(
    [-0.13193196, 0.13471916, 1.4406046, 59.395, -43.585613, -76.49974],
    dtype=np.float32,
)
DEFAULT_STATE_STD = np.array(
    [1.05345368e-01, 1.75726220e-01, 7.28756189e-02, 7.79652252e+01, 3.07379379e+01, 1.22978004e+02],
    dtype=np.float32,
)

# Active module-level stats
_GLOBAL_STATS = {
    "action_mean": DEFAULT_ACTION_MEAN,
    "action_std": DEFAULT_ACTION_STD,
    "state_mean": DEFAULT_STATE_MEAN,
    "state_std": DEFAULT_STATE_STD,
}


def load_stats_file(stats_file_path: str):
    """Loads normalization statistics from a JSON file."""
    with open(stats_file_path, "r") as f:
        data = json.load(f)
    if "action_mean" in data:
        _GLOBAL_STATS["action_mean"] = np.array(data["action_mean"], dtype=np.float32)
    if "action_std" in data:
        _GLOBAL_STATS["action_std"] = np.array(data["action_std"], dtype=np.float32)
    if "state_mean" in data:
        _GLOBAL_STATS["state_mean"] = np.array(data["state_mean"], dtype=np.float32)
    if "state_std" in data:
        _GLOBAL_STATS["state_std"] = np.array(data["state_std"], dtype=np.float32)
    logger.info(f"Loaded dataset statistics from {stats_file_path}")


def standardize_states(states: torch.Tensor) -> torch.Tensor:
    state_mean = torch.tensor(_GLOBAL_STATS["state_mean"], dtype=torch.float32, device=states.device)
    state_std = torch.tensor(_GLOBAL_STATS["state_std"], dtype=torch.float32, device=states.device)
    return (states - state_mean) / (state_std + 1e-6)


def standardize_actions(actions: torch.Tensor) -> torch.Tensor:
    action_mean = torch.tensor(_GLOBAL_STATS["action_mean"], dtype=torch.float32, device=actions.device)
    action_std = torch.tensor(_GLOBAL_STATS["action_std"], dtype=torch.float32, device=actions.device)
    return (actions - action_mean) / (action_std + 1e-6)


class LLProbeGuidanceV2Dataset(Dataset):
    """
    Standalone Dataset loader for V-JEPA 2 Action-Conditioned Transformer Predictor.
    Exclusively supports multi-modal recording sessions (ultrasound_bmode.mp4 + trajectory.csv).
    """

    def __init__(
        self,
        data_root: str,
        frames_per_clip: int = 16,
        frames_per_second: int = 4,
        frame_skip: int = 1,
        transform=None,
        is_train: bool = True,
        pose_source: str = "tip",  # "tip" or "centroid"
        stats_file: Optional[str] = None,
        avg_fps: float = DEFAULT_AVG_FPS,
    ):
        self.data_root = data_root
        self.frames_per_clip = frames_per_clip
        self.frames_per_second = frames_per_second
        self.frame_skip = frame_skip
        self.transform = transform
        self.is_train = is_train
        self.pose_source = pose_source
        self.avg_fps = avg_fps

        if stats_file and os.path.exists(stats_file):
            load_stats_file(stats_file)

        # Discover session directories
        raw_sessions = self._find_session_directories(data_root)

        self.valid_starts_map: Dict[str, List[int]] = {}
        self.states_map: Dict[str, np.ndarray] = {}
        self.video_path_map: Dict[str, str] = {}

        self.frame_step = max(1, math.ceil(self.avg_fps / self.frames_per_second))
        clip_len = self.frames_per_clip * self.frame_step

        print(f"Scanning {len(raw_sessions)} sessions in {data_root} for valid tracking clips...")
        for session_path in tqdm(raw_sessions, total=len(raw_sessions), desc="Indexing sessions"):
            session_data = self._load_session_data(session_path)
            if session_data is None:
                continue

            states, tracking, num_frames, video_path = session_data

            if num_frames < clip_len:
                continue

            valid_starts_for_this_session = []
            for start_idx in range(num_frames - clip_len + 1):
                indices = np.arange(start_idx, start_idx + clip_len, self.frame_step)
                clip_tracking = tracking[indices]

                # Only keep clips where all sample frames have valid tracking
                if np.all(clip_tracking == 1):
                    valid_starts_for_this_session.append(start_idx)

            if len(valid_starts_for_this_session) > 0:
                self.valid_starts_map[session_path] = valid_starts_for_this_session
                self.states_map[session_path] = states
                self.video_path_map[session_path] = video_path

        self.episodes = list(self.valid_starts_map.keys())
        print(f"Retained {len(self.episodes)} valid sessions out of {len(raw_sessions)}.")

        # Build eval index for deterministic evaluation
        if not self.is_train:
            self.eval_index = []
            for ep in self.episodes:
                valid_starts = self.valid_starts_map[ep]
                for start in valid_starts:
                    self.eval_index.append((ep, start))

    def _find_session_directories(self, data_root: str) -> List[str]:
        """Finds session directories containing ultrasound_bmode.mp4 and trajectory.csv."""
        if not os.path.exists(data_root):
            logger.warning(f"Data root {data_root} does not exist.")
            return []

        candidates = []
        for root, dirs, files in os.walk(data_root):
            if "trajectory.csv" in files and "ultrasound_bmode.mp4" in files:
                candidates.append(root)

        return sorted(list(set(candidates)))

    def _load_session_data(
        self, session_path: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int, str]]:
        """Loads probe states, tracking flags, and video metadata from a session directory."""
        csv_path = os.path.join(session_path, "trajectory.csv")
        mp4_path = os.path.join(session_path, "ultrasound_bmode.mp4")

        if not (os.path.exists(csv_path) and os.path.exists(mp4_path)):
            return None

        try:
            # Determine total video frames
            if _HAS_DECORD:
                vr = VideoReader(mp4_path, ctx=cpu(0))
                total_frames = len(vr)
            else:
                cap = cv2.VideoCapture(mp4_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

            if total_frames <= 0:
                return None

            df = pd.read_csv(csv_path)
            # Filter valid US frames (frame_idx_us >= 1 mapped to 0-indexed video frames: us_frame_idx = frame_idx_us - 1)
            df_valid_us = df[(df["frame_idx_us"] >= 1) & (df["frame_idx_us"] <= total_frames)].copy()
            df_valid_us["us_frame_idx"] = df_valid_us["frame_idx_us"].astype(int) - 1

            # Select pose column prefix based on pose_source
            prefix = "tip" if self.pose_source == "tip" and "tip_x" in df.columns else "centroid"
            pos_cols = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
            rot_cols = [f"{prefix}_roll", f"{prefix}_pitch", f"{prefix}_yaw"]

            # Group by US frame index, selecting latest valid tracking pose for each frame
            grouped = df_valid_us.groupby("us_frame_idx").agg({
                pos_cols[0]: "last",
                pos_cols[1]: "last",
                pos_cols[2]: "last",
                rot_cols[0]: "last",
                rot_cols[1]: "last",
                rot_cols[2]: "last",
                "tracking_valid": "max",
            }).reindex(range(total_frames))

            states = grouped[pos_cols + rot_cols].values.astype(np.float32)
            tracking = grouped["tracking_valid"].fillna(0).values.astype(np.uint8)

            # Invalidate frames containing NaN
            nan_mask = np.isnan(states).any(axis=1)
            tracking[nan_mask] = 0

            return states, tracking, total_frames, mp4_path

        except Exception as e:
            logger.warning(f"Failed to parse session {session_path}: {e}")
            return None

    def poses_to_diffs(self, states: np.ndarray) -> np.ndarray:
        """
        Converts 6-DoF absolute poses [T, 6] (x, y, z, roll_deg, pitch_deg, yaw_deg)
        into Egocentric relative Actions [T-1, 6].
        """
        states = np.asarray(states, dtype=np.float32)
        if len(states) < 2:
            return np.zeros((0, 6), dtype=np.float32)

        xyz = states[:, :3]
        rvecs_deg = states[:, 3:]

        # Convert Euler angles in degrees ('xyz' convention) to 3x3 rotation matrices
        rotations = Rotation.from_euler("xyz", rvecs_deg, degrees=True)
        matrices = rotations.as_matrix()  # [T, 3, 3]

        actions = []
        for t in range(len(states) - 1):
            R_t = matrices[t]
            R_t_inv = R_t.T
            R_next = matrices[t + 1]

            # Relative translation in current local frame
            delta_xyz = R_t_inv @ (xyz[t + 1] - xyz[t])

            # Relative rotation in current local frame
            delta_R_mat = R_t_inv @ R_next

            # SVD stabilization for numerical robustness
            u, _, vh = np.linalg.svd(delta_R_mat)
            delta_R_ortho = u @ vh
            if np.linalg.det(delta_R_ortho) < 0:
                u[:, -1] *= -1
                delta_R_ortho = u @ vh

            delta_rvec_rad = Rotation.from_matrix(delta_R_ortho).as_rotvec()
            delta_rvec_deg = np.rad2deg(delta_rvec_rad)

            action = np.concatenate([delta_xyz, delta_rvec_deg])
            actions.append(action)

        return np.array(actions, dtype=np.float32)

    def _read_video_frames(self, session_path: str, indices: np.ndarray) -> np.ndarray:
        """Reads specific frames by index from ultrasound_bmode.mp4."""
        video_path = self.video_path_map[session_path]
        if _HAS_DECORD:
            vr = VideoReader(video_path, ctx=cpu(0))
            frames = vr.get_batch(indices.tolist()).asnumpy()
            return frames
        else:
            cap = cv2.VideoCapture(video_path)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                else:
                    frames.append(np.zeros((256, 256, 3), dtype=np.uint8))
            cap.release()
            return np.stack(frames, axis=0)

    def __len__(self) -> int:
        if self.is_train:
            return len(self.episodes)
        else:
            return len(self.eval_index)

    def __getitem__(self, index: int):
        if self.is_train:
            session_path = self.episodes[index]
            valid_starts = self.valid_starts_map[session_path]
            start_idx = np.random.choice(valid_starts)
        else:
            session_path, start_idx = self.eval_index[index]

        clip_len = self.frames_per_clip * self.frame_step
        indices = np.arange(start_idx, start_idx + clip_len, self.frame_step)

        states = self.states_map[session_path][indices][:: self.frame_skip]
        extrinsics = np.zeros_like(states)[:: self.frame_skip]

        # Read frames: [T, H, W, C]
        frames_np = self._read_video_frames(session_path, indices)

        # Apply transformation: outputs [C, T, H, W]
        if self.transform is not None:
            buffer = self.transform(frames_np)
        else:
            buffer = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()

        # Compute egocentric relative actions: [T-1, 6]
        actions = self.poses_to_diffs(states)

        return buffer, actions, states, extrinsics, indices


def init_data(
    data_root: str,
    batch_size: int,
    frames_per_clip: int = 16,
    frame_skip: int = 1,
    fps: int = 4,
    crop_size: int = 256,
    rank: int = 0,
    world_size: int = 1,
    drop_last: bool = True,
    num_workers: int = 4,
    pin_mem: bool = True,
    persistent_workers: bool = True,
    collator=None,
    transform=None,
    shuffle: bool = True,
    is_train: bool = True,
    pose_source: str = "tip",
    stats_file: Optional[str] = None,
    **kwargs,
):
    dataset = LLProbeGuidanceV2Dataset(
        data_root=data_root,
        frames_per_clip=frames_per_clip,
        frame_skip=frame_skip,
        frames_per_second=fps,
        transform=transform,
        is_train=is_train,
        pose_source=pose_source,
        stats_file=stats_file,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    return data_loader, dist_sampler
