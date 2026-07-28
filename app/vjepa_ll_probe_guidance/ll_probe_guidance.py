import os
import glob
import math

import torch
import numpy as np
import torchvision
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm

# Assume avg. FPS is 15 for all of the ultrasound videos we collected
AVG_FPS = 15

class LLProbeGuidanceDataset(Dataset):
    def __init__(
        self,
        data_root,
        frames_per_clip=16,
        frames_per_second=4,
        frame_skip=2, 
        transform=None,
        is_train=True
    ):
        self.data_root = data_root
        self.frames_per_clip = frames_per_clip
        self.frames_per_second = frames_per_second
        self.frame_skip = frame_skip
        self.transform = transform
        self.is_train = is_train

        raw_episodes = sorted([
            os.path.join(data_root, d) for d in os.listdir(data_root)
            if d.startswith("episode_") and os.path.isdir(os.path.join(data_root, d))
        ])

        self.valid_starts_map = {}
        
        self.frame_step = math.ceil(AVG_FPS / self.frames_per_second)
        clip_len = self.frames_per_clip * self.frame_step

        print(f"Scanning {len(raw_episodes)} episodes for valid tracking clips...")
        for ep_path in tqdm(raw_episodes, total=len(raw_episodes)):
            tracking_path = os.path.join(ep_path, "tracking_status.npy")
            if not os.path.exists(tracking_path):
                continue
            
            tracking = np.load(tracking_path)
            
            # TODO: What's the point of this? Sanity check that tracking length = total images?
            us_dir = os.path.join(ep_path, "ultrasound")
            num_images = len(glob.glob(os.path.join(us_dir, "*.jpg")))
            total_frames = min(len(tracking), num_images)

            if len(tracking) != num_images:
                print(f"tracking status doesn't match number of images in {ep_path}")

            if total_frames < clip_len:
                continue

            valid_starts_for_this_ep = []
            
            # Slide window to find valid starting indices
            for start_idx in range(total_frames - clip_len + 1):
                indices = np.arange(start_idx, start_idx + clip_len, self.frame_step)
                clip_tracking = tracking[indices]

                # Keep if tracked (1) or interpolated (2)
                if np.all((clip_tracking == 1) | (clip_tracking == 2)):
                    valid_starts_for_this_ep.append(start_idx)
            
            # Only store episodes that actually have at least one valid clip
            if len(valid_starts_for_this_ep) > 0:
                self.valid_starts_map[ep_path] = valid_starts_for_this_ep

        self.episodes = list(self.valid_starts_map.keys())
        print(f"Retained {len(self.episodes)} episodes.")

        # FOR TESTING: WE NEED TO EVALUATE ON EVERY CLIP
        if not self.is_train:
            self.eval_index = []
            for ep in self.episodes:
                valid_starts = self.valid_starts_map[ep]
                for start in valid_starts:
                    self.eval_index.append((ep, start))

    def poses_to_diffs(self, states):
        """ Converts 6-DoF absolute states into Egocentric Actions. """
        xyz = states[:, :3]
        rvecs_deg = states[:, 3:]
    
        rvecs_rad = np.deg2rad(rvecs_deg)
        rotations = Rotation.from_rotvec(rvecs_rad)
        matrices = rotations.as_matrix()  # [T, 3, 3]

        actions = []
        for t in range(len(states) - 1):
            R_t = matrices[t]
            R_t_inv = R_t.T
            R_next = matrices[t + 1]

            delta_xyz = R_t_inv @ (xyz[t + 1] - xyz[t])
            delta_R_mat = R_t_inv @ R_next
            try:
                delta_rvec_rad = Rotation.from_matrix(delta_R_mat).as_rotvec()
                delta_rvec_deg = np.rad2deg(delta_rvec_rad)
            except Exception as e:
                print(f"Error: {e}")
                print(f"R_t_inv (Current Pose Inverse):\n{R_t_inv}")
                print(f"R_next (Next Pose):\n{R_next}")
                print(f"delta_R_mat (The Squashed Result):\n{delta_R_mat}")
                raise e

            action = np.concatenate([delta_xyz, delta_rvec_deg])
            actions.append(action)

        actions = np.array(actions, dtype=np.float32)

        return actions

    def __len__(self):
        if self.is_train:
            return len(self.episodes)
        else:
            return len(self.eval_index)

    def __getitem__(self, index):
        if self.is_train:
            episode_path = self.episodes[index]
            valid_starts = self.valid_starts_map[episode_path]
            start_idx = np.random.choice(valid_starts)
        else:
            episode_path, start_idx = self.eval_index[index]

        clip_len = self.frames_per_clip * self.frame_step
        indices = np.arange(start_idx, start_idx + clip_len, self.frame_step)

        states_path = os.path.join(episode_path, "states_6dof.npy")
        states = np.load(states_path).astype(np.float32)
        states = states[indices][:: self.frame_skip]

        # We don't have camera intrinsics, but keep to reuse collator for DROID
        extrinsics = np.zeros_like(states)[:: self.frame_skip]

        us_dir = os.path.join(episode_path, "ultrasound")
        
        clip_images = []
        for i in indices:
            # NOTE: frame name format is hardcoded here. it's 5 digits with leading zeros for each frame
            img_path = os.path.join(us_dir, f"{i:05d}.jpg")
            img = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
            clip_images.append(img)

        buffer = torch.stack(clip_images).permute(0, 2, 3, 1).numpy()

        if self.transform is not None:
            buffer = self.transform(buffer)

        # NOTE: Doing state standardization BEFORE calculating actions is incorrect. It must happen AFTER
        actions = self.poses_to_diffs(states)

        return buffer, actions, states, extrinsics, indices


def standardize_states(states):
    # TODO: don't hardcode standardization values, find better way
    state_mean = np.array([ -0.13193196, 0.13471916, 1.4406046, 
        59.395, -43.585613, -76.49974], dtype=np.float32)
    state_std = np.array([1.05345368e-01, 1.75726220e-01, 7.28756189e-02,
        7.79652252e+01, 3.07379379e+01, 1.22978004e+02], dtype=np.float32)
    
    return (states - state_mean) / (state_std + 1e-6)


def standardize_actions(actions):
    # TODO: don't hardcode standardization values, find better way
    action_mean = np.array([-2.7906366e-05,  5.4028669e-06,  6.0837847e-05,
        -1.7707590e-03, 2.4102912e-03, -3.7514704e-04], dtype=np.float32)
    action_std = np.array([0.00498742, 0.00453256, 0.00675807,
        0.55955255, 0.43809873, 0.73082167], dtype=np.float32)

    return (actions - action_mean) / (action_std + 1e-6)


def init_data(
    data_root,
    batch_size,
    frames_per_clip=16,
    frame_skip=1,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    shuffle=True,
    is_train=True,
    **kwargs # Catch any extra DROID args we don't use
):
    dataset = LLProbeGuidanceDataset(
        data_root=data_root,
        frames_per_clip=frames_per_clip,
        frame_skip=frame_skip,
        frames_per_second=fps,
        transform=transform,
        is_train=is_train
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
