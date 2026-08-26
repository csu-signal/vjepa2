import sys
sys.path.insert(0, "/home/jack/code/vjepa2-probe-guidance/vjepa2")

from pathlib import Path
import copy
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.nn import functional as F
import torchvision.transforms.v2 as T
from torchvision.io import read_image, ImageReadMode
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import cv2

from app.vjepa_ll_probe_guidance.utils import init_video_model
from app.vjepa_ll_probe_guidance.transforms import make_transforms


STATE_MEAN = torch.tensor([-0.13193196, 0.13471916, 1.4406046, 59.395, -43.585613, -76.49974], dtype=torch.float32, device="cuda:0")
STATE_STD = torch.tensor([1.05345368e-01, 1.75726220e-01, 7.28756189e-02, 7.79652252e+01, 3.07379379e+01, 1.22978004e+02], dtype=torch.float32, device="cuda:0")

ACTION_MEAN = torch.tensor([-2.7906366e-05, 5.4028669e-06, 6.0837847e-05, -1.7707590e-03, 2.4102912e-03, -3.7514704e-04], dtype=torch.float32, device="cuda:0")
ACTION_STD = torch.tensor([0.00498742, 0.00453256, 0.00675807, 0.55955255, 0.43809873, 0.73082167], dtype=torch.float32, device="cuda:0")


def standardize_states(states):
    return (states - STATE_MEAN) / (STATE_STD + 1e-6)


def standardize_actions(actions):
    return (actions - ACTION_MEAN) / (ACTION_STD + 1e-6)


def l1(a, b):
    return torch.mean(torch.abs(a - b), dim=-1)


def round_small_elements(tensor, threshold):
    mask = torch.abs(tensor) < threshold
    new_tensor = tensor.clone()
    new_tensor[mask] = 0
    return new_tensor


def cem(
    context_frame,
    context_pose,
    goal_frame,
    world_model,
    rollout=1,
    cem_steps=100,
    momentum_mean=0.25,
    momentum_std=0.95,
    samples=100,
    topk=10,
    verbose=False,
    maxnorm=0.05,
    axis={},
    objective=l1,
    device="cpu"
):
    """
    :param context_frame: [B=1, T=1, HW, D]
    :param goal_frame: [B=1, T=1, HW, D]
    :param world_model: f(context_frame, action) -> next_frame [B, 1, HW, D]
    :return: [B=1, rollout, 7] an action trajectory over rollout horizon

    Cross-Entropy Method
    -----------------------
    1. for rollout horizon:
    1.1. sample several actions
    1.2. compute next states using WM
    3. compute similarity of final states to goal_frames
    4. select topk samples and update mean and std using topk action trajs
    5. choose final action to be mean of distribution
    """
    context_frame = context_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    goal_frame = goal_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    context_pose = context_pose.repeat(samples, 1, 1)  # Reshape to [S, 1, 7]

    # Current estimate of the mean/std of distribution over action trajectories
    # Start with a normal distribution (mean=0 and std=1)
    mean = torch.zeros((rollout, 3), device=device)
    std = torch.ones((rollout, 3), device=device) * maxnorm

    for ax in axis.keys():
        mean[:, ax] = axis[ax]

    def sample_action_traj():
        """Sample several action trajectories"""
        action_traj, frame_traj, pose_traj = None, context_frame, context_pose

        for h in range(rollout):

            # -- sample new action
            action_samples = torch.randn(samples, mean.size(1), device=device) * std[h] + mean[h]
            action_samples[:, :3] = torch.clip(action_samples[:, :3], min=-maxnorm, max=maxnorm)

            for ax in axis.keys():
                action_samples[:, ax] = axis[ax]

                
            action_samples = torch.cat(
                [
                    action_samples[:, :3],
                    # NOTE: This is ignoring rotation from my understanding, undo this???
                    torch.zeros((len(action_samples), 3), device=device),
                ],
                dim=-1,
            )[:, None]

            action_traj = (
                torch.cat([action_traj, action_samples], dim=1) if action_traj is not None else action_samples
            )

            # -- compute next state
            next_frame, next_pose = world_model(frame_traj, action_traj, pose_traj)
            frame_traj = torch.cat([frame_traj, next_frame], dim=1)
            pose_traj = torch.cat([pose_traj, next_pose], dim=1)

        return action_traj, frame_traj, pose_traj

    def select_topk_action_traj(final_state, goal_state, actions):
        """Get the topk action trajectories that bring us closest to goal"""
        sims = objective(final_state.flatten(1), goal_state.flatten(1))
        indices = sims.topk(topk, largest=False).indices
        selected_actions = actions[indices]
        return selected_actions

    for step in tqdm(range(cem_steps), disable=True):
        action_traj, frame_traj, _ = sample_action_traj()
        selected_actions = select_topk_action_traj(
            final_state=frame_traj[:, -1], goal_state=goal_frame, actions=action_traj
        )
        mean_selected_actions = selected_actions.mean(dim=0)
        std_selected_actions = selected_actions.std(dim=0)

        # -- Update new sampling mean and std based on the top-k samples
        mean = torch.cat(
            [
                mean_selected_actions[..., :3] * (1.0 - momentum_mean) + mean[..., :3] * momentum_mean,
            ],
            dim=-1,
        )
        std = torch.cat(
            [
                std_selected_actions[..., :3] * (1.0 - momentum_std) + std[..., :3] * momentum_std,
            ],
            dim=-1,
        )

        if verbose:
            print(f"Step {step+1} | Mean Actions:\n{mean.detach().cpu().numpy()}")
            print(f"         | Std Actions:\n{std.detach().cpu().numpy()}\n")

    final_action_traj = torch.cat(
        [
            mean[..., :3],
            torch.zeros((rollout, 3), device=device),
        ],
        dim=-1,
    )[None, :]

    # Evaluate the final mean action trajectory once
    single_context_frame = context_frame[:1]
    single_context_pose = context_pose[:1]
    
    curr_frame, curr_pose = single_context_frame, single_context_pose
    for step in range(rollout):
        step_action = final_action_traj[:, :step+1]
        next_frame, next_pose = world_model(curr_frame, step_action, curr_pose)
        curr_frame = torch.cat([curr_frame, next_frame], dim=1)
        curr_pose = torch.cat([curr_pose, next_pose], dim=1)

    final_frame = curr_frame[:, -1:]  # Final latent frame [1, 1, HW, D]
    final_pose = curr_pose[:, -1:]    # Final pose [1, 1, 6]

    return final_action_traj, final_frame, final_pose


# NOTE: the actions are in the probe's coordinate frame, while the states are 
# in the camera's coordinate frame. Does this have a negative impact on the model?
# Nonetheless, computing the new pose needs to account for this discrepancy.


def euler_to_matrix_xyz_degrees(euler_deg):
    """
    Converts [B, 3] Euler angles in degrees (order 'xyz') to [B, 3, 3] rotation matrices.
    Runs 100% on CUDA in pure PyTorch.
    """
    rad = torch.deg2rad(euler_deg)
    cx, cy, cz = torch.cos(rad[:, 0]), torch.cos(rad[:, 1]), torch.cos(rad[:, 2])
    sx, sy, sz = torch.sin(rad[:, 0]), torch.sin(rad[:, 1]), torch.sin(rad[:, 2])

    # Construct Extrinsic XYZ rotation matrix
    R = torch.stack([
        cy * cz, -cy * sz, sy,
        cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx,
        sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy
    ], dim=-1).reshape(-1, 3, 3)
    
    return R


def matrix_to_euler_xyz_degrees(R):
    """
    Converts [B, 3, 3] rotation matrices back to [B, 3] Euler angles in degrees ('xyz').
    """
    sy = R[:, 0, 2]
    # Clamp to avoid numerical precision errors near gimbal lock
    sy = torch.clamp(sy, -0.999999, 0.999999)
    
    cy = torch.sqrt(1.0 - sy ** 2)
    
    x = torch.atan2(-R[:, 1, 2], R[:, 2, 2])
    y = torch.atan2(sy, cy)
    z = torch.atan2(-R[:, 0, 1], R[:, 0, 0])

    euler_rad = torch.stack([x, y, z], dim=-1)
    return torch.rad2deg(euler_rad)


def compute_new_pose(pose, action):
    device, dtype = pose.device, pose.dtype

    pose_vec = pose[:, -1]     # [B, 6]
    action_vec = action[:, -1] # [B, 6]

    # 1. Position update: transform local dxyz by pose rotation matrix
    R_pose_mat = euler_to_matrix_xyz_degrees(pose_vec[:, 3:6]) # [B, 3, 3]
    local_dxyz = action_vec[:, :3].unsqueeze(-1)               # [B, 3, 1]
    global_dxyz = torch.bmm(R_pose_mat, local_dxyz).squeeze(-1) # [B, 3]
    new_xyz = pose_vec[:, :3] + global_dxyz

    # 2. Rotation update: composite matrix multiplication R_new = R_pose * R_action
    R_action_mat = euler_to_matrix_xyz_degrees(action_vec[:, 3:6]) # [B, 3, 3]
    R_new_mat = torch.bmm(R_pose_mat, R_action_mat)               # [B, 3, 3]
    new_angles = matrix_to_euler_xyz_degrees(R_new_mat)          # [B, 3]

    new_pose = torch.cat([new_xyz, new_angles], dim=-1).to(device=device, dtype=dtype) # [B, 6]
    return new_pose[:, None, :] # Restore shape [B, T=1, 6]


class WorldModel(object):
    def __init__(
        self,
        encoder,
        predictor,
        tokens_per_frame,
        mpc_args={
            "rollout": 2,
            "samples": 400,
            "topk": 10,
            "cem_steps": 10,
            "momentum_mean": 0.15,
            "momentum_std": 0.15,
            "maxnorm": 0.05,
            "verbose": True,
        },
        normalize_reps=True,
        device="cpu",
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.normalize_reps = normalize_reps
        self.tokens_per_frame = tokens_per_frame
        self.device = device
        self.mpc_args = mpc_args

    def encode(self, c):
        c = c[None, :, None, ...].repeat(1, 1, 2, 1, 1)
        h = self.encoder(c)
        h = h.view(1, 1, -1, h.size(-1)).flatten(1, 2)
        if self.normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h
        
    def step_predictor(self, reps, actions, poses):
        B, T, N_T, D = reps.size()
        reps = reps.flatten(1, 2)
        standardized_poses = standardize_states(poses)
        standardized_actions = standardize_actions(actions)
        
        next_rep = self.predictor(reps, standardized_actions, standardized_poses)[:, -self.tokens_per_frame:]
        if self.normalize_reps:
            next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
        next_rep = next_rep.view(B, 1, N_T, D)
        next_pose = compute_new_pose(poses[:, -1:], actions[:, -1:])
        return next_rep, next_pose

    def evaluate_action_trajectory(self, rep, pose, goal_rep):
        action_traj, final_rep, final_pose = cem(
            context_frame=rep,
            context_pose=pose,
            goal_frame=goal_rep,
            world_model=self.step_predictor,
            device=self.device,
            **self.mpc_args,
        )

        return action_traj, final_rep, final_pose
        
    def infer_next_action(self, rep, pose, goal_rep, close_gripper=None):
        action_traj, _, _ = cem(
            context_frame=rep,
            context_pose=pose,
            goal_frame=goal_rep,
            world_model=self.step_predictor,
            device=self.device,
            **self.mpc_args,
        )
        
        # Return the first action of the trajectory
        return action_traj[0]


def forward_target(encoder, c, normalize_reps=True):
    c = c[None, :, None, ...].repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(1, 1, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


def main():
    # TODO: visualize the world model's rollouts? maybe just the final predicted representation?
    encoder, predictor = init_video_model(
            device="cuda:0",
            patch_size=16,
            max_num_frames=512,
            tubelet_size=2,
            model_name="vit_large",
            crop_size=256,
            pred_depth=12,
            pred_num_heads=12,
            pred_embed_dim=768,
            action_embed_dim=6,
            predictor_type="ac",
            pred_is_frame_causal=True,
            use_extrinsics=False,
            use_sdpa=True,
            use_rope=True
        )

    encoder.eval()
    predictor.eval()

    def load_state_dict_with_ddp_fix(model, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict, strict=True)
        return model

    resume_path = os.path.join("/home/jack/code/vjepa2-probe-guidance/vjepa2/outputs/ll_probe_guidance_vitl_4", "best.pt")
    if os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=torch.device("cpu"))
        encoder = load_state_dict_with_ddp_fix(encoder, checkpoint["encoder"])
        predictor = load_state_dict_with_ddp_fix(predictor, checkpoint["predictor"])
    else:
        print(f"Checkpoint not found at {resume_path}")

    crop_size = 256
    tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)

    ultrasound_transform = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Resize((crop_size, crop_size), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    display_resize = T.Resize((crop_size, crop_size), antialias=True)

    data_root = "/home/jack/data/probe_guidance_dataset_june/test"
    episode_paths = sorted([
        os.path.join(data_root, d) for d in os.listdir(data_root)
        if d.startswith("episode_") and os.path.isdir(os.path.join(data_root, d))
    ])

    episode_path = episode_paths[0]

    states_path = os.path.join(episode_path, "states_6dof.npy")
    states = np.load(states_path).astype(np.float32)

    rgb_dir = os.path.join(episode_path, "realsense_rgb")
    us_dir = os.path.join(episode_path, "ultrasound")

    rgb_file_count = sum(1 for item in Path(rgb_dir).iterdir() if item.is_file())
    us_file_count = sum(1 for item in Path(us_dir).iterdir() if item.is_file())

    assert rgb_file_count == us_file_count == len(states)

    world_model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        mpc_args={
            "rollout": 1,
            "samples": 10,
            "topk": 10,
            "cem_steps": 10,
            "momentum_mean": 0.15,
            "momentum_std": 0.75,
            "maxnorm": 0.075,
            "verbose": False
        },
        normalize_reps=True,
        device="cuda:0"
    )

    curr_state = None
    goal_img = None

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in tqdm(range(us_file_count), total=us_file_count):
            us_img_path = os.path.join(us_dir, f"{i:05d}.jpg")
            us_raw = read_image(us_img_path, mode=ImageReadMode.RGB).to("cuda:0", non_blocking=True)
            curr_wm_us_img = ultrasound_transform(us_raw)

            rgb_img_path = os.path.join(rgb_dir, f"{i:05d}.jpg")
            rgb_raw = read_image(rgb_img_path, mode=ImageReadMode.RGB).to("cuda:0", non_blocking=True)

            gt_us_img = display_resize(us_raw).permute(1, 2, 0).cpu().numpy()
            rgb_img = display_resize(rgb_raw).permute(1, 2, 0).cpu().numpy()

            if goal_img is not None:
                curr_rep = world_model.encode(curr_wm_us_img)
                goal_rep = world_model.encode(goal_img)

                curr_state = states[i] if not np.any(np.isnan(states[i])) else None
                if curr_state is not None:
                    curr_state = torch.from_numpy(curr_state).to("cuda:0")
                    # Do world model step
                    wm_actions = world_model.infer_next_action(curr_rep, curr_state, goal_rep).cpu().numpy()
                    print(f"World model predicted action on frame {i}: {wm_actions}")

            combined_view = np.hstack([rgb_img, gt_us_img])
            display_large = cv2.resize(combined_view, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
            cv2.imshow("VJEPA2 Probe Guidance Demo: RGB | Ground Truth US", display_large[..., ::-1])
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('g'):
                goal_img = curr_wm_us_img

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
