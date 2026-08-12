import torch
from tqdm import tqdm

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

# NOTE: state and action may be standardized, may need to account for this or rethink dataset/dataloader
# to not do standardization. Perhaps standardization should happen in training/testing and before passing to model
# forward functions (or within the forward functions themselves). doing it in dataset makes it hard to do visualizations.
# breaking standardization out can be good from a single-responsibility principle standpoint.

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
    """
    Pure PyTorch GPU replacement for your SciPy pose integration.
    
    :param pose:   [B, T=1, 6] (x, y, z, rx, ry, rz in Euler degrees)
    :param action: [B, T=1, 6] (dx, dy, dz, drx, dry, drz)
    :return:      [B, T=1, 6] updated pose on GPU
    """
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