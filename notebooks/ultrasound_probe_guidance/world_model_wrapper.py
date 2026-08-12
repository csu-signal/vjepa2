import torch
from torch.nn import functional as F

from app.vjepa_ll_probe_guidance.ll_probe_guidance import standardize_actions, standardize_states
from notebooks.ultrasound_probe_guidance.mpc_utils import compute_new_pose, cem


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

    def encode(self, clips):
        B, C, T, H, W = clips.size()
        clips = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        clips = clips.to(self.device, non_blocking=True)
        h = self.encoder(clips)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
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