# TODO: think about adding "world model simulation" steps. 
# So in other words, simulate actually taking actions from the world model for a certain amount of steps.
# However, the best way to evaluate this is probably to just look at the entire action trajectories from CEM
# and not just the first action. Does the entire action trajectory actually get us closer to the goal state?
# The problem with checking the first action is that it makes comparing parameter values choices for the rollout steps
# not very meaningful. If the model only has 1 action to get to the goal state, It will just try to jump there right away.
# But if it has many rollout steps available, it might select a bunch of random intermediate movements to "stall for time" almost.
# Ideally if it has many rollout steps available, it will still find a decent/best path to the goal, and then literally do nothing
# with the remaining rollout steps once it reaches the goal (i.e. predict a bunch of [0, 0, 0, 0, 0, 0] actions).

# TODO: What is the right way to evaluate the world model's actions? 
# Is it only important that the world model's predicted representations match the goal representations?
# For the ultrasound dataset, maybe we just care that the states end up the same. Which is synonymous with
# reaching the goal state, but only assuming that the patient hasn't moved at all... (i.e. nothing else about the environment changed)


import sys
sys.path.insert(0, "/home/jack/code/vjepa2-probe-guidance/vjepa2")
print(sys.path)

import time
import os
import json
from pathlib import Path
from datetime import datetime
import random

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader

from app.vjepa_ll_probe_guidance.ll_probe_guidance import LLProbeGuidanceDataset
from app.vjepa_ll_probe_guidance.utils import init_video_model
from app.vjepa_ll_probe_guidance.transforms import make_transforms
from notebooks.ultrasound_probe_guidance.world_model_wrapper import WorldModel


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_models(vjepa2_ac_model_path):
    encoder, predictor = init_video_model(
        device=DEVICE,
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
            # Remove 'module.' prefix if it exists
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
    
        model.load_state_dict(new_state_dict, strict=True)
        return model
    
    if os.path.exists(vjepa2_ac_model_path):
        print(f"Loading checkpoint from {vjepa2_ac_model_path}")
        checkpoint = torch.load(vjepa2_ac_model_path, map_location=torch.device("cpu"))
        encoder = load_state_dict_with_ddp_fix(encoder, checkpoint["encoder"])
        predictor = load_state_dict_with_ddp_fix(predictor, checkpoint["predictor"])
        return encoder, predictor
    else:
        print(f"Checkpoint not found at {vjepa2_ac_model_path}")

    return None


def load_clips(sample, device):
    clips = sample[0].to(device, non_blocking=True)  # [B C T H W]
    actions = sample[1].to(device, non_blocking=True)  # [B T-1 6]
    states = sample[2].to(device, non_blocking=True)  # [B T 6]
    extrinsics = sample[3].to(device, non_blocking=True)  # [B T 6]
    return (clips, actions, states, extrinsics)


def main():
    # Load config parameters
    with open("test_cem_config.json", "r") as f:
        config = json.load(f)

    data_root = config["data_root"]
    cem_config = config["cem_config"]
    T = config["clip_size"]
    crop_size = config["crop_size"]
    vjepa2_ac_model_path = config["vjepa2_ac_model_path"]

    # Set randomization seeds
    random.seed(42)    
    np.random.seed(42)

    # Load models
    encoder, predictor = init_models(vjepa2_ac_model_path)
    compiled_predictor = torch.compile(predictor, dynamic=True)

    tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
    
    # Load dataset and dataloader
    test_dataset = LLProbeGuidanceDataset(
        data_root=data_root,
        frames_per_clip=T,
        frame_skip=1,
        frames_per_second=4,
        transform=make_transforms(crop_size=crop_size),
        is_train=False
    )

#    random_indices = np.random.choice(len(test_dataset), size=2500, replace=False).tolist()
#    fast_subset = Subset(test_dataset, random_indices)
    
    loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
    )
   
    # Get starting time, also useful for storing results later
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Starting CEM hyperparameter search at: {timestamp}")

    world_model = WorldModel(
        encoder=encoder,
        predictor=compiled_predictor,
        tokens_per_frame=tokens_per_frame,
        mpc_args=cem_config,
        device=DEVICE,
    )
    
    inference_latencies_ms = []
    rep_l1_errors = []
    pose_position_errors = []
    
    with torch.no_grad():
        for sample in tqdm(loader, total=len(loader)):
            clips, actions, states, _ = load_clips(sample, DEVICE)
            start_image = clips
        
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            h = world_model.encode(clips)

            # NOTE: randomizing distance from starting frame/state to goal frame/state
            # TODO: test on all possible start frame and end frame pairs?
            start_index = random.randint(0, 6)
            end_index = random.randint(start_index+1, 7)

            #print(f"state => {start_index=}, {end_index=}")

            context_start_token = start_index * tokens_per_frame
            context_end_token = (start_index + 1) * tokens_per_frame

            #print(f"context => {context_start_token=}, {context_end_token=}")

            goal_start_token = end_index * tokens_per_frame
            goal_end_token = (end_index + 1) * tokens_per_frame

            #print(f"goal => {goal_start_token=}, {goal_end_token=}")

            z_n, z_goal = h[:, context_start_token:context_end_token], h[:, goal_start_token:goal_end_token]
            s_n, s_goal = states[:, start_index:start_index+1], states[:, end_index:end_index+1]
            
            action_traj, final_rep, final_pose = world_model.evaluate_action_trajectory(
                rep=z_n, pose=s_n, goal_rep=z_goal
            )
    
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
            inference_latencies_ms.append(elapsed_time_ms)
    
            rep_err = torch.mean(torch.abs(final_rep - z_goal)).item()
            rep_l1_errors.append(rep_err)

            pose_error = torch.norm(final_pose[0, 0, :3] - s_goal[0, 0, :3], p=2).item()
            pose_position_errors.append(pose_error)

    mean_inference_latency_ms = np.mean(inference_latencies_ms)
    std_inference_latency_ms = np.std(inference_latencies_ms)
    mean_rep_l1_error = np.mean(rep_l1_errors)
    std_rep_l1_error = np.std(rep_l1_errors)
    mean_pose_error_m = np.mean(pose_position_errors)
    std_pose_error_m = np.std(pose_position_errors)
    
    # TODO save results to csv?
    print(f"RESULTS FOR CEM CONFIG: {cem_config=}")
    print(f"\t{mean_inference_latency_ms=}, {std_inference_latency_ms=}")
    print(f"\t{mean_rep_l1_error=}, {std_rep_l1_error=}")
    print(f"\t{mean_pose_error_m=}, {std_pose_error_m=}")


if __name__ == "__main__":
    main()
