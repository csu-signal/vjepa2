import sys
sys.path.insert(0, "/home/jack/code/vjepa2-probe-guidance/vjepa2")
print(sys.path)

import itertools
import time

import pandas as pd
from torch.utils.data import Subset, DataLoader

from app.vjepa_ll_probe_guidance.ll_probe_guidance import LLProbeGuidanceDataset
from app.vjepa_ll_probe_guidance.utils import init_video_model
from app.vjepa_ll_probe_guidance.transforms import make_transforms
from notebooks.ultrasound_probe_guidance.world_model_wrapper import WorldModel

PARAM_GRID = {
    "rollout": [1, 2, 3],
    "samples": [5, 10, 15],
    "topk": [10],
    "cem_steps": [5, 10, 15],
}
VJEPA2_AC_MODEL_PATH = "/home/jack/code/vjepa2-probe-guidance/vjepa2/outputs/ll_probe_guidance_vitl_4/best.pt"


def init_models():
    encoder, predictor = init_video_model(
        device=device,
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
    
    if os.path.exists(VJEPA2_AC_MODEL_PATH):
        print(f"Loading checkpoint from {VJEPA2_AC_MODEL_PATH}")
        checkpoint = torch.load(VJEPA2_AC_MODEL_PATH, map_location=torch.device("cpu"))
        encoder = load_state_dict_with_ddp_fix(encoder, checkpoint["encoder"])
        predictor = load_state_dict_with_ddp_fix(predictor, checkpoint["predictor"])
        return encoder, predictor
    else:
        print(f"Checkpoint not found at {VJEPA2_AC_MODEL_PATH}")

    return None


def main():

    encoder, predictor = init_models()
    
    np.random.seed(42)
    random_indices = np.random.choice(len(val_dataset), size=250, replace=False).tolist()
    
    fast_subset = Subset(val_dataset, random_indices)
    
    loader = torch.utils.data.DataLoader(
        fast_subset,
        shuffle=False,
        batch_size=1,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
    )
    
    # TODO: think about adding "world model simulation" steps. 
    # So in other words, simulate actually taking actions from the world model for a certain amount of steps.
    # However, the best way to evaluate this is probably to just look at the entire action trajectories from CEM
    # and not just the first action. Does the entire action trajectory actually get us closer to the goal state?
    # The problem with checking the first action is that it makes comparing parameter values choices for the rollout steps
    # not very meaningful. If the model only has 1 action to get to the goal state, It will just try to jump there right away.
    # But if it has many rollout steps available, it might select a bunch of random intermediate movements to "stall for time" almost.
    # Ideally if it has many rollout steps available, it will still find a decent/best path to the goal, and then literally do nothing
    # with the remaining rollout steps once it reaches the goal (i.e. predict a bunch of [0, 0, 0, 0, 0, 0] actions).
    
    compiled_predictor = torch.compile(predictor, dynamic=True)
    
    # unzip back to keys and values separately
    keys, values = zip(*PARAM_GRID.items())
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # topk cannot be greater than # of samples
    valid_configs = [cfg for cfg in all_configs if cfg["topk"] <= cfg["samples"]]
    
    print(f"Total grid combinations to run: {len(valid_configs)}")
    
    results = []
    
    for idx, config in enumerate(valid_configs):
        print(f"[{idx+1}/{len(valid_configs)}] Running config: {config}")
        
        world_model = WorldModel(
            encoder=encoder,
            predictor=compiled_predictor,
            tokens_per_frame=tokens_per_frame,
            mpc_args=config,
            device=device,
        )
        
        # TODO: What is the right way to evaluate the world model's actions? 
        # Is it only important that the world model's predicted representations match the goal representations?
        # For the ultrasound dataset, maybe we just care that the states end up the same. Which is synonymous with
        # reaching the goal state, but only assuming that the patient hasn't moved at all... (i.e. nothing else about the environment changed)
        # NOTE: checking state error should be good for now, but give it more thought!
        inference_latencies_ms = []
        rep_l1_errors = []
        pose_position_errors = []
        
        with torch.no_grad():
            for sample in tqdm(loader, total=len(loader)):
                clips, actions, states, _ = load_clips(sample, device)
                start_image = clips
            
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                h = world_model.encode(clips)
                
                # NOTE: Using only the first two frames in each clip here
                # TODO: Explore how to vary this appropriately (distance based?)
                z_n, z_goal = h[:, :tokens_per_frame], h[:, tokens_per_frame:tokens_per_frame*2]
                s_n, s_goal = states[:, :1], states[:, 1:2]
                
                action_traj, final_rep, final_pose = world_model.evaluate_action_trajectory(
                    rep=z_n, pose=s_n, goal_rep=z_goal
                )
        
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
                inference_latencies_ms.append(elapsed_time_ms)
        
                rep_err = torch.mean(torch.abs(final_rep - z_goal)).item()
                rep_l1_errors.append(rep_err)
    
                pose_error = torch.norm(final_pose[0, 0, :3] - s_goal[0, 0, :3], p=2).item()
                pose_position_errors.append(pose_error)
    
        record = {
            **config,
            "mean_inference_latency_ms": np.mean(inference_latencies_ms),
            "std_inference_latency_ms": np.std(inference_latencies_ms),
            "mean_rep_l1_error": np.mean(rep_l1_errors),
            "std_rep_l1_error": np.std(rep_l1_errors),
            "mean_pose_error_m": np.mean(pose_position_errors),
            "std_pose_error_m": np.std(pose_position_errors),
        }
        results.append(record)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("cem_hyperparameter_search.csv", index=False)


if __name__ == "__main__":
    main()