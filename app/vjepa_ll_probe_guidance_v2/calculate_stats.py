import os
import json
import argparse
import numpy as np

from app.vjepa_ll_probe_guidance_v2.ll_probe_guidance import LLProbeGuidanceV2Dataset


def calculate_action_stats(data_root: str, pose_source: str = "tip", output_json: str = None):
    # Initialize dataset to discover and parse all sessions
    dataset = LLProbeGuidanceV2Dataset(
        data_root=data_root,
        transform=None,
        pose_source=pose_source,
    )
    
    all_actions = []
    all_states = []
    
    for ep_path in dataset.episodes:
        states = dataset.states_map[ep_path]
        
        # Compute diffs for valid consecutive frames
        for i in range(len(states) - 1):
            if not np.isnan(states[i]).any() and not np.isnan(states[i + 1]).any():
                valid_segment = states[i : i + 2]
                try:
                    act = dataset.poses_to_diffs(valid_segment)
                    if len(act) > 0 and np.isfinite(act).all():
                        all_actions.append(act)
                        all_states.append(np.expand_dims(states[i], axis=0))
                        all_states.append(np.expand_dims(states[i + 1], axis=0))
                except Exception:
                    pass
        
    if len(all_actions) == 0:
        print("No valid actions found in dataset!")
        return

    all_actions = np.concatenate(all_actions, axis=0)
    all_states = np.concatenate(all_states, axis=0)
    
    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0)

    states_mean = np.mean(all_states, axis=0)
    states_std = np.std(all_states, axis=0)
    
    print("=" * 60)
    print(f"Calculated statistics over {len(all_actions)} action transitions ({len(dataset.episodes)} sessions):")
    print("ACTION_MEAN = np." + repr(action_mean))
    print("ACTION_STD  = np." + repr(action_std))
    print("STATE_MEAN  = np." + repr(states_mean))
    print("STATE_STD   = np." + repr(states_std))
    print("=" * 60)

    if output_json:
        stats_dict = {
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
            "state_mean": states_mean.tolist(),
            "state_std": states_std.tolist(),
            "num_transitions": int(len(all_actions)),
            "num_sessions": int(len(dataset.episodes)),
            "pose_source": pose_source,
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(stats_dict, f, indent=2)
        print(f"Saved stats JSON to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate action and state normalization statistics for dataset.")
    parser.add_argument("--dataset-train-path", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--pose-source", type=str, default="tip", choices=["tip", "centroid"], help="Pose source to extract")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to output stats JSON")
    args = parser.parse_args()
    
    calculate_action_stats(args.dataset_train_path, pose_source=args.pose_source, output_json=args.output_json)
