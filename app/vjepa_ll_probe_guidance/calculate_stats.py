import os
import sys
import json
import argparse
import numpy as np

# Ensure vjepa2 parent directory is in sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.vjepa_ll_probe_guidance.ll_probe_guidance import LLProbeGuidanceDataset


def calculate_probe_local_stats(
    data_root: str,
    pose_source: str = "centroid",
    probe_tip_offset=None,
    fps: int = 4,
    frames_per_clip: int = 16,
    output_json: str = None,
):
    dataset = LLProbeGuidanceDataset(
        data_root=data_root,
        frames_per_clip=frames_per_clip,
        frames_per_second=fps,
        transform=None,
        pose_source=pose_source,
        probe_tip_offset=probe_tip_offset,
        is_train=True,
    )

    all_actions = []
    all_local_states = []

    print(f"Extracting probe-local actions and states across {len(dataset.episodes)} sessions...")
    for session_path in dataset.episodes:
        valid_starts = dataset.valid_starts_map[session_path]
        states_all = dataset.states_map[session_path]

        for start_idx in valid_starts:
            indices = np.arange(start_idx, start_idx + dataset.frames_per_clip * dataset.frame_step, dataset.frame_step)
            states_clip = states_all[indices]

            act = dataset.poses_to_diffs(states_clip)
            local_st = dataset.poses_to_local_states(states_clip)

            if len(act) > 0 and np.isfinite(act).all() and np.isfinite(local_st).all():
                all_actions.append(act)
                all_local_states.append(local_st)

    if len(all_actions) == 0:
        print("No valid actions found in dataset!")
        return

    all_actions = np.concatenate(all_actions, axis=0)
    all_local_states = np.concatenate(all_local_states, axis=0)

    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0)

    state_mean = np.mean(all_local_states, axis=0)
    state_std = np.std(all_local_states, axis=0)

    print("=" * 70)
    print(f"Calculated statistics in PROBE LOCAL FRAME over {len(all_actions)} actions ({len(dataset.episodes)} sessions):")
    print("ACTION_MEAN = np." + repr(action_mean))
    print("ACTION_STD  = np." + repr(action_std))
    print("STATE_MEAN  = np." + repr(state_mean))
    print("STATE_STD   = np." + repr(state_std))
    print("=" * 70)

    if output_json:
        stats_dict = {
            "coordinate_frame": "probe_local",
            "pose_source": pose_source,
            "probe_tip_offset": dataset.probe_tip_offset.tolist() if dataset.probe_tip_offset is not None else None,
            "fps": fps,
            "frames_per_clip": frames_per_clip,
            "num_actions": int(len(all_actions)),
            "num_states": int(len(all_local_states)),
            "num_sessions": int(len(dataset.episodes)),
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(stats_dict, f, indent=2)
        print(f"Saved stats JSON to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate normalization stats in probe local coordinate frame.")
    parser.add_argument("--dataset-train-path", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--pose-source", type=str, default="centroid", choices=["centroid", "tip"], help="Pose source")
    parser.add_argument(
        "--probe-tip-offset",
        type=float,
        nargs=6,
        default=None,
        help="6D probe tip offset [x y z roll pitch yaw] in meters and degrees",
    )
    parser.add_argument("--fps", type=int, default=4, help="Target frames per second")
    parser.add_argument("--frames-per-clip", type=int, default=16, help="Frames per clip")
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    calculate_probe_local_stats(
        args.dataset_train_path,
        pose_source=args.pose_source,
        probe_tip_offset=args.probe_tip_offset,
        fps=args.fps,
        frames_per_clip=args.frames_per_clip,
        output_json=args.output_json,
    )
