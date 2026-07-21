import os
import numpy as np
from app.vjepa_ll_probe_guidance.ll_probe_guidance import LLProbeGuidanceDataset
import argparse

def calculate_action_stats(data_root):
    # Initialize the dataset just to access its path discovery and math functions
    dataset = LLProbeGuidanceDataset(data_root=data_root, transform=None)
    
    all_actions = []
    all_states = []
    
    for ep_path in dataset.episodes:
        states_path = os.path.join(ep_path, "states_6dof.npy")
        states = np.load(states_path).astype(np.float32)
        
        # We need to compute diffs only for valid consecutive frames
        # to avoid the NaN rotation matrix crash and avoid fake "jump" actions
        for i in range(len(states) - 1):
            if not np.isnan(states[i]).any() and not np.isnan(states[i+1]).any():
                valid_segment = states[i:i+2] 
                
                try:
                    act = dataset.poses_to_diffs(valid_segment)
                    all_actions.append(act)
                    all_states.append(np.expand_dims(states[i], axis=0))
                    all_states.append(np.expand_dims(states[i+1], axis=0))
                except Exception:
                    pass
        
    all_actions = np.concatenate(all_actions, axis=0)
    all_states = np.concatenate(all_states, axis=0)
    
    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0)

    states_mean = np.mean(all_states, axis=0)
    states_std = np.std(all_states, axis=0)
    
    print("ACTION_MEAN = np." + repr(action_mean))
    print("ACTION_STD = np." + repr(action_std))

    print("STATE_MEAN = np." + repr(states_mean))
    print("STATE_STD = np." + repr(states_std))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-train-path", type=str, required=True)
    args = parser.parse_args()
    
    calculate_action_stats(args.dataset_train_path)
