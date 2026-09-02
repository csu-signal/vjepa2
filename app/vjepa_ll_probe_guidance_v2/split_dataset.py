import os
import random
import shutil
import argparse
from pathlib import Path


def split_dataset(data_dir: str, train_ratio: float = 0.8, action: str = "move", seed: int = 42):
    data_path = Path(data_dir)
    test_val_ratio = (1.0 - train_ratio) / 2.0
    
    # Find all session directories containing trajectory.csv and ultrasound_bmode.mp4
    sessions = []
    for d in data_path.iterdir():
        if d.is_dir() and d.name not in ("train", "val", "test"):
            if (d / "trajectory.csv").exists() and (d / "ultrasound_bmode.mp4").exists():
                sessions.append(d)
    
    sessions = sorted(sessions)
    if len(sessions) == 0:
        print(f"No valid session directories (containing trajectory.csv and ultrasound_bmode.mp4) found directly under {data_dir}")
        return

    random.seed(seed)
    random.shuffle(sessions)
    
    train_split_idx = int(len(sessions) * train_ratio)
    test_val_split_idx = train_split_idx + int(len(sessions) * test_val_ratio)
    
    train_eps = sessions[:train_split_idx]
    test_eps = sessions[train_split_idx:test_val_split_idx]
    val_eps = sessions[test_val_split_idx:]
    
    print(f"Total sessions found: {len(sessions)}")
    print(f"Allocating {len(train_eps)} to Train, {len(test_eps)} to Test, {len(val_eps)} to Val.")
    
    if action == "dry-run":
        print("[Dry-run complete. No files moved.]")
        return

    train_dir = data_path / "train"
    test_dir = data_path / "test"
    val_dir = data_path / "val"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    transfer_func = shutil.move if action == "move" else shutil.copytree

    print(f"Processing training sessions ({action})...")
    for ep in train_eps:
        transfer_func(str(ep), str(train_dir / ep.name))
        
    print(f"Processing testing sessions ({action})...")
    for ep in test_eps:
        transfer_func(str(ep), str(test_dir / ep.name))

    print(f"Processing validation sessions ({action})...")
    for ep in val_eps:
        transfer_func(str(ep), str(val_dir / ep.name))

    print("Dataset split complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset sessions into train, test, and val subdirectories.")
    parser.add_argument("--dataset-path", required=True, help="Path to base dataset directory containing sessions")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of sessions to allocate to train (default: 0.8)")
    parser.add_argument("--action", type=str, default="move", choices=["move", "copy", "dry-run"], help="Action to perform (default: move)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    
    split_dataset(args.dataset_path, train_ratio=args.train_ratio, action=args.action, seed=args.seed)
