import os
import random
import shutil
from pathlib import Path
import argparse


def split_dataset(data_dir, train_ratio=0.8):
    data_path = Path(data_dir)
    test_val_ratio = (1 - train_ratio) / 2
    
    episodes = [d for d in data_path.rglob("*") if d.is_dir() and d.name.startswith("episode_")]
    
    # deterministic shuffle to reproduce results if needed
    random.seed(42)
    random.shuffle(episodes)
    
    train_split_idx = int(len(episodes) * train_ratio)
    test_val_split_idx = train_split_idx + int(len(episodes) * test_val_ratio)
    train_eps = episodes[:train_split_idx]
    test_eps = episodes[train_split_idx:test_val_split_idx]
    val_eps = episodes[test_val_split_idx:]
    
    print(f"Total episodes: {len(episodes)}")
    print(f"Allocating {len(train_eps)} to Train, {len(test_eps)} to Test, {len(val_eps)} to Val.")
    
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    val_dir = data_path / "val"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    print("Moving training episodes...")
    for ep in train_eps:
        shutil.move(str(ep), str(train_dir / ep.name))
        
    print("Moving testing episodes...")
    for ep in test_eps:
        shutil.move(str(ep), str(test_dir / ep.name))

    print("Moving validation episodes...")
    for ep in val_eps:
        shutil.move(str(ep), str(val_dir / ep.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()
    split_dataset(args.dataset_path, train_ratio=0.8)
