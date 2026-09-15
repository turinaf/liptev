import argparse
import glob
import json
import os
import random


def create_digit_splits(config_path="config.json", processed_root=None):
    with open(config_path, "r") as f:
        config = json.load(f)

    out_dir = processed_root or config.get("digit_processed_data_dir", "./data/digit")

    if not os.path.exists(out_dir):
        print(f"Processed digit data directory not found: {out_dir}")
        return

    # Match the real project layout:
    # ./data/digit/subset_X/speaker/video/*.mp4
    search_pattern = os.path.join(out_dir, "**", "video", "*.mp4")
    all_videos = sorted(glob.glob(search_pattern, recursive=True))

    if not all_videos:
        print(f"No digit videos found. Checked pattern: {search_pattern}")
        print("Expected structure: <digit_processed_data_dir>/subset_X/speaker/video/*.mp4")
        return

    all_videos = [v.replace("\\", "/") for v in all_videos]

    random.seed(42)
    random.shuffle(all_videos)

    total_size = len(all_videos)
    test_size = max(1, int(total_size * 0.1))
    val_size = max(1, int(total_size * 0.1))
    train_size = total_size - val_size - test_size

    if train_size <= 0:
        train_size = max(1, total_size - val_size)

    train_videos = all_videos[:train_size]
    val_videos = all_videos[train_size:train_size + val_size]
    test_videos = all_videos[train_size + val_size:]

    with open(os.path.join(out_dir, "train_split.json"), "w") as f:
        json.dump(train_videos, f)
    with open(os.path.join(out_dir, "val_split.json"), "w") as f:
        json.dump(val_videos, f)
    with open(os.path.join(out_dir, "test_split.json"), "w") as f:
        json.dump(test_videos, f)

    print(f"Data split saved in {out_dir}:")
    print(f" - Train: {len(train_videos)}")
    print(f" - Val:   {len(val_videos)}")
    print(f" - Test:  {len(test_videos)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val/test splits for processed digit data.")
    parser.add_argument("--config", default="config.json", help="Path to the JSON config file")
    parser.add_argument(
        "--processed-root",
        default=None,
        help="Optional override for digit_processed_data_dir (defaults to config value)",
    )
    args = parser.parse_args()

    create_digit_splits(config_path=args.config, processed_root=args.processed_root)
