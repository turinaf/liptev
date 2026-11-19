import os
import glob
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone


def partition_videos(video_paths, dataset):
    paths = sorted(str(path).replace('\\', '/') for path in video_paths)
    if dataset == "grid":
        groups = sorted({path.split('/')[-3] for path in paths})
        if len(groups) < 3:
            raise ValueError("GRID speaker-disjoint splitting requires at least 3 speakers")
    elif dataset == "digit":
        subset_counts = Counter(path.split('/')[-4] for path in paths)
        if len(subset_counts) < 2:
            raise ValueError("Digit splitting requires at least 2 nonempty subsets")
        heldout_subset = min(subset_counts, key=lambda name: (subset_counts[name], name))
        groups = sorted({
            path.split('/')[-3] for path in paths
            if path.split('/')[-4] == heldout_subset
        })
        if len(groups) < 2:
            raise ValueError(f"Smallest digit subset '{heldout_subset}' needs at least 2 speakers")
        random.Random(42).shuffle(groups)
        test_speakers = set(groups[:len(groups) // 2])
        return {
            "train": [p for p in paths if p.split('/')[-4] != heldout_subset],
            "val": [p for p in paths if p.split('/')[-4] == heldout_subset
                    and p.split('/')[-3] not in test_speakers],
            "test": [p for p in paths if p.split('/')[-4] == heldout_subset
                     and p.split('/')[-3] in test_speakers],
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    random.Random(42).shuffle(groups)
    test_size = max(1, int(len(groups) * 0.1))
    val_size = max(1, int(len(groups) * 0.1))
    train_size = len(groups) - val_size - test_size
    assignments = {
        "train": groups[:train_size],
        "val": groups[train_size:train_size + val_size],
        "test": groups[train_size + val_size:],
    }
    if dataset == "grid":
        assignments = {
            split: [path for path in paths if path.split('/')[-3] in speakers]
            for split, speakers in assignments.items()
        }
    return assignments


def create_splits(config_path="config.json", dataset=None, processed_root=None):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    dataset = dataset if dataset is not None else config.get("dataset", "grid")
    
    if dataset == "grid":
        out_dir = processed_root if processed_root is not None else config.get("grid_processed_data_dir", "./data/grid")
        # Grid structure: data/grid/speaker/video/*.mp4
        search_pattern = os.path.join(out_dir, '*', 'video', '*.mp4')
    elif dataset == "digit":
        out_dir = processed_root if processed_root is not None else config.get("digit_processed_data_dir", "./data/digit")
        # Digit structure: data/digit/subset_X/speaker/video/*.mp4
        search_pattern = os.path.join(out_dir, 'subset_*', '*', 'video', '*.mp4')
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
        
    print(f"\nCreating Train/Val/Test splits for {dataset.upper()} dataset...")
    
    all_videos = glob.glob(search_pattern)
    
    if not all_videos:
        raise FileNotFoundError(f"No videos found. Checked pattern: {search_pattern}")

    all_videos = [os.path.relpath(path, out_dir) for path in all_videos]
    assignments = partition_videos(all_videos, dataset)
    existing = {}
    for split in assignments:
        path = os.path.join(out_dir, f"{split}_split.json")
        if os.path.exists(path):
            with open(path) as f:
                existing[split] = json.load(f)
    if any(existing[split] != assignments[split] for split in existing):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_dir = os.path.join(out_dir, "split_backups", stamp)
        os.makedirs(backup_dir)
        backup_files = [f"{split}_split.json" for split in existing] + ["split_summary.json"]
        for filename in backup_files:
            path = os.path.join(out_dir, filename)
            if os.path.exists(path):
                shutil.copy2(path, os.path.join(backup_dir, filename))
        print(f"Backed up previous splits to {backup_dir}")

    summary = {
        "dataset": dataset, "seed": 42,
        "strategy": ("speaker-disjoint, approximately 80/10/10 by speaker" if dataset == "grid"
                     else "smallest subset held out, its speakers divided between validation/test"),
        "splits": {},
    }
    if dataset == "digit":
        counts = Counter(path.split('/')[-4] for paths in assignments.values() for path in paths)
        summary["subset_sample_counts"] = dict(sorted(counts.items()))
        summary["heldout_subset"] = min(counts, key=lambda name: (counts[name], name))
        print(f"Digit subset sample counts: {summary['subset_sample_counts']}")
        print(f"Splitting speakers of smallest subset '{summary['heldout_subset']}' between val/test")
    for split, paths in assignments.items():
        with open(os.path.join(out_dir, f"{split}_split.json"), "w") as f:
            json.dump(paths, f, indent=2)
        speakers = sorted({
            '/'.join(path.split('/')[-4:-2]) if dataset == "digit" else path.split('/')[-3]
            for path in paths
        })
        summary["splits"][split] = {"samples": len(paths), "speakers": speakers}
    with open(os.path.join(out_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Data splits saved in {out_dir}:")
    for split, details in summary["splits"].items():
        print(f" - {split}: {details['samples']} clips, {len(details['speakers'])} disjoint speakers")
    return assignments

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create speaker-disjoint GRID or subset-held-out digit splits")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dataset", choices=("grid", "digit"), help="Override the dataset for split generation")
    parser.add_argument("--processed-root", help="Override the processed dataset directory")
    args = parser.parse_args()
    create_splits(args.config, dataset=args.dataset, processed_root=args.processed_root)
