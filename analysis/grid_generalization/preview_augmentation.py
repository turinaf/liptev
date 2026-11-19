"""Save a qualitative preview of training augmentation on one processed clip."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.digit_loader import DigitDataset
from dataset.grid_loader import GRIDDataset
from dataset.augmentation import VideoAugmentation
from utils.dataset_utils import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.grid.json")
    parser.add_argument("--output", default=str(Path(__file__).with_name("augmentation_preview.png")))
    args = parser.parse_args()
    config = Config(args.config)
    if config.dataset == "digit":
        dataset = DigitDataset(config.processed_data_dir, max_frames=config.max_frames,
                               source_fps=config.digit_source_fps)
    else:
        dataset = GRIDDataset(config.processed_data_dir, max_frames=config.max_frames)
    if not len(dataset):
        raise ValueError("No training clips available for the augmentation preview")
    sample = dataset[0]
    if sample["video_len"] < 1:
        raise ValueError("The preview clip contains no decoded frames")
    video = sample["video"][:sample["video_len"]]
    frames = torch.linspace(0, len(video) - 1, 5).round().long().tolist()
    rows = [("Original", video)]
    augment = VideoAugmentation()
    for seed in (7, 42, 123):
        torch.manual_seed(seed)
        rows.append((f"Augmented: seed {seed}", augment(video)))
    fig, axes = plt.subplots(len(rows), len(frames), figsize=(11, 8))
    for row, (label, clip) in enumerate(rows):
        for col, frame in enumerate(frames):
            ax = axes[row, col]
            image = ((clip[frame] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Frame {frame}")
            if col == 0:
                ax.set_ylabel(label)
    relative = Path(dataset.samples[0]["video"]).relative_to(config.processed_data_dir)
    fig.suptitle(f"{config.dataset.upper()} training clip: {relative}\n"
                 "Spatial/photometric changes only; identical frame positions in every row")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(json.dumps({"output": args.output, "clip": str(relative),
                      "frames": frames, "seeds": [7, 42, 123]}))


if __name__ == "__main__":
    main()
