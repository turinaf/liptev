import json

import cv2
import numpy as np
import pytest
import torch

from dataset.augmentation import VideoAugmentation
from dataset.digit_loader import DigitDataset
from dataset.grid_loader import GRIDDataset
from utils.dataset_utils import Config


def test_augmentation_is_clip_consistent_seeded_and_does_not_mutate_input():
    video = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32).repeat(5, 1, 1, 1)
    original = video.clone()
    augment = VideoAugmentation()
    torch.manual_seed(7)
    first = augment(video)
    torch.manual_seed(7)
    second = augment(video)
    assert torch.equal(first, second)
    assert torch.equal(video, original)
    assert torch.equal(first, first[:1].expand_as(first))
    assert not torch.equal(first, original)
    assert first.shape == video.shape and first.dtype == video.dtype
    assert torch.isfinite(first).all() and first.min() >= -1 and first.max() <= 1


@pytest.mark.parametrize("kind", ["grid", "digit"])
def test_dataset_augmentation_preserves_timing_and_leaves_evaluation_unchanged(tmp_path, kind):
    relative = "s1/video/clip.mp4" if kind == "grid" else "subset_01/s1/video/clip.mp4"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 25, (32, 32))
    assert writer.isOpened()
    pattern = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    for _ in range(12):
        writer.write(pattern)
    writer.release()
    speaker = path.parent.parent
    if kind == "grid":
        (speaker / "align").mkdir()
        (speaker / "align" / "clip.align").write_text("0 75000 bin\n")
        dataset_type = GRIDDataset
        timing_key = "word_times"
    else:
        (speaker / "text").mkdir()
        (speaker / "lab").mkdir()
        (speaker / "text" / "clip.txt").write_text("1 2")
        (speaker / "lab" / "clip.lab").write_text("1 2\n0.1-0.2 0.2-0.4\n")
        (speaker / "meta.json").write_text(json.dumps({"clip": {"source_fps": 20.0}}))
        dataset_type = DigitDataset
        timing_key = "digit_times"
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}_split.json").write_text(json.dumps([relative]))
    baseline = dataset_type(str(tmp_path), max_frames=16)[0]
    torch.manual_seed(7)
    augmented = dataset_type(str(tmp_path), max_frames=16, augment=True)[0]
    assert augmented["video_len"] == baseline["video_len"] == 12
    assert augmented[timing_key] == baseline[timing_key]
    if kind == "digit":
        assert augmented["digit_times"] == [("1", 2, 4), ("2", 4, 8)]
    assert augmented["transcript"] == baseline["transcript"]
    assert augmented["video"].shape == baseline["video"].shape == (16, 3, 32, 32)
    assert torch.equal(augmented["video"][11:], augmented["video"][11:12].expand(5, -1, -1, -1))
    assert not torch.equal(augmented["video"], baseline["video"])
    for split in ("val", "test"):
        evaluation = dataset_type(str(tmp_path), split=split, max_frames=16)[0]
        assert torch.equal(evaluation["video"], baseline["video"])
        with pytest.raises(ValueError, match="only supported for the train split"):
            dataset_type(str(tmp_path), split=split, augment=True)


def test_augmentation_config_requires_explicit_boolean(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}")
    assert not Config(config).train_augmentation
    config.write_text(json.dumps({"train_augmentation": "false"}))
    with pytest.raises(ValueError, match="must be a boolean"):
        Config(config)
