"""DigitDataset .lab supervision tests.

The regenerated .lab format includes "!" tokens on line 1 with their own
time ranges — transcripts and .lab line 1 must match token-for-token.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_digit_loader.py -q
(or: python tests/test_digit_loader.py)
"""

import os
import sys
import shutil
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.digit_loader import DigitDataset


def _make_dataset(txt, lab_line1, lab_times, n_frames=30):
    """Temp dataset with one synthetic 30-frame clip; source_fps=10."""
    root = tempfile.mkdtemp()
    speaker_dir = os.path.join(root, "subset_01", "1000")
    for sub in ("video", "text", "lab"):
        os.makedirs(os.path.join(speaker_dir, sub))
    base = "clip"
    video_path = os.path.join(speaker_dir, "video", f"{base}.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for _ in range(n_frames):
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    with open(os.path.join(speaker_dir, "text", f"{base}.txt"), "w") as f:
        f.write(txt)
    with open(os.path.join(speaker_dir, "lab", f"{base}.lab"), "w") as f:
        f.write(f"{lab_line1}\n{lab_times}\n")
    with open(os.path.join(root, "train_split.json"), "w") as f:
        import json
        json.dump([f"subset_01/1000/video/{base}.mp4"], f)
    return root, speaker_dir


def test_lab_with_bang_is_not_mismatch():
    # "!" present (with a time range) on .lab line 1: token-for-token match.
    root, _ = _make_dataset("0 ! 2", "0 ! 2", "0.1-0.2 0.3-0.4 0.5-0.6")
    try:
        ds = DigitDataset(data_dir=root, split="train", max_frames=125,
                          use_audio=False, source_fps=10.0)
        assert len(ds.samples) == 1
        assert ds.samples[0].get("lab_times") == [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
    finally:
        shutil.rmtree(root)


def test_digit_times_consume_bang_range():
    # The "!" token keeps its own .lab range; digits after it must not shift.
    root, _ = _make_dataset("0 ! 2", "0 ! 2", "0.1-0.2 0.3-0.4 0.5-0.6")
    try:
        ds = DigitDataset(data_dir=root, split="train", max_frames=125,
                          use_audio=False, source_fps=10.0)
        sample = ds[0]
        assert sample["digit_times"] == [("0", 1, 2), ("!", 3, 4), ("2", 5, 6)]
    finally:
        shutil.rmtree(root)


def test_stale_lab_without_bang_is_mismatch():
    # Old-format .lab (no "!" entry) no longer matches: exclude from
    # supervision rather than silently truncating the time walk.
    root, _ = _make_dataset("0 ! 2", "0 2", "0.1-0.2 0.5-0.6")
    try:
        ds = DigitDataset(data_dir=root, split="train", max_frames=125,
                          use_audio=False, source_fps=10.0)
        assert len(ds.samples) == 1
        assert "lab_times" not in ds.samples[0]
        assert "digit_times" not in ds[0]
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
