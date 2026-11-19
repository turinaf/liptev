import json
import random
from pathlib import Path

import pytest

from create_splits import create_splits, partition_videos
from digit_create_splits import create_digit_splits
from dataset.digit_loader import DigitDataset
from dataset.grid_loader import GRIDDataset


def _speakers(paths):
    return {Path(path).parent.parent.name for path in paths}


def test_grid_split_is_speaker_disjoint_complete_and_order_independent():
    paths = [
        f"s{speaker}/video/{clip}.mp4"
        for speaker in range(33) for clip in range(1 + speaker % 4)
    ]
    state = random.getstate()
    splits = partition_videos(paths, "grid")
    assert random.getstate() == state
    assert splits == partition_videos(list(reversed(paths)), "grid")
    groups = {name: _speakers(videos) for name, videos in splits.items()}
    assert [len(groups[name]) for name in ("train", "val", "test")] == [27, 3, 3]
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    combined = [path for videos in splits.values() for path in videos]
    assert sorted(combined) == sorted(paths)


def test_grid_rejects_too_few_speakers_and_supports_three():
    with pytest.raises(ValueError, match="at least 3 speakers"):
        partition_videos(["s1/video/a.mp4", "s2/video/b.mp4"], "grid")
    splits = partition_videos([f"s{i}/video/a.mp4" for i in range(3)], "grid")
    assert all(len(paths) == 1 for paths in splits.values())


@pytest.mark.parametrize("entrypoint", ["standalone", "preprocessing"])
def test_common_grid_manifests_back_up_old_splits_and_load_correctly(tmp_path, monkeypatch, entrypoint):
    data = tmp_path / "processed"
    raw = tmp_path / "raw"
    raw.mkdir()
    for speaker in range(10):
        for clip in range(2):
            folder = data / f"s{speaker}"
            (folder / "video").mkdir(parents=True, exist_ok=True)
            (folder / "align").mkdir(exist_ok=True)
            (folder / "video" / f"{clip}.mp4").touch()
            (folder / "align" / f"{clip}.align").write_text("0 75000 bin\n")
    legacy = ["s0/video/0.mp4"]
    for name in ("train", "val", "test"):
        (data / f"{name}_split.json").write_text(json.dumps(legacy))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "dataset": "grid", "grid_raw_data_dir": str(raw),
        "grid_processed_data_dir": str(data),
        "audio_sample_rate": 16000, "mp_model_path": "unused.task",
    }))
    if entrypoint == "standalone":
        create_splits(config)
    else:
        from pipeline import data_preparation
        monkeypatch.setattr(data_preparation.vision.FaceLandmarker, "create_from_options", lambda options: None)
        data_preparation.prepare_dataset(config)
    groups = {}
    backups = list((data / "split_backups").iterdir())
    assert len(backups) == 1
    for name in ("train", "val", "test"):
        assert json.loads((backups[0] / f"{name}_split.json").read_text()) == legacy
        manifest = json.loads((data / f"{name}_split.json").read_text())
        dataset = GRIDDataset(str(data), split=name)
        assert len(dataset) == len(manifest)
        assert all(Path(sample["video"]).is_relative_to(data) for sample in dataset.samples)
        groups[name] = _speakers(manifest)
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    create_splits(config)
    assert len(list((data / "split_backups").iterdir())) == 1


def _digit_paths():
    paths = []
    for subset, speakers, clips in ((1, 2, 5), (2, 2, 6), (3, 4, 1), (4, 2, 4), (5, 2, 3)):
        paths.extend(
            f"subset_{subset:02}/s{speaker}/video/{clip}.mp4"
            for speaker in range(speakers) for clip in range(clips)
        )
    return paths


def test_digit_holds_out_smallest_subset_and_separates_its_speakers():
    paths = _digit_paths()
    state = random.getstate()
    splits = partition_videos(paths, "digit")
    assert random.getstate() == state
    assert splits == partition_videos(list(reversed(paths)), "digit")
    assert {Path(p).parts[0] for p in splits["train"]} == {
        "subset_01", "subset_02", "subset_04", "subset_05"}
    for split in ("val", "test"):
        assert {Path(p).parts[0] for p in splits[split]} == {"subset_03"}
        assert len(_speakers(splits[split])) == 2
    assert not _speakers(splits["val"]) & _speakers(splits["test"])
    combined = [p for values in splits.values() for p in values]
    assert sorted(combined) == sorted(paths)


@pytest.mark.parametrize("entrypoint", ["standalone", "dataset_override", "digit_wrapper", "preprocessing"])
def test_digit_entrypoints_write_portable_common_manifests(tmp_path, monkeypatch, entrypoint):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data" / "digit"
    raw = tmp_path / "raw"
    raw.mkdir()
    for relative in _digit_paths():
        video = data / relative
        video.parent.mkdir(parents=True, exist_ok=True)
        video.touch()
        text = video.parent.parent / "text" / f"{video.stem}.txt"
        text.parent.mkdir(exist_ok=True)
        text.write_text("1 2")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "dataset": "grid" if entrypoint in ("dataset_override", "digit_wrapper") else "digit",
        "digit_processed_data_dir": "./data/digit", "digit_raw_data_dir": str(raw),
        "audio_sample_rate": 16000, "mp_model_path": "unused.task",
    }))
    if entrypoint == "digit_wrapper":
        create_digit_splits(config)
    elif entrypoint == "dataset_override":
        create_splits(config, dataset="digit")
    elif entrypoint == "preprocessing":
        from pipeline import data_preparation
        monkeypatch.setattr(data_preparation.vision.FaceLandmarker, "create_from_options", lambda options: None)
        data_preparation.prepare_dataset(config)
    else:
        create_splits(config)
    summary = json.loads((data / "split_summary.json").read_text())
    assert summary["subset_sample_counts"] == {
        "subset_01": 10, "subset_02": 12, "subset_03": 4, "subset_04": 8, "subset_05": 6}
    assert summary["heldout_subset"] == "subset_03"
    for split in ("train", "val", "test"):
        manifest = json.loads((data / f"{split}_split.json").read_text())
        assert all(not Path(p).is_absolute() and p.startswith("subset_") for p in manifest)
        dataset = DigitDataset(str(data), split=split)
        assert len(dataset) == len(manifest)
        assert all(Path(sample["video"]).is_relative_to(data) for sample in dataset.samples)


def test_digit_ties_are_stable_and_invalid_holdout_fails_explicitly():
    paths = [f"{subset}/s{speaker}/video/a.mp4"
             for subset in ("subset_02", "subset_01") for speaker in range(2)]
    assert all(p.startswith("subset_02/") for p in partition_videos(paths, "digit")["train"])
    with pytest.raises(ValueError, match="at least 2 nonempty subsets"):
        partition_videos(paths[:2], "digit")
    with pytest.raises(ValueError, match="at least 2 speakers"):
        partition_videos(["subset_01/s1/video/a.mp4", *paths[:2]], "digit")


def test_digit_missing_data_fails_without_writing_manifests(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"digit_processed_data_dir": str(tmp_path / "missing")}))
    with pytest.raises(FileNotFoundError, match="No videos found"):
        create_digit_splits(config)
