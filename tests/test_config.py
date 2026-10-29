"""Config must expose every key downstream code relies on.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_config.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_utils import Config


def test_config_exposes_mp_model_path():
    # The repo config.json sets it; inference.py needs it for the lip cropper.
    cfg = Config("config.json")
    assert cfg.mp_model_path == "face_landmarker.task"


def test_config_mp_model_path_has_default():
    # Missing file -> empty cfg -> the default must still be there so
    # inference degrades to the repo-root landmarker, not an AttributeError.
    cfg = Config("tests/definitely_missing_config.json")
    assert cfg.mp_model_path == "face_landmarker.task"


def test_vfa_defaults():
    cfg = Config()
    assert cfg.lambda_frame_ce == 1.0
    assert cfg.lambda_boundary == 1.0
    assert cfg.lambda_word_verif == 1.0
    assert cfg.vfa_boundary_threshold == 0.8


def test_per_dataset_configs():
    grid = Config("config.grid.json")
    assert grid.dataset == "grid"
    assert grid.processed_data_dir == "/mnt/lustre/ifm-lustre-prod-us-west1-a/av-data/grid-data"
    assert "liptev-exp/grid" in grid.output_dir

    digit = Config("config.digit.json")
    assert digit.dataset == "digit"
    assert digit.max_frames == 125  # dataset=digit selects the digit frame budget
    assert "liptev-exp/digit" in digit.output_dir


def test_entry_points_accept_config_flag():
    # --config is how the slurm scripts select the dataset; --help must work
    # without a GPU or data present.
    import subprocess
    import sys
    for entry in ("main.py", "test.py", "pipeline/data_preparation.py"):
        res = subprocess.run(
            [sys.executable, entry, "--help"], capture_output=True, text=True
        )
        assert res.returncode == 0, f"{entry} --help failed:\n{res.stderr}"
        assert "--config" in res.stdout
