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
