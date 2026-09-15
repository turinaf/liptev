"""Threshold auto-discovery and single-pair EER decisions in inference.py.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_inference_thresholds.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import format_score_decision, resolve_thresholds_path


def test_resolve_thresholds_prefers_explicit_path():
    assert resolve_thresholds_path("/tmp/custom.json", "/ckpt/best.pth") == "/tmp/custom.json"


def test_resolve_thresholds_auto_discovers_beside_checkpoint(tmp_path):
    ckpt = tmp_path / "best_lip_verification_model.pth"
    ckpt.write_text("x")
    calib = tmp_path / "eer_thresholds.json"
    calib.write_text("{}")
    assert resolve_thresholds_path(None, str(ckpt)) == str(calib)


def test_resolve_thresholds_none_when_absent(tmp_path):
    ckpt = tmp_path / "best.pth"
    ckpt.write_text("x")
    assert resolve_thresholds_path(None, str(ckpt)) is None
    assert resolve_thresholds_path(None, None) is None


def test_format_decision_match_and_mismatch():
    block = {"threshold": 0.31, "eer": 0.42, "accuracy_at_eer": 0.6, "roc_auc": 0.61}
    line = format_score_decision("Verification score", 0.0001, block)
    assert "MISMATCH" in line
    assert "0.3100" in line and "0.4200" in line  # threshold + EER shown
    assert "MATCH" in format_score_decision("Verification score", 0.9, block)
    # Boundary: score exactly at the threshold counts as MATCH (>= cut).
    assert "MATCH" in format_score_decision("Verification score", 0.31, block)


def test_format_decision_uncalibrated_without_threshold():
    assert "uncalibrated" in format_score_decision("Verification score", 0.5, None)
    nan_block = {"threshold": None, "eer": 0.5}
    assert "uncalibrated" in format_score_decision("Verification score", 0.5, nan_block)
