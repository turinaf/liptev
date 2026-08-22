"""Sanity tests for evaluation metrics (EER + threshold-free alignment).

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_metrics.py -q
(or: python tests/test_metrics.py)
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import (
    compute_metrics,
    equal_error_rate,
    load_eer_thresholds,
    save_eer_thresholds,
)


def test_eer_perfect_separation():
    labels = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    scores = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
    eer, thr = equal_error_rate(labels, scores)
    assert eer == 0.0
    acc = ((scores >= thr).float() == labels).float().mean().item()
    assert acc == 1.0


def test_eer_constant_scores_is_chance():
    # Uninformative scores: EER must be chance level (0.5), not 0 or 1.
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    scores = torch.tensor([0.7, 0.7, 0.7, 0.7])
    eer, _ = equal_error_rate(labels, scores)
    assert abs(eer - 0.5) < 1e-6


def test_eer_inverted_scores():
    # Perfectly anti-correlated scores: every threshold errs on one class.
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    scores = torch.tensor([0.9, 0.9, 0.1, 0.1])
    eer, _ = equal_error_rate(labels, scores)
    assert abs(eer - 1.0) < 1e-6


def test_eer_partial_overlap_between_zero_and_half():
    labels = torch.tensor([0.0] * 10 + [1.0] * 10)
    scores = torch.tensor(
        [0.05, 0.10, 0.15, 0.20, 0.60, 0.30, 0.35, 0.40, 0.45, 0.50] +
        [0.55, 0.65, 0.70, 0.75, 0.10, 0.80, 0.85, 0.90, 0.95, 0.58]
    )
    eer, _ = equal_error_rate(labels, scores)
    assert 0.0 < eer < 0.5


def test_eer_single_class_returns_chance():
    # Degenerate val split: no discrimination measurable.
    labels = torch.tensor([1.0, 1.0, 1.0])
    scores = torch.tensor([0.2, 0.5, 0.9])
    eer, thr = equal_error_rate(labels, scores)
    assert abs(eer - 0.5) < 1e-6


def test_compute_metrics_reports_eer_for_both_scores():
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    cons = torch.tensor([0.2, 0.3, 0.8, 0.9])
    align = torch.tensor([0.6, 0.65, 0.9, 0.95])  # all above 0.5, still separable
    m = compute_metrics(labels, cons, align)
    for key in (
        "roc_auc", "verification_accuracy",
        "verification_eer", "verification_accuracy_at_eer",
        "alignment_roc_auc", "alignment_eer", "alignment_accuracy",
    ):
        assert key in m, f"missing metric: {key}"
    assert m["verification_eer"] == 0.0
    assert m["verification_accuracy_at_eer"] == 1.0
    assert m["alignment_roc_auc"] == 1.0
    assert m["alignment_eer"] == 0.0


def test_alignment_accuracy_uses_eer_threshold_not_05():
    # The observed failure mode: every alignment score sits above 0.5
    # (cosine > 0 for deep features), so a fixed 0.5 cut predicts positive
    # for everything -> exactly 0.5 accuracy on the balanced eval set.
    # The scores are separable, so accuracy at the EER threshold is 1.0.
    labels = torch.tensor([0.0] * 8 + [1.0] * 8)
    align = torch.tensor([0.72, 0.74, 0.71, 0.75, 0.73, 0.76, 0.72, 0.74] +
                         [0.88, 0.91, 0.89, 0.92, 0.90, 0.93, 0.89, 0.91])
    cons = torch.tensor([0.1] * 8 + [0.9] * 8)
    m = compute_metrics(labels, cons, align)
    assert ((align >= 0.5).float() == 1.0).all()  # fixed 0.5 would give 0.5 acc
    assert m["alignment_accuracy"] == 1.0


def test_score_margin_unbalanced_labels():
    # Elementwise pos-neg subtraction crashes when class counts differ;
    # the mean-separation form must not.
    labels = torch.tensor([0.0] * 5 + [1.0] * 3)
    cons = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.2, 0.8, 0.9, 0.7])
    align = torch.tensor([0.6, 0.6, 0.6, 0.6, 0.6, 0.9, 0.9, 0.9])
    m = compute_metrics(labels, cons, align)
    assert abs(m["score_margin"] - (0.8 - 0.18)) < 1e-6


def _separable_metrics():
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    cons = torch.tensor([0.2, 0.3, 0.8, 0.9])
    align = torch.tensor([0.6, 0.65, 0.9, 0.95])
    return compute_metrics(labels, cons, align)


def test_compute_metrics_returns_eer_thresholds():
    # Single-pair inference needs the EER-optimal cut, not just the rate.
    m = _separable_metrics()
    assert "verification_eer_threshold" in m
    assert "alignment_eer_threshold" in m
    cons = torch.tensor([0.2, 0.3, 0.8, 0.9])
    align = torch.tensor([0.6, 0.65, 0.9, 0.95])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert ((cons >= m["verification_eer_threshold"]).float() == labels).all()
    assert ((align >= m["alignment_eer_threshold"]).float() == labels).all()


def test_save_load_eer_thresholds_roundtrip(tmp_path):
    m = _separable_metrics()
    path = str(tmp_path / "eer_thresholds.json")
    save_eer_thresholds(m, {"checkpoint": "best.pth", "dataset": "grid", "n_samples": 4}, path)

    loaded = load_eer_thresholds(path)
    assert loaded["checkpoint"] == "best.pth"
    assert loaded["dataset"] == "grid"
    assert loaded["n_samples"] == 4
    assert isinstance(loaded["timestamp"], str) and loaded["timestamp"]
    assert abs(loaded["verification"]["eer"] - m["verification_eer"]) < 1e-9
    assert abs(loaded["verification"]["threshold"] - m["verification_eer_threshold"]) < 1e-9
    assert abs(loaded["verification"]["accuracy_at_eer"] - m["verification_accuracy_at_eer"]) < 1e-9
    assert abs(loaded["verification"]["roc_auc"] - m["roc_auc"]) < 1e-9
    assert abs(loaded["alignment"]["eer"] - m["alignment_eer"]) < 1e-9
    assert abs(loaded["alignment"]["threshold"] - m["alignment_eer_threshold"]) < 1e-9
    assert abs(loaded["alignment"]["accuracy_at_eer"] - m["alignment_accuracy"]) < 1e-9


def test_saved_thresholds_json_is_valid_and_nan_becomes_null(tmp_path):
    # Degenerate eval (single class) yields NaN thresholds; json.dump would
    # emit bare NaN (invalid JSON). It must serialize as null instead.
    labels = torch.tensor([1.0, 1.0])
    m = compute_metrics(labels, torch.tensor([0.2, 0.8]), torch.tensor([0.3, 0.7]))
    assert m["verification_eer_threshold"] != m["verification_eer_threshold"]  # NaN

    path = str(tmp_path / "eer_thresholds.json")
    save_eer_thresholds(m, {"checkpoint": "c.pt", "dataset": "digit", "n_samples": 2}, path)

    raw = json.loads(open(path).read())  # strict parse fails on bare NaN
    assert raw["verification"]["threshold"] is None
    assert raw["alignment"]["threshold"] is None
    assert load_eer_thresholds(path)["alignment"]["threshold"] is None


def test_load_eer_thresholds_missing_file_returns_none(tmp_path):
    assert load_eer_thresholds(str(tmp_path / "nope.json")) is None


def test_load_eer_thresholds_corrupt_file_returns_none(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    assert load_eer_thresholds(str(path)) is None
