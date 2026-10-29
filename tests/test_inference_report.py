"""Per-token report construction for inference.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_inference_report.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import spans_to_onehot, build_token_report


def test_spans_to_onehot():
    onehot = spans_to_onehot([(0, 0, 2), (1, 2, 4)], T=5, L=2)
    assert onehot.shape == (5, 2)
    assert onehot[:2, 0].all() and onehot[2:4, 1].all()
    assert onehot[4].sum() == 0


def test_build_token_report_values_and_verdicts():
    tokens = ["1", "2"]
    spans = [(0, 0, 10), (1, 10, 20)]
    rows = build_token_report(
        tokens, spans, seconds_per_frame=0.04,
        posterior_scores=[0.9, 0.6],
        word_scores=[0.8, 0.4],
        verification_scores=[0.95, 0.20],
        thresholds={"verification": 0.5},
        fallback=False,
    )
    assert rows[0]["token"] == "1"
    assert rows[0]["start_s"] == 0.0 and rows[0]["end_s"] == 0.4
    # alignment = mean(posterior, word_score)
    assert abs(rows[0]["alignment_score"] - 0.85) < 1e-6
    assert rows[0]["verdict"] == "MATCH"
    assert rows[1]["verdict"] == "MISMATCH"
    assert rows[0]["low_confidence"] is False


def test_build_token_report_fallback_flagged():
    rows = build_token_report(
        ["3"], [(0, 0, 5)], seconds_per_frame=0.05,
        posterior_scores=[0.5], word_scores=[0.5], verification_scores=[0.5],
        thresholds={}, fallback=True,
    )
    assert rows[0]["low_confidence"] is True
    assert rows[0]["verdict"] == "uncalibrated"


def test_build_token_report_missing_token_flagged():
    # token 1 got no frames from the aligner
    rows = build_token_report(
        ["1", "2"], [(0, 0, 10)], seconds_per_frame=0.04,
        posterior_scores=[0.5, None], word_scores=[0.5, None],
        verification_scores=[0.5, None], thresholds={}, fallback=False,
    )
    assert rows[1]["low_confidence"] is True
    assert rows[1]["start_s"] is None


def test_build_token_report_alignment_score_signals():
    # posterior None + word present -> alignment is the word score alone
    rows = build_token_report(
        ["1"], [(0, 0, 10)], seconds_per_frame=0.04,
        posterior_scores=[None], word_scores=[0.7], verification_scores=[0.9],
        thresholds={"verification": 0.5}, fallback=False,
    )
    assert abs(rows[0]["alignment_score"] - 0.7) < 1e-6
    assert rows[0]["verification_score"] == 0.9
    # word None + posterior present -> alignment is the posterior alone
    rows = build_token_report(
        ["2"], [(0, 0, 10)], seconds_per_frame=0.04,
        posterior_scores=[0.3], word_scores=[None], verification_scores=[0.1],
        thresholds={"verification": 0.5}, fallback=False,
    )
    assert abs(rows[0]["alignment_score"] - 0.3) < 1e-6
    # both present -> mean (distinct signals must not be conflated)
    rows = build_token_report(
        ["3"], [(0, 0, 10)], seconds_per_frame=0.04,
        posterior_scores=[0.4], word_scores=[0.8], verification_scores=[0.1],
        thresholds={"verification": 0.5}, fallback=False,
    )
    assert abs(rows[0]["alignment_score"] - 0.6) < 1e-6
