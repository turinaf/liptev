"""Per-token report construction for inference.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_inference_report.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import (
    spans_to_onehot, build_token_report, build_word_report, spans_to_score_maps,
)


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
        thresholds={"word_verification": 0.5},
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
        thresholds={"word_verification": 0.5}, fallback=False,
    )
    assert abs(rows[0]["alignment_score"] - 0.7) < 1e-6
    assert rows[0]["verification_score"] == 0.9
    # word None + posterior present -> alignment is the posterior alone
    rows = build_token_report(
        ["2"], [(0, 0, 10)], seconds_per_frame=0.04,
        posterior_scores=[0.3], word_scores=[None], verification_scores=[0.1],
        thresholds={"word_verification": 0.5}, fallback=False,
    )
    assert abs(rows[0]["alignment_score"] - 0.3) < 1e-6
    # both present -> mean (distinct signals must not be conflated)
    rows = build_token_report(
        ["3"], [(0, 0, 10)], seconds_per_frame=0.04,
        posterior_scores=[0.4], word_scores=[0.8], verification_scores=[0.1],
        thresholds={"word_verification": 0.5}, fallback=False,
    )
    assert abs(rows[0]["alignment_score"] - 0.6) < 1e-6


def _phoneme_rows():
    return build_token_report(
        ["B", "IH", "N", "AE", "T"],
        [(i, i * 2, (i + 1) * 2) for i in range(5)],
        seconds_per_frame=0.04,
        posterior_scores=[0.9, 0.8, 0.7, 0.6, 0.9],
        word_scores=[0.9, 0.8, 0.7, 0.6, 0.9],
        verification_scores=[0.9, 0.6, 0.8, 0.2, 0.9],
        thresholds={"word_verification": {"threshold": 0.5}},
        fallback=False,
    )


def test_word_report_uses_full_word_timestamps_and_minimum_scores():
    rows = build_word_report(
        [("bin", [8, 18, 23]), ("at", [3, 32])], _phoneme_rows())
    assert rows == [
        {"token": "bin", "start_s": 0.0, "end_s": 0.24,
         "alignment_score": 0.7, "verification_score": 0.6,
         "verdict": "MATCH", "low_confidence": False},
        {"token": "at", "start_s": 0.24, "end_s": 0.4,
         "alignment_score": 0.6, "verification_score": 0.2,
         "verdict": "MISMATCH", "low_confidence": False},
    ]


def test_word_report_keeps_repeated_words_separate():
    rows = build_word_report([("a", [1]), ("a", [1])], _phoneme_rows()[:2])
    assert [r["token"] for r in rows] == ["a", "a"]
    assert [r["start_s"] for r in rows] == [0.0, 0.08]


@pytest.mark.parametrize("missing", ["span", "truncated", "empty"])
def test_word_report_incomplete_words_are_unavailable(missing):
    parts = _phoneme_rows()
    per_word = [("bin", [8, 18, 23]), ("at", [3, 32])]
    if missing == "span":
        parts[3]["start_s"] = parts[3]["end_s"] = None
    elif missing == "truncated":
        parts = parts[:4]
    else:
        per_word[1] = ("at", [])
    rows = build_word_report(per_word, parts)
    assert rows[0]["verdict"] == "MATCH"
    assert rows[1] == {
        "token": "at", "start_s": None, "end_s": None,
        "alignment_score": None, "verification_score": None,
        "verdict": "uncalibrated", "low_confidence": True,
    }


def test_word_report_propagates_fallback_and_uncalibrated_verdict():
    parts = _phoneme_rows()
    parts[1]["low_confidence"] = True
    for part in parts:
        part["verdict"] = "uncalibrated"
    rows = build_word_report([("bin", [8, 18, 23])], parts)
    assert rows[0]["low_confidence"] is True
    assert rows[0]["verdict"] == "uncalibrated"
    assert rows[0]["verification_score"] == 0.6


def test_word_report_preserves_threshold_decisions_before_rounding():
    parts = build_token_report(
        ["AE", "T"], [(0, 0, 1), (1, 1, 2)], seconds_per_frame=0.04,
        posterior_scores=[0.8, 0.8], word_scores=[0.8, 0.8],
        verification_scores=[0.50001, 0.49999],
        thresholds={"word_verification": 0.5}, fallback=False,
    )
    assert parts[0]["verification_score"] == parts[1]["verification_score"] == 0.5
    assert build_word_report([("at", [3, 32])], parts)[0]["verdict"] == "MISMATCH"


def test_score_maps_preserve_token_positions_when_spans_are_missing():
    post, word, verif = spans_to_score_maps(
        [(0, 0, 2), (2, 2, 4)], [0.9, 0.7],
        [0.8, 0.5, 0.2], [0.9, 0.6, 0.3],
    )
    assert post == {0: 0.9, 2: 0.7}
    assert word == {0: 0.9, 2: 0.3}
    assert verif == {0: 0.8, 2: 0.2}
