"""Tests for the Viterbi forced aligner port.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_viterbi.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.viterbi import viterbi_align, token_spans, uniform_token_spans


def _logits_from_path(path, n_classes):
    """Deterministic logits: +5 for the path class at each frame, 0 elsewhere."""
    T = len(path)
    logits = torch.zeros(T, n_classes)
    for t, c in enumerate(path):
        logits[t, c] = 5.0
    return logits


def test_viterbi_recovers_clean_path():
    # 2 tokens (ids 7, 3) framed by silence (class 0): optimal path is the
    # generating path itself.
    path = [0, 0, 7, 7, 7, 3, 3, 0, 0]
    labels = [0, 7, 3, 0]
    out = viterbi_align(_logits_from_path(path, 8), labels)
    assert out == path


def test_viterbi_forces_transition_at_high_boundary_prob():
    # Without boundary hint the DP may prefer staying on token 7; a >threshold
    # boundary probability at the transition frame forces the advance, so the
    # second token must start at that frame.
    logits = torch.zeros(6, 4)
    logits[:, 1] = 4.0   # token 7 -> class 1... (use class ids 1 and 2 as tokens)
    labels = [0, 1, 2, 0]
    bound = [0.0, 0.0, 0.9, 0.0, 0.0, 0.0]
    out = viterbi_align(logits, labels, bound_probs=bound, boundary_threshold=0.8)
    assert out[2] == 2   # transition forced at frame 2


def test_viterbi_short_video_returns_uniform():
    # T=2 frames, 4 labels: DP is impossible; aligner falls back to uniform.
    out = viterbi_align(torch.randn(2, 5), [0, 1, 2, 0])
    assert len(out) == 2
    assert set(out) <= {1, 2}   # every real token covered, no silence padding


def test_token_spans_groups_consecutive_frames():
    spans = token_spans([0, 0, 5, 5, 5, 9, 0], [0, 5, 9, 0])
    # frame_to_label holds class ids; position = labels index - 1
    assert spans == [(0, 2, 5), (1, 5, 6)]


def test_token_spans_omits_unassigned_tokens():
    assert token_spans([0, 0, 5, 5], [0, 5, 9, 0]) == [(0, 2, 4)]


def test_token_spans_splits_repeated_token_run():
    # "7 7" style transcripts: one run of class 5 covering two occurrences
    # in labels is split uniformly into contiguous per-occurrence spans.
    spans = token_spans([0, 5, 5, 5, 5, 5, 5, 0], [0, 5, 5, 0])
    assert spans == [(0, 1, 4), (1, 4, 7)]


def test_token_spans_omits_trailing_occurrences_of_short_run():
    # A run shorter than the occurrence count covers as far as possible;
    # trailing occurrences are omitted, never duplicated.
    spans = token_spans([0, 0, 5, 0, 0], [0, 5, 5, 0])
    assert spans == [(0, 2, 3)]


def test_uniform_token_spans_contiguous_cover():
    spans = uniform_token_spans(7, 3)
    assert spans == [(0, 0, 3), (1, 3, 5), (2, 5, 7)]
