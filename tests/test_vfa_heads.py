"""Shape/masking tests for the VFA heads.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_vfa_heads.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vfa_heads import VFAHeads, WordVerificationHead, pool_tokens


def test_frame_and_boundary_head_shapes():
    heads = VFAHeads(vocab_size=13, feature_dim=64)
    feats = torch.randn(2, 10, 64)
    assert heads.frame(feats).shape == (2, 10, 13)
    assert heads.boundary(feats).shape == (2, 10)


def test_pool_tokens_is_masked_mean():
    feats = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)  # frames 0..3
    spans = torch.zeros(1, 4, 2)
    spans[0, 0:2, 0] = 1.0     # token 0: frames 0-1
    spans[0, 2:4, 1] = 1.0     # token 1: frames 2-3
    pooled = pool_tokens(feats, spans)
    assert torch.allclose(pooled[0, 0], (feats[0, 0] + feats[0, 1]) / 2)
    assert torch.allclose(pooled[0, 1], (feats[0, 2] + feats[0, 3]) / 2)


def test_pool_tokens_empty_span_is_zero():
    feats = torch.randn(1, 4, 3)
    spans = torch.zeros(1, 4, 2)
    spans[0, 0:4, 0] = 1.0     # token 1 has no frames
    pooled = pool_tokens(feats, spans)
    assert pooled[0, 1].abs().sum() == 0


def test_word_verification_head_masks_invalid_tokens():
    head = WordVerificationHead(feature_dim=64)
    pooled = torch.randn(1, 3, 64)
    text = torch.randn(1, 3, 64)
    mask = torch.tensor([[True, True, False]])
    logits = head(pooled, text, mask)
    assert logits.shape == (1, 3)
    assert logits[0, 2].item() == 0.0
