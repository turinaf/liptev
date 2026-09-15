"""Numerical stability of the verification (BCE) term.

The BCE term must be computed on the raw classifier logit
(BCEWithLogitsLoss), never on a post-sigmoid probability (BCELoss).
Empirically in torch 2.9.0: a saturated probability (sigmoid(30) rounds
to exactly 1.0 in float32) makes BCELoss emit a huge clamped upstream
gradient (~5e11) which the in-graph sigmoid then kills with
sigma'(z) = p(1-p) = 0 — the classifier receives exactly zero gradient,
so confidently-wrong pairs silently stop training (and the loss still
looks finite). Unclamped kernels in other builds return inf/NaN from
the same code. BCEWithLogitsLoss keeps the analytic gradient
sigma(z) - y alive and stable.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_training_loss.py -q
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.verification_model import LipTextVerificationModel
from training.train import train_one_epoch


def _tiny_model(vocab_size=13):
    torch.manual_seed(0)
    return LipTextVerificationModel(
        vocab_size=vocab_size, feature_dim=64, lip_backbone_pretrained=False
    )


def _tiny_batch(vocab_size=13, B=2, T=8, L=5):
    # One positive pair, one negative pair, partial padding on sample 2 —
    # the same key set the real collate_fn emits (minus audio).
    return {
        "video": torch.rand(B, T, 3, 96, 96) * 2 - 1,
        "phonemes": torch.randint(2, vocab_size, (B, L)),
        "label": torch.tensor([1.0, 0.0]),
        "phoneme_padding_mask": _pad_mask(B, L, [L, 3]),
        "video_padding_mask": _pad_mask(B, T, [T, 6]),
        "alignment_targets": torch.zeros(B, T, L),
        "speech_mask": torch.zeros(B, T, dtype=torch.bool),
        "has_align_target": torch.tensor([True, False]),
    }


def _pad_mask(B, L, lens):
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, n in enumerate(lens):
        mask[i, n:] = True
    return mask


def _final_linear(model):
    return [m for m in model.classifier if isinstance(m, torch.nn.Linear)][-1]


def test_model_exposes_raw_logit():
    # The training loop needs the pre-sigmoid logit; downstream consumers
    # (inference, eval) keep receiving the sigmoid probability.
    model = _tiny_model()
    batch = _tiny_batch()
    with torch.no_grad():
        score, out = model(
            batch["video"], batch["phonemes"],
            video_padding_mask=batch["video_padding_mask"],
            phoneme_padding_mask=batch["phoneme_padding_mask"],
        )

    assert "consistency_logit" in out, \
        "model must expose the pre-sigmoid logit for BCEWithLogitsLoss"
    logit = out["consistency_logit"]
    assert logit.shape == score.shape == (2, 1)
    assert torch.allclose(score, torch.sigmoid(logit), atol=1e-6)
    assert (score >= 0).all() and (score <= 1).all()


def test_model_exposes_raw_logit_with_audio():
    # Same invariant on the audio-enabled forward branch.
    torch.manual_seed(0)
    model = LipTextVerificationModel(
        vocab_size=13, feature_dim=64, lip_backbone_pretrained=False
    )
    real_lens = [16000, 9000]
    audio = torch.zeros(2, 1, max(real_lens))
    for i, n in enumerate(real_lens):
        audio[i, :, :n] = torch.randn(1, n) * 0.1
    audio_padding_mask = torch.arange(max(real_lens)).unsqueeze(0) >= torch.tensor(real_lens).unsqueeze(1)

    batch = _tiny_batch()
    with torch.no_grad():
        score, out = model(
            batch["video"], batch["phonemes"], audio,
            video_padding_mask=batch["video_padding_mask"],
            phoneme_padding_mask=batch["phoneme_padding_mask"],
            audio_padding_mask=audio_padding_mask,
        )

    assert "consistency_logit" in out, "audio branch must expose the pre-sigmoid logit too"
    assert torch.allclose(score, torch.sigmoid(out["consistency_logit"]), atol=1e-6)
    assert (score >= 0).all() and (score <= 1).all()


def test_training_gradient_survives_sigmoid_saturation():
    # Regression: with BCELoss on probabilities, a saturated classifier
    # (sigmoid(30) == 1.0 exactly in float32) sends the classifier exactly
    # zero gradient — measured abs-sum 0.0 on this code path — so a
    # confidently-wrong pair never gets corrected.
    model = _tiny_model()
    with torch.no_grad():
        _final_linear(model).bias.fill_(30.0)

    batch = _tiny_batch()
    batch["alignment_targets"][:, 0:3, 2] = 1.0
    batch["speech_mask"][:, 0:3] = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = train_one_epoch(model, [batch], optimizer, torch.device("cpu"), epoch=1)

    assert math.isfinite(loss), f"training loss not finite on saturated logits: {loss}"
    grad = _final_linear(model).weight.grad
    assert grad is not None, "classifier must receive gradient from the BCE term"
    assert torch.isfinite(grad).all(), "saturated logits must not blow up gradients"
    assert grad.abs().sum().item() > 1e-6, \
        "BCE gradient is dead on saturated samples — use BCEWithLogitsLoss on the raw logit"
