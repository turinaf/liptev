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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.verification_model import LipTextVerificationModel
from training.train import (
    train_one_epoch,
    vfa_frame_ce_loss,
    vfa_boundary_loss,
    word_verification_loss,
)
from utils.alignment import frame_token_targets


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


# --- Task 7: VFA training losses -------------------------------------------
# The three VFA losses are module-level free functions so they can be
# verified numerically without a model or dataloader.

def test_vfa_frame_ce_ignores_unsupervised_frames():
    logits = torch.zeros(1, 3, 5)
    logits[0, 0, 2] = 5.0; logits[0, 1, 2] = 5.0; logits[0, 2, 4] = 5.0
    # Frame 2 is unsupervised and confidently wrong (argmax 4 vs id 2), so an
    # implementation that ignores frame_valid gets ~1.69 instead of ~0.0266.
    ids = torch.tensor([[2, 2, 2]])
    valid = torch.tensor([[True, True, False]])
    loss = vfa_frame_ce_loss(logits, ids, valid)
    # Supervised frames are argmax-correct (target column holds the only 5.0),
    # so the loss is the small residual softmax mass on the zero logits, not
    # exactly zero — and frame 2 must contribute nothing.
    expected = F.cross_entropy(logits[0, :2], ids[0, :2])
    assert torch.allclose(loss, expected), f"{loss.item()} vs {expected.item()}"
    assert loss.item() < 0.05


def test_vfa_frame_ce_penalizes_wrong_frame():
    logits = torch.zeros(1, 1, 5)
    logits[0, 0, 1] = 5.0       # argmax 1, target 2
    ids = torch.tensor([[2]])
    valid = torch.tensor([[True]])
    assert vfa_frame_ce_loss(logits, ids, valid).item() > 0


def test_vfa_boundary_loss_masks_frames():
    bound_logits = torch.tensor([[5.0, 5.0, -5.0]])
    targets = torch.tensor([[0.0, 1.0, 1.0]])
    valid = torch.tensor([[True, True, False]])
    # frames 0,1 supervised; frame 2 is wrong (cost ~5.0) and must not pull
    # the mean — the loss must equal BCE over the supervised frames only.
    loss = vfa_boundary_loss(bound_logits, targets, valid)
    expected = F.binary_cross_entropy_with_logits(bound_logits[0, :2], targets[0, :2])
    assert torch.allclose(loss, expected), f"{loss.item()} vs {expected.item()}"
    assert torch.isfinite(loss)


def test_word_verification_loss_only_on_valid_tokens():
    logits = torch.tensor([[5.0, -5.0, 5.0]])   # token 1 wrong
    labels = torch.tensor([[1.0, 1.0, 1.0]])
    valid = torch.tensor([[True, True, False]])
    loss = word_verification_loss(logits, labels, valid)
    # only tokens 0 (correct) and 1 (wrong) count — each contributes
    # softplus(-logit * label): token 0 correct at +5, token 1 wrong at -5.
    expected = 0.5 * (F.softplus(torch.tensor(-5.0)) + F.softplus(torch.tensor(5.0)))
    assert torch.allclose(loss, expected)


def test_losses_zero_when_nothing_valid():
    assert vfa_frame_ce_loss(torch.zeros(1, 2, 3), torch.zeros(1, 2).long(),
                             torch.zeros(1, 2, dtype=torch.bool)).item() == 0.0
    assert vfa_boundary_loss(torch.zeros(1, 2), torch.zeros(1, 2),
                             torch.zeros(1, 2, dtype=torch.bool)).item() == 0.0
    assert word_verification_loss(torch.zeros(1, 2), torch.zeros(1, 2),
                                  torch.zeros(1, 2, dtype=torch.bool)).item() == 0.0
