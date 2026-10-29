"""use_vfa gating and VFA outputs of the verification model.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_vfa_model.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.verification_model import LipTextVerificationModel


def _model(use_vfa):
    # backbone_pretrained=False keeps the test off torchvision weights
    return LipTextVerificationModel(
        vocab_size=13, feature_dim=64, lip_backbone_pretrained=False, use_vfa=use_vfa
    )


def _inputs():
    torch.manual_seed(0)
    video = torch.randn(2, 8, 3, 96, 96)
    phones = torch.tensor([[1, 2, 3], [4, 5, 0]])
    video_mask = torch.zeros(2, 8, dtype=torch.bool)
    video_mask[1, 5:] = True
    phone_mask = phones.eq(0)
    spans = torch.zeros(2, 8, 3)
    spans[0, 0:3, 0] = 1.0; spans[0, 3:5, 1] = 1.0; spans[0, 5:8, 2] = 1.0
    spans[1, 0:2, 0] = 1.0; spans[1, 2:5, 1] = 1.0  # token 2 unsupervised
    return video, phones, video_mask, phone_mask, spans


def test_use_vfa_false_matches_legacy_state_dict_keys():
    model = _model(False)
    names = [n for n, _ in model.named_parameters()]
    assert not any(n.startswith(("vfa_heads", "word_head")) for n in names)


def test_use_vfa_false_has_no_vfa_outputs():
    video, phones, vmask, pmask, _ = _inputs()
    _, align_out = _model(False)(video, phones, None, video_padding_mask=vmask,
                                 phoneme_padding_mask=pmask)
    assert "vfa_frame_logits" not in align_out


def test_use_vfa_true_emits_frame_boundary_word_logits():
    video, phones, vmask, pmask, spans = _inputs()
    _, align_out = _model(True)(video, phones, None, video_padding_mask=vmask,
                                phoneme_padding_mask=pmask, word_spans=spans)
    assert align_out["vfa_frame_logits"].shape == (2, 8, 13)
    assert align_out["vfa_bound_logits"].shape == (2, 8)
    assert align_out["word_verif_logits"].shape == (2, 3)
    # token 2 of sample 1 has no span -> masked
    assert align_out["word_token_mask"][1, 2].item() is False or \
           align_out["word_token_mask"][1, 2].item() == 0


def test_word_logits_absent_without_spans():
    video, phones, vmask, pmask, _ = _inputs()
    _, align_out = _model(True)(video, phones, None, video_padding_mask=vmask,
                                phoneme_padding_mask=pmask)
    assert "word_verif_logits" not in align_out
    assert align_out["vfa_frame_logits"].shape == (2, 8, 13)


def test_score_word_spans_public_method():
    video, phones, vmask, pmask, spans = _inputs()
    model = _model(True)
    with torch.no_grad():
        _, align_out = model(video, phones, None, video_padding_mask=vmask,
                             phoneme_padding_mask=pmask)
        logits, mask = model.score_word_spans(
            align_out["lip_feats"], align_out["text_feats"], spans
        )
    assert logits.shape == (2, 3)
    assert mask[0].all()
    assert not mask[1, 2]


def test_legacy_checkpoint_loads_strict_into_use_vfa_false():
    # A state dict saved without VFA heads must strict-load into use_vfa=False.
    old = _model(False)
    sd = old.state_dict()
    fresh = _model(False)
    fresh.load_state_dict(sd)  # raises on any mismatch
