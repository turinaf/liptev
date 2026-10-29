"""Final-review fixes: legacy checkpoints in test.py, NaN-free JSON reports,
per-token score scales, digit fps resolution, global JSON verdict.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_final_fixes.py -q
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import (
    build_token_report,
    format_score_decision,
    resolve_video_fps,
    rescale_word_scores,
    score_verdict,
    spans_to_score_maps,
)
from models.verification_model import LipTextVerificationModel
from utils.checkpoint import save_checkpoint


# --- Fix 1: legacy checkpoints must strict-load through test.py ---


def _small_model(use_vfa):
    # feature_dim must match build_eval_model's hardcoded 256 so the saved
    # state dict strict-loads through the entrypoint's construction.
    return LipTextVerificationModel(
        vocab_size=13, feature_dim=256, lip_backbone_pretrained=False, use_vfa=use_vfa
    )


def test_legacy_checkpoint_loads_through_test_entrypoint(tmp_path):
    # test.py must build use_vfa from the checkpoint stamp: a legacy state
    # dict (no VFA heads) strict-loads into a use_vfa=False model only.
    from types import SimpleNamespace

    import test as test_entrypoint

    model = _small_model(use_vfa=False)
    path = str(tmp_path / "legacy.pt")
    save_checkpoint(model.state_dict(), str(path), alignment_mode="sim", vfa_heads=False)

    checkpoint = torch.load(path)
    cfg = SimpleNamespace(alignment_mode="sim", lip_backbone_pretrained=False)
    rebuilt = test_entrypoint.build_eval_model(
        checkpoint, cfg, vocab_size=13, device=torch.device("cpu")
    )
    assert rebuilt.use_vfa is False
    rebuilt.load_state_dict(checkpoint["model_state_dict"])  # strict


def test_vfa_checkpoint_loads_through_test_entrypoint(tmp_path):
    from types import SimpleNamespace

    import test as test_entrypoint

    model = _small_model(use_vfa=True)
    path = str(tmp_path / "vfa.pt")
    save_checkpoint(model.state_dict(), str(path), alignment_mode="both", vfa_heads=True)

    checkpoint = torch.load(path)
    cfg = SimpleNamespace(alignment_mode="sim", lip_backbone_pretrained=False)
    rebuilt = test_entrypoint.build_eval_model(
        checkpoint, cfg, vocab_size=13, device=torch.device("cpu")
    )
    assert rebuilt.use_vfa is True
    assert rebuilt.alignment_lip_text.mode == "both"  # stamp wins over config
    rebuilt.load_state_dict(checkpoint["model_state_dict"])  # strict


# --- Fix 2: zero-length spans must not produce NaN scores ---


def test_spans_to_score_maps_drops_zero_length_spans():
    maps = spans_to_score_maps(
        [(0, 0, 5), (1, 5, 5), (2, 5, 8)],
        posterior_scores=[0.9, 0.123, 0.4],
        verif_scores=[0.8, 0.456, 0.3],
        word_scores=[0.7, 0.789, 0.2],
    )
    post, ws, ver = maps
    assert 0 in post and 2 in post
    assert 1 not in post and 1 not in ws and 1 not in ver


def test_spans_to_score_maps_without_word_scores():
    post, ws, ver = spans_to_score_maps(
        [(0, 0, 5)], posterior_scores=[0.9], verif_scores=[0.8], word_scores=None
    )
    assert post[0] == 0.9 and ver[0] == 0.8 and ws == {}


def test_zero_length_span_token_takes_missing_path():
    # Video shorter than the transcript: the uniform fallback degenerates the
    # last token to a zero-length span -> report nulls + low confidence, not NaN.
    rows = build_token_report(
        ["1", "2", "3"],
        [(0, 0, 5), (1, 5, 5)],
        seconds_per_frame=0.04,
        posterior_scores=[0.5],
        word_scores=[0.5],
        verification_scores=[0.5],
        thresholds={"verification": 0.5},
        fallback=True,
    )
    assert rows[0]["alignment_score"] == 0.5
    assert rows[1]["start_s"] is None and rows[1]["end_s"] is None
    assert rows[1]["alignment_score"] is None
    assert rows[1]["low_confidence"] is True


def test_json_report_has_no_nan_for_zero_length_spans():
    rows = build_token_report(
        ["1", "2"],
        [(0, 0, 5), (1, 5, 5)],
        seconds_per_frame=0.04,
        posterior_scores=[0.5],
        word_scores=[0.5],
        verification_scores=[0.5],
        thresholds={},
        fallback=True,
    )
    text = json.dumps({"global": {}, "tokens": rows})
    assert "NaN" not in text and "Infinity" not in text


# --- Fix 3: per-token word scores on the same [0, 1] scale ---


def test_rescale_word_scores_maps_cosine_to_unit_interval():
    t = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    out = rescale_word_scores(t)
    assert torch.allclose(out, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert out.min() >= 0.0 and out.max() <= 1.0


# --- Fix 4: digit fps resolution for processed videos ---


def test_resolve_fps_digit_meta_json_beside_video(tmp_path):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    (video_dir / "meta.json").write_text(json.dumps({"f01": {"source_fps": 29.87}}))
    fps, source = resolve_video_fps(str(video_dir / "f01.mp4"), "digit", cap_fps=25.0)
    assert fps == 29.87
    assert "meta.json" in source


def test_resolve_fps_digit_meta_json_speaker_level(tmp_path):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    (tmp_path / "meta.json").write_text(json.dumps({"f01": {"source_fps": 25.0}}))
    fps, source = resolve_video_fps(str(video_dir / "f01.mp4"), "digit", cap_fps=25.0)
    assert fps == 25.0
    assert "meta.json" in source


def test_resolve_fps_digit_falls_back_to_container_without_meta(tmp_path):
    # No meta.json, no sibling wav -> the container probe is the last resort.
    fps, source = resolve_video_fps(str(tmp_path / "f01.mp4"), "digit", cap_fps=25.0)
    assert fps == 25.0
    assert "container" in source


def test_resolve_fps_digit_defaults_without_container_label(tmp_path):
    fps, source = resolve_video_fps(str(tmp_path / "f01.mp4"), "digit", cap_fps=0.0)
    assert fps == 25.0
    assert "default" in source


def test_resolve_fps_grid_keeps_container_probe(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"v1": {"source_fps": 99.0}}))
    fps, source = resolve_video_fps(str(tmp_path / "v1.mp4"), "grid", cap_fps=25.0)
    assert fps == 25.0
    assert "container" in source


# --- Fix 5: global JSON verdict ---


def test_score_verdict_match_mismatch_uncalibrated():
    block = {"threshold": 0.31, "eer": 0.42}
    assert score_verdict(0.31, block) == "MATCH"
    assert score_verdict(0.30, block) == "MISMATCH"
    assert score_verdict(0.5, None) == "uncalibrated"
    assert score_verdict(0.5, {"threshold": None}) == "uncalibrated"
    assert score_verdict(0.5, {"eer": 0.5}) == "uncalibrated"


def test_format_decision_uses_same_verdict_rule():
    block = {"threshold": 0.31, "eer": 0.42}
    assert "MATCH" in format_score_decision("Score", 0.31, block)
    assert "MISMATCH" in format_score_decision("Score", 0.30, block)
    assert "uncalibrated" in format_score_decision("Score", 0.5, None)
