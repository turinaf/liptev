"""Checkpoint/run-config integrity: alignment_mode must travel with the weights.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_checkpoint_mode.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.checkpoint import (
    save_checkpoint,
    resolve_alignment_mode,
    check_alignment_supervision,
    resolve_vfa_heads,
)


def test_save_checkpoint_stamps_alignment_mode(tmp_path):
    # main.py must record the scoring mode next to the weights so an eval run
    # can't silently score a checkpoint that was trained under another mode.
    path = tmp_path / "checkpoint_ep1.pt"
    save_checkpoint({"lip_encoder.0.weight": torch.zeros(1)}, str(path), alignment_mode="attn")

    loaded = torch.load(path)
    assert loaded["alignment_mode"] == "attn"
    assert loaded["model_state_dict"]["lip_encoder.0.weight"].shape == (1,)


def test_saved_format_round_trips_through_existing_loader_unwrap(tmp_path):
    # test.py/inference.py unwrap with checkpoint.get('model_state_dict',
    # checkpoint); the wrapped format must keep working through that exact
    # expression, and weights_only loading must accept the str field.
    path = tmp_path / "checkpoint_ep2.pt"
    save_checkpoint({}, str(path), "both")

    checkpoint = torch.load(path, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    assert state_dict == {}


def test_resolve_prefers_checkpoint_mode_and_warns_on_mismatch(capsys):
    # The stamp wins over config: the classifier input distribution and the
    # eer_thresholds.json calibration belong to the checkpoint's mode.
    checkpoint = {"model_state_dict": {}, "alignment_mode": "attn"}
    assert resolve_alignment_mode(checkpoint, "sim") == "attn"

    out = capsys.readouterr().out
    assert "Warning" in out
    assert "attn" in out and "sim" in out


def test_resolve_matching_checkpoint_mode_is_silent(capsys):
    checkpoint = {"model_state_dict": {}, "alignment_mode": "sim"}
    assert resolve_alignment_mode(checkpoint, "sim") == "sim"
    assert "Warning" not in capsys.readouterr().out


def test_resolve_falls_back_to_config_for_legacy_bare_state_dict(capsys):
    # Pre-change checkpoints are bare state dicts: no stamp, no warning —
    # config (default "sim") stays authoritative for them.
    checkpoint = {"lip_encoder.0.weight": torch.zeros(1)}
    assert resolve_alignment_mode(checkpoint, "sim") == "sim"
    assert "Warning" not in capsys.readouterr().out


def test_resolve_falls_back_when_wrapped_dict_lacks_mode(capsys):
    checkpoint = {"model_state_dict": {}}
    assert resolve_alignment_mode(checkpoint, "both") == "both"


def test_check_alignment_supervision_warns_on_noop_lambda_align(capsys):
    # 'attn' mode has no sim_matrix, so lambda_align does nothing — the run
    # must not pretend otherwise.
    check_alignment_supervision("attn", lambda_align=5.0)

    out = capsys.readouterr().out
    assert "Warning" in out
    assert "lambda_align" in out


def test_check_alignment_supervision_silent_when_supervision_active(capsys):
    # 'sim'/'both' supervise the sim matrix; lambda_align 0 opts out on purpose.
    check_alignment_supervision("sim", 5.0)
    check_alignment_supervision("both", 5.0)
    check_alignment_supervision("attn", 0.0)
    assert capsys.readouterr().out == ""


def test_vfa_stamp_roundtrip(tmp_path):
    path = str(tmp_path / "ck.pt")
    save_checkpoint({}, path, alignment_mode="sim", vfa_heads=True)
    ckpt = torch.load(path)
    assert resolve_vfa_heads(ckpt) is True


def test_missing_stamp_resolves_false(capsys):
    assert resolve_vfa_heads({"alignment_mode": "sim"}) is False
    assert "vfa" in capsys.readouterr().out.lower()
