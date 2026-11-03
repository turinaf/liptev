import collections
import json
import math
import random
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from evaluation.evaluate import evaluate_model, forced_word_scores
from evaluation.metrics import compute_word_metrics, save_eer_thresholds, checkpoint_sha256
from inference import build_token_report, validate_calibration
from models.verification_model import LipTextVerificationModel
from training.train import train_one_epoch, word_verification_loss, vfa_frame_ce_loss
from utils.dataset_utils import collate_fn_factory
from utils.negative_sampler import GRID_WORD_CLASSES, NegativeSampler
from utils.tokenizer import TranscriptTokenizer
from utils.viterbi import forced_token_spans


def _item():
    words = "bin blue at b eight now".split()
    return {
        "video": torch.randn(16, 3, 32, 32), "video_len": 14,
        "transcript": " ".join(words),
        "word_times": [(word, 1 + 2 * pos, 3 + 2 * pos) for pos, word in enumerate(words)],
    }


def test_balanced_word_loss_gives_equal_total_class_weight():
    logits = torch.zeros(1, 34, requires_grad=True)
    labels = torch.tensor([[1.0] * 33 + [0.0]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    loss = word_verification_loss(logits, labels, valid, balanced=True)
    assert torch.allclose(loss, torch.tensor(math.log(2)))
    loss.backward()
    assert torch.allclose(logits.grad[0, :33].sum(), torch.tensor(-0.25))
    assert torch.allclose(logits.grad[0, 33], torch.tensor(0.25))


@pytest.mark.parametrize("target", [0.0, 1.0])
def test_balanced_loss_single_class_and_empty_mask_are_finite(target):
    logits = torch.tensor([[30.0, -30.0]], requires_grad=True)
    labels = torch.full_like(logits, target)
    valid = torch.tensor([[True, False]])
    loss = word_verification_loss(logits, labels, valid, balanced=True)
    assert torch.allclose(loss, F.binary_cross_entropy_with_logits(logits[:, :1], labels[:, :1]))
    loss.backward()
    assert logits.grad[0, 1] == 0
    assert word_verification_loss(logits, labels, ~torch.ones_like(valid), balanced=True) == 0


def test_controlled_sampling_covers_single_multi_and_cross_class():
    sampler = NegativeSampler([], rng=random.Random(42))
    tokenizer = TranscriptTokenizer("grid", "word")
    positive = "bin blue at b eight now"
    counts = collections.Counter()
    cross = 0
    for _ in range(2000):
        negative = sampler.word_substitution(positive)
        assert len(tokenizer.encode(negative)) == 6
        words = negative.split()
        changed = sum(a != b for a, b in zip(positive.split(), words))
        counts[changed] += 1
        cross += any(word not in GRID_WORD_CLASSES[pos] for pos, word in enumerate(words))
    assert set(counts) == set(range(1, 7))
    assert 0.45 < counts[1] / 2000 < 0.55
    assert cross > 100
    with pytest.raises(ValueError, match="empty"):
        sampler.word_substitution("")


def test_controlled_multi_negatives_preserve_spans_and_label_every_word():
    item = _item()
    collate = collate_fn_factory("grid", token_level="word", balanced_word_supervision=True)
    with (mock.patch("utils.negative_sampler.random.random", return_value=0.1),
          mock.patch.object(NegativeSampler, "word_substitution",
                            return_value="green white at eight nine now")):
        batch = collate([item])
    assert batch["label"].tolist() == [1, 0]
    assert batch["word_labels"][1].tolist() == [0, 0, 1, 0, 0, 1]
    assert batch["word_label_mask"].all()
    assert torch.equal(batch["word_spans"][0], batch["word_spans"][1])
    assert batch["alignment_targets"][1].sum() == 0
    assert batch["has_align_target"].tolist() == [1, 0]
    assert not batch["word_spans"][:, 14:].any()


def test_generic_multi_negatives_remain_masked_and_validation_is_repeatable():
    collate = collate_fn_factory(
        "grid", token_level="word", balanced_word_supervision=True, seed=42)
    item = _item()
    state = random.getstate()
    first, second = collate([item]), collate([item])
    assert first["transcript"] == second["transcript"]
    assert random.getstate() == state
    generic = collate_fn_factory("grid", token_level="word", balanced_word_supervision=True)
    with (mock.patch("utils.negative_sampler.random.random", return_value=0.99),
          mock.patch.object(NegativeSampler, "sample", return_value="lay red in z zero soon")):
        batch = generic([item])
    assert not batch["word_label_mask"][1].any()


def test_balanced_mode_never_labels_identical_transcript_as_negative():
    item = _item()
    collate = collate_fn_factory("grid", token_level="word", balanced_word_supervision=True)
    with (mock.patch("utils.negative_sampler.random.random", return_value=0.99),
          mock.patch.object(NegativeSampler, "sample", return_value=item["transcript"])):
        batch = collate([item])
    assert batch["transcript"][0] != batch["transcript"][1]
    assert (batch["word_labels"][1] == 0).any()


def test_silence_supervision_includes_real_silence_not_padding_or_negative_frames():
    with (mock.patch("utils.negative_sampler.random.random", return_value=0.1),
          mock.patch.object(NegativeSampler, "word_substitution", return_value="lay red at b eight now")):
        batch = collate_fn_factory(
            "grid", token_level="word", balanced_word_supervision=True)([_item()])
    model = LipTextVerificationModel(
        vocab_size=53, feature_dim=16, lip_backbone_pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with mock.patch("training.train.vfa_frame_ce_loss", wraps=vfa_frame_ce_loss) as frame_loss:
        loss = train_one_epoch(
            model, [batch], optimizer, torch.device("cpu"), epoch=1,
            balanced_word_supervision=True, supervise_silence=True)
    assert math.isfinite(loss)
    _, ids, valid = frame_loss.call_args.args
    assert valid[0, :14].all()
    assert not valid[0, 14:].any() and not valid[1].any()
    assert ids[0, 0] == ids[0, 13] == 0
    assert (ids[0, 1:13] != 0).all()


def test_word_metrics_need_both_classes_and_reject_nonfinite_scores():
    with pytest.raises(ValueError, match="both classes"):
        compute_word_metrics(torch.ones(3), torch.ones(3))
    with pytest.raises(ValueError, match="finite"):
        compute_word_metrics(torch.tensor([0, 1]), torch.tensor([float("nan"), 0.5]))
    metrics = compute_word_metrics(torch.tensor([0, 0, 1, 1]), torch.tensor([0.1, 0.2, 0.8, 0.9]))
    assert metrics["word_eer"] == 0
    assert metrics["word_positive_count"] == metrics["word_negative_count"] == 2
    assert metrics["word_false_match_rate"] == metrics["word_false_reject_rate"] == 0


def test_forced_scoring_uses_decoded_spans_and_masks_missing_annotation_and_fallback():
    batch = {
        "video": torch.zeros(3, 8, 3, 2, 2),
        "phonemes": torch.tensor([[2, 3], [2, 3], [2, 3]]),
        "phoneme_padding_mask": torch.zeros(3, 2, dtype=torch.bool),
        "video_padding_mask": torch.tensor([[False] * 8, [False] * 8, [False] + [True] * 7]),
        "word_spans": torch.ones(3, 8, 2),
        "word_label_mask": torch.ones(3, 2, dtype=torch.bool),
        "word_labels": torch.tensor([[1., 1.], [0., 1.], [0., 0.]]),
    }
    batch["word_spans"][1, :, 1] = 0
    frame_logits = torch.full((3, 8, 4), -10.)
    frame_logits[:, 0, 0] = frame_logits[:, 7, 0] = 10
    frame_logits[:, 1:4, 2] = frame_logits[:, 4:7, 3] = 10
    out = {
        "vfa_frame_logits": frame_logits,
        "vfa_bound_logits": torch.full((3, 8), -10.),
        "lip_feats": torch.zeros(3, 8, 16), "text_feats": torch.zeros(3, 2, 16),
    }
    model = mock.Mock()
    model.score_word_spans.return_value = (torch.zeros(3, 2), torch.ones(3, 2, dtype=torch.bool))
    # A Mock claims every attribute exists, so avoid the DDP module branch.
    del model.module
    scores, labels = forced_word_scores(model, out, batch, 0.8)
    weights = model.score_word_spans.call_args.args[2]
    assert weights[0, 1:4, 0].all() and weights[0, 4:7, 1].all()
    assert weights[0, 0].sum() == weights[0, 7].sum() == 0
    assert labels.tolist() == [1, 1, 0]
    assert scores.tolist() == [0.5] * 3


def test_shared_decoder_short_video_marks_fallback():
    spans, fallback = forced_token_spans(torch.zeros(1, 4), [2, 3], np.zeros(1))
    assert fallback
    assert spans == [(0, 0, 1), (1, 1, 1)]


def test_word_verdicts_never_reuse_global_threshold_or_calibrate_fallback():
    args = (["green"], [(0, 0, 2)], 0.04, [0.2], [0.3], [0.9])
    rows = build_token_report(*args, {"verification": {"threshold": 0.008}}, False)
    assert rows[0]["verdict"] == "uncalibrated"
    rows = build_token_report(*args, {"word_verification": {"threshold": 0.95}}, False)
    assert rows[0]["verdict"] == "MISMATCH"
    rows = build_token_report(*args, {"word_verification": {"threshold": 0.5}}, True)
    assert rows[0]["verdict"] == "uncalibrated" and rows[0]["low_confidence"]


def test_word_calibration_roundtrip_and_checkpoint_binding(tmp_path):
    from evaluation.metrics import compute_metrics
    labels, scores = torch.tensor([0., 1.]), torch.tensor([0.1, 0.9])
    metrics = compute_metrics(labels, scores, scores)
    metrics.update(compute_word_metrics(labels, scores))
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"weights")
    metadata = {
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "dataset": "grid", "token_level": "word", "boundary_threshold": 0.8, "split": "val",
    }
    path = tmp_path / "calibration.json"
    save_eer_thresholds(metrics, metadata, path)
    calibration = json.loads(path.read_text())
    assert calibration["word_verification"]["span_source"] == "viterbi"
    assert validate_calibration(calibration, checkpoint, "grid", "word", 0.8) == calibration
    mismatched_decoder = validate_calibration(calibration, checkpoint, "grid", "word", 0.7)
    assert "word_verification" not in mismatched_decoder
    assert "verification" in mismatched_decoder
    assert validate_calibration(calibration, checkpoint, "digit", "word", 0.8) is None
    checkpoint.write_bytes(b"new-weights-same-filename")
    assert validate_calibration(calibration, checkpoint, "grid", "word", 0.8) is None


class _DistributedWordModel(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def forward(self, video, tokens, audio=None, **kwargs):
        B, T, L = len(video), video.shape[1], tokens.shape[1]
        frame_logits = torch.full((B, T, 4), -10.)
        frame_logits[:, 0, 0] = frame_logits[:, -1, 0] = 10
        frame_logits[:, 1:4, 2] = frame_logits[:, 4:7, 3] = 10
        return torch.tensor([[0.9], [0.1]]), {
            "alignment_score": torch.tensor([0.9, 0.1]),
            "vfa_frame_logits": frame_logits, "vfa_bound_logits": torch.full((B, T), -10.),
            "lip_feats": torch.zeros(B, T, 1),
            "text_feats": torch.full((B, L, 1), 3. if self.rank == 0 else -3.),
        }

    def score_word_spans(self, lip_feats, text_feats, spans):
        return text_feats.squeeze(-1), spans.sum(dim=1) > 0


def _word_validation_worker(rank, rendezvous, output_dir):
    import torch.distributed as dist
    from pathlib import Path
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        batch = {
            "video": torch.zeros(2, 8, 3, 2, 2),
            "phonemes": torch.tensor([[2, 3], [2, 3]]),
            "label": torch.tensor([1., 0.]),
            "phoneme_padding_mask": torch.zeros(2, 2, dtype=torch.bool),
            "video_padding_mask": torch.zeros(2, 8, dtype=torch.bool),
            "word_spans": torch.ones(2, 8, 2),
            "word_labels": torch.full((2, 2), 1. if rank == 0 else 0.),
            "word_label_mask": torch.ones(2, 2, dtype=torch.bool),
        }
        metrics = evaluate_model(_DistributedWordModel(rank), [batch], torch.device("cpu"), word_metrics=True)
        Path(output_dir, f"word-rank-{rank}.json").write_text(json.dumps(metrics))
    finally:
        dist.destroy_process_group()


def test_word_metrics_gather_classes_across_distributed_ranks(tmp_path):
    torch.multiprocessing.spawn(
        _word_validation_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    metrics = [json.loads((tmp_path / f"word-rank-{rank}.json").read_text()) for rank in range(2)]
    assert metrics[0] == metrics[1]
    assert metrics[0]["word_positive_count"] == metrics[0]["word_negative_count"] == 4
    assert metrics[0]["word_eer"] == 0


def test_calibration_persists_validation_not_test_metrics(tmp_path):
    import test as entrypoint
    from utils.dataset_utils import Config
    tokenizer = TranscriptTokenizer("grid", "word")
    cfg = Config("config.grid.json")
    cfg.processed_data_dir = cfg.output_dir = str(tmp_path)
    checkpoint = {
        "model_state_dict": {}, "alignment_mode": "both", "vfa_heads": True,
        "dataset": "grid", "token_level": "word", "token_vocab": tokenizer.vocab,
    }
    val_metrics = {"roc_auc": 0.9, "verification_eer": 0.1, "verification_accuracy": 0.9}
    test_metrics = {"roc_auc": 0.8, "verification_eer": 0.2, "verification_accuracy": 0.8}
    with (mock.patch.object(entrypoint, "Config", return_value=cfg),
          mock.patch.object(entrypoint, "GRIDDataset", return_value=[_item()]),
          mock.patch.object(entrypoint, "_resolve_latest_checkpoint", return_value="word.pt"),
          mock.patch.object(entrypoint.torch, "load", return_value=checkpoint),
          mock.patch.object(entrypoint, "build_eval_model"),
          mock.patch.object(entrypoint, "evaluate_model", side_effect=[val_metrics, test_metrics]) as evaluate,
          mock.patch.object(entrypoint, "checkpoint_sha256", return_value="digest"),
          mock.patch.object(entrypoint, "save_eer_thresholds") as save):
        entrypoint.test()
    assert evaluate.call_count == 2
    assert save.call_args.args[0] == val_metrics
    assert save.call_args.args[1]["split"] == "val"


def test_digit_evaluation_gets_independent_token_calibration_without_word_sampling(tmp_path):
    import test as entrypoint
    from utils.dataset_utils import Config
    cfg = Config("config.digit.json")
    cfg.processed_data_dir = cfg.output_dir = str(tmp_path)
    checkpoint = {"model_state_dict": {}, "alignment_mode": "sim", "vfa_heads": True}
    metrics = {"roc_auc": 0.9, "verification_eer": 0.1, "verification_accuracy": 0.9}
    with (mock.patch.object(entrypoint, "Config", return_value=cfg),
          mock.patch.object(entrypoint, "DigitDataset", return_value=[{}]),
          mock.patch.object(entrypoint, "_resolve_latest_checkpoint", return_value="digit.pt"),
          mock.patch.object(entrypoint.torch, "load", return_value=checkpoint),
          mock.patch.object(entrypoint, "build_eval_model"),
          mock.patch.object(entrypoint, "evaluate_model", return_value=metrics) as evaluate,
          mock.patch.object(entrypoint, "checkpoint_sha256", return_value="digest"),
          mock.patch.object(entrypoint, "collate_fn_factory") as collate,
          mock.patch.object(entrypoint, "save_eer_thresholds")):
        entrypoint.test("config.digit.json")
    assert all(call.kwargs["word_metrics"] for call in evaluate.call_args_list)
    assert all(not call.kwargs["balanced_word_supervision"] for call in collate.call_args_list)
