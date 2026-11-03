import json
import math
import random
from unittest import mock

import pytest
import torch

from inference import run_inference
from models.verification_model import LipTextVerificationModel
from training.train import ValidationMonitor, build_optimizer, train_one_epoch
from utils.alignment import frame_token_targets
from utils.checkpoint import checkpoint_tokenizer, save_checkpoint
from utils.dataset_utils import Config, collate_fn_factory
from utils.negative_sampler import GRID_WORD_CLASSES, NegativeSampler
from utils.tokenizer import TranscriptTokenizer


def _item():
    words = "bin blue at f two now".split()
    return {
        "video": torch.randn(16, 3, 32, 32),
        "video_len": 16,
        "transcript": " ".join(words),
        "word_times": [(word, 1 + 2 * i, 3 + 2 * i) for i, word in enumerate(words)],
    }


def test_grid_word_vocabulary_is_stable_and_complete():
    tokenizer = TranscriptTokenizer("grid", "word")
    assert len(tokenizer.vocab) == 53
    assert tokenizer.vocab == TranscriptTokenizer("grid", "word").vocab
    assert tokenizer.vocab["<pad>"] == 0 and tokenizer.vocab["<unk>"] == 1
    words = set().union(*GRID_WORD_CLASSES)
    assert words == set(tokenizer.vocab) - {"<pad>", "<unk>"}
    assert "please" in words and "w" not in words
    assert tokenizer.encode("BIN blue, at f two now") == [
        tokenizer.vocab[word] for word in "bin blue at f two now".split()]
    with pytest.raises(ValueError, match="outside the GRID vocabulary"):
        tokenizer.encode("unrecognized")


def test_word_negatives_remain_in_vocabulary():
    tokenizer = TranscriptTokenizer("grid", "word")
    sampler = NegativeSampler([], rng=random.Random(42))
    for _ in range(100):
        assert len(tokenizer.encode(sampler._phoneme_similar("bin blue at f two now"))) == 6


def test_word_collation_uses_exact_word_boundaries_and_negative_labels():
    tokenizer = TranscriptTokenizer("grid", "word")
    collate = collate_fn_factory("grid", token_level="word")
    with mock.patch.object(NegativeSampler, "sample", return_value="bin blue at b two now"):
        batch = collate([_item()])
    assert batch["phonemes"].shape == (2, 6)
    assert batch["phonemes"][0].tolist() == tokenizer.encode("bin blue at f two now")
    assert batch["phonemes"][1].tolist() == tokenizer.encode("bin blue at b two now")
    assert batch["alignment_targets"].shape == (2, 16, 6)
    for pos in range(6):
        assert batch["alignment_targets"][0, 1 + 2 * pos:3 + 2 * pos, pos].all()
    assert batch["has_align_target"].tolist() == [1, 0]
    assert batch["alignment_targets"][1].sum() == 0
    assert batch["word_labels"][1].tolist() == [1, 1, 1, 0, 1, 1]
    assert batch["word_label_mask"].all()
    assert torch.equal(batch["word_spans"][0], batch["word_spans"][1])
    ids, boundaries = frame_token_targets(batch["alignment_targets"], batch["phonemes"])
    assert ids[0, 7:9].tolist() == [tokenizer.vocab["f"]] * 2
    assert boundaries[0, 1:13].sum() == 5


def test_validation_negatives_are_repeatable_without_global_rng_changes():
    collate = collate_fn_factory("grid", token_level="word", seed=42)
    item = _item()
    state = random.getstate()
    first = collate([item])
    assert random.getstate() == state
    for _ in range(10):
        random.random()
    second = collate([item])
    assert first["transcript"] == second["transcript"]
    assert torch.equal(first["phonemes"], second["phonemes"])


def test_word_checkpoint_metadata_roundtrip_and_legacy_detection(tmp_path):
    tokenizer = TranscriptTokenizer("grid", "word")
    path = tmp_path / "word.pt"
    save_checkpoint({}, path, "both", dataset="grid", token_level="word",
                    token_vocab=tokenizer.vocab, dropout=0.2)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["dropout"] == 0.2
    assert checkpoint_tokenizer(checkpoint, "grid").vocab == tokenizer.vocab
    assert checkpoint_tokenizer({}, "grid").token_level == "phoneme"
    assert len(checkpoint_tokenizer({}, "digit").vocab) == 13
    with pytest.raises(ValueError, match="dataset"):
        checkpoint_tokenizer(checkpoint, "digit")
    checkpoint["token_vocab"]["bin"] = 999
    with pytest.raises(ValueError, match="vocabulary"):
        checkpoint_tokenizer(checkpoint, "grid")


def test_grid_configs_enable_word_training_and_overfit_controls():
    for path in ("config.json", "config.grid.json"):
        cfg = Config(path)
        assert cfg.token_level == "word"
        assert cfg.epochs == 30
        assert cfg.weight_decay == 0.01
        assert cfg.dropout == 0.2
        assert cfg.early_stopping_patience == 5
        assert cfg.output_dir.endswith("grid-word")
    digit = Config("config.digit.json")
    assert digit.weight_decay == digit.dropout == digit.early_stopping_patience == 0
    assert len(TranscriptTokenizer("digit", digit.token_level).vocab) == 13


def test_validation_monitor_selects_best_and_stops_on_plateau():
    monitor = ValidationMonitor(patience=2)
    assert monitor.update(0.2)
    assert monitor.update(0.1)
    assert not monitor.update(0.1)
    assert not monitor.should_stop
    assert not monitor.update(0.15)
    assert monitor.should_stop and monitor.best_eer == 0.1
    assert monitor.update(0.05)
    assert not monitor.should_stop
    with pytest.raises(ValueError, match="not finite"):
        monitor.update(float("nan"))
    disabled = ValidationMonitor()
    for _ in range(10):
        disabled.update(0.2)
    assert not disabled.should_stop


class _ScoreModel(torch.nn.Module):
    def forward(self, video, tokens, audio=None, **kwargs):
        return video.unsqueeze(-1), {"alignment_score": video}


def _distributed_validation_worker(rank, rendezvous, output_dir):
    import torch.distributed as dist
    from pathlib import Path
    from evaluation.evaluate import evaluate_model
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        batch = {
            "video": torch.tensor([0.9, 0.8] if rank == 0 else [0.2, 0.1]),
            "phonemes": torch.ones(2, 1, dtype=torch.long),
            "label": torch.tensor([1.0, 0.0]),
        }
        metrics = evaluate_model(_ScoreModel(), [batch], torch.device("cpu"))
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(metrics))
    finally:
        dist.destroy_process_group()


def test_distributed_validation_uses_all_ranks(tmp_path):
    from evaluation.metrics import compute_metrics
    torch.multiprocessing.spawn(
        _distributed_validation_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True,
    )
    expected = compute_metrics(
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        torch.tensor([0.9, 0.8, 0.2, 0.1]),
        torch.tensor([0.9, 0.8, 0.2, 0.1]),
    )
    for rank in range(2):
        metrics = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert metrics == expected


def test_word_model_trains_all_heads_and_roundtrips(tmp_path):
    tokenizer = TranscriptTokenizer("grid", "word")
    model = LipTextVerificationModel(
        vocab_size=len(tokenizer.vocab), feature_dim=16,
        lip_backbone_pretrained=False, alignment_mode="both", dropout=0.2,
    )
    optimizer = build_optimizer(model, 1e-4, weight_decay=0.01)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert isinstance(build_optimizer(model, 1e-4), torch.optim.Adam)
    with mock.patch.object(NegativeSampler, "sample", return_value="bin blue at b two now"):
        batch = collate_fn_factory("grid", token_level="word")([_item()])
    before = model.text_encoder.embedding.weight.detach().clone()
    loss = train_one_epoch(model, [batch], optimizer, torch.device("cpu"), epoch=1)
    assert math.isfinite(loss) and loss > 0
    assert not torch.equal(before, model.text_encoder.embedding.weight)
    for layer in (model.ctc_head, model.vfa_heads.frame_head[-1],
                  model.vfa_heads.boundary_head[-1], model.word_head.net[-1]):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
    path = tmp_path / "word.pt"
    save_checkpoint(model.state_dict(), path, "both", dataset="grid",
                    token_level="word", token_vocab=tokenizer.vocab, dropout=0.2)
    checkpoint = torch.load(path)
    rebuilt = LipTextVerificationModel(
        vocab_size=len(checkpoint_tokenizer(checkpoint, "grid").vocab),
        feature_dim=16, lip_backbone_pretrained=False, alignment_mode="both", dropout=0.2)
    rebuilt.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    rebuilt.eval()
    with torch.no_grad():
        args = (batch["video"], batch["phonemes"])
        expected, out = model(*args, word_spans=batch["word_spans"])
        actual, _ = rebuilt(*args)
    assert torch.allclose(actual, expected)
    assert out["vfa_frame_logits"].shape == (2, 16, 53)
    assert out["word_verif_logits"].shape == (2, 6)


def test_word_inference_emits_direct_word_rows(tmp_path):
    tokenizer = TranscriptTokenizer("grid", "word")
    checkpoint = {
        "model_state_dict": {}, "alignment_mode": "sim", "vfa_heads": True,
        "dataset": "grid", "token_level": "word", "token_vocab": tokenizer.vocab,
    }
    model = mock.Mock(use_vfa=True)
    model.to.return_value = model
    model.return_value = (torch.tensor([[0.8]]), {
        "alignment_score": torch.tensor([0.9]),
        "vfa_frame_logits": torch.zeros(1, 75, 53),
        "vfa_bound_logits": torch.zeros(1, 75),
        "lip_feats": torch.zeros(1, 75, 256),
        "text_feats": torch.zeros(1, 6, 256),
        "word_scores": torch.ones(1, 6),
    })
    model.score_word_spans.return_value = (torch.zeros(1, 6), torch.ones(1, 6, dtype=torch.bool))
    spans = [(i, 1 + 2 * i, 3 + 2 * i) for i in range(6)]
    import numpy as np
    report_path = tmp_path / "report.json"
    with (mock.patch("inference.torch.load", return_value=checkpoint),
          mock.patch("inference.resolve_checkpoint_path", return_value="word.pt"),
          mock.patch("inference.LipTextVerificationModel", return_value=model) as constructor,
          mock.patch("inference.os.path.exists", return_value=True),
          mock.patch("inference.vision.FaceLandmarker.create_from_options"),
          mock.patch("inference.extract_lip_frames", return_value=[np.zeros((96, 96, 3))] * 16),
          mock.patch("inference.cv2.VideoCapture") as cap,
          mock.patch("inference.load_eer_thresholds", return_value=None),
          mock.patch("inference.forced_token_spans", return_value=(spans, False)),
          mock.patch("inference.build_word_report") as aggregate):
        cap.return_value.get.return_value = 25.0
        run_inference("clip.mp4", "bin blue at f two now", dataset="grid",
                      json_path=report_path, config_path="config.grid.json")
    aggregate.assert_not_called()
    assert constructor.call_args.kwargs["vocab_size"] == 53
    assert model.call_args.args[1].tolist() == [tokenizer.encode("bin blue at f two now")]
    rows = json.loads(report_path.read_text())["tokens"]
    assert [row["token"] for row in rows] == "bin blue at f two now".split()
    assert rows[3]["start_s"] == 0.28 and rows[3]["end_s"] == 0.36


def test_evaluation_uses_checkpoint_word_tokens_with_legacy_config(tmp_path):
    import test as entrypoint
    tokenizer = TranscriptTokenizer("grid", "word")
    cfg = Config("config.grid.json")
    cfg.processed_data_dir = str(tmp_path)
    cfg.output_dir = str(tmp_path)
    cfg.token_level = "phoneme"
    checkpoint = {
        "model_state_dict": {}, "alignment_mode": "sim", "vfa_heads": True,
        "dataset": "grid", "token_level": "word", "token_vocab": tokenizer.vocab,
    }

    def evaluate(model, loader, device, **kwargs):
        batch = next(iter(loader))
        assert batch["phonemes"].shape == (2, 6)
        assert batch["phonemes"][0].tolist() == tokenizer.encode(_item()["transcript"])
        return {"roc_auc": 0.8, "verification_accuracy": 0.7, "verification_eer": 0.2}

    with (mock.patch.object(entrypoint, "Config", return_value=cfg),
          mock.patch.object(entrypoint, "GRIDDataset", return_value=[_item()]),
          mock.patch.object(entrypoint, "_resolve_latest_checkpoint", return_value="word.pt"),
          mock.patch.object(entrypoint.torch, "load", return_value=checkpoint),
          mock.patch.object(entrypoint, "build_eval_model") as build,
          mock.patch.object(entrypoint, "evaluate_model", side_effect=evaluate),
          mock.patch.object(entrypoint, "checkpoint_sha256", return_value="digest"),
          mock.patch.object(entrypoint, "save_eer_thresholds") as save):
        entrypoint.test()
    assert build.call_args.args[2] == 53
    assert save.call_args.args[1]["token_level"] == "word"
