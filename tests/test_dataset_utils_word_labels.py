"""Per-token verification supervision emitted by the collate.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_dataset_utils_word_labels.py -q
"""

import os
import sys
import random
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_utils import collate_fn_factory, substitution_position
from utils.negative_sampler import NegativeSampler


def _digit_item(transcript, digit_times, video_len=20):
    return {
        'video': torch.randn(video_len, 3, 96, 96),
        'transcript': transcript,
        'digit_times': digit_times,
        'video_len': video_len,
    }


def test_substitution_position():
    assert substitution_position(["1", "2", "3"], ["1", "2", "3"]) is None      # identical
    assert substitution_position(["1", "2", "3"], ["1", "9", "3"]) == 1         # one swap
    assert substitution_position(["1", "2"], ["1", "9", "3"]) is None           # length differs
    assert substitution_position(["1", "2", "3"], ["9", "8", "3"]) is None      # two swaps
    assert substitution_position([], []) is None


def _collate_digit(items):
    # Empty transcript pool: collate builds a sampler from the batch itself.
    fn = collate_fn_factory("digit", transcript_pool=[])
    return fn(items)


def test_collate_positive_pair_word_labels_and_spans():
    random.seed(0)
    item = _digit_item("1 2 3", [("1", 0, 6), ("2", 6, 12), ("3", 12, 18)])
    batch = _collate_digit([item])
    # Batch order: [positive, negative]
    pos_labels = batch['word_labels'][0]
    pos_mask = batch['word_label_mask'][0]
    assert pos_labels[:3].tolist() == [1.0, 1.0, 1.0]
    assert pos_mask[:3].tolist() == [True, True, True]
    # Spans: token 0 owns frames 0-5, token 1 frames 6-11, token 2 frames 12-17
    spans = batch['word_spans'][0]
    assert spans[:6, 0].all() and spans[6:12, 1].all() and spans[12:18, 2].all()
    assert spans[18:].sum() == 0


def test_collate_substitution_negative_localizes_wrong_token():
    random.seed(0)
    real_choice = random.choice
    def fake_choice(seq):
        if list(seq)[:1] == ['hard']:
            return 'phoneme_similar'      # strategy list -> force substitution
        return real_choice(seq)
    item = _digit_item("1 2 3", [("1", 0, 6), ("2", 6, 12), ("3", 12, 18)])
    with mock.patch("utils.negative_sampler.random.choice", side_effect=fake_choice):
        batch = _collate_digit([item])
    neg_labels = batch['word_labels'][1]
    neg_mask = batch['word_label_mask'][1]
    assert neg_mask[:3].tolist() == [True, True, True]
    swapped = [i for i in range(3) if neg_labels[i].item() == 0.0]
    assert len(swapped) == 1              # exactly the substituted digit is 0
    # the other two tokens are 1
    others = [i for i in range(3) if i not in swapped]
    assert all(neg_labels[i].item() == 1.0 for i in others)
    # negative spans exist (copied from positive geometry)
    assert batch['word_spans'][1, :, :3].sum() > 0


def test_collate_random_negative_masks_word_labels():
    random.seed(0)
    real_choice = random.choice
    def fake_choice(seq):
        if list(seq)[:1] == ['hard']:
            return 'random'
        return real_choice(seq)
    item = _digit_item("1 2 3", [("1", 0, 6), ("2", 6, 12), ("3", 12, 18)])
    with mock.patch("utils.negative_sampler.random.choice", side_effect=fake_choice):
        batch = _collate_digit([item])
    assert not batch['word_label_mask'][1].any()
    assert batch['word_spans'][1].sum() == 0


def test_balanced_digit_sampler_preserves_pause_slots_and_varies_edit_count():
    sampler = NegativeSampler([], dataset_name="digit", rng=random.Random(42))
    positive = "2 6 8 ! 9 2 9 8".split()
    edit_counts = set()
    for _ in range(100):
        negative = sampler.word_substitution(" ".join(positive)).split()
        assert len(negative) == len(positive)
        assert negative[3] == "!"
        assert all(token in "0123456789" for i, token in enumerate(negative) if i != 3)
        count = sum(p != n for p, n in zip(positive, negative))
        assert count >= 1
        edit_counts.add(count)
    assert 1 in edit_counts and 7 in edit_counts


def test_balanced_digit_multi_substitution_labels_all_annotated_slots():
    digits = "2 6 8 0 9 2 9 8".split()
    item = _digit_item(" ".join(digits), [(d, 2 * i, 2 * i + 2) for i, d in enumerate(digits)])
    collate = collate_fn_factory("digit", balanced_word_supervision=True, seed=42)
    with mock.patch.object(NegativeSampler, "word_substitution", return_value="1 2 3 4 5 6 7 8"):
        batch = collate([item])
        untimed = collate([dict(item, digit_times=[])])

    assert batch["label"].tolist() == [1, 0]
    assert batch["word_label_mask"][1].all()
    assert batch["word_labels"][1].tolist() == [0] * 7 + [1]
    assert torch.equal(batch["word_spans"][1], batch["word_spans"][0])
    assert not batch["alignment_targets"][1].any()
    assert untimed["label"].tolist() == [1, 0]
    assert not untimed["word_label_mask"].any()


def test_balanced_digit_arbitrary_transcript_only_supervises_clip():
    item = _digit_item("1 2 3", [("1", 0, 6), ("2", 6, 12), ("3", 12, 18)])
    collate = collate_fn_factory("digit", balanced_word_supervision=True, seed=0)
    with mock.patch.object(NegativeSampler, "sample", return_value="9 3 2"):
        batch = collate([item])
    assert batch["label"].tolist() == [1, 0]
    assert batch["transcript"][1] == "9 3 2"
    assert not batch["word_label_mask"][1].any()
    assert not batch["word_spans"][1].any()
