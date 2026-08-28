"""Tests for pinyin-based phoneme conversion of digit transcripts.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_digit_pinyin.py -q
(or: python tests/test_digit_pinyin.py)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.phoneme_converter import DigitPinyinConverter, ARPABET, DIGIT_PINYIN_PHONEMES
from utils.alignment import expand_token_ranges, build_alignment_targets


def test_digit_vocab_is_compact_and_covers_all_pronunciations():
    # Digit-only vocab: <pad>/<unk> + exactly the phones the 10 digit
    # pronunciations use (15 phones -> 17 tokens), not the shared 41.
    conv = DigitPinyinConverter()
    used = {p for _, (_, phones) in DIGIT_PINYIN_PHONEMES.items() for p in phones}
    assert conv.vocab_size == 17
    assert set(conv.vocab) == {"<pad>", "<unk>"} | used
    assert conv.vocab["<pad>"] == 0 and conv.vocab["<unk>"] == 1
    for d in "0123456789":
        ids = conv.token_to_phoneme_indices(d)
        assert len(ids) >= 1, f"digit {d} has no pronunciation"
        assert all(0 < i < conv.vocab_size for i in ids), f"digit {d} -> {ids}"


def test_vocab_size_constants():
    from utils.phoneme_converter import DIGIT_VOCAB_SIZE, GRID_VOCAB_SIZE
    assert GRID_VOCAB_SIZE == len(ARPABET) == 41
    assert DIGIT_VOCAB_SIZE == DigitPinyinConverter().vocab_size == 17


def test_explicit_pinyin_mappings():
    conv = DigitPinyinConverter()
    name = {i: p for p, i in conv.vocab.items()}
    # qī  -> CH IY   (visually near-identical to yī=Y IY, but distinct phones)
    assert [name[i] for i in conv.token_to_phoneme_indices("7")] == ["CH", "IY"]
    # líng -> L IH NG
    assert [name[i] for i in conv.token_to_phoneme_indices("0")] == ["L", "IH", "NG"]
    # liù -> L IY OW
    assert [name[i] for i in conv.token_to_phoneme_indices("6")] == ["L", "IY", "OW"]


def test_digit_pronunciations_are_distinct():
    # The 10 digits must flatten to 10 distinct phone sequences, otherwise two
    # digits would be indistinguishable at the text/CTC target level.
    conv = DigitPinyinConverter()
    seqs = {d: tuple(conv.token_to_phoneme_indices(d)) for d in "0123456789"}
    assert len(set(seqs.values())) == 10


def test_bang_maps_to_empty_group():
    conv = DigitPinyinConverter()
    trans = "3 0 8 4 ! 6 8 !"
    groups = conv.transcript_to_phoneme_indices_per_token(trans)
    # 1:1 with transcript tokens ("!" keeps its slot for digit_times zipping)
    assert [tok for tok, _ in groups] == trans.split()
    assert all(ids == [] for tok, ids in groups if tok == "!")
    assert all(len(ids) > 0 for tok, ids in groups if tok != "!")
    flat = conv.flatten_per_token(groups)
    assert len(flat) == 15  # 3+3+2+2 + 0 + 3+2 + 0


def test_alignment_columns_match_flattened_sequence():
    # The single-source-of-truth property: seq indices produced from
    # (per_token x digit_times) must index the flattened encoder/CTC sequence,
    # and empty "!" groups must not shift later tokens' columns.
    conv = DigitPinyinConverter()
    trans = "3 0 8 4 ! 6 8 !"
    digit_times = [("3", 1, 3), ("0", 4, 6), ("8", 7, 9), ("4", 10, 12),
                   ("!", 13, 14), ("6", 15, 17), ("8", 18, 20), ("!", 21, 23)]
    per_token = conv.transcript_to_phoneme_indices_per_token(trans)
    assert len(per_token) == len(digit_times)
    token_times = [(ids, fs, fe) for (_, ids), (_, fs, fe) in zip(per_token, digit_times)]
    flat = conv.flatten_per_token(per_token)

    ranges = expand_token_ranges(token_times)
    assert all(0 <= seq < len(flat) for seq, _, _ in ranges)
    targets, speech = build_alignment_targets(ranges, video_len=25, target_len=len(flat))
    # "!" ranges (13-14, 21-23) get no supervised frames
    assert not speech[13:15].any() and not speech[21:24].any()
    # every digit's frames stay supervised, nothing lost
    assert speech[1:3].all() and speech[15:17].all() and speech[18:20].all()


def test_collate_digit_batch_end_to_end():
    import random
    random.seed(0)  # the negative sampler draws strategies at random
    from utils.dataset_utils import collate_fn_factory
    collate = collate_fn_factory("digit", transcript_pool=["3 0 8 4 ! 6 8 !", "7 1 9 2 5 0 3 6"])

    v = torch.zeros(26, 3, 16, 16)
    batch = [
        {"video": v, "transcript": "3 0 8 4 ! 6 8 !", "video_len": 25,
         "digit_times": [("3", 1, 3), ("0", 4, 6), ("8", 7, 9), ("4", 10, 12),
                         ("!", 13, 14), ("6", 15, 17), ("8", 18, 20), ("!", 21, 23)]},
        {"video": v, "transcript": "7 1 9 2 5 0 3 6", "video_len": 20,
         "digit_times": [("7", 2, 4), ("1", 5, 7), ("9", 8, 10), ("2", 11, 13),
                         ("5", 14, 16), ("0", 17, 19), ("3", 20, 22), ("6", 23, 24)]},
    ]
    out = collate(batch)

    conv = DigitPinyinConverter()
    flat_len = lambda t: len(conv.flatten_per_token(
        conv.transcript_to_phoneme_indices_per_token(t)))

    # B = 2 items x (1 pos + 1 neg)
    assert out["label"].tolist() == [1.0, 0.0, 1.0, 0.0]
    ids = out["phonemes"]
    assert ids.min() >= 0 and ids.max() < 17
    # positive rows: flattened digit-pinyin lengths (row 0 excludes "!")
    assert out["phoneme_padding_mask"][0].sum() == ids.shape[1] - flat_len("3 0 8 4 ! 6 8 !")
    assert out["phoneme_padding_mask"][2].sum() == ids.shape[1] - flat_len("7 1 9 2 5 0 3 6")
    # alignment targets only on positives with boundary data
    assert out["has_align_target"].tolist() == [1.0, 0.0, 1.0, 0.0]
    hot_cols = out["alignment_targets"][0].sum(dim=0) > 0
    # 12, not 13: digit "3"'s 2-frame range splits across its 3 phones, so
    # one phone gets an empty part and no supervised frames
    assert hot_cols.sum() == 12
    assert not out["speech_mask"][0][13:15].any()  # "!" frames unsupervised


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
