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


def test_digit_vocab_is_shared_arpabet():
    # Digits index into the same ARPABET vocabulary GRID fills via g2p_en
    # (CMU dict): one phoneme space, one vocab_size (41) for both datasets.
    conv = DigitPinyinConverter()
    assert conv.vocab == {p: i for i, p in enumerate(ARPABET)}
    assert conv.vocab_size == 41
    assert conv.vocab["<pad>"] == 0 and conv.vocab["<unk>"] == 1
    used = {p for _, (_, phones) in DIGIT_PINYIN_PHONEMES.items() for p in phones}
    assert used <= set(ARPABET)  # every mapped phone is in the shared vocab
    for tok in list("0123456789") + ["!"]:
        ids = conv.token_to_phoneme_indices(tok)
        assert len(ids) >= 1, f"token {tok} has no pronunciation"
        assert all(0 < i < conv.vocab_size for i in ids), f"token {tok} -> {ids}"


def test_vocab_size_constants():
    from utils.phoneme_converter import PHONEME_VOCAB_SIZE
    assert PHONEME_VOCAB_SIZE == len(ARPABET) == DigitPinyinConverter().vocab_size == 41


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


def test_bang_is_yao_not_a_pause():
    # "!" is the digit 1 read yāo (number-string style, kept distinct from
    # qī=7): a real pronunciation, not a pause — it must contribute phones
    # and therefore supervised alignment frames in its .lab range.
    conv = DigitPinyinConverter()
    name = {i: p for p, i in conv.vocab.items()}
    assert [name[i] for i in conv.token_to_phoneme_indices("!")] == ["Y", "AW"]
    # and it stays distinct from "1" (yī = Y IY)
    assert conv.token_to_phoneme_indices("!") != conv.token_to_phoneme_indices("1")

    trans = "3 0 8 4 ! 6 8 !"
    groups = conv.transcript_to_phoneme_indices_per_token(trans)
    # 1:1 with transcript tokens ("!" keeps its slot for digit_times zipping)
    assert [tok for tok, _ in groups] == trans.split()
    assert all(len(ids) > 0 for _, ids in groups)
    flat = conv.flatten_per_token(groups)
    assert len(flat) == 19  # 3+3+2+2 + 2 + 3+2 + 2


def test_alignment_columns_match_flattened_sequence():
    # The single-source-of-truth property: seq indices produced from
    # (per_token x digit_times) must index the flattened encoder/CTC sequence,
    # with every token — the pronounced "!" (yāo) included — owning columns.
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
    # "!" (yāo) ranges get supervised frames like any digit. Ranges are
    # end-exclusive: (13,14) is a single frame, (21,23) two frames.
    assert speech[13].item() and speech[21:23].all()
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
    assert ids.min() >= 0 and ids.max() < 41
    # positive rows: flattened digit-pinyin lengths ("!" counts — it is pronounced)
    assert out["phoneme_padding_mask"][0].sum() == ids.shape[1] - flat_len("3 0 8 4 ! 6 8 !")
    assert out["phoneme_padding_mask"][2].sum() == ids.shape[1] - flat_len("7 1 9 2 5 0 3 6")
    # alignment targets only on positives with boundary data
    assert out["has_align_target"].tolist() == [1.0, 0.0, 1.0, 0.0]
    hot_cols = out["alignment_targets"][0].sum(dim=0) > 0
    # 15, not 19: "3"/"0"/"6" each split a 2-frame range across 3 phones and
    # the first "!" a 1-frame range across 2 phones — one empty part each,
    # so 4 phones get no supervised frames (ranges are end-exclusive)
    assert hot_cols.sum() == 15
    assert out["speech_mask"][0][13].item()  # "!" (yāo) frame supervised


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
