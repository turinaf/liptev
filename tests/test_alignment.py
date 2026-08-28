"""Sanity tests for frame-alignment supervision.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_alignment.py -q
(or: python tests/test_alignment.py)
"""

import os
import sys
import random

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.alignment import (
    calibrate_value_to_frame,
    seconds_to_frame,
    split_frames_uniformly,
    expand_token_ranges,
    build_alignment_targets,
)
from utils.negative_sampler import NegativeSampler, GRID_WORD_CLASSES


# --- utils/alignment.py ----------------------------------------------------

def test_calibrate_real_grid_format():
    # Real-format values (ts-lip2text/data/bbaf2n.align): integers 0..74500
    # for a 75-frame clip. Proportional calibration must map them into
    # [0, 75] regardless of the timebase unit.
    max_end, video_len = 74500, 75
    assert calibrate_value_to_frame(0, max_end, video_len) == 0
    assert calibrate_value_to_frame(max_end, max_end, video_len) == video_len
    mid = calibrate_value_to_frame(37250, max_end, video_len)
    assert 36 <= mid <= 39
    assert calibrate_value_to_frame(10**9, max_end, video_len) == video_len  # clamped
    assert calibrate_value_to_frame(-5, max_end, video_len) == 0

def test_seconds_to_frame():
    assert seconds_to_frame(1.0, 29.87) == 30
    assert seconds_to_frame(0.5, 25.0) == 12  # banker's rounding of 12.5
    assert seconds_to_frame(2.0, 0) == 0      # invalid fps guarded

def test_split_frames_uniformly():
    # Even split
    assert split_frames_uniformly(0, 9, 3) == [(0, 3), (3, 6), (6, 9)]
    # Uneven range: contiguous, monotone, covers everything
    parts = split_frames_uniformly(0, 10, 3)
    assert parts[0][0] == 0 and parts[-1][1] == 10
    for (a, b), (c, d) in zip(parts, parts[1:]):
        assert b == c
    # Fewer frames than tokens -> empty parts allowed, no crash
    parts = split_frames_uniformly(4, 6, 4)
    assert sum(b - a for a, b in parts) == 2
    assert split_frames_uniformly(5, 5, 3) == [(5, 5)] * 3

def test_build_alignment_targets():
    # Token ids are arbitrary; target columns are SEQUENCE positions (the
    # sim_matrix L axis): pos0/pos1 from the two-phoneme word, pos2 the digit.
    ranges = expand_token_ranges([([7, 8], 0, 4), ([3], 6, 10)])
    targets, speech = build_alignment_targets(ranges, video_len=12, target_len=5)

    assert targets.shape == (12, 5)
    assert speech.tolist() == [True] * 4 + [False] * 2 + [True] * 4 + [False] * 2
    assert targets[:2, 0].all()          # first phoneme -> frames 0-1
    assert targets[2:4, 1].all()         # second phoneme -> frames 2-3
    assert targets[6:10, 2].all()        # digit -> frames 6-9
    assert targets.sum() == 8            # one-hot over speech frames only

def test_build_alignment_targets_clamps():
    ranges = expand_token_ranges([([1], 5, 100)])
    targets, speech = build_alignment_targets(ranges, video_len=10, target_len=2)
    assert speech.tolist() == [False] * 5 + [True] * 5
    assert targets[:, 0].sum() == 5


# --- utils/phoneme_converter.py --------------------------------------------

def test_per_word_phonemization():
    from utils.phoneme_converter import PhonemeConverter
    conv = PhonemeConverter()

    per_word = conv.text_to_phoneme_indices_per_word("bin blue at f two now")
    words = [w for w, _ in per_word]
    assert words == ["bin", "blue", "at", "f", "two", "now"]

    # No <unk> for inter-word spaces (the whole-sentence path mapped every
    # space to <unk>).
    flat = conv.flatten_per_word(per_word)
    assert 1 not in flat  # 1 == <unk>

    # Single source of truth: concatenation of per-word calls
    joined = [i for w in words for i in conv.word_to_phoneme_indices(w)]
    assert flat == joined


# --- utils/negative_sampler.py ----------------------------------------------

def test_digit_negative_sampler():
    random.seed(0)
    sampler = NegativeSampler(["7 3 0 8 4 6 5", "1 2 3 4 5 6 7"] * 10, dataset_name="digit")
    pos = "7 7 7 7 7 7 7"
    valid_digits = set("0123456789") | {"!"}
    for _ in range(200):
        neg = sampler.sample(pos)
        assert neg != pos
        # whatever the strategy, the result stays digit tokens of same length
        assert all(t in valid_digits for t in neg.split())
        assert len(neg.split()) == 7

def test_grid_phoneme_similar_stays_in_class():
    random.seed(0)
    sampler = NegativeSampler([], dataset_name="grid")
    for _ in range(100):
        neg = sampler._phoneme_similar("bin blue at f two now")
        n, p = neg.split(), "bin blue at f two now".split()
        changed = [(a, b) for a, b in zip(p, n) if a != b]
        assert changed, "phoneme_similar must change something"
        for a, b in changed:
            ca = next(c for c in GRID_WORD_CLASSES if a in c)
            assert b in ca, f"{a} swapped for out-of-class {b}"


# --- model forward smoke -----------------------------------------------------

def _smoke_forward(vocab_size, B=2, T=20, L=8, D=64):
    from models.verification_model import LipTextVerificationModel
    torch.manual_seed(0)
    model = LipTextVerificationModel(vocab_size=vocab_size, feature_dim=D, lip_backbone_pretrained=False)
    video = torch.randn(B, T, 3, 96, 96)
    phonemes = torch.randint(2, vocab_size, (B, L))
    phoneme_padding_mask = torch.zeros(B, L, dtype=torch.bool)
    phoneme_padding_mask[:, 6:] = True
    video_padding_mask = torch.zeros(B, T, dtype=torch.bool)
    video_padding_mask[:, 15:] = True

    score, out = model(video, phonemes, None,
                       video_padding_mask=video_padding_mask,
                       phoneme_padding_mask=phoneme_padding_mask)

    assert score.shape == (B, 1)
    assert out["ctc_nll_per_token"].shape == (B,)
    assert out["ctc_valid"].all()
    # Positive pairs only would select these; values finite per-token NLL
    assert torch.isfinite(out["ctc_nll_per_token"]).all()

    # AlignmentKL path with matching shapes
    from training.train import alignment_kl_loss
    targets = torch.zeros(B, T, L)
    targets[:, 0:3, 2] = 1.0  # column = sequence position, not token id
    frame_valid = torch.zeros(B, T, dtype=torch.bool)
    frame_valid[:, 0:3] = True
    frame_valid[1] = False  # one sample without targets
    loss = alignment_kl_loss(out["sim_matrix"], targets, frame_valid, temperature=1.0)
    assert torch.isfinite(loss)

def test_model_forward_grid_vocab():
    _smoke_forward(vocab_size=41)

def test_model_forward_digit_vocab():
    # Digit transcripts map into their own compact vocab: <pad>/<unk> +
    # the 15 phones the digit pronunciations use (no shared 41 with GRID).
    _smoke_forward(vocab_size=17)

def test_positives_only_mask():
    # The training loop's CTC mask: negatives must be excluded even when valid.
    labels = torch.tensor([1.0, 0.0])
    ctc_valid = torch.tensor([True, True])
    positive_mask = labels == 1
    selected = ctc_valid & positive_mask
    assert selected.tolist() == [True, False]


# --- alignment mode switch (sim / attn / both) --------------------------------

def _mode_inputs(B=2, T=20, L=6, D=64):
    torch.manual_seed(0)
    lip = torch.randn(B, T, D)
    text = torch.randn(B, L, D)
    lip_pad = torch.zeros(B, T, dtype=torch.bool)
    lip_pad[:, 15:] = True
    text_pad = torch.zeros(B, L, dtype=torch.bool)
    text_pad[:, 4:] = True
    return lip, text, lip_pad, text_pad

def test_alignment_mode_rejects_unknown():
    from models.cross_modal_alignment import CrossModalAlignment
    try:
        CrossModalAlignment(64, mode="nope")
    except ValueError:
        return
    raise AssertionError("unknown mode must raise ValueError")

def test_alignment_mode_sim_matches_legacy_output():
    # "sim" mode must reproduce the original cosine-only scoring exactly.
    from models.cross_modal_alignment import CrossModalAlignment
    lip, text, lip_pad, text_pad = _mode_inputs()
    torch.manual_seed(1)
    m = CrossModalAlignment(64, mode="sim")
    out = m(lip, text, lip_padding_mask=lip_pad, text_padding_mask=text_pad)

    lip_norm = torch.nn.functional.normalize(lip, dim=-1)
    text_norm = torch.nn.functional.normalize(text, dim=-1)
    sim = torch.bmm(lip_norm, text_norm.transpose(1, 2))
    sim = sim.masked_fill(lip_pad.unsqueeze(-1), -1e9).masked_fill(text_pad.unsqueeze(1), -1e9)
    word_scores = sim.max(dim=1).values
    valid = (~text_pad).float()
    raw = (word_scores * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)
    expected = ((raw + 1.0) / 2.0).clamp(0.0, 1.0)

    assert torch.allclose(out["alignment_score"], expected, atol=1e-6)
    assert out["sim_matrix"] is not None
    assert out["aligned_features"] is None  # attention not run in "sim" mode
    assert out["attn_weights"] is None

def test_alignment_mode_attn():
    from models.cross_modal_alignment import CrossModalAlignment
    lip, text, lip_pad, text_pad = _mode_inputs()
    m = CrossModalAlignment(64, mode="attn")
    out = m(lip, text, lip_padding_mask=lip_pad, text_padding_mask=text_pad)

    score = out["alignment_score"]
    assert score.shape == (2,)
    assert torch.isfinite(score).all() and (score >= 0).all() and (score <= 1).all()
    assert out["sim_matrix"] is None           # no cosine matrix in "attn" mode
    assert out["attn_weights"].shape == (2, 6, 20)  # B x L x T
    assert out["word_scores"].shape == (2, 6)
    # Padded word slots contribute nothing: score equals mean over valid words
    ws = out["word_scores"]
    raw = ws[:, :4].mean(dim=-1)
    expected = ((raw + 1.0) / 2.0).clamp(0.0, 1.0)
    assert torch.allclose(score, expected, atol=1e-6)

def test_alignment_mode_both_averages():
    from models.cross_modal_alignment import CrossModalAlignment
    lip, text, lip_pad, text_pad = _mode_inputs()
    torch.manual_seed(1)
    both = CrossModalAlignment(64, mode="both")
    sim_m = CrossModalAlignment(64, mode="sim")
    attn_m = CrossModalAlignment(64, mode="attn")
    both.load_state_dict(sim_m.state_dict())
    attn_m.load_state_dict(sim_m.state_dict())

    kwargs = dict(lip_padding_mask=lip_pad, text_padding_mask=text_pad)
    s_both = both(lip, text, **kwargs)["alignment_score"]
    s_sim = sim_m(lip, text, **kwargs)["alignment_score"]
    s_attn = attn_m(lip, text, **kwargs)["alignment_score"]
    assert torch.allclose(s_both, (s_sim + s_attn) / 2.0, atol=1e-6)

def test_model_forward_with_each_mode():
    from models.verification_model import LipTextVerificationModel
    for mode in ("sim", "attn", "both"):
        torch.manual_seed(0)
        model = LipTextVerificationModel(vocab_size=41, feature_dim=64,
                                         lip_backbone_pretrained=False,
                                         alignment_mode=mode)
        video = torch.randn(2, 12, 3, 48, 48)
        phonemes = torch.randint(2, 41, (2, 5))
        score, out = model(video, phonemes)
        assert torch.isfinite(score).all(), f"mode={mode}"
        # train.py reads sim_matrix from the no-audio align_out; in "attn"
        # mode it is absent/None and the KL-loss guard must skip it.
        sim = out.get("sim_matrix", out.get("lip_text", {}).get("sim_matrix"))
        if mode == "attn":
            assert sim is None, f"mode={mode}"
        else:
            assert sim is not None and torch.isfinite(sim[sim > -1e8]).all(), f"mode={mode}"


# --- audio path ---------------------------------------------------------------

def test_bang_token_keeps_sequence_slot():
    # Zero-length ranges (start == end) get zero supervised frames but keep
    # the sequence slot so later tokens don't shift.
    token_times = [([2], 0, 5), ([11], 0, 0), ([4], 10, 20)]  # digit, "!", digit
    ranges = expand_token_ranges(token_times)
    # ranges: [(0,0,5), (1,0,0), (2,10,20)]
    targets, speech = build_alignment_targets(ranges, video_len=25, target_len=3)
    assert targets[:5, 0].all()
    assert targets[:, 1].sum() == 0            # "!" gets no frames
    assert targets[10:20, 2].all()             # later digit at position 2

def test_audio_forward_with_masks():
    from models.verification_model import LipTextVerificationModel
    from models.audio_encoder import AudioEncoder
    torch.manual_seed(0)
    B, T, L = 2, 20, 6
    model = LipTextVerificationModel(vocab_size=41, feature_dim=64, lip_backbone_pretrained=False)

    # audio of different real lengths, padded to batch max
    real_lens = [16000, 9000]
    audio = torch.zeros(B, 1, max(real_lens))
    for i, n in enumerate(real_lens):
        audio[i, :, :n] = torch.randn(1, n) * 0.1
    audio_padding_mask = torch.arange(max(real_lens)).unsqueeze(0) >= torch.tensor(real_lens).unsqueeze(1)

    video = torch.randn(B, T, 3, 48, 48)
    phonemes = torch.randint(2, 41, (B, L))
    video_padding_mask = torch.zeros(B, T, dtype=torch.bool); video_padding_mask[:, 15:] = True
    phoneme_padding_mask = torch.zeros(B, L, dtype=torch.bool)

    score, out = model(video, phonemes, audio,
                       video_padding_mask=video_padding_mask,
                       phoneme_padding_mask=phoneme_padding_mask,
                       audio_padding_mask=audio_padding_mask)
    assert torch.isfinite(score).all()
    T_mel = out["audio_text"]["sim_matrix"].shape[1]
    hop = model.audio_encoder.hop_length()
    assert T_mel == max(real_lens) // hop + 1
    # padded audio mel frames masked out of the similarity matrix
    sim = out["audio_text"]["sim_matrix"]
    n_valid = real_lens[0] // hop
    assert torch.isinf(sim[1, n_valid:]).all() or (sim[1, n_valid:] <= -1e8).all()

def test_mel_domain_mask_boundary():
    from models.verification_model import LipTextVerificationModel
    mask = torch.tensor([[False, False, False, True, True]])  # 3 real samples
    mel_mask = LipTextVerificationModel._mel_domain_mask(mask, T_mel=5, hop_length=2)
    # frame t valid iff t*2 < 3 -> t in {0, 1}
    assert mel_mask.tolist() == [[False, False, True, True, True]]


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
