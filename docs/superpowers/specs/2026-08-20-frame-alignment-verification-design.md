# Design: Frame-Level Alignment for Lip-to-Text Verification

Date: 2026-08-20 (rev. 3 — audio-path fixes from sample-data review)
Branch: `feature/frame-alignment-verification`

> **Rev. 3 changelog** (issues found testing audio against the sample data
> in `./data`):
> 1. **Audio was hard-truncated to 3 s** while video kept up to 125 frames —
>   37% of digit samples had later digits with lip movement but silence.
>   Audio length now follows the kept frames per sample
>   (`samples = sr × video_len / fps`); GRID is native 75 frames/3 s
>   (verified: 984/1000 sample videos are exactly 75 frames), digit budget
>   stays 125 (measured p95) — no independent seconds cap anywhere.
> 2. **`.txt` transcripts contain `!` tokens that `.lab` omits** — strict
>   equality dropped 40% of boundary supervision (220/371 → 336/366 after
>   the fix). Matching now compares digit tokens only; `!` keeps its
>   sequence slot (CTC/embedding) with zero supervised frames.
> 3. **No audio padding mask existed** — zero-padded silence (log-mel
>   −13.8) leaked into both audio alignments and mean pooling. The collate
>   now builds `audio_padding_mask` (samples), the model converts it to the
>   mel-frame domain, and it masks the audio transformer, both cross-modal
>   alignments, and audio pooling.
> 4. **AudioEncoder had no positional encoding** — a permutation-invariant
>   transformer over time. Sinusoidal PE added (shared with the lip
>   encoder).
> 5. **fps fallback fixed at 29.87 for everyone** — sample data includes
>   ~25/26 fps speakers (misplacing frames ~20%). When `meta.json` is
>   absent the loader now infers per-video fps exactly from processed frame
>   count ÷ raw audio duration (wav header, stdlib).

> **Rev. 2 changelog** (issues found reviewing rev. 1 against the actual data):
> 1. GRID `.align` times are **not seconds** — rev. 1's example format was
>   wrong. Real files use integer values (audio samples at ~25 kHz for a 3 s
>   clip). Conversion is now auto-calibrated per sample (§2.1).
> 2. `.lab` files exist only in the raw tree; the pipeline never copied them,
>   so the loader could never find them. `pipeline/data_preparation.py` is now
>   in scope (§2.2, §9).
> 3. CTC and alignment losses are now restricted to **positive pairs only** —
>   rev. 1 would have trained the model to align mismatched text (§4).
> 4. Per-word phonemization is now the single source for both the encoder
>   input and the alignment targets; the whole-sentence `g2p` path also had a
>   latent bug (spaces mapped to `<unk>`) (§5.3).
> 5. Digit source fps is **per-video** (measured 25 fps and ~29.9 fps VFR in
>   different subsets), so the pipeline records it per video in `meta.json`;
>   `digit_max_frames` raised to 125 with an explicit skip policy for samples
>   whose `.lab` extends beyond the window (38% of videos exceed the old
>   75-frame cap; max is 337) (§2.2).
> 6. The CTC log-likelihood fed to the classifier is per-token normalized (§3.3).
> 7. Per-sample `has_align_target` mask so batches with missing `.lab` targets
>   don't bias the alignment loss (§4).

## 1. Problem

The current lip-to-text verification model does not learn which video frames
correspond to which phoneme/digit tokens. The only supervision is a global
match/mismatch signal (BCE + contrastive) on positive/negative pairs. The
cross-modal alignment module (`models/cross_modal_alignment.py`) computes a
cosine similarity matrix but it is never supervised, so the mapping it produces
is emergent and unreliable.

Goal: teach the model an explicit frame->token alignment and make the
verification score alignment-aware, for both the GRID dataset (phonemes) and
the digit dataset (digits).

## 2. Supervision Sources

### 2.1 GRID (word-level alignments)

`*.align` files contain one token per line with integer boundary values, e.g.
(from `ts-lip2text/data/bbaf2n.align`, a 3 s / 75-frame clip):

```
0 23750 sil
23750 29500 bin
29500 34000 blue
...
53000 74500 sil
```

The unit is **audio samples (~25 kHz)**, not seconds: word durations at
/25000 are plausible ("bin" 0.23 s, "at" 0.06 s) and the final value (74500)
matches the 3 s clip. Hardcoding any single unit is fragile, so the
frame conversion is auto-calibrated per sample: the align file spans the whole
clip (starts at 0, ends at the video end), therefore

```
frame = round(value * video_len / max_end_value)
```

where `video_len` is the clip's true (unpadded) frame count. This is correct
for any timebase and needs no unit constant. Frames are clamped to
`[0, video_len)`; a warning is logged if >10% of a sample's word frames land
out of range (indicates a corrupt/unusual align file).

Each word's frame range is split uniformly (contiguously) across that word's
phonemes — each speech frame is assigned to exactly one phoneme (one-hot row,
not soft). `sil`/`sp` entries are excluded from targets but included in the
calibration span.

### 2.2 Digits (digit-level alignments, `.lab` files)

Raw `.lab` files (raw tree: `subset_X/speaker/lab/<base>.lab`) contain:

```
7 3 0 8 4 6 5
0.3-0.51 0.54-0.93 0.99-1.26 1.32-1.59 1.62-1.92 2.19-2.37 2.4-2.64
```

- Line 1: digit sequence (must match the `.txt` transcript; on mismatch the
  boundary data is dropped and the sample trains CTC-only, counted + logged).
- Line 2: per-token `start-end` ranges **in seconds, in the raw video's
  timeline**.

Three data facts drive the design:

1. **Pipeline copies `lab/` → processed `lab/`** (rev. 1 omitted this; the
   loader only sees the processed tree, so without this step no `.lab` is ever
   found).
2. **Source fps is per-video.** Measured: some subsets are ~29.9 fps VFR
   (e.g. 19500/653), others exactly 25 fps. The processed re-encode keeps all
   frames but labels them 25 fps, so `frame = round(seconds * source_fps)`
   requires the *source* fps. The pipeline records it per video in
   `subset_X/speaker/meta.json` (`{"<base>": {"source_fps": 29.87, ...}}`)
   while reading the raw video anyway. Loader resolution order:
   `meta.json` → config `digit_source_fps` (default 29.87) → warn.
3. **Frame budget.** 38% of raw videos exceed the old 75-frame cap (p50=70,
   p95=112, max=337). `digit_max_frames: 125` (covers p95). Samples whose
   `.lab` spoken range extends beyond frame 125 are **skipped at scan time**
   (counted + logged) rather than trained on truncated positives — the
   transcript would claim digits the network never sees.

If no `.lab` exists for a sample, the sample still trains via CTC with no
boundary supervision (`has_align_target = 0`).

## 3. Model Architecture Changes

### 3.1 CTC Predictor head

Add to `LipTextVerificationModel`:

```python
self.ctc_head = nn.Linear(feature_dim, vocab_size + 1)  # blank appended
```

`blank = vocab_size` (41 for GRID, 13 for digits); head output `B x T x (V+1)`
over the lip encoder per-frame features (the lip encoder preserves T, so CTC
`input_lengths = video_lens`). CTC is the primary alignment mechanism for both
datasets.

### 3.2 Masked pooling

Replace un-masked `mean(dim=1)` pooling with mask-aware pooling for lip, text,
and audio features (audio only when lengths are available; currently dormant
while `use_audio=false`). Padded frames are currently averaged in, leaking
repeated last-frame padding into the pooled representation.

### 3.3 CTC log-likelihood feature

`log P(T|V)` is unbounded and length-dependent, so the classifier receives the
**per-token normalized** value `log P(T|V) / target_len` (typ. in [-10, 0]).
It is computed inside `forward` for every pair (for negatives it is naturally
low — that is the point), returned as `ctc_logp_per_token`, and reused by the
training loop as the CTC loss term (no double computation). Gradients flow
both through the loss and the classifier feature.

## 4. Losses

```
loss = BCE + contrastive + lambda_ctc * CTC + lambda_align * AlignmentKL
```

- **CTC and AlignmentKL are computed on positive pairs only.** The collate
  duplicates each video with a mismatched negative transcript; supervising
  those pairs would train the model to align text that contradicts the lips.
  (The CTC *feature* still feeds the classifier for negative pairs.)
- **CTC**: `F.ctc_loss` with `reduction='none'` on log-softmaxed per-frame
  logits; mean over positive samples. `input_lengths` from the video padding
  mask, `target_lengths` from the phoneme padding mask.
- **AlignmentKL**: temperature-scaled frame→token softmax over the
  cross-modal `sim_matrix` (`B x T x L`), cross-entropy against the one-hot
  target rows, averaged over frames valid for that sample (speech ∧
  non-padded ∧ `has_align_target`) — per-sample frame means are then averaged
  over samples that have targets, so batches with partial `.lab` coverage
  aren't biased.
- CTC already enforces monotonic alignment; no separate monotonicity
  regularizer.
- BCE + contrastive verification terms are unchanged.

New config keys (with defaults): `lambda_ctc: 1.0`, `lambda_align: 1.0`,
`alignment_temperature: 1.0`, `digit_source_fps: 29.87`,
`digit_max_frames: 125`.

## 5. Data Loading

### 5.1 `dataset/grid_loader.py`

- Parse `.align` into `word_times`: list of `(word, start_value, end_value)`
  plus the file's `max_end`; convert to frames in `__getitem__` using the
  §2.1 calibration (video_len is known there).
- Expose `word_times` (frame indices) in `__getitem__`.

### 5.2 `dataset/digit_loader.py`

- Fix the `break` bug (missing align file aborts the entire loop) → `continue`.
- Read `.lab` (probe processed `text/<base>.lab`, then `lab/<base>.lab`);
  convert seconds → frames via §2.2 fps resolution; apply the §2.2 skip
  policy. Expose `digit_times`.

### 5.3 `utils/alignment.py` (new) + `utils/phoneme_converter.py`

- `PhonemeConverter.word_to_phoneme_indices(word)` and
  `text_to_phoneme_indices_per_word(text)`: per-word `g2p`, filtering
  non-ARPABET tokens. **Single source of truth**: the concatenated per-word
  indices are used both as the text-encoder input and as the target token
  axis. This also fixes the latent whole-sentence path where every inter-word
  space became `<unk>` (`vocab.get(' ')`).
- `build_alignment_targets(token_frame_ranges, video_len, target_len)` →
  `(targets T x L one-hot rows, speech_mask T)`.
- `calibrate_frames(values, max_end, video_len)` for §2.1.

### 5.4 `utils/dataset_utils.py` (collate)

- GRID phoneme sequences come from per-word phonemization (both pairs).
- Build alignment targets for **positive** samples only (negatives keep
  zeros). Batch gains `alignment_targets` (`B x T x L`), `speech_mask`
  (`B x T`), `has_align_target` (`B`, float).

## 6. Negative Sampling (`utils/negative_sampler.py`)

- Make the sampler dataset-aware (`dataset_name` arg).
- `phoneme_similar`:
  - GRID: replace a randomly chosen word with a phonetically confusable GRID
    word from the same grammatical class (color→color, letter→letter,
    digit-word→digit-word, command/prep/adverb classes), fallback to random
    GRID vocab word.
  - Digits: replace one digit with a random different digit (retry on
    identity).
- Global guard: if the generated negative equals the positive transcript
  (e.g. shuffled "7 7 7 7"), fall back to `_random_other`.

## 7. Training Loop (`training/train.py`)

- Read `ctc_logp_per_token` from the model output; CTC loss = mean over
  positive samples.
- AlignmentKL from `sim_matrix` per §4; skipped (0) when no sample in the
  batch has targets.
- Weighted sum per config; new tensorboard scalars for CTC and alignment.

## 8. Docs & Naming

- Renamed "IPA" references to "ARPABET" — on inspection none exist in
  `LIP-TO-TEXT.MD` or the code, so no changes were needed. The vocab is
  ARPABET from `g2p_en` throughout.
- `inference.py` switched from the legacy whole-sentence phonemization to
  the per-word path so inference sequences match training sequences exactly.

## 9. Files Touched

- `models/verification_model.py`
- `models/lip_encoder.py` (shared masked-pooling helper)
- `models/audio_encoder.py` (positional encoding, padding mask)
- `dataset/grid_loader.py`
- `dataset/digit_loader.py`
- `pipeline/data_preparation.py` (copy `lab/`, write `meta.json` with source fps)
- `utils/alignment.py` (new)
- `utils/dataset_utils.py`
- `utils/negative_sampler.py`
- `utils/phoneme_converter.py`
- `training/train.py`
- `evaluation/evaluate.py` (audio mask pass-through)
- `config.json`
- `LIP-TO-TEXT.MD`
- `tests/test_alignment.py` (new)
- `inference.py` (per-word phonemization for train/inference consistency)

Out of scope (follow-ups): digit split protocol (random split ignores the
`_train_`/`_verify_` filename tags and leaks speakers across splits).

## 10. Verification Plan

1. Unit: per-word phonemization — no `<unk>` for spaces, concatenation equals
   per-word results.
2. Unit: target builder on synthetic frame ranges — one-hot rows, speech mask,
   out-of-range clamping.
3. Unit: digit negative sampler never returns the positive; GRID class-aware
   swaps stay in-class.
4. Unit: calibration maps a real-format GRID align (0–74500 → 0–75) correctly.
5. Smoke: model forward with CTC head + masked pooling on a small random
   batch for both vocab sizes (CPU).
6. Data sanity (against raw data in `../ts-lip2text`): parsed `.align` frame
   ranges ⊆ [0, 75]; `.lab` discovery + fps resolution produce in-range frames.
7. End-to-end: `torchrun main.py` on a CUDA box on existing data (out of scope
   for this machine; CPU smoke covers the code paths).
