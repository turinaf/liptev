# Design: Frame-Level Alignment for Lip-to-Text Verification

Date: 2026-08-20
Branch: `feature/frame-alignment-verification`

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

`*.align` files contain one word per line:

```
0.060 0.280 bin
...
```

`start` and `end` are in seconds. Processed videos are written at 25fps
(`pipeline/data_preparation.py`), so:

```
start_frame = int(start_seconds * 25)
end_frame   = int(end_seconds * 25)
```

Each word maps to a frame range. Within a word, frames are distributed
uniformly across that word's phonemes (CMU-dict per-word phoneme mapping via
`PhonemeConverter`). This yields a soft target matrix `B x T x L`.

### 2.2 Digits (digit-level alignments, `.lab` files)

User-provided `.lab` files contain two lines:

```
7 0 6 7 9 3 2 5
0.3-0.57 0.6-0.72 0.75-0.93 0.96-1.17 1.2-1.38 1.41-1.65 1.68-1.74 1.77-2.01
```

- Line 1: digit sequence (same tokens as the `.txt` transcript)
- Line 2: per-token `start-end` ranges in seconds

Each digit maps to a frame range -> one-hot target matrix `B x T x L`.

The `DigitDataset` loader searches for `<base>.lab` next to `<base>.txt` (same
`text/` directory, and also probes a sibling `lab/` directory). If no `.lab`
file exists for a sample, the sample still trains via CTC with no alignment
supervision loss.

## 3. Model Architecture Changes

### 3.1 CTC Predictor head

Add to `LipTextVerificationModel`:

```python
self.ctc_head = nn.Linear(feature_dim, vocab_size + 1)  # blank token appended
```

`blank = vocab_size` (41 for GRID, 13 for digits). Per-frame logits over the
lip encoder output (`B x T x V+1`). CTC loss enforces monotonic frame->token
alignment implicitly and is the primary alignment mechanism for both datasets.

### 3.2 Masked pooling

Replace un-masked `mean(dim=1)` pooling with mask-aware pooling for lip, text,
and audio features. Padded frames are currently averaged in, leaking repeated
last-frame padding into the pooled representation and into the classifier.

### 3.3 CTC log-likelihood feature

Feed `log P(T|V)` (negative of CTC loss per sample) into the classifier input
vector so the verification score is alignment-aware.

## 4. Losses

```
loss = BCE + contrastive + lambda_ctc * CTC + lambda_align * AlignmentKL
```

- **CTC**: `torch.nn.CTCLoss` between per-frame lip logits and the token
  sequence (blank = vocab_size). Uses `video_lens` as `input_lengths` and true
  token length as `target_lengths`.
- **AlignmentKL**: frame->token softmax over the cross-modal `sim_matrix`
  (`B x T x L`, temperature-scaled), cross-entropy against the target matrix,
  masked to speech frames (frames belonging to some token) and non-padded
  frames.
- CTC already enforces monotonic alignment; no separate monotonicity regularizer
  is added.
- BCE + contrastive verification terms are unchanged.

New config keys (with defaults): `lambda_ctc: 1.0`, `lambda_align: 1.0`,
`alignment_temperature: 1.0`.

## 5. Data Loading

### 5.1 `dataset/grid_loader.py`

- Parse `.align` into `word_times`: list of `(word, start_frame, end_frame)`.
- Expose in `__getitem__` as `word_times`.
- Keep existing transcript behavior.

### 5.2 `dataset/digit_loader.py`

- Fix the `break` bug (missing align file aborts the entire loop) -> `continue`.
- Read `.lab` files into `digit_times`: list of `(digit, start_frame, end_frame)`.
  Probe `<base>.lab` in `text/` and `lab/`; if absent, `digit_times = None`.
- Expose in `__getitem__` as `digit_times`.

### 5.3 `utils/alignment.py` (new)

- `build_word_to_phoneme_indices(converter, transcript)` -> list of
  `(word, [phoneme_ids])` using per-word CMU-dict mapping (deterministic).
- `build_alignment_targets(seq_len, times, token_id_fn, max_frames)` -> target
  matrix `T x L` (soft for phonemes, one-hot for digits) plus a speech-frame
  mask, given frame-index ranges per token.

### 5.4 `utils/dataset_utils.py` (collate)

- Build per-sample alignment targets from `word_times` / `digit_times`.
- Return `alignment_targets` (`B x T x L`) and `speech_mask` (`B x T`) in the
  batch dict.

## 6. Negative Sampling (`utils/negative_sampler.py`)

- Make the sampler dataset-aware (`dataset_name` arg).
- `phoneme_similar`:
  - GRID: replace a randomly chosen word position with a phonetically
    confusable GRID word (color->color, letter->letter, digit-word->digit-word,
    other GRID vocab words).
  - Digits: replace one digit with a random different digit (retry on identity).
- Global guard: if the generated negative equals the positive transcript, fall
  back to `_random_other`.

## 7. Training Loop (`training/train.py`)

- Extract per-frame lip logits for CTC.
- Compute CTC loss and alignment KL loss alongside BCE/contrastive.
- Weighted sum per config.
- Keep the existing tensorboard logging; add CTC and alignment loss scalars.

## 8. Docs & Naming

- Rename "IPA" references to "ARPABET" in `LIP-TO-TEXT.MD` and comments
  (the vocab is ARPABET from `g2p_en`).

## 9. Files Touched

- `models/verification_model.py`
- `models/lip_encoder.py` (masked pooling helper)
- `dataset/grid_loader.py`
- `dataset/digit_loader.py`
- `utils/alignment.py` (new)
- `utils/dataset_utils.py`
- `utils/negative_sampler.py`
- `utils/phoneme_converter.py`
- `training/train.py`
- `config.json`
- `LIP-TO-TEXT.MD`

## 10. Verification Plan

1. Sanity test: per-word phoneme mapping (`build_word_to_phoneme_indices`).
2. Sanity test: alignment-target builder on synthetic frame ranges.
3. Sanity test: digit negative sampler never returns `positive` and never
   produces `"4 red 6"`-style cross-vocab garbage.
4. Unit smoke test: model forward with CTC head + masked pooling on a small
   batch for both GRID and digit vocab sizes.
5. End-to-end: `torchrun --nproc_per_node=2 main.py` on the existing data.