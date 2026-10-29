# Visual Forced Alignment for Lip-Text Verification — Design

Date: 2026-10-29
Branch: `feature/visual-forced-alignment`
Status: approved by user (design sections 1–3, this conversation)

## Goal

Add the Visual-Forced-Alignment (VFA) *method* to liptev so that, when scoring a
(video, transcript) pair, the system also reports:

1. **Per-token timestamps** — when each word/digit starts and ends in the video.
2. **Per-token alignment score** — how well the visual signal matches the text
   for that token (non-semantic; purely visual-text correspondence).
3. **Per-token verification score** — whether the lips are actually articulating
   that word/digit (semantic; distinguishes e.g. "three" from "tree").

Both datasets (GRID and digit) get this from day one.

The VFA reference implementation lives in the sibling repo
`../Visual-Forced-Alignment` (model: `PhonemeSplitDecoderNet`). It has **no
pretrained checkpoints**, so we do not use its model; we port its *method* into
liptev's existing model.

## Decisions (from brainstorming)

- **VFA method inside liptev** — new heads on `LipTextVerificationModel`'s
  existing lip encoder; one model, one checkpoint. No dependency on the sibling
  repo at runtime.
- **Both datasets at once** (GRID + digit).
- **Trained per-word verification head** (not just derived signals) so the
  semantic score genuinely localizes which token mismatches.
- **Output**: human-readable summary + structured JSON.
- **Work on `feature/visual-forced-alignment`**, merged to `main` by PR.

## 1. Model changes (`models/`)

### 1a. Forced-alignment heads — new `models/vfa_heads.py`

Two heads consuming the lip encoder's output (`B×T×256`), mirroring VFA's
`frame_linear` and `bound_linear` (`MultiLinear(..., hidden=128)`-style stacks):

- **Token-posterior head**: per-frame logits over `[PAD, SIL] + vocab`
  (GRID: word vocab; digit: `digit_vocab` incl. `!`). This is the emission
  model for forced alignment.
- **Boundary head**: per-frame binary logit for "token transition here".

Training signals already exist in the pipeline:

- Per-frame token ids: argmax over the one-hot `alignment_targets` (`B×T×L`)
  built from `.align`/`.lab` boundaries (`utils/alignment.py`).
- Boundary targets: edges of token frame ranges.

New losses added to the total: frame CE (`lambda_frame_ce`) + boundary BCE
(`lambda_boundary`). Samples with `has_align_target=False` (broken `.lab`s
etc.) are masked from both, via the existing speech-mask machinery.

**Skipped on purpose**: VFA's autoregressive token-sequence decoder head. The
existing CTC head already forces the visual stream to encode the transcript;
a third sequence loss would be redundant.

### 1b. Viterbi forced aligner — new `utils/viterbi.py`

Port of `Visual-Forced-Alignment/scripts/viterbi_align.py`, cleaned up:

- Pure Python/torch, CPU-only (no `.cuda()` requirements).
- Monotonic DP over `[SIL] + tokens + [SIL]`, each token ≥ 1 frame; transition
  scores from softmaxed frame posteriors; transitions at boundary probability
  > `vfa_boundary_threshold` (default 0.8) are forced, as in VFA.
- Returns the frame→token path; consecutive runs of the same token yield
  `(token, start_frame, end_frame)` spans → seconds at the video's probed fps.

### 1c. Per-word verification head — new `WordVerificationHead` (in `models/vfa_heads.py`)

Per token: masked-mean of lip features over the token's frames, concatenated
with the token's text-encoder embedding and cross-modal aligned feature →
MLP → per-token verification logit → sigmoid → `[0,1]` score.

- **Training**: pools over **ground-truth** token ranges. The negative sampler
  already swaps one token for swap-style strategies; the collate emits
  **per-token labels**: all 1 on positives; 0 on the swapped token, 1 elsewhere
  on substitution negatives. Whole-sequence negatives (`random`, `shuffled`)
  have no well-defined per-token attribution, so their per-token labels are
  masked — the head is not supervised on them.
- **Inference**: pools over **Viterbi spans** (so the head's usefulness also
  implicitly depends on alignment quality).

Loss: `BCEWithLogitsLoss` per valid token, weight `lambda_word_verif`.

## 2. Inference (`inference.py`)

New optional flag `--json <path>` (structured result printed either way):

1. Run `viterbi_align` on the token-posterior head's frame logits with the
   transcript token sequence → per-token `(start_s, end_s)`, converted at the
   fps probed from the input video (not hardcoded 25; digit source fps varies).
2. Per-token **alignment score**: mean softmax posterior of the aligned token
   over its span, averaged with the existing cross-modal `word_scores`
   restricted to the span (simple mean of the two `[0,1]` values).
3. Per-token **verification score**: `WordVerificationHead` sigmoid over span.
4. Per-token verdict at the global EER thresholds; low-confidence flag for
   degenerate spans.

```json
{
  "global": {"verification_score": 0.93, "alignment_score": 0.87, "verdict": "MATCH"},
  "tokens": [
    {"token": "3", "start_s": 0.24, "end_s": 0.76,
     "alignment_score": 0.91, "verification_score": 0.95, "verdict": "MATCH",
     "low_confidence": false}
  ]
}
```

Human-readable summary keeps its current form, extended with a per-token table.

## 3. Config, checkpoints, error handling

- **Config** (`config.json` + `Config`): `lambda_frame_ce`,
  `lambda_boundary`, `lambda_word_verif`, `vfa_boundary_threshold: 0.8`.
  Defaults keep existing commands working.
- **Checkpoints** (`utils/checkpoint.py`): new heads are absent from old
  checkpoints — load with `strict=False` for the new heads only, print an
  explicit warning, and degrade gracefully (JSON reports null timestamps/
  per-token scores). Stamp `vfa_heads: true` in checkpoint metadata next to
  `alignment_mode`, same pattern as today.
- **Error handling**:
  - Video too short for the token count (< 1 frame/token + SIL): fall back to
    uniform frame split, flag `low_confidence: true`.
  - `!` interjection tokens participate like any token (they're in `digit_vocab`).
  - Broken `.lab`s: no alignment supervision via existing `has_align_target`.

## 4. Testing (TDD; local env `~/miniconda3/envs/torch/bin/python`)

- `utils/viterbi.py`: synthetic posteriors with known optimal paths;
  forced-transition behavior; T < n_tokens fallback.
- Collate per-token labels: swap negatives → 0 on swapped token only;
  `random`/`shuffled` → masked.
- Head shapes and masking (padding, `has_align_target=False`).
- Inference JSON schema (structure, per-token fields, graceful degradation).
- Checkpoint back-compat: loading an old checkpoint warns and degrades.

Training runs on CUDA machines, as with all prior work.

## 5. Out of scope

- Using/training VFA's standalone `PhonemeSplitDecoderNet`.
- Chinese G2P phoneme tokenization for the digit set (separate exploration,
  `utils/chinese_g2p.py`).
- Fixing the digit `_train_`/`_verify_` split-tag leakage issue (open, unrelated).
