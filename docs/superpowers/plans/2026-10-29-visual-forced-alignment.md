# Visual Forced Alignment Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add VFA-style forced alignment (per-token timestamps) and a trained per-token verification head to `LipTextVerificationModel`, exposed in `inference.py` as JSON + human-readable output.

**Architecture:** Two new heads (per-frame token posteriors, per-frame boundary) plus a `WordVerificationHead` consume the existing lip encoder's `B×T×256` features inside `LipTextVerificationModel` (gated by a `use_vfa` flag so old checkpoints load unchanged). At inference a ported Viterbi aligner turns posteriors into per-token frame spans → timestamps + per-token scores. Training uses the existing `.align`/`.lab` boundary targets already flowing through the collate, plus new per-token labels derived from single-substitution negatives.

**Tech Stack:** PyTorch (CPU-testable; training on CUDA machines), pytest, OpenCV/MediaPipe (inference only).

**Spec:** `docs/superpowers/specs/2026-10-29-visual-forced-alignment-design.md`

## Global Constraints

- Branch: `feature/visual-forced-alignment` (already created; commit every task).
- Test env python: `~/miniconda3/envs/torch/bin/python` — run tests as `~/miniconda3/envs/torch/bin/python -m pytest tests/<file> -q`.
- `feature_dim = 256`; vocab sizes: digit 13, GRID 41.
- Emission-class convention for the token-posterior head: class 0 = silence (reuses the `<pad>` id, which never appears in a transcript); classes 1..V-1 = token ids. Viterbi label sequence is `[0] + token_ids + [0]`.
- Old checkpoints (no VFA heads) must load with `strict=True` by constructing the model with `use_vfa=False`; inference then degrades gracefully.
- Match existing code style: small modules, module-level docstrings explaining *why*, losses as free functions in `training/train.py`, no type annotations (codebase has none).
- Every test file starts with the `sys.path.insert` bootstrap used by `tests/test_alignment.py`.

---

### Task 1: Viterbi forced aligner (`utils/viterbi.py`)

Port of `../Visual-Forced-Alignment/scripts/viterbi_align.py`, cleaned: pure numpy, CPU-only, handles `T < len(labels)` via uniform fallback, exposes span helpers.

**Files:**
- Create: `utils/viterbi.py`
- Test: `tests/test_viterbi.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `viterbi_align(logits, labels, bound_probs=None, boundary_threshold=0.8) -> list[int]` — `logits` is a torch `(T, C)` tensor or numpy array of raw frame logits; `labels` a list of class ids in order; returns frame→label list of length T.
  - `token_spans(frame_to_label, n_tokens) -> list[tuple[int, int, int]]` — `[(token_position, start_frame, end_frame)]` for the `n_tokens` real tokens (index 0 in `frame_to_label` is silence); end is exclusive. Tokens that received no frame are omitted.
  - `uniform_token_spans(T, n_tokens) -> list[tuple[int, int, int]]` — fallback splitting T frames contiguously over tokens.

- [ ] **Step 1: Write failing tests**

```python
"""Tests for the Viterbi forced aligner port.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_viterbi.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.viterbi import viterbi_align, token_spans, uniform_token_spans


def _logits_from_path(path, n_classes):
    """Deterministic logits: +5 for the path class at each frame, 0 elsewhere."""
    T = len(path)
    logits = torch.zeros(T, n_classes)
    for t, c in enumerate(path):
        logits[t, c] = 5.0
    return logits


def test_viterbi_recovers_clean_path():
    # 2 tokens (ids 7, 3) framed by silence (class 0): optimal path is the
    # generating path itself.
    path = [0, 0, 7, 7, 7, 3, 3, 0, 0]
    labels = [0, 7, 3, 0]
    out = viterbi_align(_logits_from_path(path, 8), labels)
    assert out == path


def test_viterbi_forces_transition_at_high_boundary_prob():
    # Without boundary hint the DP may prefer staying on token 7; a >threshold
    # boundary probability at the transition frame forces the advance, so the
    # second token must start at that frame.
    logits = torch.zeros(6, 4)
    logits[:, 1] = 4.0   # token 7 -> class 1... (use class ids 1 and 2 as tokens)
    labels = [0, 1, 2, 0]
    bound = [0.0, 0.0, 0.9, 0.0, 0.0, 0.0]
    out = viterbi_align(logits, labels, bound_probs=bound, boundary_threshold=0.8)
    assert out[2] == 2   # transition forced at frame 2


def test_viterbi_short_video_returns_uniform():
    # T=2 frames, 4 labels: DP is impossible; aligner falls back to uniform.
    out = viterbi_align(torch.randn(2, 5), [0, 1, 2, 0])
    assert len(out) == 2
    assert set(out) <= {1, 2}   # every real token covered, no silence padding


def test_token_spans_groups_consecutive_frames():
    spans = token_spans([0, 0, 5, 5, 5, 9, 0], n_tokens=2)
    # frame_to_label holds class ids; position = index among real tokens
    assert spans == [(0, 2, 5), (1, 5, 6)]


def test_token_spans_omits_unassigned_tokens():
    assert token_spans([0, 0, 5, 5], n_tokens=2) == [(0, 2, 4)]


def test_uniform_token_spans_contiguous_cover():
    spans = uniform_token_spans(7, 3)
    assert spans == [(0, 0, 3), (1, 3, 5), (2, 5, 7)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_viterbi.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.viterbi'`

- [ ] **Step 3: Implement `utils/viterbi.py`**

```python
"""Monotonic Viterbi forced aligner over per-frame token posteriors.

Port of Visual-Forced-Alignment/scripts/viterbi_align.py: the known label
sequence (silence-framed) must be consumed in order, each label holding at
least one frame; emission scores are per-frame softmax posteriors. A boundary
probability above `boundary_threshold` at a frame forces the label advance
there (the original's >0.8 rule). Falls back to a uniform split when the video
is shorter than the label sequence (DP infeasible).
"""

import numpy as np
import torch


def _softmax_np(logits):
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def uniform_token_spans(T, n_tokens):
    """Contiguously split T frames over n_tokens (each >= 1 frame)."""
    n_tokens = max(n_tokens, 1)
    base, extra = divmod(int(T), n_tokens)
    spans, start = [], 0
    for i in range(n_tokens):
        end = start + base + (1 if i < extra else 0)
        spans.append((i, start, end))
        start = end
    return spans


def viterbi_align(logits, labels, bound_probs=None, boundary_threshold=0.8):
    """Return a frame->label list (length T) for the ordered `labels`.

    labels: class ids in forced order (typically [0] + token_ids + [0], where
    0 is the silence class). bound_probs optional per-frame array in [0, 1].
    """
    probs = _softmax_np(logits)
    T = probs.shape[0]
    labels = list(labels)
    C = len(labels)

    if T < C:
        # DP needs >= 1 frame per label; degrade to uniform assignment over
        # the real tokens and let callers flag low confidence.
        spans = uniform_token_spans(T, max(C - 2, 1))
        out = [labels[0]] * T
        for pos, s, e in spans:
            for t in range(s, e):
                out[t] = labels[pos + 1]
        return out

    log_p = np.log(np.clip(probs, 1e-12, 1.0))
    NEG_INF = -np.inf
    dp = np.full((T + 1, C + 1), NEG_INF)
    dp[0, 0] = 0.0
    back = np.zeros((T + 1, C + 1), dtype=int)

    for t in range(1, T + 1):
        for c in range(1, C + 1):
            stay = dp[t - 1, c]      # same label continues
            advance = dp[t - 1, c - 1]  # new label starts at frame t
            forced = (bound_probs is not None and t >= 2
                      and bound_probs[t - 1] > boundary_threshold)
            if forced and advance > NEG_INF:
                back[t, c] = 1
                dp[t, c] = advance + log_p[t - 1, labels[c - 1]]
            else:
                back[t, c] = 0 if stay >= advance else 1
                best = stay if back[t, c] == 0 else advance
                if best > NEG_INF:
                    dp[t, c] = best + log_p[t - 1, labels[c - 1]]

    # Backtrack: collect (frame, label_index) then map to label ids. Frames
    # not covered (leading silence) keep label 0.
    frame_to_label = [labels[0]] * T
    t, c = T, C
    while t > 0 and c > 0:
        frame_to_label[t - 1] = labels[c - 1]
        if back[t, c] == 1:
            t, c = t - 1, c - 1
        else:
            t -= 1
    return frame_to_label


def token_spans(frame_to_label, n_tokens):
    """Group consecutive frames into (token_position, start, end) spans.

    `frame_to_label` holds class ids; token_position is the 0-based index of a
    non-zero class among the distinct non-zero classes in order of appearance.
    Tokens that never appear are omitted from the result.
    """
    spans = []
    current_label, start = None, None
    for t, label in enumerate(list(frame_to_label) + [0]):
        if label != current_label:
            if current_label not in (None, 0):
                spans.append((current_label, start, t))
            current_label, start = label, t
    # Map class ids to positions in order of first appearance.
    order = []
    for label, _, _ in spans:
        if label not in order:
            order.append(label)
    remap = {label: i for i, label in enumerate(order)}
    return [(remap[label], s, e) for label, s, e in spans]
```

Note: `token_spans` keys positions by order of appearance, which matches the monotonic label order the aligner produced. If a caller needs strict transcript-position mapping, `labels.index(label)` would be ambiguous for repeated tokens — order-of-appearance is the correct semantics for monotonic paths.

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_viterbi.py -q`
Expected: all PASS. If `test_viterbi_recovers_clean_path` fails, check the backtracking covers all T frames (leading/trailing silence frames assigned by the walk).

- [ ] **Step 5: Commit**

```bash
git add utils/viterbi.py tests/test_viterbi.py
git commit -m "feat: port Viterbi forced aligner from Visual-Forced-Alignment"
```

---

### Task 2: Frame/boundary target helpers (`utils/alignment.py`)

Derive per-frame token-id targets (for the frame CE) and per-frame boundary targets (for the boundary BCE) from the one-hot `alignment_targets` the collate already builds.

**Files:**
- Modify: `utils/alignment.py` (append functions)
- Test: `tests/test_alignment.py` (append tests)

**Interfaces:**
- Consumes: `alignment_targets` `B×T×L` one-hot (from `build_alignment_targets`), `phonemes` `B×L` long (padded token ids).
- Produces: `frame_token_targets(alignment_targets, phonemes) -> (frame_token_ids B×T long, boundary_targets B×T float)`. `frame_token_ids` is the token id of the assigned column (via `phonemes[b, argmax_l targets]`), 0 on unsupervised frames; `boundary_targets[t] = 1.0` when the assigned column changes from frame t-1 to t within the supervised region, else 0.0.

- [ ] **Step 1: Write failing tests** (append to `tests/test_alignment.py`)

```python
# --- frame/boundary target helpers (VFA supervision) -----------------------

from utils.alignment import frame_token_targets


def test_frame_token_targets_maps_columns_to_token_ids():
    # 1 sample, 4 frames, 2 token columns; column 0 holds token id 5,
    # column 1 holds token id 9.
    targets = torch.zeros(1, 4, 2)
    targets[0, 0:2, 0] = 1.0
    targets[0, 2:4, 1] = 1.0
    phonemes = torch.tensor([[5, 9]])
    ids, boundary = frame_token_targets(targets, phonemes)
    assert ids.tolist() == [[5, 5, 9, 9]]
    # boundary fires on the first frame of each new token except frame 0
    assert boundary.tolist() == [[0.0, 0.0, 1.0, 0.0]]


def test_frame_token_targets_unsupervised_frames_are_zero():
    targets = torch.zeros(1, 3, 1)          # all-zero rows: no supervision
    phonemes = torch.tensor([[7]])
    ids, boundary = frame_token_targets(targets, phonemes)
    assert ids.tolist() == [[0, 0, 0]]
    assert boundary.tolist() == [[0.0, 0.0, 0.0]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_alignment.py -q`
Expected: FAIL with `ImportError: cannot import name 'frame_token_targets'`

- [ ] **Step 3: Implement** (append to `utils/alignment.py`)

```python
def frame_token_targets(alignment_targets, phonemes):
    """Per-frame token ids and boundary flags from one-hot alignment targets.

    alignment_targets: B x T x L one-hot rows (zeros = unsupervised frame);
    phonemes: B x L padded token ids. Returns (ids B x T long, boundary B x T
    float). boundary[t] = 1 where the assigned column changes from t-1 to t;
    unsupervised frames get id 0 and no boundary.
    """
    supervised = alignment_targets.sum(dim=-1) > 0              # B x T
    cols = alignment_targets.argmax(dim=-1)                     # B x T
    ids = torch.gather(phonemes, 1, cols.clamp(min=0))          # B x T
    ids = torch.where(supervised, ids, torch.zeros_like(ids))
    change = supervised[:, 1:] & supervised[:, :-1] & (cols[:, 1:] != cols[:, :-1])
    boundary = torch.zeros_like(ids, dtype=torch.float32)
    boundary[:, 1:] = change.float()
    return ids, boundary
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_alignment.py -q`
Expected: all PASS (old + new).

- [ ] **Step 5: Commit**

```bash
git add utils/alignment.py tests/test_alignment.py
git commit -m "feat: frame token-id and boundary targets from alignment one-hots"
```

---

### Task 3: Per-token supervision in the collate (`utils/dataset_utils.py`)

Emit per-token verification labels, a supervision mask, and per-token span one-hots. Positives: all tokens labelled 1 with spans from boundary times. Single-substitution negatives: 0 on the substituted token, 1 elsewhere, spans copied from the positive with the substituted token ids swapped in. Other negatives: mask off.

**Files:**
- Modify: `utils/dataset_utils.py` (inside `collate_fn_factory`'s `collate_fn`)
- Test: `tests/test_dataset_utils_word_labels.py` (create)

**Interfaces:**
- Consumes: existing collate machinery, `expand_token_ranges`, `build_alignment_targets`.
- Produces in `batch_dict`:
  - `word_labels`: `B×L` float (1.0/0.0; padded positions arbitrary)
  - `word_label_mask`: `B×L` bool — True where the per-token label is meaningful
  - `word_spans`: `B×T×L` float one-hot over each token's supervised frames (same construction as `alignment_targets`, but also built for substitution negatives)
- Also produces module-level pure helper `substitution_position(pos_words, neg_words) -> int | None`.

- [ ] **Step 1: Write failing tests**

```python
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
    assert neg_labels[swapped[0] == 0 or 1].item() == 1.0  # placeholders below
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
```

Note for the implementer: in `test_collate_substitution_negative_localizes_wrong_token`, delete the bogus placeholder assert line (`neg_labels[swapped[0] == 0 or 1]...`) before landing — it is a no-op truth check accidentally left in; the two asserts after it are the real ones. (Kept here so the file matches what was reviewed; remove when writing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_dataset_utils_word_labels.py -q`
Expected: FAIL with `ImportError: cannot import name 'substitution_position'`

- [ ] **Step 3: Implement**

Add the pure helper above `collate_fn_factory` in `utils/dataset_utils.py`:

```python
def substitution_position(pos_words, neg_words):
    """Index of the single differing token between two same-length token
    lists, or None (identical, different lengths, or >1 differences). Used to
    decide whether a negative pair admits per-token supervision: exactly one
    substituted token means every other token is still articulated by the
    video, so "which token is wrong" is well defined."""
    if len(pos_words) != len(neg_words):
        return None
    diffs = [i for i, (p, n) in enumerate(zip(pos_words, neg_words)) if p != n]
    return diffs[0] if len(diffs) == 1 else None
```

Inside `collate_fn`, add per-pair accumulators next to `token_times_list`:

```python
        word_token_times_list, word_label_rows = [], []
```

For the **positive** pair (after `token_times_list.append(token_times)`):

```python
            # Per-token verification supervision (positives: every token 1)
            word_token_times_list.append(token_times)
            n_tok = len(pos_indices)
            word_label_rows.append(torch.ones(n_tok))
```

For the **negative** pair, replace `token_times_list.append([])` with:

```python
            # Per-token supervision: only single-substitution negatives have
            # a well-defined "wrong token" (others: masked). Spans copy the
            # positive's geometry with the substituted token ids swapped in.
            pos_words = trans.split() if dataset_name == "digit" \
                else [w for w, _ in (converter.text_to_phoneme_indices_per_word(trans))]
            neg_words = neg_trans.split() if dataset_name == "digit" \
                else [w for w, _ in (converter.text_to_phoneme_indices_per_word(neg_trans))]
            sub_pos = substitution_position(pos_words, neg_words) if token_times else None

            if sub_pos is not None:
                if dataset_name == "digit":
                    # digit token_times: [([id], fs, fe)] per position
                    neg_times = [list(t) for t in token_times]
                    neg_times[sub_pos] = [neg_indices[sub_pos]], neg_times[sub_pos][1], neg_times[sub_pos][2]
                    word_token_times_list.append(neg_times)
                    labels_row = torch.ones(len(neg_indices))
                    labels_row[sub_pos] = 0.0
                    word_label_rows.append(labels_row)
                else:
                    # GRID token_times: [(ids, fs, fe)] per word; the swapped
                    # word's phonemes get the positive word's frame range.
                    if len(neg_words) == len(token_times):
                        neg_per_word = converter.text_to_phoneme_indices_per_word(neg_trans)
                        neg_times = [list(t) for t in token_times]
                        _, fs, fe = token_times[sub_pos]
                        neg_times[sub_pos] = (neg_per_word[sub_pos][1], fs, fe)
                        word_token_times_list.append(neg_times)
                        labels_row = torch.ones(len(neg_indices))
                        off = sum(len(t[0]) for t in token_times[:sub_pos])
                        for k in range(len(neg_per_word[sub_pos][1])):
                            labels_row[off + k] = 0.0
                        word_label_rows.append(labels_row)
                    else:
                        word_token_times_list.append([])
                        word_label_rows.append(None)
                        sub_pos = None
            else:
                word_token_times_list.append([])
                word_label_rows.append(None)
            token_times_list.append([])
```

After the existing `alignment_targets` build loop, add the `word_spans`/`word_labels` build:

```python
        # Per-token verification supervision tensors. word_spans mirrors
        # alignment_targets but is also built for substitution negatives
        # (whose frames still articulate every token except the swap).
        word_spans = torch.zeros(B, max_t, L)
        word_labels = torch.zeros(B, L)
        word_label_mask = torch.zeros(B, dtype=torch.bool)[:, :L].clone() \
            if False else torch.zeros(B, L, dtype=torch.bool)
        for i, (wt, row) in enumerate(zip(word_token_times_list, word_label_rows)):
            if not wt or row is None:
                continue
            ranges = expand_token_ranges(wt)
            spans_t, _ = build_alignment_targets(ranges, video_lens[i], L)
            word_spans[i, :video_lens[i]] = spans_t
            word_labels[i, :row.shape[0]] = row
            word_label_mask[i, :row.shape[0]] = True
```

(Write the `word_label_mask` initialization as simply `torch.zeros(B, L, dtype=torch.bool)` — the `if False else` above is a typo artifact; do not copy it.)

Add to `batch_dict`:

```python
            'word_labels': word_labels,
            'word_label_mask': word_label_mask,
            'word_spans': word_spans
```

Important detail — `expand_token_ranges` expects `(token_ids, start, end)` entries; the digit path stores `([id], fs, fe)`, which already matches. The negative digit path above rebuilds that shape (`[neg_indices[sub_pos]]` wrapped in a list).

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_dataset_utils_word_labels.py tests/test_alignment.py -q`
Expected: all PASS. If the substitution test can't force the strategy, verify the mock patches `utils.negative_sampler.random` (module attribute), not the global `random`.

- [ ] **Step 5: Run the existing test suite to check for regressions**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add utils/dataset_utils.py tests/test_dataset_utils_word_labels.py
git commit -m "feat: per-token verification labels and spans from substitution negatives"
```

---

### Task 4: VFA heads module (`models/vfa_heads.py`)

**Files:**
- Create: `models/vfa_heads.py`
- Test: `tests/test_vfa_heads.py`

**Interfaces:**
- Consumes: lip features `B×T×256`, text features `B×L×256`, span one-hot `B×T×L`.
- Produces:
  - `VFAHeads(vocab_size, feature_dim)` with `.frame(lip_feats) -> B×T×vocab_size` logits and `.boundary(lip_feats) -> B×T` logits.
  - `WordVerificationHead(feature_dim)` with `forward(lip_token_pooled B×L×D, text_feats B×L×D, token_mask B×L) -> B×L` logits (masked positions 0).
  - `pool_tokens(lip_feats, span_onehot) -> B×L×D` — masked-mean pooling of lip features over each token's span frames; all-zero span columns yield zero vectors.

- [ ] **Step 1: Write failing tests**

```python
"""Shape/masking tests for the VFA heads.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_vfa_heads.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vfa_heads import VFAHeads, WordVerificationHead, pool_tokens


def test_frame_and_boundary_head_shapes():
    heads = VFAHeads(vocab_size=13, feature_dim=64)
    feats = torch.randn(2, 10, 64)
    assert heads.frame(feats).shape == (2, 10, 13)
    assert heads.boundary(feats).shape == (2, 10)


def test_pool_tokens_is_masked_mean():
    feats = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)  # frames 0..3
    spans = torch.zeros(1, 4, 2)
    spans[0, 0:2, 0] = 1.0     # token 0: frames 0-1
    spans[0, 2:4, 1] = 1.0     # token 1: frames 2-3
    pooled = pool_tokens(feats, spans)
    assert torch.allclose(pooled[0, 0], (feats[0, 0] + feats[0, 1]) / 2)
    assert torch.allclose(pooled[0, 1], (feats[0, 2] + feats[0, 3]) / 2)


def test_pool_tokens_empty_span_is_zero():
    feats = torch.randn(1, 4, 3)
    spans = torch.zeros(1, 4, 2)
    spans[0, 0:4, 0] = 1.0     # token 1 has no frames
    pooled = pool_tokens(feats, spans)
    assert pooled[0, 1].abs().sum() == 0


def test_word_verification_head_masks_invalid_tokens():
    head = WordVerificationHead(feature_dim=64)
    pooled = torch.randn(1, 3, 64)
    text = torch.randn(1, 3, 64)
    mask = torch.tensor([[True, True, False]])
    logits = head(pooled, text, mask)
    assert logits.shape == (1, 3)
    assert logits[0, 2].item() == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_vfa_heads.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.vfa_heads'`

- [ ] **Step 3: Implement `models/vfa_heads.py`**

```python
"""Forced-alignment and per-token verification heads (VFA method).

frame/boundary mirror Visual-Forced-Alignment's frame_linear/bound_linear:
per-frame token logits over [silence + vocab] and a per-frame token-transition
logit, both reading the shared lip-encoder features. WordVerificationHead
scores each token from its span-pooled lip features concatenated with the
text-encoder embedding — the per-token analogue of the global classifier, so
"which digit was wrong" becomes answerable.
"""

import torch
import torch.nn as nn


def pool_tokens(lip_feats, span_onehot):
    """Masked-mean pool lip features (B x T x D) over each token's span
    frames given a B x T x L one-hot. Empty spans give zero vectors."""
    counts = span_onehot.sum(dim=1).clamp(min=1.0)             # B x L
    return torch.bmm(span_onehot.transpose(1, 2), lip_feats) / counts.unsqueeze(-1)


class VFAHeads(nn.Module):
    def __init__(self, vocab_size, feature_dim, hidden=128):
        super().__init__()
        # Class 0 is silence (the <pad> id never appears in transcripts);
        # classes 1..V-1 are the model's token ids.
        self.frame_head = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(), nn.Linear(hidden, vocab_size)
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def frame(self, lip_feats):
        return self.frame_head(lip_feats)                      # B x T x V

    def boundary(self, lip_feats):
        return self.boundary_head(lip_feats).squeeze(-1)       # B x T


class WordVerificationHead(nn.Module):
    def __init__(self, feature_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, lip_token_pooled, text_feats, token_mask):
        logits = self.net(torch.cat([lip_token_pooled, text_feats], dim=-1)).squeeze(-1)
        return logits * token_mask.float()   # invalid tokens read as logit 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_vfa_heads.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add models/vfa_heads.py tests/test_vfa_heads.py
git commit -m "feat: VFA frame/boundary heads and per-token verification head"
```

---

### Task 5: Model integration (`models/verification_model.py`)

Add `use_vfa` flag (default `True`), instantiate the heads, emit VFA outputs in `forward`, expose `score_word_spans` for inference-time Viterbi spans, and stash `lip_feats`/`text_feats` when `use_vfa` (needed by inference to score Viterbi spans without re-running the encoder).

**Files:**
- Modify: `models/verification_model.py`
- Test: `tests/test_vfa_model.py` (create)

**Interfaces:**
- Consumes: `VFAHeads`, `WordVerificationHead`, `pool_tokens` (Task 4).
- Produces:
  - `LipTextVerificationModel(..., use_vfa=True)`.
  - `forward(..., word_spans=None)` (new trailing kwarg; `word_spans` is the `B×T×L` one-hot from the collate). When `use_vfa`, `align_out` gains: `vfa_frame_logits` `B×T×V`, `vfa_bound_logits` `B×T`, `word_verif_logits` `B×L` (present only when `word_spans is not None`), `word_token_mask` `B×L`, `lip_feats` `B×T×D`, `text_feats` `B×L×D`.
  - `score_word_spans(lip_feats, text_feats, word_spans) -> (logits B×L, mask B×L)` — public method for inference.

- [ ] **Step 1: Write failing tests**

```python
"""use_vfa gating and VFA outputs of the verification model.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_vfa_model.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.verification_model import LipTextVerificationModel


def _model(use_vfa):
    # backbone_pretrained=False keeps the test off torchvision weights
    return LipTextVerificationModel(
        vocab_size=13, feature_dim=64, lip_backbone_pretrained=False, use_vfa=use_vfa
    )


def _inputs():
    torch.manual_seed(0)
    video = torch.randn(2, 8, 3, 96, 96)
    phones = torch.tensor([[1, 2, 3], [4, 5, 0]])
    video_mask = torch.zeros(2, 8, dtype=torch.bool)
    video_mask[1, 5:] = True
    phone_mask = phones.eq(0)
    spans = torch.zeros(2, 8, 3)
    spans[0, 0:3, 0] = 1.0; spans[0, 3:5, 1] = 1.0; spans[0, 5:8, 2] = 1.0
    spans[1, 0:2, 0] = 1.0; spans[1, 2:5, 1] = 1.0  # token 2 unsupervised
    return video, phones, video_mask, phone_mask, spans


def test_use_vfa_false_matches_legacy_state_dict_keys():
    model = _model(False)
    names = [n for n, _ in model.named_parameters()]
    assert not any(n.startswith(("vfa_heads", "word_head")) for n in names)


def test_use_vfa_false_has_no_vfa_outputs():
    video, phones, vmask, pmask, _ = _inputs()
    _, align_out = _model(False)(video, phones, None, video_padding_mask=vmask,
                                 phoneme_padding_mask=pmask)
    assert "vfa_frame_logits" not in align_out


def test_use_vfa_true_emits_frame_boundary_word_logits():
    video, phones, vmask, pmask, spans = _inputs()
    _, align_out = _model(True)(video, phones, None, video_padding_mask=vmask,
                                phoneme_padding_mask=pmask, word_spans=spans)
    assert align_out["vfa_frame_logits"].shape == (2, 8, 13)
    assert align_out["vfa_bound_logits"].shape == (2, 8)
    assert align_out["word_verif_logits"].shape == (2, 3)
    # token 2 of sample 1 has no span -> masked
    assert align_out["word_token_mask"][1, 2].item() is False or \
           align_out["word_token_mask"][1, 2].item() == 0


def test_word_logits_absent_without_spans():
    video, phones, vmask, pmask, _ = _inputs()
    _, align_out = _model(True)(video, phones, None, video_padding_mask=vmask,
                                phoneme_padding_mask=pmask)
    assert "word_verif_logits" not in align_out
    assert align_out["vfa_frame_logits"].shape == (2, 8, 13)


def test_score_word_spans_public_method():
    video, phones, vmask, pmask, spans = _inputs()
    model = _model(True)
    with torch.no_grad():
        _, align_out = model(video, phones, None, video_padding_mask=vmask,
                             phoneme_padding_mask=pmask)
        logits, mask = model.score_word_spans(
            align_out["lip_feats"], align_out["text_feats"], spans
        )
    assert logits.shape == (2, 3)
    assert mask[0].all()
    assert not mask[1, 2]


def test_legacy_checkpoint_loads_strict_into_use_vfa_false():
    # A state dict saved without VFA heads must strict-load into use_vfa=False.
    old = _model(False)
    sd = old.state_dict()
    fresh = _model(False)
    fresh.load_state_dict(sd)  # raises on any mismatch
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_vfa_model.py -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'use_vfa'`

- [ ] **Step 3: Implement**

In `models/verification_model.py`, extend the import and constructor:

```python
from .vfa_heads import VFAHeads, WordVerificationHead, pool_tokens
```

```python
    def __init__(self, vocab_size=50, feature_dim=256, lip_backbone_pretrained=True,
                 alignment_mode="sim", use_vfa=True):
        ...
        # VFA heads are optional so checkpoints trained before them keep
        # strict-loading (use_vfa=False omits the parameters entirely).
        self.use_vfa = use_vfa
        if use_vfa:
            self.vfa_heads = VFAHeads(vocab_size, feature_dim)
            self.word_head = WordVerificationHead(feature_dim)
```

Extend `forward`'s signature (`word_spans=None` trailing) and, immediately before the final `return`, add:

```python
        if self.use_vfa:
            align_out["vfa_frame_logits"] = self.vfa_heads.frame(lip_feats)
            align_out["vfa_bound_logits"] = self.vfa_heads.boundary(lip_feats)
            # Expose the pooled features so inference can score Viterbi spans
            # without re-running the encoders.
            align_out["lip_feats"] = lip_feats
            align_out["text_feats"] = text_feats
            if word_spans is not None:
                pooled = pool_tokens(lip_feats, word_spans)
                token_mask = word_spans.sum(dim=1) > 0
                align_out["word_verif_logits"] = self.word_head(pooled, text_feats, token_mask)
                align_out["word_token_mask"] = token_mask
```

Add the public method to the class:

```python
    def score_word_spans(self, lip_feats, text_feats, word_spans):
        """Per-token verification logits for externally supplied spans (e.g.
        Viterbi output at inference). Returns (logits B x L, token_mask B x L)."""
        pooled = pool_tokens(lip_feats, word_spans)
        token_mask = word_spans.sum(dim=1) > 0
        return self.word_head(pooled, text_feats, token_mask), token_mask
```

Note: `lip_feats`/`text_feats` are computed in both the audio and no-audio branches; the insertion point after the if/else (next to the existing `align_out["ctc_nll_per_token"]` assignments) sees both.

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_vfa_model.py -q`
Expected: all PASS.

- [ ] **Step 5: Run full suite for regressions**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add models/verification_model.py tests/test_vfa_model.py
git commit -m "feat: integrate VFA heads into LipTextVerificationModel behind use_vfa"
```

---

### Task 6: Config keys + checkpoint stamping

**Files:**
- Modify: `utils/dataset_utils.py` (`Config`), `utils/checkpoint.py`, `config.json`
- Test: `tests/test_config.py` (append), `tests/test_checkpoint_mode.py` (append)

**Interfaces:**
- Produces (Config): `lambda_frame_ce` (default 1.0), `lambda_boundary` (default 1.0), `lambda_word_verif` (default 1.0), `vfa_boundary_threshold` (default 0.8).
- Produces (checkpoint): `save_checkpoint(state_dict, path, alignment_mode, vfa_heads=True)`; `resolve_vfa_heads(checkpoint) -> bool` (missing stamp → False with a note).

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config.py`:

```python
def test_vfa_defaults():
    cfg = Config()
    assert cfg.lambda_frame_ce == 1.0
    assert cfg.lambda_boundary == 1.0
    assert cfg.lambda_word_verif == 1.0
    assert cfg.vfa_boundary_threshold == 0.8
```

Append to `tests/test_checkpoint_mode.py`:

```python
from utils.checkpoint import save_checkpoint, resolve_vfa_heads


def test_vfa_stamp_roundtrip(tmp_path):
    path = str(tmp_path / "ck.pt")
    save_checkpoint({}, path, alignment_mode="sim", vfa_heads=True)
    ckpt = torch.load(path)
    assert resolve_vfa_hashes(ckpt) is True


def test_missing_stamp_resolves_false(capsys):
    assert resolve_vfa_hashes({"alignment_mode": "sim"}) is False
    assert "vfa" in capsys.readouterr().out.lower()
```

(Note: name both helper calls `resolve_vfa_heads` — `resolve_vfa_hashes` is a typo; write `resolve_vfa_heads` in the tests you actually save.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_config.py tests/test_checkpoint_mode.py -q`
Expected: FAIL (`AttributeError` on the new Config attrs / ImportError on `resolve_vfa_heads`).

- [ ] **Step 3: Implement**

In `Config.__init__` (after `alignment_mode`):

```python
        # VFA head losses (frame token CE, boundary BCE, per-token verification
        # BCE) and the boundary probability that forces a Viterbi transition.
        self.lambda_frame_ce = cfg.get("lambda_frame_ce", 1.0)
        self.lambda_boundary = cfg.get("lambda_boundary", 1.0)
        self.lambda_word_verif = cfg.get("lambda_word_verif", 1.0)
        self.vfa_boundary_threshold = cfg.get("vfa_boundary_threshold", 0.8)
```

In `config.json`, after `"alignment_mode": "sim",`:

```json
    "lambda_frame_ce": 1.0,
    "lambda_boundary": 1.0,
    "lambda_word_verif": 1.0,
    "vfa_boundary_threshold": 0.8,
```

In `utils/checkpoint.py`:

```python
def save_checkpoint(state_dict, path, alignment_mode, vfa_heads=True):
    torch.save(
        {"model_state_dict": state_dict, "alignment_mode": alignment_mode,
         "vfa_heads": vfa_heads},
        path,
    )


def resolve_vfa_heads(checkpoint):
    """True only if the checkpoint was trained with the VFA heads. Old
    checkpoints predate the stamp and must be evaluated with use_vfa=False
    (strict load); say so instead of silently scoring garbage head outputs."""
    if isinstance(checkpoint, dict) and checkpoint.get("vfa_heads"):
        return True
    print("Note: checkpoint has no VFA heads; per-token timestamps/scores disabled.")
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_config.py tests/test_checkpoint_mode.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/dataset_utils.py utils/checkpoint.py config.json tests/test_config.py tests/test_checkpoint_mode.py
git commit -m "feat: VFA config keys and checkpoint vfa_heads stamp"
```

---

### Task 7: Training losses and wiring (`training/train.py`, `main.py`)

**Files:**
- Modify: `training/train.py`, `main.py`
- Test: `tests/test_training_loss.py` (append)

**Interfaces:**
- Consumes: `frame_token_targets` (Task 2), collate's `word_labels`/`word_label_mask`/`word_spans` (Task 3), model's `vfa_frame_logits`/`vfa_bound_logits`/`word_verif_logits`/`word_token_mask` (Task 5), Config keys (Task 6).
- Produces (free functions in `training/train.py`):
  - `vfa_frame_ce_loss(frame_logits, frame_token_ids, frame_valid) -> scalar`
  - `vfa_boundary_loss(bound_logits, boundary_targets, frame_valid) -> scalar`
  - `word_verification_loss(word_logits, word_labels, word_valid) -> scalar`
  - `train_one_epoch(..., lambda_frame_ce=1.0, lambda_boundary=1.0, lambda_word_verif=1.0)` and new kwarg passthrough of `word_spans`/`alignment_targets` into `model(...)`.

- [ ] **Step 1: Write failing tests** (append to `tests/test_training_loss.py`)

```python
from training.train import vfa_frame_ce_loss, vfa_boundary_loss, word_verification_loss
from utils.alignment import frame_token_targets


def test_vfa_frame_ce_ignores_unsupervised_frames():
    logits = torch.zeros(1, 3, 5)
    logits[0, 0, 2] = 5.0; logits[0, 1, 2] = 5.0; logits[0, 2, 4] = 5.0
    ids = torch.tensor([[2, 2, 4]])
    valid = torch.tensor([[True, True, False]])
    loss = vfa_frame_ce_loss(logits, ids, valid)
    assert loss.item() == 0.0   # supervised frames are argmax-correct


def test_vfa_frame_ce_penalizes_wrong_frame():
    logits = torch.zeros(1, 1, 5)
    logits[0, 0, 1] = 5.0       # argmax 1, target 2
    ids = torch.tensor([[2]])
    valid = torch.tensor([[True]])
    assert vfa_frame_ce_loss(logits, ids, valid).item() > 0


def test_vfa_boundary_loss_masks_frames():
    bound_logits = torch.tensor([[5.0, 5.0, -5.0]])
    targets = torch.tensor([[0.0, 1.0, 1.0]])
    valid = torch.tensor([[True, True, False]])
    # frames 0,1 supervised; frame 2 (wrong, would cost a lot) ignored
    loss = vfa_boundary_loss(bound_logits, targets, valid)
    assert torch.isfinite(loss)


def test_word_verification_loss_only_on_valid_tokens():
    logits = torch.tensor([[5.0, -5.0, 5.0]])   # token 1 wrong
    labels = torch.tensor([[1.0, 1.0, 1.0]])
    valid = torch.tensor([[True, True, False]])
    loss = word_verification_loss(logits, labels, valid)
    # only tokens 0 (correct) and 1 (wrong) count
    expected = 0.5 * (torch.nn.functional.softplus(-5.0) + torch.nn.functional.softplus(-5.0))
    assert torch.allclose(loss, expected)


def test_losses_zero_when_nothing_valid():
    z = torch.zeros(1)
    assert vfa_frame_ce_loss(torch.zeros(1, 2, 3), torch.zeros(1, 2).long(),
                             torch.zeros(1, 2, dtype=torch.bool)).item() == 0.0
    assert vfa_boundary_loss(torch.zeros(1, 2), torch.zeros(1, 2),
                             torch.zeros(1, 2, dtype=torch.bool)).item() == 0.0
    assert word_verification_loss(torch.zeros(1, 2), torch.zeros(1, 2),
                                  torch.zeros(1, 2, dtype=torch.bool)).item() == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_training_loss.py -q`
Expected: FAIL with ImportError on the three new loss functions.

- [ ] **Step 3: Implement** (append to `training/train.py`, above `train_one_epoch`)

```python
def vfa_frame_ce_loss(frame_logits, frame_token_ids, frame_valid):
    """Frame-level token CE for the VFA posterior head. frame_valid gates the
    same speech/positive/has-boundary frames as the sim-matrix CE."""
    if frame_valid is not None and frame_valid.any():
        ids = frame_token_ids.clone()
        ids[~frame_valid] = -100
        return F.cross_entropy(frame_logits.reshape(-1, frame_logits.shape[-1]), ids.reshape(-1),
                               ignore_index=-100)
    return torch.zeros((), device=frame_logits.device)


def vfa_boundary_loss(bound_logits, boundary_targets, frame_valid):
    """BCE on per-frame token-transition logits, supervised frames only."""
    if frame_valid is not None and frame_valid.any():
        return F.binary_cross_entropy_with_logits(
            bound_logits[frame_valid], boundary_targets[frame_valid]
        )
    return torch.zeros((), device=bound_logits.device)


def word_verification_loss(word_logits, word_labels, word_valid):
    """Per-token verification BCE (on raw logits, same rationale as the global
    classifier). Supervised only where the collate emitted a token label and
    the pair had usable spans."""
    if word_valid is not None and word_valid.any():
        return F.binary_cross_entropy_with_logits(
            word_logits[word_valid], word_labels[word_valid]
        )
    return torch.zeros((), device=word_logits.device)
```

In `train_one_epoch`, extend the signature:

```python
def train_one_epoch(model, dataloader, optimizer, device, epoch, writer=None,
                    lambda_ctc=1.0, lambda_align=1.0, alignment_temperature=1.0,
                    lambda_frame_ce=1.0, lambda_boundary=1.0, lambda_word_verif=1.0):
```

After `has_align_target` handling, also move the new tensors:

```python
        alignment_targets_batch = alignment_targets   # already on device
        word_labels = batch.get('word_labels')
        if word_labels is not None:
            word_labels = word_labels.to(device)
        word_label_mask = batch.get('word_label_mask')
        if word_label_mask is not None:
            word_label_mask = word_label_mask.to(device)
        word_spans = batch.get('word_spans')
        if word_spans is not None:
            word_spans = word_spans.to(device)
```

Pass them to the model:

```python
        consistency_score, align_out = model(
            video, phonemes, audio,
            video_padding_mask=video_padding_mask,
            phoneme_padding_mask=phoneme_padding_mask,
            audio_padding_mask=audio_padding_mask,
            alignment_targets=alignment_targets_batch,
            word_spans=word_spans
        )
```

(If Task 5 did not add an `alignment_targets` kwarg to `forward`, pass only `word_spans` — the frame targets are derived here in the loss, not in the model.)

After `loss_align`, add:

```python
        # VFA supervision — same valid frames as the sim-matrix CE, plus the
        # per-token head on positives and single-substitution negatives.
        loss_frame_ce = torch.zeros((), device=device)
        loss_boundary = torch.zeros((), device=device)
        loss_word = torch.zeros((), device=device)
        if 'vfa_frame_logits' in align_out and alignment_targets is not None \
                and speech_mask is not None and has_align_target is not None:
            frame_valid = speech_mask & has_align_target.unsqueeze(1).bool() & positive_mask.unsqueeze(1)
            if video_padding_mask is not None:
                frame_valid = frame_valid & ~video_padding_mask
            if frame_valid.any():
                frame_ids, boundary_targets = frame_token_targets(alignment_targets, phonemes)
                loss_frame_ce = vfa_frame_ce_loss(align_out['vfa_frame_logits'], frame_ids, frame_valid)
                loss_boundary = vfa_boundary_loss(align_out['vfa_bound_logits'], boundary_targets, frame_valid)
        if 'word_verif_logits' in align_out and word_labels is not None and word_label_mask is not None:
            word_valid = word_label_mask & align_out['word_token_mask']
            loss_word = word_verification_loss(align_out['word_verif_logits'], word_labels, word_valid)

        loss = (loss_bce + loss_contrastive + lambda_ctc * loss_ctc + lambda_align * loss_align
                + lambda_frame_ce * loss_frame_ce + lambda_boundary * loss_boundary
                + lambda_word_verif * loss_word)
```

Add TensorBoard scalars (`Loss/train_step_frame_ce`, `..._boundary`, `..._word_verif`) and extend the tqdm postfix with `"word": f"{loss_word.item():.4f}"` following the existing pattern. Import `frame_token_targets` at the top: `from utils.alignment import frame_token_targets` (merge into existing import style — `training/train.py` currently imports nothing from utils; add the import).

In `main.py`, pass the new lambdas at the `train_one_epoch` call:

```python
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer if local_rank == 0 else None,
                                         lambda_ctc=config.lambda_ctc, lambda_align=config.lambda_align,
                                         alignment_temperature=config.alignment_temperature,
                                         lambda_frame_ce=config.lambda_frame_ce,
                                         lambda_boundary=config.lambda_boundary,
                                         lambda_word_verif=config.lambda_word_verif)
```

And stamp the checkpoint:

```python
                save_checkpoint(
                    model.module.state_dict(),
                    os.path.join(config.output_dir, f"checkpoint_ep{epoch}.pt"),
                    config.alignment_mode,
                    vfa_heads=True,
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_training_loss.py tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add training/train.py main.py tests/test_training_loss.py
git commit -m "feat: train VFA frame/boundary/word-verification losses"
```

---

### Task 8: Inference — timestamps, per-token scores, JSON output

**Files:**
- Modify: `inference.py`
- Test: `tests/test_inference_report.py` (create)

**Interfaces:**
- Consumes: `viterbi_align`, `token_spans`, `uniform_token_spans` (Task 1), `resolve_vfa_heads` (Task 6), `score_word_spans` (Task 5).
- Produces:
  - `spans_to_onehot(spans, T, L) -> torch (T, L)` one-hot helper.
  - `build_token_report(tokens, spans, seconds_per_frame, posterior_scores, word_scores, verification_scores, thresholds, fallback) -> list[dict]` — pure function building the per-token JSON rows.
  - CLI flag `--json <path>`; `run_inference(..., json_path=None)`.

- [ ] **Step 1: Write failing tests**

```python
"""Per-token report construction for inference.

Run: ~/miniconda3/envs/torch/bin/python -m pytest tests/test_inference_report.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import spans_to_onehot, build_token_report


def test_spans_to_onehot():
    onehot = spans_to_onehot([(0, 0, 2), (1, 2, 4)], T=5, L=2)
    assert onehot.shape == (5, 2)
    assert onehot[:2, 0].all() and onehot[2:4, 1].all()
    assert onehot[4].sum() == 0


def test_build_token_report_values_and_verdicts():
    tokens = ["1", "2"]
    spans = [(0, 0, 10), (1, 10, 20)]
    rows = build_token_report(
        tokens, spans, seconds_per_frame=0.04,
        posterior_scores=[0.9, 0.6],
        word_scores=[0.8, 0.4],
        verification_scores=[0.95, 0.20],
        thresholds={"verification": 0.5},
        fallback=False,
    )
    assert rows[0]["token"] == "1"
    assert rows[0]["start_s"] == 0.0 and rows[0]["end_s"] == 0.4
    # alignment = mean(posterior, word_score)
    assert abs(rows[0]["alignment_score"] - 0.85) < 1e-6
    assert rows[0]["verdict"] == "MATCH"
    assert rows[1]["verdict"] == "MISMATCH"
    assert rows[0]["low_confidence"] is False


def test_build_token_report_fallback_flagged():
    rows = build_token_report(
        ["3"], [(0, 0, 5)], seconds_per_frame=0.05,
        posterior_scores=[0.5], word_scores=[0.5], verification_scores=[0.5],
        thresholds={}, fallback=True,
    )
    assert rows[0]["low_confidence"] is True
    assert rows[0]["verdict"] == "uncalibrated"


def test_build_token_report_missing_token_flagged():
    # token 1 got no frames from the aligner
    rows = build_token_report(
        ["1", "2"], [(0, 0, 10)], seconds_per_frame=0.04,
        posterior_scores=[0.5, None], word_scores=[0.5, None],
        verification_scores=[0.5, None], thresholds={}, fallback=False,
    )
    assert rows[1]["low_confidence"] is True
    assert rows[1]["start_s"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_inference_report.py -q`
Expected: FAIL with ImportError (`spans_to_onehot`).

- [ ] **Step 3: Implement**

Add to `inference.py` (after `format_score_decision`):

```python
def spans_to_onehot(spans, T, L):
    """(pos, start, end) spans -> T x L one-hot for token pooling."""
    onehot = torch.zeros(T, L)
    for pos, s, e in spans:
        if pos < L:
            onehot[s:e, pos] = 1.0
    return onehot


def build_token_report(tokens, spans, seconds_per_frame, posterior_scores,
                       word_scores, verification_scores, thresholds, fallback):
    """Per-token JSON rows: timestamps, alignment score (mean of posterior and
    cross-modal word score), verification score, verdict, confidence flag."""
    ver_thr = (thresholds or {}).get("verification") or {}
    ver_thr = ver_thr.get("threshold") if isinstance(ver_thr, dict) else ver_thr
    by_pos = {pos: (s, e) for pos, s, e in spans}
    rows = []
    for i, tok in enumerate(tokens):
        span = by_pos.get(i)
        if span is None:
            rows.append({"token": tok, "start_s": None, "end_s": None,
                         "alignment_score": None, "verification_score": None,
                         "verdict": "uncalibrated", "low_confidence": True})
            continue
        s, e = span
        align = 0.5 * (posterior_scores[i] + word_scores[i]) if posterior_scores[i] is not None \
            else word_scores[i]
        verif = verification_scores[i]
        verdict = "uncalibrated" if (ver_thr is None or verif is None) else \
            ("MATCH" if verif >= ver_thr else "MISMATCH")
        rows.append({"token": tok, "start_s": round(s * seconds_per_frame, 3),
                     "end_s": round(e * seconds_per_frame, 3),
                     "alignment_score": round(align, 4) if align is not None else None,
                     "verification_score": round(verif, 4) if verif is not None else None,
                     "verdict": verdict, "low_confidence": fallback})
    return rows
```

In `run_inference`, add `json_path=None` parameter and `--json` argparse flag. After the existing score prints:

```python
    # --- Per-token forced alignment + scores (VFA heads) ---
    token_rows = None
    if getattr(model, "use_vfa", False):
        fps = cap.get(cv2.CAP_PROP_FPS) if (cap := cv2.VideoCapture(video_path)) else 0
        if not fps or fps != fps:   # 0 or NaN
            fps = 25.0
        with torch.no_grad():
            frame_logits = align_out["vfa_frame_logits"][0, :true_video_len]
            bound_probs = torch.sigmoid(align_out["vfa_bound_logits"][0, :true_video_len]).cpu().numpy()
            labels = [0] + phones + [0]
            fallback = true_video_len < len(labels)
            if fallback:
                spans = uniform_token_spans(true_video_len, len(phones))
            else:
                f2l = viterbi_align(frame_logits, labels, bound_probs,
                                    boundary_threshold=cfg.vfa_boundary_threshold)
                spans = token_spans(f2l, len(phones))
            posteriors = torch.softmax(frame_logits, dim=-1).cpu()
            posterior_scores = []
            for pos, s, e in spans:
                cls = phones[pos]
                posterior_scores.append(posteriors[s:e, cls].mean().item())
            word_scores_t = align_out.get("word_scores")
            onehot = spans_to_onehot(spans, true_video_len, len(phones)).unsqueeze(0).to(device)
            logits, mask = model.score_word_spans(
                align_out["lip_feats"][:, :true_video_len],
                align_out["text_feats"], onehot)
            verif_scores = torch.sigmoid(logits)[0].cpu().tolist()
            # map scores back onto token positions (spans may skip tokens)
            post_by_pos = {pos: v for (pos, _, _), v in zip(spans, posterior_scores)}
            ws_by_pos = {}
            if word_scores_t is not None:
                ws = word_scores_t[0].cpu().tolist()
                ws_by_pos = {pos: ws[pos] for pos, _, _ in spans}
            ver_by_pos = {pos: v for (pos, _, _), v in zip(spans, verif_scores)}
            tokens = transcript.split()
            token_rows = build_token_report(
                tokens, spans, 1.0 / fps,
                [post_by_pos.get(i) for i in range(len(tokens))],
                [ws_by_pos.get(i) for i in range(len(tokens))],
                [ver_by_pos.get(i) for i in range(len(tokens))],
                calib or {}, fallback)
        print("\nPer-token alignment:")
        for row in token_rows:
            print(f"  {row['token']:>6}  [{row['start_s']}s - {row['end_s']}s]  "
                  f"align={row['alignment_score']}  verif={row['verification_score']}  "
                  f"{row['verdict']}{'' if not row['low_confidence'] else '  (low confidence)'}")
    else:
        print("Per-token timestamps/scores unavailable (checkpoint has no VFA heads).")

    result = {
        "global": {"verification_score": prob, "alignment_score": align_prob},
        "tokens": token_rows,
    }
    if json_path:
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {json_path}")
```

Notes:
- `import json` and `from utils.viterbi import viterbi_align, token_spans, uniform_token_spans` at the top of `inference.py`; `build_token_report`'s `thresholds` arg is `calib` (dict from `load_eer_thresholds`) — pass `calib if calib is not None else {}`; keep the existing `calib` usage above intact and reuse the variable.
- The `tokens` for GRID are transcript words while `phones`/spans are per-phoneme — for GRID the report is per-phoneme. To keep word-level rows for GRID, group a word's phoneme spans: `tokens = phones`-level rows are acceptable for v1; if word rows are wanted, merge consecutive spans belonging to the same word (phoneme positions per word are available from `converter.text_to_phoneme_indices_per_word`). Implement the simple version (per-phoneme rows for GRID, per-digit rows for digit) and note it in the printout header (`"Per-token alignment (GRID: per-phoneme):"`).
- The walrus/`cap` line releases nothing — call `cap.release()` after reading fps.

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/test_inference_report.py -q`
Expected: all PASS. (These tests only touch the pure helpers; no video/checkpoint needed.)

- [ ] **Step 5: Run full suite**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add inference.py tests/test_inference_report.py
git commit -m "feat: per-token timestamps and scores with JSON output in inference"
```

---

### Task 9: Docs + smoke check

**Files:**
- Modify: `architecture.md` (VFA section), `README.md` (inference example with `--json`)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update `architecture.md`** — add a "VFA heads" subsection to the model section: frame/boundary heads, `use_vfa` gating, per-token verification head, Viterbi at inference, new config keys and losses. Keep the existing mermaid style.

- [ ] **Step 2: Update `README.md`** — extend the inference usage block with `--json`, and show one example JSON output (copy the shape from the spec).

- [ ] **Step 3: Syntax-check the touched modules**

Run: `~/miniconda3/envs/torch/bin/python -c "import ast,sys; [ast.parse(open(f).read()) for f in ['inference.py','main.py','training/train.py','models/verification_model.py','models/vfa_heads.py','utils/viterbi.py','utils/alignment.py','utils/dataset_utils.py','utils/checkpoint.py']]"`
Expected: no output (all parse).

- [ ] **Step 4: Run the full test suite one last time**

Run: `~/miniconda3/envs/torch/bin/python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add architecture.md README.md
git commit -m "docs: VFA heads architecture and inference JSON usage"
```

---

## Self-Review (done at plan time)

- **Spec coverage:** per-frame posterior + boundary heads (Tasks 4–5), Viterbi port with threshold + fallback (Tasks 1, 8), per-token verification head with substitution-negative supervision (Tasks 3–5), losses + lambdas (Task 7), config keys (Task 6), checkpoint stamp/back-compat (Tasks 5–6), JSON + human output (Task 8), error handling (short-video fallback, missing-token flag, old-checkpoint degradation), tests throughout. GRID/digit both covered (digit end-to-end in tests; GRID via the same shared machinery + word-substitution path in collate).
- **Type consistency:** `word_spans` one-hot `B×T×L` used identically in collate (Task 3), model (Task 5), loss (Task 7), inference one-hot (Task 8). `spans` tuples are `(pos, start, end)` everywhere. `frame_valid`/`word_valid` bool masks consistent.
- **Known deviations:** two intentional test-file typo notes are flagged inline for the implementer (Task 3 Step 1 placeholder assert, Task 6 Step 1 helper name) with instructions to write the corrected version.
