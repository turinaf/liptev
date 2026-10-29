"""Monotonic Viterbi forced aligner over per-frame token posteriors.

Port of Visual-Forced-Alignment/scripts/viterbi_align.py: the known label
sequence (silence-framed) must be consumed in order, each label holding at
least one frame; emission scores are per-frame softmax posteriors. A boundary
probability above `boundary_threshold` at a frame forces the label advance
there (the original's >0.8 rule). liptev boundary targets fire only where one
supervised token column hands over to the next, so the forced advance must be
a token-to-token transition; a boundary peak cannot be dodged by (re)starting
the first token there, which the original's per-state rule allowed. Falls back
to a uniform split when the video is shorter than the label sequence (DP
infeasible).
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
            if forced:
                # A boundary pins a token-to-token transition to this frame:
                # the advance is mandatory and must start a non-silence label
                # whose predecessor was also non-silence. States that cannot
                # make such a transition are unreachable here.
                if (c >= 2 and labels[c - 2] != 0 and labels[c - 1] != 0
                        and advance > NEG_INF):
                    back[t, c] = 1
                    dp[t, c] = advance + log_p[t - 1, labels[c - 1]]
            else:
                back[t, c] = 0 if stay >= advance else 1
                best = stay if back[t, c] == 0 else advance
                if best > NEG_INF:
                    dp[t, c] = best + log_p[t - 1, labels[c - 1]]

    # A boundary peak that cannot be honoured (e.g. a misfire inside leading
    # silence leaves no room for the required token-to-token transition)
    # leaves the DP infeasible; retry without boundary hints instead of
    # backtracking through unreachable states.
    if dp[T, C] == NEG_INF and bound_probs is not None:
        return viterbi_align(logits, labels,
                             boundary_threshold=boundary_threshold)

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


def token_spans(frame_to_label, labels):
    """Group consecutive frames into (token_position, start, end) spans.

    `frame_to_label` holds class ids from a monotonic alignment; `labels` is
    the forced-order label sequence that produced it (typically
    [0] + token_ids + [0], class 0 = silence). Positions are per occurrence:
    the entry at labels[i] gets position i - 1, so repeated tokens keep one
    span each. Runs and label entries are paired in order; a run covering k
    consecutive identical entries is split uniformly across them, and a run
    shorter than k covers as far as possible with trailing occurrences
    omitted.
    """
    labels = list(labels)

    # Maximal runs of one non-zero class id in the alignment.
    runs = []
    current_label, start = None, None
    for t, label in enumerate(list(frame_to_label) + [0]):
        if label != current_label:
            if current_label not in (None, 0):
                runs.append((current_label, start, t))
            current_label, start = label, t

    spans = []
    ri = 0
    i = 0
    while ri < len(runs) and i < len(labels):
        if labels[i] == 0:
            i += 1
            continue
        # Consecutive identical entries are covered by one run.
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        k = j - i
        cls, s, e = runs[ri]
        ri += 1
        length = e - s
        if length <= k:
            # Short run: 1 frame per occurrence as far as possible; the
            # trailing occurrences get no span.
            for off in range(length):
                spans.append((i + off - 1, s + off, s + off + 1))
        else:
            base, extra = divmod(length, k)
            cursor = s
            for off in range(k):
                end = cursor + base + (1 if off < extra else 0)
                spans.append((i + off - 1, cursor, end))
                cursor = end
        i = j
    return spans
