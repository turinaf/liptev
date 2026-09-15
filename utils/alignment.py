"""
Frame-alignment target construction for lip-to-text verification.

Builds per-frame token targets from dataset-supplied boundary times:
- GRID: .align word boundaries (raw values in an arbitrary timebase, see
  calibrate_value_to_frame) distributed across per-word phonemes.
- Digits: .lab per-digit start/end seconds converted at the raw video's fps.

Targets are one-hot rows over speech frames (each speech frame is assigned to
exactly one token); frames outside every token range are masked out via the
speech mask.
"""

import torch


def calibrate_value_to_frame(value, max_end, video_len):
    """Map a raw .align boundary value to a frame index.

    GRID .align values are in an undocumented per-file timebase (measured:
    ~25 kHz audio samples for 3 s clips). The file spans the whole clip
    (starts at 0, ends at the video end), so a proportional mapping against
    the clip's true frame count is correct regardless of the unit.
    """
    if max_end <= 0:
        return 0
    frame = int(round(value * video_len / max_end))
    return max(0, min(frame, video_len))


def seconds_to_frame(seconds, fps):
    """Map a .lab timestamp (seconds, raw-video timeline) to a frame index.

    Processed videos keep every source frame (the re-encode only relabels
    them 25 fps), so the source fps is the correct conversion rate.
    """
    if fps <= 0:
        return 0
    return int(round(seconds * fps))


def split_frames_uniformly(start, end, n):
    """Contiguously split the frame range [start, end) into n parts.

    Parts may be empty when the range is shorter than n (a short word with
    many phonemes): those tokens simply receive no supervised frames.
    """
    start, end, n = int(start), int(end), int(n)
    if n <= 0:
        return []
    total = max(end - start, 0)
    bounds = [start + (total * i) // n for i in range(n + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(n)]


def expand_token_ranges(word_infos):
    """Flatten per-word token/frame ranges into per-token sequence ranges.

    word_infos: list of (token_ids, start_frame, end_frame) — one entry per
    word (GRID, token_ids = that word's phoneme ids) or per digit (single-id
    list). Returns list of (seq_index, start_frame, end_frame) where seq_index
    is the token's position in the concatenated target sequence.
    """
    ranges = []
    seq_index = 0
    for token_ids, start, end in word_infos:
        for i, (part_start, part_end) in enumerate(split_frames_uniformly(start, end, len(token_ids))):
            ranges.append((seq_index + i, part_start, part_end))
        seq_index += len(token_ids)
    return ranges


def build_alignment_targets(token_ranges, video_len, target_len):
    """Build a one-hot target matrix and speech mask from token frame ranges.

    token_ranges: list of (token_seq_index, start_frame, end_frame) as produced
    by expand_token_ranges. Ranges are clamped to [0, video_len).
    Returns (targets video_len x target_len, speech_mask video_len).
    """
    targets = torch.zeros(video_len, target_len)
    speech_mask = torch.zeros(video_len, dtype=torch.bool)

    for seq_index, start, end in token_ranges:
        if seq_index < 0 or seq_index >= target_len:
            continue
        start = max(0, min(int(start), video_len))
        end = max(start, min(int(end), video_len))
        if end > start:
            targets[start:end, seq_index] = 1.0
            speech_mask[start:end] = True

    return targets, speech_mask
