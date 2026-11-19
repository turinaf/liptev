"""Normalize a newly recorded video toward the training data before lip cropping.

Steps: timestamp-based resampling to the dataset's frame rate, upright
orientation, face-scale normalization (GRID only), optional trimming, and
input-quality checks. The lip crop itself is the training crop
(``crop_lip_region``: fixed 12 px padding, 96x96).
"""

import cv2
import mediapipe as mp
import numpy as np

from dataset.preprocessing import LIP_LANDMARKS, crop_lip_region

# Mouth width (px) of lip landmarks in raw GRID training videos (360x288),
# measured on 299 speaker-disjoint train clips: 5th percentile, median, 95th.
# Because the crop padding is a fixed 12 px, a recording whose mouth width is
# outside this range produces a crop scale the GRID model never saw.
GRID_MOUTH_WIDTH_PX = (29.3, 37.5, 46.0)

MIN_DETECTION_RATE = 0.8
MIN_DURATION_S = 1.0
SCALE_PROBE_FRAMES = 8

_ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
}


def dataset_profile(dataset, digit_source_fps=29.87):
    """Per-dataset targets that match how each training set was prepared.

    GRID: studio clips at 25 fps and a fixed 360x288 framing, so recordings are
    resampled to 25 fps and their face scale normalized to the GRID range.
    Digit: smartphone clips (iPhone/Android, variable resolution and frame
    rate) whose frames are kept at the native rate (~29.9 fps, sometimes 25)
    and cropped at native resolution, so recordings are resampled to the
    configured digit source rate and their scale is left untouched.
    """
    if dataset == "grid":
        return {"target_fps": 25.0, "mouth_width_px": GRID_MOUTH_WIDTH_PX}
    if dataset == "digit":
        return {"target_fps": float(digit_source_fps), "mouth_width_px": None}
    raise ValueError(f"Unknown dataset: {dataset}")


def _rotate(frame, degrees):
    code = _ROTATIONS[degrees]
    return frame if code is None else cv2.rotate(frame, code)


def _detect(detector, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return rgb, detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))


def _mouth_width(result, width):
    landmarks = result.face_landmarks[0]
    xs = [landmarks[i].x * width for i in LIP_LANDMARKS]
    return max(xs) - min(xs)


def _read_timestamps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    container_fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    times = []
    while cap.grab():
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    cap.release()
    return times, container_fps, (width, height)


def _source_times(times, container_fps, source_fps_override):
    """Per-frame source times; falls back to a uniform timeline when the
    container timestamps are missing or non-monotonic."""
    if source_fps_override:
        return np.arange(len(times)) / float(source_fps_override), "override"
    t = np.asarray(times, dtype=np.float64)
    if len(t) > 1 and np.all(np.diff(t) > 0):
        return t - t[0], "timestamps"
    fps = container_fps if container_fps and container_fps > 0 else 25.0
    return np.arange(len(times)) / fps, "container fps"


def _resample_indices(src_times, target_fps, start=None, end=None):
    """Nearest source frame for each target time k / target_fps."""
    if len(src_times) == 0:
        return np.zeros(0, dtype=int)
    first = src_times[0] if start is None else max(start, src_times[0])
    last = src_times[-1] if end is None else min(end, src_times[-1])
    if last < first:
        return np.zeros(0, dtype=int)
    targets = first + np.arange(int(np.floor((last - first) * target_fps + 1e-6)) + 1) / target_fps
    right = np.clip(np.searchsorted(src_times, targets), 1, len(src_times) - 1)
    left = right - 1
    pick_right = np.abs(src_times[right] - targets) < np.abs(targets - src_times[left])
    return np.where(pick_right, right, left) if len(src_times) > 1 else np.zeros(len(targets), dtype=int)


def _probe_orientation_and_scale(video_path, indices, detector, mouth_range):
    """Pick the rotation that exposes a face on sampled frames and the scale
    that brings the mouth width into the training range."""
    probe = np.unique(indices[np.linspace(0, len(indices) - 1,
                                          min(SCALE_PROBE_FRAMES, len(indices))).astype(int)])
    wanted = set(probe.tolist())
    cap = cv2.VideoCapture(video_path)
    votes, widths = {}, []
    idx = 0
    while wanted and cap.grab():
        if idx in wanted:
            wanted.discard(idx)
            ok, frame = cap.retrieve()
            if ok:
                for degrees in (0, 90, 270, 180):
                    rotated = _rotate(frame, degrees)
                    _, result = _detect(detector, rotated)
                    if result.face_landmarks:
                        votes[degrees] = votes.get(degrees, 0) + 1
                        widths.append((degrees, _mouth_width(result, rotated.shape[1])))
                        break
        idx += 1
    cap.release()
    rotation = max(votes, key=lambda d: (votes[d], d == 0)) if votes else 0
    mouth = [w for d, w in widths if d == rotation]
    mouth_width = float(np.median(mouth)) if mouth else None
    scale = 1.0
    if mouth_range is not None and mouth_width:
        low, target, high = mouth_range
        if not low <= mouth_width <= high:
            scale = target / mouth_width
    return rotation, mouth_width, scale


def prepare_recording(video_path, detector, dataset, max_frames, digit_source_fps=29.87,
                      target_fps=None, start=None, end=None, source_fps_override=None):
    """Return (lip_crops, report) for a raw recording.

    lip_crops are 96x96 RGB arrays at the dataset frame rate, at most
    ``max_frames`` long. ``report`` describes the source, the applied
    normalization and whether the input is usable for a decision.
    """
    profile = dataset_profile(dataset, digit_source_fps)
    target_fps = float(target_fps or profile["target_fps"])
    times, container_fps, (width, height) = _read_timestamps(video_path)
    src_times, timeline = _source_times(times, container_fps, source_fps_override)
    indices = _resample_indices(src_times, target_fps, start, end)

    report = {
        "source": {
            "resolution": [width, height],
            "frames": len(times),
            "container_fps": round(container_fps, 3) if container_fps else None,
            "measured_fps": (round((len(src_times) - 1) / (src_times[-1] - src_times[0]), 3)
                             if len(src_times) > 1 and src_times[-1] > src_times[0] else None),
            "variable_frame_rate": bool(len(src_times) > 2 and
                                        np.std(np.diff(src_times)) > 0.1 * np.mean(np.diff(src_times))),
            "duration_s": round(float(src_times[-1] - src_times[0]), 3) if len(src_times) else 0.0,
            "timeline": timeline,
        },
        "target_fps": target_fps,
        "trim_s": [start, end],
        "warnings": [],
    }
    if len(indices) == 0:
        report.update(usable=False, frames=0)
        report["warnings"].append("No frames inside the requested time range.")
        return [], report

    rotation, mouth_width, scale = _probe_orientation_and_scale(
        video_path, indices, detector, profile["mouth_width_px"])
    report.update(rotation_degrees=rotation,
                  mouth_width_px=round(mouth_width, 1) if mouth_width else None,
                  scale=round(scale, 4))

    total = len(indices)
    kept = indices[:max_frames]
    needed = {}
    for position, src in enumerate(kept):
        needed.setdefault(int(src), []).append(position)

    crops = [None] * len(kept)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    last_needed = max(needed)
    while idx <= last_needed and cap.grab():
        if idx in needed:
            ok, frame = cap.retrieve()
            crop = None
            if ok:
                frame = _rotate(frame, rotation)
                if scale != 1.0:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                rgb, result = _detect(detector, frame)
                crop = crop_lip_region(rgb, result, padding=12, size=(96, 96))
            for position in needed[idx]:
                crops[position] = crop
        idx += 1
    cap.release()

    detected_positions = sum(c is not None for c in crops)
    if detected_positions == 0:
        report.update(usable=False, frames=0, detection_rate=0.0)
        report["warnings"].append("No face/lips detected in any frame.")
        return [], report
    # Missed frames reuse the nearest earlier crop (leading misses the first
    # detected crop), keeping the timeline intact as in training's fallback.
    first = next(c for c in crops if c is not None)
    previous = first
    for i, crop in enumerate(crops):
        if crop is None:
            crops[i] = previous
        else:
            previous = crop

    detection_rate = detected_positions / len(crops)
    kept_s = len(kept) / target_fps
    report.update(frames=len(kept), resampled_frames=total,
                  kept_duration_s=round(kept_s, 3),
                  detection_rate=round(detection_rate, 3))
    if total > max_frames:
        report["warnings"].append(
            f"Recording is {total / target_fps:.2f}s at {target_fps:g} fps; only the first "
            f"{max_frames} frames ({kept_s:.2f}s) are scored. Trim to the spoken phrase (--start/--end).")
    if kept_s < MIN_DURATION_S:
        report["warnings"].append(f"Only {kept_s:.2f}s of video; the phrase is likely incomplete.")
    if detection_rate < MIN_DETECTION_RATE:
        report["warnings"].append(
            f"Lips detected in {detection_rate:.0%} of frames (< {MIN_DETECTION_RATE:.0%}); "
            "face/lips are occluded, out of frame or poorly lit.")
    if mouth_width is not None and mouth_width * scale < 15:
        report["warnings"].append(
            f"Mouth is only {mouth_width * scale:.0f}px wide; move closer or record at higher resolution.")
    report["usable"] = detection_rate >= MIN_DETECTION_RATE and kept_s >= MIN_DURATION_S
    return crops, report
