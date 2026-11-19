import cv2
import numpy as np
import pytest
from types import SimpleNamespace

from dataset.preprocessing import LIP_LANDMARKS
from dataset.recording import GRID_MOUTH_WIDTH_PX, _resample_indices, prepare_recording


class FakeDetector:
    """Lips centred in the frame, ``mouth_frac`` of its width wide; frames
    whose blue channel is 255 have no detectable face."""

    def __init__(self, mouth_frac):
        self.mouth_frac = mouth_frac

    def detect(self, image):
        frame = image.numpy_view()
        if frame[0, 0, 2] > 200:
            return SimpleNamespace(face_landmarks=[])
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        for i, idx in enumerate(LIP_LANDMARKS):
            points[idx] = SimpleNamespace(x=0.5 + self.mouth_frac * (i / (len(LIP_LANDMARKS) - 1) - 0.5),
                                          y=0.5 + 0.05 * (i % 2))
        return SimpleNamespace(face_landmarks=[points])


def write_video(path, frames, fps, size=(160, 120), missing=()):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), 60, np.uint8)
        if i in missing:
            frame[:] = (255, 0, 0)  # BGR blue -> RGB channel 2 saturated
        writer.write(frame)
    writer.release()
    return str(path)


def test_resampling_uses_timestamps_and_trim_window():
    variable = np.array([0.0, 0.03, 0.09, 0.1, 0.17, 0.2])
    assert _resample_indices(variable, 25.0).tolist() == [0, 1, 2, 3, 4, 5]
    times = np.arange(120) / 60.0
    assert len(_resample_indices(times, 25.0)) == 50
    assert _resample_indices(times, 25.0, start=1.0, end=1.4).tolist() == [60, 62, 65, 67, 70, 72, 74, 77, 79, 82, 84]


def test_grid_recording_is_resampled_scaled_and_truncated(tmp_path):
    video = write_video(tmp_path / "phone.mp4", 240, 60.0)  # 4 s at 60 fps
    crops, report = prepare_recording(video, FakeDetector(0.5), "grid", max_frames=75)
    assert len(crops) == 75 and crops[0].shape == (96, 96, 3)
    assert report["target_fps"] == 25.0 and report["resampled_frames"] == 100
    assert report["source"]["measured_fps"] == pytest.approx(60.0, rel=1e-3)
    assert report["mouth_width_px"] == pytest.approx(80, abs=1)
    assert report["scale"] == pytest.approx(GRID_MOUTH_WIDTH_PX[1] / report["mouth_width_px"], rel=1e-3)
    assert report["usable"] and any("first 75 frames" in w for w in report["warnings"])


def test_grid_scale_untouched_inside_training_range(tmp_path):
    video = write_video(tmp_path / "grid.mp4", 50, 25.0)
    _, report = prepare_recording(video, FakeDetector(0.25), "grid", max_frames=75)
    assert report["mouth_width_px"] == pytest.approx(40, abs=1) and report["scale"] == 1.0


def test_digit_uses_source_rate_without_rescaling(tmp_path):
    video = write_video(tmp_path / "iphone.mp4", 120, 60.0)
    crops, report = prepare_recording(video, FakeDetector(0.5), "digit", max_frames=125,
                                      digit_source_fps=29.87)
    assert report["target_fps"] == 29.87 and report["scale"] == 1.0
    assert len(crops) == report["frames"] == 60


def test_poor_detection_marks_input_unusable(tmp_path):
    video = write_video(tmp_path / "occluded.mp4", 50, 25.0, missing=set(range(10, 40)))
    crops, report = prepare_recording(video, FakeDetector(0.25), "grid", max_frames=75)
    assert len(crops) == 50 and report["detection_rate"] == pytest.approx(0.4)
    assert not report["usable"]
    assert crops[20] is crops[9]  # missed frames reuse the previous crop
