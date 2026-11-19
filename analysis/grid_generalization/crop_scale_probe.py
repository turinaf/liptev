"""Compare fixed and scaled lip-crop padding on resized GRID source videos."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from audit_grid_generalization import digest, read_transcript, video_tensor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, help="Override the configured manifest directory")
    parser.add_argument("--checkpoint", type=Path, help="Override the configured best checkpoint")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    sys.path.insert(0, str(args.repo))
    from dataset.preprocessing import crop_lip_region
    from models.verification_model import LipTextVerificationModel
    from utils.checkpoint import checkpoint_tokenizer

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    config = json.loads(args.config.read_text())
    if config["dataset"] != "grid" or config.get("use_audio", False):
        raise ValueError("This probe implements video-only GRID analysis")
    root = Path(config["grid_processed_data_dir"])
    split_dir = args.split_dir or root
    if not split_dir.is_absolute():
        split_dir = args.repo / split_dir
    raw_root = Path(config["grid_raw_data_dir"])
    checkpoint_path = args.checkpoint or Path(config["output_dir"]) / "best_lip_verification_model.pth"
    checkpoint_hash = digest(checkpoint_path)
    calibration = json.loads((checkpoint_path.parent / "eer_thresholds.json").read_text())
    if calibration.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Calibration hash does not match checkpoint")
    threshold = calibration["verification"]["threshold"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tokenizer = checkpoint_tokenizer(checkpoint, "grid")
    model = LipTextVerificationModel(
        len(tokenizer.vocab), lip_backbone_pretrained=False,
        alignment_mode=checkpoint["alignment_mode"],
        dropout=checkpoint.get("dropout", 0),
        use_vfa=checkpoint.get("vfa_heads", False)).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    landmarker_path = Path(config["mp_model_path"])
    if not landmarker_path.is_absolute():
        landmarker_path = args.repo / landmarker_path
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(landmarker_path)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    manifest = json.loads((split_dir / "test_split.json").read_text())
    selected, speakers = [], set()
    for path in manifest:
        p = Path(path)
        speaker = p.parent.parent.name
        if speaker not in speakers:
            speakers.add(speaker)
            selected.append(p)
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise ValueError(f"Only {len(selected)} distinct test speakers available")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with vision.FaceLandmarker.create_from_options(options) as detector, torch.inference_mode():
        for p in selected:
            speaker = p.parent.parent.name
            source = raw_root / speaker / f"{p.stem}.mpg"
            annotation = root / speaker / "align" / f"{p.stem}.align"
            transcript = read_transcript(annotation)
            tokens = torch.tensor([tokenizer.encode(transcript)])
            cap = cv2.VideoCapture(str(source))
            fps = cap.get(cv2.CAP_PROP_FPS)
            variants = {
                (scale, padding): []
                for scale in (1, 2, 4)
                for padding in sorted({12, 12 * scale})
            }
            missing = dict.fromkeys(variants, 0)
            frame_count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for scale in (1, 2, 4):
                    image = rgb if scale == 1 else cv2.resize(
                        rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale))
                    detection = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
                    for padding in sorted({12, 12 * scale}):
                        key = (scale, padding)
                        crop = crop_lip_region(image, detection, padding=padding, size=(96, 96))
                        if crop is not None:
                            variants[key].append(crop)
                        else:
                            missing[key] += 1
                            if variants[key]:
                                variants[key].append(variants[key][-1])
            cap.release()
            if frame_count == 0:
                raise RuntimeError(f"No decoded frames: {source}")
            row = {
                "clip": f"{speaker}/{p.name}", "transcript": transcript,
                "source_fps": fps, "source_frames": frame_count, "variants": {},
            }
            for (scale, padding), frames in variants.items():
                if not frames:
                    raise RuntimeError(f"No face detected: {source} at {scale}x")
                video, mask = video_tensor(np.stack(frames), config["max_frames"])
                score, out = model(
                    video, tokens, video_padding_mask=mask,
                    phoneme_padding_mask=tokens.eq(0))
                value = float(score.item())
                row["variants"][f"{scale}x_padding_{padding}"] = {
                    "verification_score": value, "match": value >= threshold,
                    "ctc_nll_per_token": float(out["ctc_nll_per_token"].item()),
                    "missing_detections": missing[(scale, padding)],
                }
            results.append(row)
            print(json.dumps(row), flush=True)

    report = {
        "method": (
            "Resize the same raw GRID clip 1x, 2x and 4x, rerun MediaPipe at each "
            "resolution, and compare the current fixed 12px padding with padding "
            "scaled by the same factor. All clips retain the same transcript."
        ),
        "limitations": (
            "Synthetic resizing of five in-domain test clips is a sensitivity "
            "experiment, not an external-speaker or real phone-video evaluation."
        ),
        "checkpoint_sha256": checkpoint_hash,
        "validation_verification_threshold": threshold,
        "rows": results,
    }
    (args.output / "crop_scale_control.json").write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for row in results:
        for index, scaled in enumerate((False, True)):
            scores = [
                row["variants"][f"{scale}x_padding_{12 * scale if scaled else 12}"]["verification_score"]
                for scale in (1, 2, 4)
            ]
            axes[index].plot((1, 2, 4), scores, marker="o", label=row["clip"].split("/")[0])
    for ax, title in zip(axes, ("Current fixed 12px padding", "Padding scaled with source resolution")):
        ax.axhline(threshold, color="black", linestyle="--", label="Validation threshold")
        ax.set_yscale("log")
        ax.set_xticks((1, 2, 4), ("1x", "2x", "4x"))
        ax.set_xlabel("Source-frame resizing factor")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Verification score (log scale)")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Same GRID videos and transcripts; MediaPipe redetected at each size")
    fig.tight_layout()
    fig.savefig(args.output / "crop_scale_sensitivity.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
