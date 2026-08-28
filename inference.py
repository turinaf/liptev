import torch
import cv2
import numpy as np
import os

os.environ['GLOG_minloglevel'] = '2' # Suppress mediapipe C++ logging

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataset.preprocessing import extract_lip_frames
from models.verification_model import LipTextVerificationModel
from utils.phoneme_converter import PhonemeConverter, DigitPinyinConverter
from utils.dataset_utils import Config
from utils.checkpoint import resolve_alignment_mode
from evaluation.metrics import load_eer_thresholds
import argparse
import sys
import glob
import re

EER_THRESHOLDS_FILENAME = "eer_thresholds.json"

def resolve_thresholds_path(requested_path, checkpoint_path):
    """Locate the EER calibration file: --thresholds wins, else look for
    eer_thresholds.json next to the resolved checkpoint (written by test.py)."""
    if requested_path:
        return requested_path
    if checkpoint_path:
        candidate = os.path.join(os.path.dirname(checkpoint_path), EER_THRESHOLDS_FILENAME)
        if os.path.exists(candidate):
            return candidate
    return None

def format_score_decision(label, score, block):
    """One report line: raw score plus the MATCH/MISMATCH call at the
    EER-optimal threshold. Without calibration the raw score has no
    meaningful 0.5 cut, so say so instead of pretending."""
    if not block or block.get("threshold") is None:
        return f"{label}: {score:.4f}  ->  uncalibrated (no EER threshold available)"
    decision = "MATCH" if score >= block["threshold"] else "MISMATCH"
    return (f"{label}: {score:.4f}  ->  {decision} "
            f"(EER thr {block['threshold']:.4f}, EER {block['eer']:.4f})")

def parse_input_text(text, dataset):
    if dataset == 'grid':
        words = []
        # Handle literal '\n' from command line
        lines = text.replace('\\n', '\n').strip().split('\n')
        if len(lines) > 1 or (len(lines) == 1 and len(lines[0].split()) == 3 and lines[0].split()[0].isdigit()):
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3:
                    start, end, word = parts
                    if word not in ['sil', 'sp']:
                        words.append(word)
            return " ".join(words)
        else:
            # Also filter 'sil' and 'sp' if plain text is provided
            return " ".join([w for w in text.split() if w not in ['sil', 'sp']])
    else:
        return text

def resolve_checkpoint_path(requested_model_path, config_output_dir):
    if requested_model_path and os.path.exists(requested_model_path):
        return requested_model_path

    preferred = [
        os.path.join(config_output_dir, "best_lip_verification_model.pth"),
        os.path.join(config_output_dir, "best_lip_verification_model.pt"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path

    patterns = [
        "checkpoint_ep*.pt",
        "checkpoint_ep*.pth",
        "checkpoint_epoch_*.pt",
        "checkpoint_epoch_*.pth",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(config_output_dir, pattern)))

    if not candidates:
        return None

    def extract_epoch(path):
        name = os.path.basename(path)
        m = re.search(r"(?:checkpoint_ep|checkpoint_epoch_)(\d+)", name)
        return int(m.group(1)) if m else -1

    candidates.sort(key=extract_epoch)
    return candidates[-1]


def run_inference(video_path, transcript, model_path=None, dataset=None, thresholds_path=None):
    cfg = Config("config.json")
    dataset = dataset or cfg.dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    proj_root = os.path.dirname(os.path.abspath(__file__))
    output_dir_abs = cfg.output_dir if os.path.isabs(cfg.output_dir) else os.path.abspath(os.path.join(proj_root, cfg.output_dir))
    resolved_model_path = resolve_checkpoint_path(model_path, output_dir_abs)

    if resolved_model_path is None:
        print(f"No checkpoint found. Checked requested path and output dir: {output_dir_abs}")
        sys.exit(1)

    print(f"Loading Model from {resolved_model_path} onto {device}...")
    
    transcript = parse_input_text(transcript, dataset)
    
    checkpoint = torch.load(resolved_model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    # The checkpoint's stamp wins over config: the classifier and the EER
    # calibration were produced under the checkpoint's scoring mode.
    alignment_mode = resolve_alignment_mode(checkpoint, cfg.alignment_mode)

    vocab_size = 41  # shared ARPABET vocab (39 + pad + unk) for both datasets
    # Needs to match architecture in main
    model = LipTextVerificationModel(
        vocab_size=vocab_size,
        feature_dim=256,
        lip_backbone_pretrained=cfg.lip_backbone_pretrained,
        alignment_mode=alignment_mode
    ).to(device)

    model.load_state_dict(clean_state_dict)
    model.eval()

    print(f"Parsing transcript: '{transcript}'")
    if dataset == "digit":
        digit_converter = DigitPinyinConverter()
        phones = digit_converter.flatten_per_token(
            digit_converter.transcript_to_phoneme_indices_per_token(transcript))
    else:
        converter = PhonemeConverter()
        # Per-word phonemization — the same source the model is trained on
        # (the legacy whole-sentence path emitted <unk> for every space).
        phones = converter.flatten_per_word(converter.text_to_phoneme_indices_per_word(transcript))
        
    if len(phones) == 0:
        phones = [0]

    # Keep a bounded sequence length to match training regime while retaining more transcript context.
    max_phone_len = 128
    phones = phones[:max_phone_len]
    phone_tensor = torch.tensor(phones, dtype=torch.long).unsqueeze(0).to(device) # Add batch dimension
    phoneme_padding_mask = phone_tensor.eq(0)

    print(f"Extracting lip-cropped frames from {video_path}...")
    mp_model_path = cfg.mp_model_path if os.path.isabs(cfg.mp_model_path) else os.path.join(proj_root, cfg.mp_model_path)
    if not os.path.exists(mp_model_path):
        print(f"MediaPipe model not found at {mp_model_path}.")
        sys.exit(1)
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=mp_model_path),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Same preprocessing as training (pipeline/data_preparation.py):
    # mediapipe lip crop (padding=12) resized to 96x96 RGB.
    correct_orientation = (dataset == "digit")
    frames = extract_lip_frames(video_path, detector=detector, correct_orientation=correct_orientation)

    if len(frames) == 0:
        print("Failed to load video frames.")
        sys.exit(1)

    max_frames = cfg.max_frames
    true_video_len = min(len(frames), max_frames)
    if len(frames) > max_frames:
        frames = frames[:max_frames]
    elif len(frames) < max_frames:
        last = frames[-1]
        frames.extend([last] * (max_frames - len(frames)))

    frames = np.stack(frames)
    # Models expect shape: (Batch, Time, Channels, Height, Width)
    # frames is (Time, Height, Width, Channels)
    video_tensor = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    # Apply the same normalization as train/test: T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    video_tensor = (video_tensor - 0.5) / 0.5
    video_tensor = video_tensor.unsqueeze(0).to(device)
    video_padding_mask = (torch.arange(max_frames, device=device).unsqueeze(0) >= true_video_len)

    print(f"Evaluating alignment...")
    with torch.no_grad():
        score, align_out = model(
            video_tensor,
            phone_tensor,
            audio_features=None,
            video_padding_mask=video_padding_mask,
            phoneme_padding_mask=phoneme_padding_mask
        )

    # Output Confidence
    prob = score.squeeze().item()
    align_prob = align_out["alignment_score"].squeeze().item()

    print(f"=====================================")
    calib_path = resolve_thresholds_path(thresholds_path, resolved_model_path)
    calib = load_eer_thresholds(calib_path)
    if calib is not None:
        calib_ckpt = calib.get("checkpoint")
        if calib_ckpt and os.path.basename(calib_ckpt) != os.path.basename(resolved_model_path):
            print(f"Warning: EER thresholds calibrated for '{calib_ckpt}' but loaded "
                  f"'{os.path.basename(resolved_model_path)}' - decisions may be off.")
        print(format_score_decision("Verification score", prob, calib.get("verification")))
        print(format_score_decision("Alignment score    ", align_prob, calib.get("alignment")))
        print(f"Decisions at EER-optimal thresholds from {calib_path}")
    else:
        # The classifier probability is not calibrated to a 0.5 cut (the
        # observed 0.000-for-everything); only a set-level EER threshold
        # makes these numbers decidable.
        print(f"Verification score (uncalibrated): {prob:.4f}")
        print(f"Alignment score (uncalibrated):    {align_prob:.4f}")
        print("No eer_thresholds.json found next to the checkpoint.")
        print("Run test.py to calibrate EER thresholds for single-pair decisions.")
    print(f"=====================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lip-to-Text Inference Verification")
    parser.add_argument("--video", type=str, required=True, help="Path to raw video file; the lip region is cropped automatically (MediaPipe)")
    parser.add_argument("--text", type=str, required=True, help="Ground truth or hypothesized text")
    parser.add_argument("--model", type=str, default=None, help="Optional checkpoint model path; if omitted, uses latest from config output_dir")
    parser.add_argument("--dataset", type=str, default=None, choices=["grid", "digit"], help="Optional dataset override; defaults to config dataset")
    parser.add_argument("--thresholds", type=str, default=None, help="Optional eer_thresholds.json path; if omitted, auto-discovered next to the checkpoint")

    args = parser.parse_args()
    run_inference(args.video, args.text, args.model, args.dataset, args.thresholds)
