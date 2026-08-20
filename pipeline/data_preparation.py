import os
os.environ['GLOG_minloglevel'] = '2' # Suppress mediapipe C++ logging

import sys
# Ensure module resolution across repo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json
import cv2
import shutil
import numpy as np
import random
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from dataset.preprocessing import extract_lip_frames
from dataset.audio_extraction import extract_audio

def save_video_from_frames(frames, output_path, fps=25):
    """
    Saves a list of numpy RGB frames to a video file.
    """
    if not frames:
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    height, width, _ = frames[0].shape
    
    # Use standard fourcc method
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB back to BGR for OpenCV writing
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
        
    out.release()

def prepare_dataset(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    dataset = config.get("dataset", "grid")
    
    if dataset == "grid":
        raw_dir = config["grid_raw_data_dir"]
        out_dir = config["grid_processed_data_dir"]
    else:
        raw_dir = config.get("digit_raw_data_dir")
        out_dir = config.get("digit_processed_data_dir")
        
    sample_rate = config["audio_sample_rate"]
    mp_model_path = config["mp_model_path"]
    
    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory '{raw_dir}' does not exist.")
        return
        
    # Initialize MediaPipe Face Landmarker ONCE
    print(f"Initializing MediaPipe Face Landmarker...")
    base_options = python.BaseOptions(model_asset_path=mp_model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
        
    print(f"Scanning '{raw_dir}' for {dataset.upper()} speakers...")
    
    if dataset == "grid":
        speakers = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
        skip_speakers = ["s14_processed", "s9_processed", "s27_processed", "s22_processed", 
                         "s1_processed", " s29_processed", "s18_processed"]
    else:
        # digit dataset structure: root_dir (data/digit_raw) -> subset_X -> speaker
        speakers = []
        for subset in os.listdir(raw_dir):
            if subset.startswith("subset") and os.path.isdir(os.path.join(raw_dir, subset)):
                subset_dir = os.path.join(raw_dir, subset)
                for spk in os.listdir(subset_dir):
                    if os.path.isdir(os.path.join(subset_dir, spk)):
                        speakers.append(os.path.join(subset, spk))
        skip_speakers = []
    
    for speaker in speakers:
        if speaker in skip_speakers:
            print(f"Skipping speaker {speaker} (already processed completely).")
            continue
            
        print(f"Processing speaker: {speaker}")
        speaker_raw_dir = os.path.join(raw_dir, speaker)
        if dataset == "grid":
            speaker_out_dir = os.path.join(out_dir, speaker)
            raw_video_dir = speaker_raw_dir
            raw_align_dir = os.path.join(speaker_raw_dir, "align")
            videos = glob.glob(os.path.join(raw_video_dir, "*.mpg"))
            out_align_dir_name = "align"
            align_ext = ".align"
        else:
            # digit dataset

            speaker_out_dir = os.path.join(out_dir, speaker)
            raw_video_dir = os.path.join(speaker_raw_dir, "video")
            raw_align_dir = os.path.join(speaker_raw_dir, "text")
            raw_lab_dir = os.path.join(speaker_raw_dir, "lab")
            videos = glob.glob(os.path.join(raw_video_dir, "*.mp4"))
            out_align_dir_name = "text"
            align_ext = ".txt"

        out_video_dir = os.path.join(speaker_out_dir, "video")
        out_audio_dir = os.path.join(speaker_out_dir, "audio")
        out_align_dir = os.path.join(speaker_out_dir, out_align_dir_name)
        # Per-video source metadata (fps) for datasets whose raw timeline
        # differs from the 25 fps re-encode; also receives copied .lab files.
        out_lab_dir = os.path.join(speaker_out_dir, "lab") if dataset == "digit" else None
        meta_path = os.path.join(speaker_out_dir, "meta.json") if dataset == "digit" else None

        os.makedirs(out_video_dir, exist_ok=True)
        os.makedirs(out_audio_dir, exist_ok=True)
        os.makedirs(out_align_dir, exist_ok=True)
        if out_lab_dir is not None:
            os.makedirs(out_lab_dir, exist_ok=True)

        # {base_name: {"source_fps": float}} — consumed by DigitDataset to
        # convert .lab seconds to frame indices.
        video_meta = {}
        if meta_path is not None and os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    video_meta = json.load(f)
            except (OSError, ValueError):
                video_meta = {}
        
        print(f"  Found {len(videos)} videos.")
        
        for vid_path in tqdm(videos, desc=f"Preprocessing {speaker}"):
            base_name = os.path.splitext(os.path.basename(vid_path))[0]

            out_vid_path = os.path.join(out_video_dir, f"{base_name}.mp4")
            out_aud_path = os.path.join(out_audio_dir, f"{base_name}.wav")
            out_aln_path = os.path.join(out_align_dir, f"{base_name}{align_ext}")

            src_aln_path = os.path.join(raw_align_dir, f"{base_name}{align_ext}")
            src_lab_path = os.path.join(raw_lab_dir, f"{base_name}.lab") if dataset == "digit" else None

            if not os.path.exists(src_aln_path):
                continue

            # Record the raw video's fps before any processing (also on the
            # already-processed skip path, so existing data gets backfilled):
            # the 25 fps re-encode preserves frame indices but relabels the
            # timebase, so .lab seconds must be converted at the source rate.
            # Measured rates vary per recording device (~29.9 fps VFR, 25 fps).
            if meta_path is not None and base_name not in video_meta:
                probe = cv2.VideoCapture(vid_path)
                fps = probe.get(cv2.CAP_PROP_FPS)
                probe.release()
                video_meta[base_name] = {"source_fps": fps if fps and fps > 0 else 25.0}
                with open(meta_path, "w") as f:
                    json.dump(video_meta, f)

            # Boundary-supervision files for the digit dataset (kept on the
            # skip path as well — see above).
            if src_lab_path is not None and os.path.exists(src_lab_path):
                out_lab_path = os.path.join(out_lab_dir, f"{base_name}.lab")
                if not os.path.exists(out_lab_path):
                    shutil.copy2(src_lab_path, out_lab_path)

            # Skip video explicitly if it's already completely extracted
            if os.path.exists(out_vid_path) and os.path.exists(out_aln_path):
                continue

            if not os.path.exists(out_vid_path):
                correct_orientation = (dataset == "digit")
                cropped_frames = extract_lip_frames(vid_path, detector=detector, correct_orientation=correct_orientation)
                if cropped_frames:
                    save_video_from_frames(cropped_frames, out_vid_path, fps=25)
                else:
                    continue

            if not os.path.exists(out_aud_path) and config.get("use_audio", False):
                extract_audio(vid_path, out_aud_path, sample_rate=sample_rate)

            if not os.path.exists(out_aln_path):
                shutil.copy2(src_aln_path, out_aln_path)

    print("\nCreating Train/Val/Test splits...")
    all_videos = glob.glob(os.path.join(out_dir, '*', 'video', '*.mp4'))
    
    random.seed(42) 
    random.shuffle(all_videos)
    
    total_size = len(all_videos)
    test_size = max(1, int(total_size * 0.1))
    val_size = max(1, int(total_size * 0.1))
    train_size = total_size - val_size - test_size
    
    train_videos = all_videos[:train_size]
    val_videos = all_videos[train_size:train_size + val_size]
    test_videos = all_videos[train_size + val_size:]
    
    with open(os.path.join(out_dir, "train_split.json"), "w") as f:
        json.dump(train_videos, f)
    with open(os.path.join(out_dir, "val_split.json"), "w") as f:
        json.dump(val_videos, f)
    with open(os.path.join(out_dir, "test_split.json"), "w") as f:
        json.dump(test_videos, f)
        
    print(f"Data split saved in {out_dir}:")
    print(f" - Train: {len(train_videos)}")
    print(f" - Val:   {len(val_videos)}")
    print(f" - Test:  {len(test_videos)}")

if __name__ == "__main__":
    prepare_dataset()
