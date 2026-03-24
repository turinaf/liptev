import os
import sys
# Ensure module resolution across repo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json
import cv2
import shutil
import numpy as np
from tqdm import tqdm

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
    
    # Use mp4v codec for standard mp4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB back to BGR for OpenCV writing
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
        
    out.release()

def prepare_dataset(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    raw_dir = config["raw_data_dir"]
    out_dir = config["processed_data_dir"]
    sample_rate = config["audio_sample_rate"]
    max_frames = config["max_frames"]
    mp_model_path = config["mp_model_path"]
    
    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory '{raw_dir}' does not exist.")
        return
        
    print(f"Scanning '{raw_dir}' for GRID speakers...")
    
    # Iterate over speakers (e.g. s1, s2, s10...)
    speakers = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    for speaker in speakers:
        print(f"Processing speaker: {speaker}")
        speaker_raw_dir = os.path.join(raw_dir, speaker)
        speaker_out_dir = os.path.join(out_dir, speaker)
        
        # Paths according to new layout:
        # data/speaker_dir/video1.mpg
        # data/speaker_dir/align/video1.align
        raw_video_dir = speaker_raw_dir
        raw_align_dir = os.path.join(speaker_raw_dir, "align")
        
        # Target Preprocessed Layout
        out_video_dir = os.path.join(speaker_out_dir, "video")
        out_audio_dir = os.path.join(speaker_out_dir, "audio")
        out_align_dir = os.path.join(speaker_out_dir, "align")
        
        os.makedirs(out_video_dir, exist_ok=True)
        os.makedirs(out_audio_dir, exist_ok=True)
        os.makedirs(out_align_dir, exist_ok=True)
        
        # Get all videos directly in the speaker directory
        videos = glob.glob(os.path.join(raw_video_dir, "*.mpg"))
        print(f"  Found {len(videos)} videos.")
        
        for vid_path in tqdm(videos, desc=f"Preprocessing {speaker}"):
            base_name = os.path.basename(vid_path).replace('.mpg', '')
            
            # 1. Output Paths
            out_vid_path = os.path.join(out_video_dir, f"{base_name}.mp4")
            out_aud_path = os.path.join(out_audio_dir, f"{base_name}.wav")
            out_aln_path = os.path.join(out_align_dir, f"{base_name}.align")
            
            # Source align path
            src_aln_path = os.path.join(raw_align_dir, f"{base_name}.align")
            
            # Skip if align missing
            if not os.path.exists(src_aln_path):
                continue
                
            # 2. Extract and Save Cropped Lip Video
            if not os.path.exists(out_vid_path):
                cropped_frames = extract_lip_frames(vid_path, max_frames=max_frames, model_path=mp_model_path)
                save_video_from_frames(cropped_frames, out_vid_path, fps=25)
                
            # 3. Extract Audio
            if not os.path.exists(out_aud_path):
                extract_audio(vid_path, out_aud_path, sample_rate=sample_rate)
                
            # 4. Copy Align File
            if not os.path.exists(out_aln_path):
                shutil.copy2(src_aln_path, out_aln_path)

if __name__ == "__main__":
    prepare_dataset()
