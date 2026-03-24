import torch
import cv2
import numpy as np
from models.verification_model import LipTextVerificationModel
from utils.phoneme_converter import PhonemeConverter
import argparse
import sys

def run_inference(video_path, transcript, model_path="output/best_lip_verification_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Model from {model_path} onto {device}...")
    
    # Needs to match architecture in main
    model = LipTextVerificationModel(vocab_size=50, feature_dim=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    converter = PhonemeConverter()
    print(f"Parsing transcript: '{transcript}'")
    phones = converter.text_to_phonemes(transcript)
    phones = phones[:30] + [0] * max(0, 30 - len(phones))
    phone_tensor = torch.tensor(phones).unsqueeze(0).to(device) # Add batch dimension

    print(f"Loading specific video frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Assuming crop matches your 96x96 preprocessing requirement
        frame = cv2.resize(frame, (96, 96))
        frames.append(frame)
        
    cap.release()
    if len(frames) == 0:
        print("Failed to load video frames.")
        sys.exit(1)

    frames = np.stack(frames)
    # Shape needed: (Batch, Channels, Time, Height, Width)
    video_tensor = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
    video_tensor = video_tensor.unsqueeze(0).to(device)

    print(f"Evaluating alignment...")
    with torch.no_grad():
        score = model(video_tensor, phone_tensor)

    # Output Confidence 
    prob = torch.sigmoid(score).item()
    result = "MATCH" if prob > 0.5 else "MISMATCH"
    
    print(f"=====================================")
    print(f"Result: {result}")
    print(f"Confidence (Probability): {prob:.4f}")
    print(f"=====================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lip-to-Text Inference Verification")
    parser.add_argument("--video", type=str, required=True, help="Path to cropped MP4 lip file")
    parser.add_argument("--text", type=str, required=True, help="Ground truth or hypothesized text")
    parser.add_argument("--model", type=str, default="output/best_lip_verification_model.pth", help="Checkpoint model path")
    
    args = parser.parse_args()
    run_inference(args.video, args.text, args.model)
