import torch
import cv2
import numpy as np
from models.verification_model import LipTextVerificationModel
from utils.phoneme_converter import PhonemeConverter
import argparse
import sys

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

def run_inference(video_path, transcript, model_path="output/best_lip_verification_model.pth", dataset="grid"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Model from {model_path} onto {device}...")
    
    transcript = parse_input_text(transcript, dataset)
    
    vocab_size = 13 if dataset == "digit" else 41
    # Needs to match architecture in main
    model = LipTextVerificationModel(vocab_size=vocab_size, feature_dim=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Parsing transcript: '{transcript}'")
    if dataset == "digit":
        digit_vocab = {str(i): i+1 for i in range(10)} 
        digit_vocab["!"] = 11
        digit_vocab["<pad>"] = 0
        digit_vocab["<unk>"] = 12
        phones = [digit_vocab.get(d, digit_vocab["<unk>"]) for d in transcript.split()]
    else:
        converter = PhonemeConverter()
        _, phones = converter.text_to_phonemes(transcript)
        
    phones = phones[:30] + [0] * max(0, 30 - len(phones))
    phone_tensor = torch.tensor(phones).unsqueeze(0).to(device) # Add batch dimension

    print(f"Loading specific video frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Assuming crop matches your 96x96 preprocessing requirement
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (96, 96))
        frames.append(frame)
        
    cap.release()
    if len(frames) == 0:
        print("Failed to load video frames.")
        sys.exit(1)

    frames = np.stack(frames)
    # Models expect shape: (Batch, Time, Channels, Height, Width)
    # frames is (Time, Height, Width, Channels)
    video_tensor = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    # Apply the same normalization as train/test: T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    video_tensor = (video_tensor - 0.5) / 0.5
    video_tensor = video_tensor.unsqueeze(0).to(device)

    print(f"Evaluating alignment...")
    with torch.no_grad():
        score, align_out = model(video_tensor, phone_tensor)

    # Output Confidence 
    prob = torch.sigmoid(score).item()
    # result = "MATCH" if prob > 0.5 else "MISMATCH"
    
    print(f"=====================================")
    # print(f"Result: {result}")
    print(f"Alignment score (Probability): {prob:.4f}")
    print(f"=====================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lip-to-Text Inference Verification")
    parser.add_argument("--video", type=str, required=True, help="Path to cropped MP4 lip file")
    parser.add_argument("--text", type=str, required=True, help="Ground truth or hypothesized text")
    parser.add_argument("--model", type=str, default="output/best_lip_verification_model.pth", help="Checkpoint model path")
    parser.add_argument("--dataset", type=str, default="grid", choices=["grid", "digit"], help="Dataset to use (grid or digit)")
    
    args = parser.parse_args()
    run_inference(args.video, args.text, args.model, args.dataset)
