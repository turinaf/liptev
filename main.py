import torch
import os
import sys
import json
from torch.optim import Adam
from dataset.grid_loader import GRIDDataset
from utils.phoneme_converter import PhonemeConverter
from utils.negative_sampler import NegativeSampler
from models.verification_model import LipTextVerificationModel
from training.train import train_one_epoch
from evaluation.evaluate import evaluate_model
from torch.utils.data import DataLoader, random_split

class Config:
    def __init__(self, config_path="config.json"):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        self.processed_data_dir = cfg.get("processed_data_dir", "./data")
        self.use_audio = cfg.get("use_audio", False)
        self.batch_size = cfg.get("batch_size", 4)
        self.learning_rate = cfg.get("learning_rate", 1e-4)
        self.epochs = cfg.get("epochs", 5)
        self.max_frames = cfg.get("max_frames", 75)

def collate_fn(batch):
    converter = PhonemeConverter()
    all_trans = [item['transcript'] for item in batch]
    sampler = NegativeSampler(all_trans)
    
    videos = []
    phonemes = []
    labels = []
    transcripts = []
    audios = []
    
    use_audio = 'audio' in batch[0]
    
    for item in batch:
        vid = item['video']
        trans = item['transcript']
        if use_audio:
            aud = item['audio']
        
        # --- Positive Sample ---
        videos.append(vid)
        p_seq = converter.text_to_phonemes(trans)
        p_seq = p_seq[:30] + [0] * max(0, 30 - len(p_seq))
        phonemes.append(torch.tensor(p_seq))
        labels.append(1.0)
        transcripts.append(trans)
        if use_audio: audios.append(aud)
        
        # --- Negative Sample ---
        neg_trans = sampler.sample(trans)
        videos.append(vid)
        n_seq = converter.text_to_phonemes(neg_trans)
        n_seq = n_seq[:30] + [0] * max(0, 30 - len(n_seq))
        phonemes.append(torch.tensor(n_seq))
        labels.append(0.0)
        transcripts.append(neg_trans)
        # Note: audio remains the SAME for the negative text sample since text is what mismatches 
        if use_audio: audios.append(aud)
        
    ret = {
        'video': torch.stack(videos),
        'phonemes': torch.stack(phonemes),
        'label': torch.tensor(labels, dtype=torch.float32),
        'transcripts': transcripts
    }
    
    if use_audio:
        ret['audio'] = torch.stack(audios)
        
    return ret

def main():
    cfg = Config("config.json")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset_path = os.path.abspath(cfg.processed_data_dir)
    
    if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
        raise ValueError(f"Processed dataset directory ({dataset_path}) is missing. Please run pipeline/data_preparation.py first.")
        
    full_dataset = GRIDDataset(data_dir=dataset_path, split='train', max_frames=cfg.max_frames, use_audio=cfg.use_audio)
    
    if len(full_dataset) == 0:
        raise ValueError(f"No samples were found in {dataset_path}. Did you run preparation script?")
        
    print("Initializing model...")
    # NOTE: The current LipTextVerificationModel uses only video/text. If `cfg.use_audio` is True, 
    # you would need an audio encoder integrated into your forward pass later.
    model = LipTextVerificationModel(vocab_size=50, feature_dim=256).to(device)
    
    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples.")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    
    print("\nStarting training...")
    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch} | Train Loss: {loss:.4f}")
        
    print("\nEvaluating Model on Validation Set...")
    evaluate_model(model, val_loader, device)

if __name__ == "__main__":
    main()
