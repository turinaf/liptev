import torch
import os
import sys
from torch.optim import Adam
from dataset.grid_loader import GRIDDataset
from utils.phoneme_converter import PhonemeConverter
from utils.negative_sampler import NegativeSampler
from models.verification_model import LipTextVerificationModel
from training.train import train_one_epoch
from evaluation.evaluate import evaluate_model
from torch.utils.data import DataLoader, random_split

def collate_fn(batch):
    converter = PhonemeConverter()
    # Gather all transcripts in the batch to serve as initial negatives
    all_trans = [item['transcript'] for item in batch]
    sampler = NegativeSampler(all_trans)
    
    videos = []
    phonemes = []
    labels = []
    transcripts = []
    
    for item in batch:
        vid = item['video']
        trans = item['transcript']
        
        # --- Positive Sample ---
        videos.append(vid)
        p_seq = converter.text_to_phonemes(trans)
        p_seq = p_seq[:30] + [0] * max(0, 30 - len(p_seq)) # Padding changed to 30
        phonemes.append(torch.tensor(p_seq))
        labels.append(1.0)
        transcripts.append(trans)
        
        # --- Negative Sample ---
        neg_trans = sampler.sample(trans)
        videos.append(vid)
        n_seq = converter.text_to_phonemes(neg_trans)
        n_seq = n_seq[:30] + [0] * max(0, 30 - len(n_seq))
        phonemes.append(torch.tensor(n_seq))
        labels.append(0.0)
        transcripts.append(neg_trans)
        
    return {
        'video': torch.stack(videos),
        'phonemes': torch.stack(phonemes),
        'label': torch.tensor(labels, dtype=torch.float32),
        'transcripts': transcripts
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Using the real dataset path containing GRID speaker folders
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    
    if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
        raise ValueError(f"Dataset directory ({dataset_path}) is missing or empty. Please supply the true GRID dataset.")
        
    full_dataset = GRIDDataset(data_dir=dataset_path)
    
    if len(full_dataset) == 0:
        raise ValueError(f"No actual video/alignment samples were found in {dataset_path}.")
        
    print("Initializing model...")
    model = LipTextVerificationModel(vocab_size=50, feature_dim=256).to(device)
    
    # Split Dataset into Train / Val
    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples.")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    print("\nStarting training...")
    for epoch in range(1, 6):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch} | Train Loss: {loss:.4f}")
        
    print("\nEvaluating Model on Validation Set...")
    evaluate_model(model, val_loader, device)
    
    print("\nInference Output Demo:")
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        vids = batch['video'].to(device)
        phons = batch['phonemes'].to(device)
        trans = batch['transcripts']
        lbls = batch['label'].tolist()
        
        cons_scores, align_outs = model(vids, phons)
        for i in range(len(vids)):
            transcript = trans[i]
            align_score = align_outs['alignment_score'][i].item()
            final_score = cons_scores[i].item()
            decision = "Match" if final_score >= 0.5 else "Mismatch"
            
            print(f"Transcript: {transcript}")
            print(f"Ground Truth Label: {lbls[i]}")
            print(f"Alignment Score: {align_score:.4f}")
            print(f"Final Consistency Score: {final_score:.4f}")
            print(f"Decision: {decision}")
            print("-" * 30)

if __name__ == "__main__":
    main()
