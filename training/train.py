import torch
import torch.nn as nn

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    bce_loss_fn = nn.BCELoss()
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)
    
    total_loss = 0.0
    
    for batch in dataloader:
        video = batch['video'].to(device)
        phonemes = batch['phonemes'].to(device)
        labels = batch['label'].to(device) # 1 For match, 0 for negative samples
        
        optimizer.zero_grad()
        consistency_score, align_out = model(video, phonemes)
        
        # Binary Cross Entropy (maximize verification score)
        loss_bce = bce_loss_fn(consistency_score.view(-1), labels)
        
        # Contrastive Logic via Margin objective
        align_score = align_out['alignment_score']
        # Maximize match score, minimize mismatch score
        loss_contrastive = torch.mean(
            labels * (1.0 - align_score) + 
            (1.0 - labels) * torch.clamp(align_score - 0.5, min=0.0) # margin is implicit here
        )
        
        loss = loss_bce + loss_contrastive
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)