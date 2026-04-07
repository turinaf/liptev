from tqdm import tqdm
import torch
import torch.nn as nn

def train_one_epoch(model, dataloader, optimizer, device, epoch, writer=None):
    model.train()
    bce_loss_fn = nn.BCELoss()
    # triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)
    
    total_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} Training")
    for batch_idx, batch in pbar:
        video = batch['video'].to(device)
        phonemes = batch['phonemes'].to(device)
        labels = batch['label'].to(device) # 1 For match, 0 for negative samples
        
        if 'audio' in batch:
            audio = batch['audio'].to(device)
        else:
            audio = None
        
        optimizer.zero_grad()
        consistency_score, align_out = model(video, phonemes, audio)
        
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
        
        # Log to tensorboard step by step
        if writer is not None:
            global_step = (epoch - 1) * len(dataloader) + batch_idx
            writer.add_scalar('Loss/train_step_bce', loss_bce.item(), global_step)
            writer.add_scalar('Loss/train_step_contrastive', loss_contrastive.item(), global_step)
            writer.add_scalar('Loss/train_step_total', loss.item(), global_step)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
    return total_loss / len(dataloader)
