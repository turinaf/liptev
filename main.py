import torch
import os
import json
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from dataset.grid_loader import GRIDDataset
from utils.phoneme_converter import PhonemeConverter
from utils.negative_sampler import NegativeSampler
from models.verification_model import LipTextVerificationModel
from training.train import train_one_epoch
from evaluation.evaluate import evaluate_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Config:
    def __init__(self, config_path="config.json"):
        full_path = os.path.join(os.path.dirname(__file__), config_path)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                cfg = json.load(f)
        else:
            cfg = {}
        self.processed_data_dir = cfg.get("processed_data_dir", "./data")
        self.use_audio = cfg.get("use_audio", False)
        self.batch_size = cfg.get("batch_size", 4)
        self.learning_rate = cfg.get("learning_rate", 1e-4)
        self.epochs = cfg.get("epochs", 5)
        self.max_frames = cfg.get("max_frames", 75)
        self.output_dir = cfg.get("output_dir", "./output")

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
        if use_audio: aud = item['audio']
        
        # Positive
        videos.append(vid)
        p_seq = converter.text_to_phonemes(trans)
        p_seq = p_seq[:30] + [0] * max(0, 30 - len(p_seq))
        phonemes.append(torch.tensor(p_seq))
        labels.append(1.0)
        transcripts.append(trans)
        if use_audio: audios.append(aud)
        
        # Negative
        neg_trans = sampler.sample(trans)
        videos.append(vid)
        n_seq = converter.text_to_phonemes(neg_trans)
        n_seq = n_seq[:30] + [0] * max(0, 30 - len(n_seq))
        phonemes.append(torch.tensor(n_seq))
        labels.append(0.0)
        transcripts.append(neg_trans)
        if use_audio: audios.append(aud) 
        
    ret = {
        'video': torch.stack(videos),
        'phonemes': torch.stack(phonemes),
        'label': torch.tensor(labels, dtype=torch.float32),
        'transcripts': transcripts
    }
    
    if use_audio: ret['audio'] = torch.stack(audios)
    return ret

def main():
    # --- DDP Initialization ---
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main_process = (local_rank == 0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main_process = True
        local_rank = 0

    cfg = Config("config.json")
    proj_root = os.path.dirname(__file__)
    output_dir_abs = os.path.join(proj_root, cfg.output_dir)
    
    writer = None
    if is_main_process:
        os.makedirs(output_dir_abs, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(output_dir_abs, "logs"))
        print(f"Using device: {device} | Distributed: {is_distributed}")
    
    dataset_path = os.path.abspath(os.path.join(proj_root, cfg.processed_data_dir))
    
    if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
        if is_main_process:
            print(f"Error: Processed dataset directory ({dataset_path}) is missing.")
        return
        
    train_dataset = GRIDDataset(data_dir=dataset_path, split='train', max_frames=cfg.max_frames, use_audio=cfg.use_audio)
    val_dataset = GRIDDataset(data_dir=dataset_path, split='val', max_frames=cfg.max_frames, use_audio=cfg.use_audio)
    test_dataset = GRIDDataset(data_dir=dataset_path, split='test', max_frames=cfg.max_frames, use_audio=cfg.use_audio)
    
    if is_main_process:
        print(f"Splits Loaded -> Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # --- Distributed Samplers ---
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    
    # Validation/Test do not use DistributedSampler here to keep evaluation full and simple on all GPUs.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    
    if is_main_process: print(r"Initializing model...")
    model = LipTextVerificationModel(vocab_size=50, feature_dim=256).to(device)
    
    # --- Wrap Model in DDP ---
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    best_val_auc = 0.0
    model_save_path = os.path.join(output_dir_abs, "best_lip_verification_model.pth")
    
    if is_main_process: print(r"[--- Starting Training ---]")
    for epoch in range(1, cfg.epochs + 1):
        if is_distributed:
            # Important: Ensures shuffling depends on epoch
            train_sampler.set_epoch(epoch)
            
        # Only pass writer to rank 0 to avoid multiple tqdm/tensorboard writes
        loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer=writer if is_main_process else None)
        
        if is_main_process:
            print(f"Epoch {epoch} | Train Loss: {loss:.4f}")
            print(f"Evaluating Model on Validation Set (Epoch {epoch})...")
            
        val_metrics = evaluate_model(model, val_loader, device)
        
        if is_main_process:
            writer.add_scalar('Metrics/val_roc_auc', val_metrics['roc_auc'], epoch)
            writer.add_scalar('Metrics/val_verify_acc', val_metrics['verification_accuracy'], epoch)
            
            if val_metrics['roc_auc'] > best_val_auc:
                best_val_auc = val_metrics['roc_auc']
                torch.save(model.module.state_dict() if is_distributed else model.state_dict(), model_save_path)
                print(f"--> Saved new BEST model with AUC {best_val_auc:.4f} to {model_save_path}")

    if is_main_process:
        print("\n[--- Testing Phase ---]")
        if os.path.exists(model_save_path):
            state_dict = torch.load(model_save_path, map_location=device)
            if is_distributed:
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
                
            print("Evaluating Model on Hold-out Test Set...")
            test_metrics = evaluate_model(model, test_loader, device)
            
            writer.add_scalar('Metrics/test_roc_auc', test_metrics['roc_auc'], 1)
            writer.add_scalar('Metrics/test_verify_acc', test_metrics['verification_accuracy'], 1)
            print(f"Final Test Result -> AUC: {test_metrics['roc_auc']:.4f} | Acc: {test_metrics['verification_accuracy']:.4f}")
            
        writer.close()
        print("Complete!")

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
    