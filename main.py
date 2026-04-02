import torch
import os
import json
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from dataset.grid_loader import GRIDDataset
from dataset.digit_loader import DigitDataset
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
        self.dataset = cfg.get("dataset")
        self.processed_data_dir = cfg.get("digit_processed_data_dir") if self.dataset == "digit" else cfg.get("grid_processed_data_dir")
        self.use_audio = cfg.get("use_audio")
        self.batch_size = cfg.get("batch_size")
        self.learning_rate = cfg.get("learning_rate")
        self.epochs = cfg.get("epochs")
        self.max_frames = cfg.get("max_frames")
        self.output_dir = cfg.get("output_dir")

def collate_fn_factory(dataset_name):
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
            
            # Positive sample
            videos.append(vid)
            if dataset_name == "digit":
                # Use digits directly as phonemes
                pos_phonemes = trans.split()
            else:
                pos_phonemes = converter.get_phonemes(trans)
            phonemes.append(pos_phonemes)
            labels.append(1.0)
            transcripts.append(trans)
            if use_audio: audios.append(aud)
            
            # Negative sample
            videos.append(vid)
            neg_trans = sampler.sample(trans)
            if dataset_name == "digit":
                # Use digits directly as phonemes
                neg_phonemes = neg_trans.split()
            else:
                neg_phonemes = converter.get_phonemes(neg_trans)
            phonemes.append(neg_phonemes)
            labels.append(0.0)
            transcripts.append(neg_trans)
            if use_audio: audios.append(aud)
            
        videos = torch.stack(videos)
        labels = torch.tensor(labels, dtype=torch.float32)
        if use_audio: 
            audios = torch.stack(audios)
            
        # Note: phonemes is left as a list of lists/strings, model should handle padding/embedding
        # Model's text encoder needs to correctly map digits if they are passed as words. 
        batch_dict = {
            'video': videos,
            'phonemes': phonemes,
            'label': labels,
            'transcript': transcripts
        }
        if use_audio:
            batch_dict['audio'] = audios
            
        return batch_dict
    return collate_fn

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    config = Config()
    
    device = torch.device(f"cuda:{rank}")
    
    if config.dataset == "digit":
        train_dataset = DigitDataset(config.processed_data_dir, split="train", max_frames=config.max_frames, use_audio=config.use_audio)
        val_dataset = DigitDataset(config.processed_data_dir, split="val", max_frames=config.max_frames, use_audio=config.use_audio)
    else:
        train_dataset = GRIDDataset(config.processed_data_dir, split="train", max_frames=config.max_frames, use_audio=config.use_audio)
        val_dataset = GRIDDataset(config.processed_data_dir, split="val", max_frames=config.max_frames, use_audio=config.use_audio)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    custom_collate = collate_fn_factory(config.dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler, collate_fn=custom_collate)
    
    # Model uses a hardcoded vocab size. If using raw digits vs phonemes, make sure vocab size accounts for "0"-"9" characters
    # Usually digit tokens "0", "1"... can just be mapped inside the LipTextVerificationModel's text encoder
    model = LipTextVerificationModel().to(device)
    model = DDP(model, device_ids=[rank])
    
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    
    writer = None
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'logs'))
        
    for epoch in range(1, config.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer if rank == 0 else None)
        
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        
        if rank == 0:
            print(f"Epoch {epoch}/{config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            # Save checkpoint
            torch.save(model.module.state_dict(), os.path.join(config.output_dir, f"checkpoint_ep{epoch}.pt"))
            
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 0:
        import torch.multiprocessing as mp
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("No CUDA devices available for DDP.")
