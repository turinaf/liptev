import torch
import multiprocessing
import os
import json
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from dataset.grid_loader import GRIDDataset
from dataset.digit_loader import DigitDataset
from models.verification_model import LipTextVerificationModel
from training.train import train_one_epoch, build_optimizer, ValidationMonitor
from evaluation.evaluate import evaluate_model
from evaluation.metrics import save_eer_thresholds, checkpoint_sha256
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataset_utils import Config, collate_fn_factory
from utils.checkpoint import save_checkpoint, check_alignment_supervision
from utils.tokenizer import TranscriptTokenizer



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(config_path="config.json"):
    writer = None
    try:
        # Python 3.14 switched the POSIX default start method to 'forkserver',
        # which pickles DataLoader worker args; collate_fn_factory returns a
        # closure that can't be pickled. Restore the pre-3.14 'fork' behavior.
        multiprocessing.set_start_method("fork", force=True)
        # Detect if spawned via torchrun
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            # Only initialize if NOT ALREADY initialized by torchrun natively
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            device = torch.device(f"cuda:{local_rank}")
        else:
            # Standalone spawn logic
            setup(0, 1)
            local_rank = 0
            world_size = 1
            device = torch.device("cuda:0")

        config = Config(config_path)
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        check_alignment_supervision(config.alignment_mode, config.lambda_align)

        if config.dataset == "digit":
            train_dataset = DigitDataset(config.processed_data_dir, split="train", max_frames=config.max_frames, use_audio=config.use_audio, source_fps=config.digit_source_fps)
            val_dataset = DigitDataset(config.processed_data_dir, split="val", max_frames=config.max_frames, use_audio=config.use_audio, source_fps=config.digit_source_fps)
        else:
            train_dataset = GRIDDataset(config.processed_data_dir, split="train", max_frames=config.max_frames, use_audio=config.use_audio)
            val_dataset = GRIDDataset(config.processed_data_dir, split="val", max_frames=config.max_frames, use_audio=config.use_audio)

        for split_name, ds in (("train", train_dataset), ("val", val_dataset)):
            if len(ds) == 0:
                raise RuntimeError(
                    f"{config.dataset} '{split_name}' split is empty in '{config.processed_data_dir}'. "
                    f"Check the processed data dir in '{config_path}' or run pipeline/data_preparation.py."
                )
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        transcript_pool = [s.get("transcript", "") for s in getattr(train_dataset, "samples", [])]
        custom_collate = collate_fn_factory(
            config.dataset, transcript_pool=transcript_pool, token_level=config.token_level,
            balanced_word_supervision=config.balanced_word_supervision)
        val_collate = collate_fn_factory(
            config.dataset,
            transcript_pool=[s.get("transcript", "") for s in getattr(val_dataset, "samples", [])],
            token_level=config.token_level, seed=config.validation_seed,
            balanced_word_supervision=config.balanced_word_supervision)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, collate_fn=custom_collate, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler, collate_fn=val_collate, num_workers=4, pin_memory=True)
        
        tokenizer = TranscriptTokenizer(config.dataset, config.token_level)
        vocab_size = len(tokenizer.vocab)
        model = LipTextVerificationModel(
            vocab_size=vocab_size,
            lip_backbone_pretrained=config.lip_backbone_pretrained,
            alignment_mode=config.alignment_mode,
            dropout=config.dropout,
        ).to(device)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        
        optimizer = build_optimizer(model, config.learning_rate, config.weight_decay)
        monitor = ValidationMonitor(config.early_stopping_patience)
        
        if rank == 0:
            os.makedirs(config.output_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'logs'))
            print(f"Training {config.dataset} with {config.token_level} tokens, "
                  f"vocab_size={vocab_size}, dropout={config.dropout}, "
                  f"weight_decay={config.weight_decay}, up to {config.epochs} epochs")
            
        for epoch in range(1, config.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer if rank == 0 else None,
                                         lambda_ctc=config.lambda_ctc, lambda_align=config.lambda_align,
                                         alignment_temperature=config.alignment_temperature,
                                         lambda_frame_ce=config.lambda_frame_ce,
                                         lambda_boundary=config.lambda_boundary,
                                         lambda_word_verif=config.lambda_word_verif,
                                         balanced_word_supervision=config.balanced_word_supervision,
                                         supervise_silence=config.supervise_silence)
            
            metrics = evaluate_model(
                model, val_loader, device, boundary_threshold=config.vfa_boundary_threshold,
                word_metrics=config.balanced_word_supervision)
            selection_eer = metrics["verification_eer"]
            if config.balanced_word_supervision:
                selection_eer = 0.5 * (selection_eer + metrics["word_eer"])
            improved = monitor.update(selection_eer)
            
            if rank == 0:
                print(
                    f"Epoch {epoch}/{config.epochs} | Train Loss: {train_loss:.4f} | "
                    f"Val ROC AUC: {metrics['roc_auc']:.4f} | "
                    f"Val EER: {metrics['verification_eer']:.4f} | "
                    f"Val Ver Acc: {metrics['verification_accuracy']:.4f}"
                )
                writer.add_scalar('Logic/roc_auc', metrics["roc_auc"], epoch)
                writer.add_scalar('Logic/align_roc_auc', metrics["alignment_roc_auc"], epoch)
                writer.add_scalar('EER/verify', metrics["verification_eer"], epoch)
                writer.add_scalar('EER/align', metrics["alignment_eer"], epoch)
                writer.add_scalar('Accuracy/verify', metrics["verification_accuracy"], epoch)
                writer.add_scalar('Accuracy/verify_at_eer', metrics["verification_accuracy_at_eer"], epoch)
                writer.add_scalar('EER/selection', selection_eer, epoch)
                if config.balanced_word_supervision:
                    writer.add_scalar('EER/word', metrics["word_eer"], epoch)
                    writer.add_scalar('Logic/word_roc_auc', metrics["word_roc_auc"], epoch)
                
                # Save checkpoint (mode stamped next to the weights so eval
                # runs can't silently score it under a different mode)
                save_checkpoint(
                    model.module.state_dict(),
                    os.path.join(config.output_dir, f"checkpoint_ep{epoch}.pt"),
                    config.alignment_mode,
                    vfa_heads=True,
                    dataset=config.dataset, token_level=config.token_level,
                    token_vocab=tokenizer.vocab, dropout=config.dropout,
                )
                if improved:
                    best_path = os.path.join(config.output_dir, "best_lip_verification_model.pth")
                    save_checkpoint(
                        model.module.state_dict(),
                        best_path,
                        config.alignment_mode, vfa_heads=True,
                        dataset=config.dataset, token_level=config.token_level,
                        token_vocab=tokenizer.vocab, dropout=config.dropout,
                    )
                    save_eer_thresholds(
                        metrics,
                        {"checkpoint": os.path.basename(best_path),
                         "checkpoint_sha256": checkpoint_sha256(best_path),
                         "dataset": config.dataset, "token_level": config.token_level,
                         "split": "val", "n_samples": len(val_dataset),
                         "boundary_threshold": config.vfa_boundary_threshold},
                        os.path.join(config.output_dir, "eer_thresholds.json"),
                    )
                    print(f"Saved best checkpoint at epoch {epoch} (selection EER {monitor.best_eer:.4f})")
            # evaluate_model gathers every rank, so all ranks make the same
            # stop decision and exit DDP together.
            if monitor.should_stop:
                if rank == 0:
                    print(f"Early stopping after {monitor.bad_epochs} epochs without selection EER improvement.")
                break
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lip-Text Verification Training")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Config file selecting dataset, data dirs, and checkpoint output dir")
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        main(args.config)
    else:
        world_size = torch.cuda.device_count()
        if world_size > 0:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            import torch.multiprocessing as mp
            # If using mp.spawn manually
            def entry(rank, ws):
                os.environ["LOCAL_RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(ws)
                main(args.config)
            mp.spawn(entry, args=(world_size,), nprocs=world_size, join=True)
        else:
            print("No CUDA devices available.")
