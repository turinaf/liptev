import torch
import os
import glob
import re
from torch.utils.data import DataLoader
from dataset.grid_loader import GRIDDataset
from models.verification_model import LipTextVerificationModel
from evaluation.evaluate import evaluate_model
from evaluation.metrics import save_eer_thresholds, checkpoint_sha256
from utils.dataset_utils import collate_fn_factory, Config
from utils.checkpoint import resolve_alignment_mode, resolve_vfa_heads, checkpoint_tokenizer
from dataset.digit_loader import DigitDataset


def _resolve_latest_checkpoint(output_dir_abs):
    # Priority order: explicit best checkpoint, then latest epoch checkpoint.
    preferred = [
        os.path.join(output_dir_abs, "best_lip_verification_model.pth"),
        os.path.join(output_dir_abs, "best_lip_verification_model.pt"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path

    patterns = [
        "checkpoint_ep*.pt",       # current training format
        "checkpoint_ep*.pth",
        "checkpoint_epoch_*.pth",  # legacy format
        "checkpoint_epoch_*.pt",
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(output_dir_abs, pattern)))

    if not candidates:
        return None

    def extract_epoch(path):
        name = os.path.basename(path)
        match = re.search(r"(?:checkpoint_ep|checkpoint_epoch_)(\d+)", name)
        return int(match.group(1)) if match else -1

    candidates.sort(key=extract_epoch)
    return candidates[-1]

def build_eval_model(checkpoint, cfg, vocab_size, device):
    """Model for strict checkpoint loading. Both stamps come from the
    checkpoint, never config: the scoring mode because the classifier and the
    EER calibration were produced under it, and the VFA-head flag because a
    legacy state dict (pre-VFA) strict-loads only into use_vfa=False."""
    alignment_mode = resolve_alignment_mode(checkpoint, cfg.alignment_mode)
    return LipTextVerificationModel(
        vocab_size=vocab_size,
        feature_dim=256,
        lip_backbone_pretrained=cfg.lip_backbone_pretrained,
        alignment_mode=alignment_mode,
        use_vfa=resolve_vfa_heads(checkpoint),
        dropout=checkpoint.get("dropout", 0.0),
    ).to(device)

def test(config_path="config.json"):
    cfg = Config(config_path)
    proj_root = os.path.dirname(os.path.abspath(__file__))
    output_dir_abs = cfg.output_dir if os.path.isabs(cfg.output_dir) else os.path.abspath(os.path.join(proj_root, cfg.output_dir))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset_path = os.path.abspath(os.path.join(proj_root, cfg.processed_data_dir))
    
    if not os.path.exists(dataset_path):
        print(f"Error: Processed dataset directory ({dataset_path}) is missing.")
        return
        
    print(f"Loading test dataset from {dataset_path}...")
    if cfg.dataset == "digit":
        test_dataset = DigitDataset(data_dir=dataset_path, split='test', max_frames=cfg.max_frames, use_audio=cfg.use_audio)
    else:
        test_dataset = GRIDDataset(data_dir=dataset_path, split='test', max_frames=cfg.max_frames, use_audio=cfg.use_audio)
    
    if len(test_dataset) == 0:
        print("Test dataset is empty. Check your data splits.")
        return
        
    print(f"Test split loaded -> {len(test_dataset)} samples")
    
    model_save_path = _resolve_latest_checkpoint(output_dir_abs)
    if model_save_path is None:
        print(f"No checkpoints found under {output_dir_abs}. Please train the model first.")
        return

    print(f"Loading weights from {model_save_path}...")
    checkpoint = torch.load(model_save_path, map_location=device)
    tokenizer = checkpoint_tokenizer(checkpoint, cfg.dataset)
    controlled_grid = cfg.dataset == "grid" and tokenizer.token_level == "word"
    word_metrics = bool(checkpoint.get("vfa_heads")) and (controlled_grid or cfg.dataset == "digit")
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn_factory(
            cfg.dataset, token_level=tokenizer.token_level, seed=cfg.validation_seed,
            balanced_word_supervision=controlled_grid),
    )

    # Handle both standalone weights and wrapped checkpoint dictionaries
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # Handle DDP weights (stripping the 'module.' prefix if the model was trained with Multi-GPU)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    print("Initializing model...")
    vocab_size = len(tokenizer.vocab)
    model = build_eval_model(checkpoint, cfg, vocab_size, device)

    model.load_state_dict(clean_state_dict)
    
    dataset_class = DigitDataset if cfg.dataset == "digit" else GRIDDataset
    val_dataset = dataset_class(
        data_dir=dataset_path, split="val", max_frames=cfg.max_frames, use_audio=cfg.use_audio)
    if len(val_dataset) == 0:
        raise RuntimeError("Validation split is empty; cannot calibrate on the test set")
    val_pool = [sample["transcript"] for sample in getattr(val_dataset, "samples", [])]
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn_factory(
            cfg.dataset, transcript_pool=val_pool, token_level=tokenizer.token_level,
            seed=cfg.validation_seed, balanced_word_supervision=controlled_grid),
    )
    print("\n[--- Validation Calibration ---]")
    val_metrics = evaluate_model(
        model, val_loader, device, boundary_threshold=cfg.vfa_boundary_threshold,
        word_metrics=word_metrics)
    thresholds_path = os.path.join(output_dir_abs, "eer_thresholds.json")
    save_eer_thresholds(
        val_metrics,
        {
            "checkpoint": os.path.basename(model_save_path),
            "checkpoint_sha256": checkpoint_sha256(model_save_path),
            "dataset": cfg.dataset,
            "token_level": tokenizer.token_level,
            "split": "val", "n_samples": len(val_dataset),
            "boundary_threshold": cfg.vfa_boundary_threshold,
        },
        thresholds_path,
    )
    print(f"EER thresholds saved to {thresholds_path}")

    print("\n[--- Testing Phase ---]")
    print("Evaluating Model on Hold-out Test Set (not used for calibration)...")
    test_metrics = evaluate_model(
        model, test_loader, device, boundary_threshold=cfg.vfa_boundary_threshold,
        word_metrics=word_metrics)
    print(f"\nFinal Test Result -> AUC: {test_metrics['roc_auc']:.4f} | Acc: {test_metrics['verification_accuracy']:.4f} | EER: {test_metrics['verification_eer']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lip-Text Verification test-set evaluation + EER calibration")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Config file selecting dataset, data dir, and checkpoint output dir")
    args = parser.parse_args()
    test(args.config)
