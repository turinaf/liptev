import torch
import os
from torch.utils.data import DataLoader
from dataset.grid_loader import GRIDDataset
from models.verification_model import LipTextVerificationModel
from evaluation.evaluate import evaluate_model
from main import collate_fn, Config

def test():
    cfg = Config("config.json")
    proj_root = os.path.dirname(os.path.abspath(__file__))
    output_dir_abs = os.path.join(proj_root, cfg.output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset_path = os.path.abspath(os.path.join(proj_root, cfg.processed_data_dir))
    
    if not os.path.exists(dataset_path):
        print(f"Error: Processed dataset directory ({dataset_path}) is missing.")
        return
        
    print(f"Loading test dataset from {dataset_path}...")
    test_dataset = GRIDDataset(data_dir=dataset_path, split='test', max_frames=cfg.max_frames, use_audio=cfg.use_audio)
    
    if len(test_dataset) == 0:
        print("Test dataset is empty. Check your data splits.")
        return
        
    print(f"Test split loaded -> {len(test_dataset)} samples")
    
    # Standard PyTorch DataLoader for evaluation
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print("Initializing model...")
    model = LipTextVerificationModel(vocab_size=50, feature_dim=256).to(device)
    
    model_save_path = os.path.join(output_dir_abs, "best_lip_verification_model.pth")
    if not os.path.exists(model_save_path):
        # Fallback to check for standard checkpoints
        print(f"Could not find {model_save_path}, looking for other checkpoints...")
        import glob
        checkpoints = glob.glob(os.path.join(output_dir_abs, "checkpoint_epoch_*.pth"))
        if not checkpoints:
            print("No checkpoints found. Please train the model first.")
            return
        checkpoints.sort(key=lambda x: int(x.split("_epoch_")[-1].split(".pth")[0]))
        model_save_path = checkpoints[-1]
    
    print(f"Loading weights from {model_save_path}...")
    checkpoint = torch.load(model_save_path, map_location=device)
    
    # Handle both standalone weights and wrapped checkpoint dictionaries
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Handle DDP weights (stripping the 'module.' prefix if the model was trained with Multi-GPU)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    
    print("\n[--- Testing Phase ---]")
    print("Evaluating Model on Hold-out Test Set...")
    test_metrics = evaluate_model(model, test_loader, device)
    
    print(f"\nFinal Test Result -> AUC: {test_metrics['roc_auc']:.4f} | Acc: {test_metrics['verification_accuracy']:.4f}")

if __name__ == "__main__":
    test()
