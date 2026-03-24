import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from .preprocessing import extract_lip_frames

class GRIDDataset(Dataset):
    def __init__(self, data_dir, split='train', max_frames=75, transform=None, mp_model_path="face_landmarker.task"):
        """
        Loads the actual GRID dataset.
        Looks for speaker directories, searching for both videos (.mpg) and alignment files (.align).
        """
        self.data_dir = data_dir
        self.split = split
        self.max_frames = max_frames
        self.mp_model_path = mp_model_path
        self.transform = transform or T.Compose([
            T.ToTensor(),
            # Normalization typically handled if not resenting to specific domain, keeping simple for PyTorch compat.
        ])
        
        self.samples = self._load_dataset_paths()

    def _load_dataset_paths(self):
        samples = []
        # Find all .mpg files for all speakers in the dataset directory
        video_paths = glob.glob(os.path.join(self.data_dir, '*', 'video', 'mac', '*.mpg'))
        video_paths += glob.glob(os.path.join(self.data_dir, '*', 'video', '*.mpg')) # fallback if paths differ
        
        for vid_path in video_paths:
            # Reconstruct corresponding align file path
            base_name = os.path.basename(vid_path).replace('.mpg', '')
            
            # Navigate back from video/mac to sX/align
            speaker_dir = os.path.dirname(os.path.dirname(os.path.dirname(vid_path)))
            if os.path.basename(os.path.dirname(vid_path)) == 'video':
                speaker_dir = os.path.dirname(os.path.dirname(vid_path))
                
            align_path = os.path.join(speaker_dir, 'align', f"{base_name}.align")
            
            if os.path.exists(align_path):
                transcript = self._read_align_file(align_path)
                if transcript:
                    samples.append({
                        "video": vid_path,
                        "transcript": transcript,
                        "align_path": align_path
                    })
        return samples

    def _read_align_file(self, align_path):
        """
        Reads a GRID alignment file and extracts the spoken words.
        Returns the reconstructed transcript.
        """
        words = []
        with open(align_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start, end, word = parts
                    # Standard words to exclude from transcript
                    if word not in ['sil', 'sp']:
                        words.append(word)
        return " ".join(words)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract actual video frames & dynamically crop lip region using the MediaPipe Taks!
        frames = extract_lip_frames(sample["video"], self.max_frames, self.mp_model_path)
        
        # Apply transforms and stack temporally
        tensor_list = []
        for f in frames:
            tf = self.transform(f) if self.transform else torch.tensor(f)
            # Ensure type safety and validation that output is absolutely a Tensor
            if not isinstance(tf, torch.Tensor):
                tf = torch.tensor(tf)
            tensor_list.append(tf)

        frames_tensor = torch.stack(tensor_list) # T x C x H x W
        
        return {
            "video": frames_tensor,
            "transcript": sample["transcript"]
        }
