import os
import glob
import torch
import cv2
import json
import torchaudio
from torch.utils.data import Dataset
import torchvision.transforms as T

class GRIDDataset(Dataset):
    def __init__(self, data_dir, split='train', max_frames=75, transform=None, use_audio=False):
        """
        Loads the preprocessed GRID dataset using pre-computed split files.
        data_dir should point to the processed directory (e.g. ./data) containing 
        the train_split.json, val_split.json, and test_split.json manifests.
        """
        self.data_dir = data_dir
        self.split = split
        self.max_frames = max_frames
        self.use_audio = use_audio
        self.transform = transform or T.Compose([
            T.ToTensor(),
        ])
        
        self.samples = self._load_dataset_paths()

    def _load_dataset_paths(self):
        samples = []
        
        split_file = os.path.join(self.data_dir, f"{self.split}_split.json")
        
        if not os.path.exists(split_file):
            print(f"Warning: Split file {split_file} not found. Please run pipeline/data_preparation.py first.")
            return []
            
        with open(split_file, "r") as f:
            video_paths = json.load(f)
            
        for vid_path in video_paths:
            base_name = os.path.basename(vid_path).replace('.mp4', '')
            speaker_dir = os.path.dirname(os.path.dirname(vid_path))
            
            align_path = os.path.join(speaker_dir, 'align', f"{base_name}.align")
            audio_path = os.path.join(speaker_dir, 'audio', f"{base_name}.wav")
            
            if os.path.exists(align_path):
                transcript = self._read_align_file(align_path)
                
                # Check for audio if requested
                if self.use_audio and not os.path.exists(audio_path):
                    continue
                    
                if transcript:
                    sample_dict = {
                        "video": vid_path,
                        "transcript": transcript,
                        "align_path": align_path
                    }
                    if self.use_audio:
                        sample_dict["audio"] = audio_path
                        
                    samples.append(sample_dict)
                    
        return samples

    def _read_align_file(self, align_path):
        words = []
        with open(align_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start, end, word = parts
                    if word not in ['sil', 'sp']:
                        words.append(word)
        return " ".join(words)

    def _load_video_frames(self, path):
        """ Loads already pre-cropped and pre-sized video """
        cap = cv2.VideoCapture(path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # Temporal padding just in case length doesn't strictly match max_frames
        if len(frames) == 0:
            frames = [torch.zeros((3, 96, 96)) for _ in range(self.max_frames)]
            return torch.stack(frames)
            
        elif len(frames) > self.max_frames:
            frames = frames[:self.max_frames]
        else:
            pad_len = self.max_frames - len(frames)
            last_frame = frames[-1]
            frames.extend([last_frame] * pad_len)
            
        tensor_list = []
        for f in frames:
            tf = self.transform(f) if self.transform else torch.tensor(f)
            if not isinstance(tf, torch.Tensor):
                tf = torch.tensor(tf)
            tensor_list.append(tf)
            
        return torch.stack(tensor_list)

    def _load_audio(self, path):
        # returns simple raw audio form (can be expanded to mel spectrograms depending on audio features)
        waveform, sr = torchaudio.load(path)
        # Flatten and truncate / pad to standard length depending on implementation
        # For ~3 seconds of 16kHz audio = 48000 samples
        target_len = sr * 3 
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        else:
            pad = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        frames_tensor = self._load_video_frames(sample["video"])
        
        ret_dict = {
            "video": frames_tensor,
            "transcript": sample["transcript"]
        }
        
        if self.use_audio and "audio" in sample:
            audio_tensor = self._load_audio(sample["audio"])
            ret_dict["audio"] = audio_tensor
            
        return ret_dict
