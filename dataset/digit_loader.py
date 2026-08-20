import os
import glob
import wave
import torch
import cv2
import json
import torchaudio
from torch.utils.data import Dataset
import torchvision.transforms as T

class DigitDataset(Dataset):
    def __init__(self, data_dir, split='train', max_frames=125, transform=None, use_audio=False,
                 source_fps=29.87):
        """
        Loads the preprocessed Digit dataset using pre-computed split files.
        """
        self.data_dir = data_dir
        self.split = split
        self.max_frames = max_frames
        self.use_audio = use_audio
        # Fallback fps for converting .lab seconds to frames when the
        # pipeline's per-video meta.json is unavailable (measured source
        # rates vary per device: ~29.9 fps VFR and 25 fps both occur).
        self.source_fps = source_fps
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self._fps_meta_cache = {}

        self.samples = self._load_dataset_paths()

    def _speaker_fps(self, subset_name, speaker_name, base_name, vid_path=None, audio_path=None):
        """Per-video source fps.

        Resolution order: the pipeline's meta.json (authoritative) -> an
        exact inference from the processed video's frame count over the raw
        audio's duration (frame indices are preserved by the 25 fps
        re-encode, so frames / duration == source fps) -> the configured
        default. The inference matters because measured source rates vary
        per recording device (~29.9 fps VFR and 25 fps both occur) and the
        fixed default misplaces .lab frames by up to ~20% for 25 fps
        speakers.
        """
        key = (subset_name, speaker_name)
        if key not in self._fps_meta_cache:
            meta_path = os.path.join(self.data_dir, subset_name, speaker_name, "meta.json")
            try:
                with open(meta_path, "r") as f:
                    self._fps_meta_cache[key] = json.load(f)
            except (OSError, ValueError):
                self._fps_meta_cache[key] = {}
        entry = self._fps_meta_cache[key].get(base_name)
        if entry and entry.get("source_fps", 0) > 0:
            return entry["source_fps"]
        # Fallback inference (cached back into the meta dict for this run).
        if vid_path and audio_path:
            fps = self._infer_fps(vid_path, audio_path)
            if fps:
                self._fps_meta_cache[key][base_name] = {"source_fps": fps}
                return fps
        return self.source_fps

    def _infer_fps(self, vid_path, audio_path):
        try:
            cap = cv2.VideoCapture(vid_path)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            with wave.open(audio_path, "rb") as w:
                duration = w.getnframes() / float(w.getframerate())
            if frames > 0 and duration > 0:
                fps = frames / duration
                if 20.0 <= fps <= 35.0:  # sanity window around real capture rates
                    return round(fps, 3)
        except Exception:
            pass
        return None

    def _find_lab_file(self, subset_name, speaker_name, base_name):
        """Processed .lab location: text/ (alongside .txt) or lab/ (copied
        from the raw tree by the pipeline)."""
        for dirname in ("text", "lab"):
            path = os.path.join(self.data_dir, subset_name, speaker_name, dirname, f"{base_name}.lab")
            if os.path.exists(path):
                return path
        return None

    def _parse_lab_file(self, lab_path):
        """Parse .lab: line 1 = digit sequence, line 2 = per-digit
        'start-end' seconds in the raw video's timeline."""
        with open(lab_path, "r") as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        if len(lines) < 2:
            return None, None
        digits = lines[0].split()
        try:
            times = [tuple(float(x) for x in rng.split("-")) for rng in lines[1].split()]
        except ValueError:
            return None, None
        if len(digits) != len(times):
            return None, None
        return digits, times

    def _load_dataset_paths(self):
        samples = []
        skipped_lab_out_of_window = 0
        skipped_lab_mismatch = 0

        split_file = os.path.join(self.data_dir, f"{self.split}_split.json")

        if not os.path.exists(split_file):
            print(f"Warning: Split file {split_file} not found. Please run pipeline/data_preparation.py first.")
            return []

        with open(split_file, "r") as f:
            video_paths = json.load(f)

        for vid_path_raw in video_paths:
            base_name = os.path.basename(vid_path_raw).replace('.mp4', '')
            # original_speaker_dir = "video"
            # os.path.dirname(original_speaker_dir) = "1011"
            speaker_dir_path = os.path.dirname(os.path.dirname(vid_path_raw))
            speaker_name = os.path.basename(speaker_dir_path)
            subset_name = os.path.basename(os.path.dirname(speaker_dir_path))
            
            # The digit dataset uses 'text' directory and .txt files
            # Reconstruct paths using the current data_dir
            vid_path = os.path.join(self.data_dir, subset_name, speaker_name, 'video', f"{base_name}.mp4")
            align_path = os.path.join(self.data_dir, subset_name, speaker_name, 'text', f"{base_name}.txt")
            audio_path = os.path.join(self.data_dir, subset_name, speaker_name, 'audio', f"{base_name}.wav")
            
            # The paths in the JSON already contain the data_dir prefix! So we don't need to rebuild them if they are absolute/relative from root
            if "data/digit" in vid_path_raw: # The original passed in vid_path
                vid_path = vid_path_raw.replace('./', '')
                align_path = vid_path.replace('/video/', '/text/').replace('.mp4', '.txt')
                audio_path = vid_path.replace('/video/', '/audio/').replace('.mp4', '.wav')
                
            # Allow searching since exlamation points (!) might be escaped/missing and names might vary slightly in the actual filesystem
            if not os.path.exists(vid_path):
                 # Try matching without escaping/special characters if exact match fails
                 potential_matches = glob.glob(os.path.join(self.data_dir, subset_name, speaker_name, 'video', f"*{base_name.split('_')[-2]}*.mp4"))
                 if potential_matches:
                     vid_path = potential_matches[0]
                     base_name = os.path.basename(vid_path).replace('.mp4', '')
                     align_path = os.path.join(self.data_dir, subset_name, speaker_name, 'text', f"{base_name}.txt")
                     audio_path = os.path.join(self.data_dir, subset_name, speaker_name, 'audio', f"{base_name}.wav")
                     
            if os.path.exists(align_path):
                transcript = self._read_text_file(align_path)
                
                # Check for audio if requested
                if self.use_audio and not os.path.exists(audio_path):
                    print(f"Missing audio: {audio_path}")
                    continue
                    
                if transcript:
                    sample_dict = {
                        "video": vid_path,
                        "transcript": transcript,
                        "align_path": align_path
                    }

                    # Optional .lab boundary supervision. Frames beyond the
                    # max_frames window can't be supervised (and the visible
                    # content would contradict the transcript), so those
                    # samples are skipped rather than truncated. .txt
                    # transcripts may contain "!" tokens (non-speech events)
                    # that .lab line 1 omits — compare on the digit tokens
                    # only.
                    lab_path = self._find_lab_file(subset_name, speaker_name, base_name)
                    if lab_path is not None:
                        lab_digits, lab_times = self._parse_lab_file(lab_path)
                        fps = self._speaker_fps(subset_name, speaker_name, base_name,
                                                 vid_path=vid_path, audio_path=audio_path)
                        if lab_digits is None or lab_digits != [t for t in transcript.split() if t != "!"]:
                            skipped_lab_mismatch += 1
                        elif any(int(round(e * fps)) > self.max_frames for _, e in lab_times):
                            skipped_lab_out_of_window += 1
                            continue
                        else:
                            sample_dict["lab_times"] = lab_times
                            sample_dict["source_fps"] = fps

                    if self.use_audio:
                        sample_dict["audio"] = audio_path

                    samples.append(sample_dict)
            else:
                print(f"Missing: align={align_path}, audio={audio_path}, video={vid_path}")
                continue

        if skipped_lab_out_of_window or skipped_lab_mismatch:
            print(f"Digit[{self.split}]: skipped {skipped_lab_out_of_window} samples whose .lab "
                  f"extends beyond {self.max_frames} frames, {skipped_lab_mismatch} with invalid/mismatched .lab")

        return samples

    def _read_text_file(self, align_path):
        # The digit text file just contains digits e.g. "4 5 6 3 8 7 1 6"
        with open(align_path, 'r') as f:
            content = f.read().strip()
        return content

    def _load_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        true_len = min(len(frames), self.max_frames)

        if len(frames) == 0:
            frames = [torch.zeros((3, 96, 96)) for _ in range(self.max_frames)]
            return torch.stack(frames), 0
            
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
            
        return torch.stack(tensor_list), true_len

    def _load_audio(self, path, video_len, fps):
        """Audio trimmed/padded to exactly the span the kept frames cover.

        A fixed 3 s budget used to cut audio while the video kept up to
        max_frames (~4.2 s at ~30 fps source), leaving later digits with lip
        movement but silence. Length now follows the video: samples =
        sr * video_len / fps (frame indices are preserved by the 25 fps
        re-encode).
        """
        waveform, sr = torchaudio.load(path)
        if fps <= 0 or video_len <= 0:
            return waveform, waveform.shape[1]
        target_len = int(sr * video_len / fps)
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        else:
            pad = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform, waveform.shape[1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        frames_tensor, video_len = self._load_video_frames(sample["video"])
        fps = sample.get("source_fps", self.source_fps)

        ret_dict = {
            "video": frames_tensor,
            "transcript": sample["transcript"],
            "video_len": video_len
        }

        # .lab seconds -> frames at the raw video's fps (frame indices are
        # preserved by the 25 fps re-encode). The walk covers the FULL token
        # sequence: "!" tokens (non-speech events without .lab entries) keep
        # their sequence slot but receive zero supervised frames, so later
        # digits don't shift positions.
        if "lab_times" in sample:
            times = iter(sample["lab_times"])
            digit_times = []
            for tok in sample["transcript"].split():
                if tok == "!":
                    digit_times.append((tok, 0, 0))
                else:
                    s, e = next(times)
                    digit_times.append((
                        tok,
                        max(0, min(int(round(s * fps)), video_len)),
                        max(0, min(int(round(e * fps)), video_len))
                    ))
            ret_dict["digit_times"] = digit_times

        if self.use_audio and "audio" in sample:
            audio_tensor, audio_len = self._load_audio(sample["audio"], video_len, fps)
            ret_dict["audio"] = audio_tensor
            ret_dict["audio_len"] = audio_len

        return ret_dict

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.dataset_utils import Config
    from utils.negative_sampler import NegativeSampler
    
    config = Config()
    test_data_dir = config.processed_data_dir
    
    print(f"--- Debugging DigitDataset ---")
    print(f"Looking for data in: {test_data_dir}")
    print(f"Using audio: {config.use_audio}")
    
    try:
        dataset = DigitDataset(data_dir=test_data_dir, split='train', max_frames=config.max_frames, use_audio=config.use_audio)
        print(f"\nSuccessfully loaded train split with {len(dataset)} samples.")

        digit_vocab = {str(i): i + 1 for i in range(10)}
        digit_vocab["!"] = 11
        digit_vocab["<pad>"] = 0
        digit_vocab["<unk>"] = 12

        def transcript_to_sequence(transcript):
            return [digit_vocab.get(token, digit_vocab["<unk>"]) for token in transcript.split()]
        
        if len(dataset) > 0:
            print("\n--- Inspecting Samples ---")
            transcript_pool = [sample_item["transcript"] for sample_item in dataset.samples]
            sampler = NegativeSampler(transcript_pool)

            sample_count = min(3, len(dataset))
            for idx in range(sample_count):
                sample = dataset[idx]
                positive_seq = transcript_to_sequence(sample["transcript"])
                negative_transcript = sampler.sample(sample["transcript"])
                negative_seq = transcript_to_sequence(negative_transcript)

                print(f"\nSample {idx + 1} Positive:")
                print(f"Transcript: '{sample['transcript']}'")
                print(f"Sequence: {positive_seq}")
                print(f"Video Tensor Shape: {sample['video'].shape}")
                print(f"Video Tensor Data Type: {sample['video'].dtype}")
                print(f"Max Val: {sample['video'].max():.4f}, Min Val: {sample['video'].min():.4f}")
                if "audio" in sample:
                    print(f"Audio Tensor Shape: {sample['audio'].shape}")

                print(f"Sample {idx + 1} Negative:")
                print(f"Transcript: '{negative_transcript}'")
                print(f"Sequence: {negative_seq}")
            
    except Exception as e:
        print(f"\nError initializing dataset: {e}")
