import os
import json
import torch
from utils.phoneme_converter import PhonemeConverter
from utils.negative_sampler import NegativeSampler
from utils.alignment import expand_token_ranges, build_alignment_targets

class Config:
    def __init__(self, config_path="config.json"):
        full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                cfg = json.load(f)
        else:
            cfg = {}
        self.dataset = cfg.get("dataset", "grid")
        self.processed_data_dir = cfg.get("digit_processed_data_dir") if self.dataset == "digit" else cfg.get("grid_processed_data_dir")
        self.use_audio = cfg.get("use_audio", False)
        self.batch_size = cfg.get("batch_size", 32)
        self.learning_rate = cfg.get("learning_rate", 0.0001)
        self.epochs = cfg.get("epochs", 100)
        # Frame budgets: GRID clips are 3 s @ 25 fps; digit clips are longer
        # (p95 = 112 frames at ~30 fps source rate).
        if self.dataset == "digit":
            self.max_frames = cfg.get("digit_max_frames", 125)
        else:
            self.max_frames = cfg.get("max_frames", 75)
        self.output_dir = cfg.get("output_dir", "./exp/no_audio")
        self.lip_backbone_pretrained = cfg.get("lip_backbone_pretrained", True)
        # Boundary-supervision losses
        self.lambda_ctc = cfg.get("lambda_ctc", 1.0)
        self.lambda_align = cfg.get("lambda_align", 1.0)
        self.alignment_temperature = cfg.get("alignment_temperature", 1.0)
        # Fallback fps for .lab second->frame conversion when meta.json is absent
        self.digit_source_fps = cfg.get("digit_source_fps", 29.87)

def collate_fn_factory(dataset_name, transcript_pool=None):
    converter = PhonemeConverter() if dataset_name != "digit" else None
    global_sampler = NegativeSampler(transcript_pool or [], dataset_name=dataset_name)

    digit_vocab = {str(i): i+1 for i in range(10)}
    digit_vocab["!"] = 11
    digit_vocab["<pad>"] = 0
    digit_vocab["<unk>"] = 12

    def collate_fn(batch):
        all_trans = [item['transcript'] for item in batch]
        sampler = global_sampler if len(global_sampler.all_transcripts) > 0 else NegativeSampler(all_trans, dataset_name=dataset_name)

        videos, phonemes_indices_list, labels, transcripts, audios, video_lens = [], [], [], [], [], []
        # Per-sample boundary times for the POSITIVE pair only (a negative
        # transcript contradicts the video, so it has no valid alignment).
        token_times_list, use_audio = [], 'audio' in batch[0]

        for item in batch:
            vid, trans = item['video'], item['transcript']
            if use_audio: aud = item['audio']

            # --- POSITIVE SAMPLE ---
            videos.append(vid)
            if dataset_name == "digit":
                pos_indices = [digit_vocab.get(d, digit_vocab["<unk>"]) for d in trans.split()]
                # digit_times: [(digit, start_frame, end_frame)] — one token each
                token_times = [
                    ([digit_vocab.get(d, digit_vocab["<unk>"])], fs, fe)
                    for d, fs, fe in item.get('digit_times') or []
                ]
            else:
                per_word = converter.text_to_phoneme_indices_per_word(trans)
                pos_indices = converter.flatten_per_word(per_word)
                word_times = item.get('word_times') or []
                if len(word_times) == len(per_word):
                    token_times = [(ids, fs, fe) for (_, ids), (_, fs, fe) in zip(per_word, word_times)]
                else:
                    token_times = []
            phonemes_indices_list.append(torch.tensor(pos_indices, dtype=torch.long))
            token_times_list.append(token_times)

            labels.append(1.0)
            transcripts.append(trans)
            video_lens.append(int(item.get('video_len', vid.shape[0])))
            if use_audio: audios.append(aud)

            # --- NEGATIVE SAMPLE ---
            videos.append(vid)
            neg_trans = sampler.sample(trans)
            if dataset_name == "digit":
                neg_indices = [digit_vocab.get(d, digit_vocab["<unk>"]) for d in neg_trans.split()]
            else:
                neg_indices = converter.flatten_per_word(converter.text_to_phoneme_indices_per_word(neg_trans))
            phonemes_indices_list.append(torch.tensor(neg_indices, dtype=torch.long))
            token_times_list.append([])

            labels.append(0.0)
            transcripts.append(neg_trans)
            video_lens.append(int(item.get('video_len', vid.shape[0])))
            if use_audio: audios.append(aud)

        videos = torch.stack(videos)
        labels = torch.tensor(labels, dtype=torch.float32)
        if use_audio: audios = torch.stack(audios)

        padded_phonemes = torch.nn.utils.rnn.pad_sequence(phonemes_indices_list, batch_first=True, padding_value=0)
        phoneme_padding_mask = padded_phonemes.eq(0)

        max_t = videos.shape[1]
        video_lens_t = torch.tensor(video_lens, dtype=torch.long)
        video_padding_mask = torch.arange(max_t).unsqueeze(0) >= video_lens_t.unsqueeze(1)

        # Alignment targets over the padded video/time grid. Samples without
        # usable boundary times keep has_align_target = 0 and contribute no
        # supervised frames.
        B, L = len(labels), padded_phonemes.shape[1]
        alignment_targets = torch.zeros(B, max_t, L)
        speech_mask = torch.zeros(B, max_t, dtype=torch.bool)
        has_align_target = torch.zeros(B)
        for i, token_times in enumerate(token_times_list):
            if not token_times:
                continue
            token_ranges = expand_token_ranges(token_times)
            targets, speech = build_alignment_targets(token_ranges, video_lens[i], L)
            alignment_targets[i, :video_lens[i]] = targets
            speech_mask[i, :video_lens[i]] = speech
            has_align_target[i] = 1.0

        batch_dict = {
            'video': videos,
            'phonemes': padded_phonemes,
            'phoneme_padding_mask': phoneme_padding_mask,
            'video_padding_mask': video_padding_mask,
            'label': labels,
            'transcript': transcripts,
            'alignment_targets': alignment_targets,
            'speech_mask': speech_mask,
            'has_align_target': has_align_target
        }
        if use_audio: batch_dict['audio'] = audios

        return batch_dict
    return collate_fn

# Default instances for tests
default_config = Config()
collate_fn = collate_fn_factory(default_config.dataset)
