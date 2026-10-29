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
        # MediaPipe face landmarker used to crop lip regions (pipeline + inference)
        self.mp_model_path = cfg.get("mp_model_path", "face_landmarker.task")
        # Boundary-supervision losses
        self.lambda_ctc = cfg.get("lambda_ctc", 1.0)
        self.lambda_align = cfg.get("lambda_align", 1.0)
        self.alignment_temperature = cfg.get("alignment_temperature", 1.0)
        # Alignment scoring path: "sim" (cosine matrix), "attn"
        # (cross-attention), or "both" (average)
        self.alignment_mode = cfg.get("alignment_mode", "sim")
        # Fallback fps for .lab second->frame conversion when meta.json is absent
        self.digit_source_fps = cfg.get("digit_source_fps", 29.87)

def substitution_position(pos_words, neg_words):
    """Index of the single differing token between two same-length token
    lists, or None (identical, different lengths, or >1 differences). Used to
    decide whether a negative pair admits per-token supervision: exactly one
    substituted token means every other token is still articulated by the
    video, so "which token is wrong" is well defined."""
    if len(pos_words) != len(neg_words):
        return None
    diffs = [i for i, (p, n) in enumerate(zip(pos_words, neg_words)) if p != n]
    return diffs[0] if len(diffs) == 1 else None


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

        videos, phonemes_indices_list, labels, transcripts, audios, audio_lens, video_lens = [], [], [], [], [], [], []
        # Per-sample boundary times for the POSITIVE pair only (a negative
        # transcript contradicts the video, so it has no valid alignment).
        token_times_list, use_audio = [], 'audio' in batch[0]
        # Per-token verification supervision: boundary times and 0/1 labels
        # for pairs where per-token supervision is meaningful.
        word_token_times_list, word_label_rows = [], []

        for item in batch:
            vid, trans = item['video'], item['transcript']
            if use_audio: aud = item['audio']

            # --- POSITIVE SAMPLE ---
            videos.append(vid)
            if dataset_name == "digit":
                pos_indices = [digit_vocab.get(d, digit_vocab["<unk>"]) for d in trans.split()]
                # digit_times: [(token, start_frame, end_frame)] over the full
                # token sequence ("!" tokens carry their own .lab ranges)
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

            # Per-token verification supervision (positives: every token 1)
            word_token_times_list.append(token_times)
            n_tok = len(pos_indices)
            word_label_rows.append(torch.ones(n_tok))

            labels.append(1.0)
            transcripts.append(trans)
            video_lens.append(int(item.get('video_len', vid.shape[0])))
            if use_audio:
                audios.append(aud)
                audio_lens.append(int(item.get('audio_len', aud.shape[1])))

            # --- NEGATIVE SAMPLE ---
            videos.append(vid)
            neg_trans = sampler.sample(trans)
            if dataset_name == "digit":
                neg_indices = [digit_vocab.get(d, digit_vocab["<unk>"]) for d in neg_trans.split()]
            else:
                neg_indices = converter.flatten_per_word(converter.text_to_phoneme_indices_per_word(neg_trans))
            phonemes_indices_list.append(torch.tensor(neg_indices, dtype=torch.long))

            # Per-token supervision: only single-substitution negatives have
            # a well-defined "wrong token" (others: masked). Spans copy the
            # positive's geometry with the substituted token ids swapped in.
            pos_words = trans.split() if dataset_name == "digit" \
                else [w for w, _ in (converter.text_to_phoneme_indices_per_word(trans))]
            neg_words = neg_trans.split() if dataset_name == "digit" \
                else [w for w, _ in (converter.text_to_phoneme_indices_per_word(neg_trans))]
            sub_pos = substitution_position(pos_words, neg_words) if token_times else None

            if sub_pos is not None:
                if dataset_name == "digit":
                    # digit token_times: [([id], fs, fe)] per position
                    neg_times = [list(t) for t in token_times]
                    neg_times[sub_pos] = [neg_indices[sub_pos]], neg_times[sub_pos][1], neg_times[sub_pos][2]
                    word_token_times_list.append(neg_times)
                    labels_row = torch.ones(len(neg_indices))
                    labels_row[sub_pos] = 0.0
                    word_label_rows.append(labels_row)
                else:
                    # GRID token_times: [(ids, fs, fe)] per word; the swapped
                    # word's phonemes get the positive word's frame range.
                    if len(neg_words) == len(token_times):
                        neg_per_word = converter.text_to_phoneme_indices_per_word(neg_trans)
                        neg_times = [list(t) for t in token_times]
                        _, fs, fe = token_times[sub_pos]
                        neg_times[sub_pos] = (neg_per_word[sub_pos][1], fs, fe)
                        word_token_times_list.append(neg_times)
                        labels_row = torch.ones(len(neg_indices))
                        off = sum(len(t[0]) for t in token_times[:sub_pos])
                        for k in range(len(neg_per_word[sub_pos][1])):
                            labels_row[off + k] = 0.0
                        word_label_rows.append(labels_row)
                    else:
                        word_token_times_list.append([])
                        word_label_rows.append(None)
                        sub_pos = None
            else:
                word_token_times_list.append([])
                word_label_rows.append(None)
            token_times_list.append([])

            labels.append(0.0)
            transcripts.append(neg_trans)
            video_lens.append(int(item.get('video_len', vid.shape[0])))
            if use_audio:
                audios.append(aud)
                audio_lens.append(int(item.get('audio_len', aud.shape[1])))

        videos = torch.stack(videos)
        labels = torch.tensor(labels, dtype=torch.float32)
        if use_audio:
            max_audio = max(a.shape[1] for a in audios)
            audios = torch.stack([
                torch.nn.functional.pad(a, (0, max_audio - a.shape[1])) for a in audios
            ])
            audio_lens_t = torch.tensor(audio_lens, dtype=torch.long)
            audio_padding_mask = torch.arange(max_audio).unsqueeze(0) >= audio_lens_t.unsqueeze(1)
        else:
            audio_padding_mask = None

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

        # Per-token verification supervision tensors. word_spans mirrors
        # alignment_targets but is also built for substitution negatives
        # (whose frames still articulate every token except the swap).
        word_spans = torch.zeros(B, max_t, L)
        word_labels = torch.zeros(B, L)
        word_label_mask = torch.zeros(B, L, dtype=torch.bool)
        for i, (wt, row) in enumerate(zip(word_token_times_list, word_label_rows)):
            if not wt or row is None:
                continue
            ranges = expand_token_ranges(wt)
            spans_t, _ = build_alignment_targets(ranges, video_lens[i], L)
            word_spans[i, :video_lens[i]] = spans_t
            word_labels[i, :row.shape[0]] = row
            word_label_mask[i, :row.shape[0]] = True

        batch_dict = {
            'video': videos,
            'phonemes': padded_phonemes,
            'phoneme_padding_mask': phoneme_padding_mask,
            'video_padding_mask': video_padding_mask,
            'label': labels,
            'transcript': transcripts,
            'alignment_targets': alignment_targets,
            'speech_mask': speech_mask,
            'has_align_target': has_align_target,
            'word_labels': word_labels,
            'word_label_mask': word_label_mask,
            'word_spans': word_spans
        }
        if use_audio:
            batch_dict['audio'] = audios
            batch_dict['audio_padding_mask'] = audio_padding_mask

        return batch_dict
    return collate_fn

# Default instances for tests
default_config = Config()
collate_fn = collate_fn_factory(default_config.dataset)
