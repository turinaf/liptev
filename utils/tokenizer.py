"""Transcript token IDs shared by training, evaluation and inference."""

import re

from utils.negative_sampler import GRID_WORD_CLASSES


class TranscriptTokenizer:
    def __init__(self, dataset, token_level="phoneme"):
        if dataset not in ("grid", "digit"):
            raise ValueError(f"Unknown dataset: {dataset}")
        if token_level not in ("word", "phoneme"):
            raise ValueError(f"Unknown token_level: {token_level}")
        self.dataset = dataset
        self.token_level = token_level
        self.converter = None
        if dataset == "digit":
            self.vocab = {str(i): i + 1 for i in range(10)}
            self.vocab.update({"!": 11, "<pad>": 0, "<unk>": 12})
        elif token_level == "word":
            words = sorted(set().union(*GRID_WORD_CLASSES))
            self.vocab = {"<pad>": 0, "<unk>": 1}
            self.vocab.update({word: i + 2 for i, word in enumerate(words)})
        else:
            from utils.phoneme_converter import PhonemeConverter
            self.converter = PhonemeConverter()
            self.vocab = self.converter.vocab

    def per_word(self, text):
        if self.converter is not None:
            return self.converter.text_to_phoneme_indices_per_word(text)
        words = text.split() if self.dataset == "digit" else re.findall(r"[a-z0-9]+", text.lower())
        if self.dataset == "grid":
            unknown = [word for word in words if word not in self.vocab]
            if unknown:
                raise ValueError(f"Words outside the GRID vocabulary: {', '.join(unknown)}")
        return [(word, [self.vocab.get(word, self.vocab["<unk>"])]) for word in words]

    def encode(self, text):
        return [idx for _, ids in self.per_word(text) for idx in ids]
