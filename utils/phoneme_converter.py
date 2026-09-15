import re
import nltk
# nltk.download('averaged_perceptron_tagger_eng', quiet=True)
class PhonemeConverter:
    def __init__(self):
        try:
            from g2p_en import G2p
            self.g2p = G2p()
        except ImportError:
            raise ImportError("pip install g2p_en")

        arpabet = [
            "<pad>", "<unk>",
            "AA","AE","AH","AO","AW","AY",
            "B","CH","D","DH","EH","ER","EY",
            "F","G","HH","IH","IY","JH","K","L",
            "M","N","NG","OW","OY","P","R","S",
            "SH","T","TH","UH","UW","V","W","Y","Z","ZH"
        ]

        self.vocab = {p:i for i,p in enumerate(arpabet)}

    def normalize_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text

    def remove_stress(self, phoneme):
        return ''.join([c for c in phoneme if not c.isdigit()])

    def text_to_phonemes(self, text):
        text = self.normalize_text(text)
        phonemes = self.g2p(text)

        indices = []

        for p in phonemes:
            base = self.remove_stress(p)
            indices.append(self.vocab.get(base, self.vocab["<unk>"]))

        return phonemes, indices

    def _phoneme_index(self, token):
        """Index of a g2p output token, or None if it is not an ARPABET phoneme.

        g2p passes spaces and punctuation through unchanged, so whole-sentence
        output contains non-phoneme tokens (the legacy text_to_phonemes path
        mapped every inter-word space to <unk>).
        """
        return self.vocab.get(self.remove_stress(token))

    def word_to_phoneme_indices(self, word):
        """Deterministic per-word phoneme ids (CMU-dict lookup for known words).

        Single source of truth for both the text-encoder input and the
        alignment-target token axis — the two must index identical phonemes.
        """
        phonemes = self.g2p(self.normalize_text(word))
        return [idx for p in phonemes if (idx := self._phoneme_index(p)) is not None]

    def text_to_phoneme_indices_per_word(self, text):
        """Per-word [(word, [phoneme_ids])] for a transcript."""
        words = self.normalize_text(text).split()
        return [(w, self.word_to_phoneme_indices(w)) for w in words]

    @staticmethod
    def flatten_per_word(per_word):
        """Concatenate per-word ids into the full target/encoder sequence."""
        return [idx for _, ids in per_word for idx in ids]

    @property
    def vocab_size(self):
        return len(self.vocab)