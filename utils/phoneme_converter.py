import re
import nltk
# nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Full ARPABET vocabulary for English (GRID) transcripts: 39 phonemes +
# <pad> + <unk>. Digit transcripts use their own compact DIGIT_ARPABET
# (only the phones digits actually pronounce), so each dataset's text
# encoder embeddings and CTC head are sized to what it transcribes.
ARPABET = [
    "<pad>", "<unk>",
    "AA","AE","AH","AO","AW","AY",
    "B","CH","D","DH","EH","ER","EY",
    "F","G","HH","IH","IY","JH","K","L",
    "M","N","NG","OW","OY","P","R","S",
    "SH","T","TH","UH","UW","V","W","Y","Z","ZH"
]
GRID_VOCAB_SIZE = len(ARPABET)  # 41

class PhonemeConverter:
    def __init__(self):
        try:
            from g2p_en import G2p
            self.g2p = G2p()
        except ImportError:
            raise ImportError("pip install g2p_en")

        self.vocab = {p:i for i,p in enumerate(ARPABET)}

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


# Mandarin digits: toneless pinyin syllable -> ARPABET phone approximation.
# Initials map to their closest English phones (q->CH, j->JH, y->Y, w->W);
# finals decompose articulatorily (an->AA N, ing->IH NG, iu->IY OW,
# i->IY, u->UW, a->AA, er->ER). This replaces the old opaque 10-class digit
# vocabulary with pronunciation-level targets, mirroring what g2p_en does
# for English GRID transcripts.
DIGIT_PINYIN_PHONEMES = {
    "0": ("líng", ["L", "IH", "NG"]),
    "1": ("yī",   ["Y", "IY"]),
    "2": ("èr",   ["ER"]),
    "3": ("sān",  ["S", "AA", "N"]),
    "4": ("sì",   ["S", "IY"]),
    "5": ("wǔ",   ["W", "UW"]),
    "6": ("liù",  ["L", "IY", "OW"]),
    "7": ("qī",   ["CH", "IY"]),
    "8": ("bā",   ["B", "AA"]),
    "9": ("jiǔ",  ["JH", "IY", "OW"]),
}

# Compact digit-only vocab: <pad>/<unk> + exactly the phones the digit
# pronunciations above use (15 phones -> 17 tokens). Derived from the table
# (sorted for determinism) so it can never drift when a pronunciation
# changes; every mapped phone is in the vocab by construction.
DIGIT_ARPABET = [
    "<pad>", "<unk>",
    *sorted({p for _, phones in DIGIT_PINYIN_PHONEMES.values() for p in phones}),
]
DIGIT_VOCAB_SIZE = len(DIGIT_ARPABET)  # 17

class DigitPinyinConverter:
    """Digit transcripts -> ARPABET ids via toneless pinyin pronunciation.

    The digit dataset's counterpart of PhonemeConverter: "7" is pronounced
    qī and articulated as CH IY rather than treated as an opaque word class,
    so the text encoder and the CTC head see sub-lexical units the lips
    actually produce. Indexes into the compact digit-only vocabulary
    DIGIT_ARPABET (vocab_size 17: <pad>/<unk> + the 15 phones digits use).

    "!" (a pause marker: it carries its own .lab range but no articulation)
    maps to an EMPTY phoneme group — CTC blank absorbs the pause and the
    alignment loss gets no supervised frames there. Groups stay 1:1 with
    transcript tokens so they can be zipped against digit_times.
    """

    def __init__(self):
        self.vocab = {p: i for i, p in enumerate(DIGIT_ARPABET)}

    def token_to_phoneme_indices(self, token):
        """Phoneme ids for one transcript token ('0'-'9' or '!')."""
        entry = DIGIT_PINYIN_PHONEMES.get(token)
        return [self.vocab[p] for p in entry[1]] if entry else []

    def transcript_to_phoneme_indices_per_token(self, transcript):
        """Per-token [(token, [phoneme_ids])] — one group per transcript
        token, empty lists included, so the output zips 1:1 with digit_times."""
        return [(tok, self.token_to_phoneme_indices(tok)) for tok in transcript.split()]

    @staticmethod
    def flatten_per_token(per_token):
        """Concatenate per-token ids into the full target/encoder sequence."""
        return [idx for _, ids in per_token for idx in ids]

    @property
    def vocab_size(self):
        return len(self.vocab)