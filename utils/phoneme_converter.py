import os
import re

# This environment is behind a proxy that requires explicit opt-in for NLTK's
# proxied URL fetching. The project assumes the proxy is trusted and safe.
os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")

# Shared ARPABET vocabulary: 39 phonemes + <pad> + <unk>. GRID fills it from
# the CMU dict via g2p_en; digit transcripts index into the same list through
# their pinyin pronunciations — one phoneme space, one vocab_size (41) for
# both datasets.
ARPABET = [
    "<pad>", "<unk>",
    "AA","AE","AH","AO","AW","AY",
    "B","CH","D","DH","EH","ER","EY",
    "F","G","HH","IH","IY","JH","K","L",
    "M","N","NG","OW","OY","P","R","S",
    "SH","T","TH","UH","UW","V","W","Y","Z","ZH"
]
PHONEME_VOCAB_SIZE = len(ARPABET)  # 41

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
# i->IY, u->UW, a->AA, ao->AW, er->ER). This replaces the old opaque 10-class
# digit vocabulary with pronunciation-level targets, mirroring what g2p_en
# does for English GRID transcripts.
#
# "!" is the digit 1 read yāo — the number-string reading that keeps it
# distinct from qī (7) — so it is a real pronunciation, not a pause marker.
DIGIT_PINYIN_PHONEMES = {
    "0": ("líng", ["L", "IH", "NG"]),
    "1": ("yī",   ["Y", "IY"]),
    "!": ("yāo",  ["Y", "AW"]),
    "2": ("èr",   ["ER"]),
    "3": ("sān",  ["S", "AA", "N"]),
    "4": ("sì",   ["S", "IY"]),
    "5": ("wǔ",   ["W", "UW"]),
    "6": ("liù",  ["L", "IY", "OW"]),
    "7": ("qī",   ["CH", "IY"]),
    "8": ("bā",   ["B", "AA"]),
    "9": ("jiǔ",  ["JH", "IY", "OW"]),
}

class DigitPinyinConverter:
    """Digit transcripts -> ARPABET ids via toneless pinyin pronunciation.

    The digit dataset's counterpart of PhonemeConverter: "7" is pronounced
    qī and articulated as CH IY rather than treated as an opaque word class,
    so the text encoder and the CTC head see sub-lexical units the lips
    actually produce. Indexes into the shared ARPABET vocabulary (41 tokens,
    the same one GRID's CMU-dict/g2p_en path uses).

    "!" is the digit 1 read yāo (distinct from yī): it carries its own .lab
    range and contributes Y AW phones, so its frames are supervised like any
    other token's. Groups stay 1:1 with transcript tokens so they can be
    zipped against digit_times.
    """

    def __init__(self):
        self.vocab = {p: i for i, p in enumerate(ARPABET)}

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