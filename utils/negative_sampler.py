import random
import string

# GRID's fixed grammar swaps phonetically confusable words within the same
# grammatical class (color->color, letter->letter, digit-word->digit-word).
GRID_WORD_CLASSES = [
    {"bin", "lay", "place", "set"},                                  # commands
    {"blue", "green", "red", "white"},                               # colors
    {"at", "by", "in", "with"},                                      # prepositions
    set(string.ascii_lowercase) - {"w"},                             # GRID letters
    {"zero", "one", "two", "three", "four",
     "five", "six", "seven", "eight", "nine"},                       # digit words
    {"again", "now", "please", "soon"},                             # adverbs
]

class NegativeSampler:
    def __init__(self, all_transcripts, dataset_name="grid", rng=None):
        self.dataset_name = dataset_name
        self.rng = rng if rng is not None else random
        self.all_transcripts = [t for t in all_transcripts if isinstance(t, str) and t.strip()]

    def _random_other(self, positive_transcript, candidates):
        if not candidates:
            return positive_transcript
        neg = self.rng.choice(candidates)
        tries = 0
        while neg == positive_transcript and tries < 20:
            neg = self.rng.choice(candidates)
            tries += 1
        return neg

    def _token_overlap(self, a, b):
        a_set = set(a.split())
        b_set = set(b.split())
        if not a_set and not b_set:
            return 0.0
        denom = len(a_set | b_set)
        return (len(a_set & b_set) / denom) if denom > 0 else 0.0

    def _hard_negative(self, positive_transcript):
        # Approximate hard negative: pick the most lexically similar transcript
        # from a random candidate subset to keep this lightweight.
        candidates = [t for t in self.all_transcripts if t != positive_transcript]
        if not candidates:
            return positive_transcript

        sampled = self.rng.sample(candidates, k=min(64, len(candidates)))
        sampled.sort(key=lambda t: self._token_overlap(positive_transcript, t), reverse=True)
        return sampled[0]

    def _phoneme_similar(self, positive_transcript):
        words = positive_transcript.split()
        if not words:
            return positive_transcript

        if self.dataset_name == "digit":
            # Swap one digit for a different digit (identity is useless).
            idx = self.rng.randrange(len(words))
            words[idx] = self.rng.choice([d for d in map(str, range(10)) if d != words[idx]])
            return " ".join(words)

        idx = self.rng.randrange(len(words))
        word = words[idx].lower()
        same_class = next((c for c in GRID_WORD_CLASSES if word in c), None)
        if same_class:
            words[idx] = self.rng.choice(sorted(same_class - {word}))
        else:
            # Not in any GRID class (e.g. "please"): swap for another GRID word.
            pool = sorted(set().union(*GRID_WORD_CLASSES))
            words[idx] = self.rng.choice([w for w in pool if w != word])
        return " ".join(words)

    def word_substitution(self, positive_transcript):
        """Controlled replacements with known position-wise labels.

        Half are single substitutions, half change 2..N words. Some draws use
        the full vocabulary to cover out-of-grammar transcripts as well.
        """
        words = positive_transcript.split()
        if not words:
            raise ValueError("Cannot substitute words in an empty transcript")
        n = len(words)
        count = 1 if n == 1 or self.rng.random() < 0.5 else self.rng.randint(2, n)
        cross_class = self.rng.random() < 0.25
        vocabulary = set().union(*GRID_WORD_CLASSES)
        for pos in self.rng.sample(range(n), count):
            word = words[pos].lower()
            same_class = next((c for c in GRID_WORD_CLASSES if word in c), vocabulary)
            pool = vocabulary if cross_class else same_class
            words[pos] = self.rng.choice(sorted(pool - {word}))
        return " ".join(words)

    def sample(self, positive_transcript):
        strategy = self.rng.choice(['hard', 'wrong', 'shuffled', 'phoneme_similar', 'random'])
        candidates = self.all_transcripts

        if strategy == 'hard':
            neg = self._hard_negative(positive_transcript)

        elif strategy in ('wrong', 'random'):
            neg = self._random_other(positive_transcript, candidates)

        elif strategy == 'shuffled':
            words = positive_transcript.split()
            self.rng.shuffle(words)
            neg = " ".join(words)

        else:  # phoneme_similar
            neg = self._phoneme_similar(positive_transcript)

        # Global guard: a "negative" identical to the positive (e.g. shuffled
        # "7 7 7 7") would poison training with a mislabeled pair.
        if neg == positive_transcript:
            neg = self._random_other(positive_transcript, candidates)
        return neg
