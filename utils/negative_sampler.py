import random
import string

# GRID's fixed grammar swaps phonetically confusable words within the same
# grammatical class (color->color, letter->letter, digit-word->digit-word).
GRID_WORD_CLASSES = [
    {"bin", "lay", "place", "set"},                                  # commands
    {"blue", "green", "red", "white"},                               # colors
    {"at", "by", "in", "with"},                                      # prepositions
    set(string.ascii_lowercase),                                     # letters
    {"zero", "one", "two", "three", "four",
     "five", "six", "seven", "eight", "nine"},                       # digit words
    {"again", "continue", "now", "ready", "soon", "stop"},           # adverbs
]

class NegativeSampler:
    def __init__(self, all_transcripts, dataset_name="grid"):
        self.dataset_name = dataset_name
        self.all_transcripts = [t for t in all_transcripts if isinstance(t, str) and t.strip()]

    def _random_other(self, positive_transcript, candidates):
        if not candidates:
            return positive_transcript
        neg = random.choice(candidates)
        tries = 0
        while neg == positive_transcript and tries < 20:
            neg = random.choice(candidates)
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

        sampled = random.sample(candidates, k=min(64, len(candidates)))
        sampled.sort(key=lambda t: self._token_overlap(positive_transcript, t), reverse=True)
        return sampled[0]

    def _phoneme_similar(self, positive_transcript):
        words = positive_transcript.split()
        if not words:
            return positive_transcript

        if self.dataset_name == "digit":
            # Swap one digit for a different digit (identity is useless).
            idx = random.randrange(len(words))
            words[idx] = random.choice([d for d in map(str, range(10)) if d != words[idx]])
            return " ".join(words)

        idx = random.randrange(len(words))
        word = words[idx].lower()
        same_class = next((c for c in GRID_WORD_CLASSES if word in c), None)
        if same_class:
            words[idx] = random.choice(sorted(same_class - {word}))
        else:
            # Not in any GRID class (e.g. "please"): swap for another GRID word.
            pool = sorted(set().union(*GRID_WORD_CLASSES))
            words[idx] = random.choice([w for w in pool if w != word])
        return " ".join(words)

    def sample(self, positive_transcript):
        strategy = random.choice(['hard', 'wrong', 'shuffled', 'phoneme_similar', 'random'])
        candidates = self.all_transcripts

        if strategy == 'hard':
            neg = self._hard_negative(positive_transcript)

        elif strategy in ('wrong', 'random'):
            neg = self._random_other(positive_transcript, candidates)

        elif strategy == 'shuffled':
            words = positive_transcript.split()
            random.shuffle(words)
            neg = " ".join(words)

        else:  # phoneme_similar
            neg = self._phoneme_similar(positive_transcript)

        # Global guard: a "negative" identical to the positive (e.g. shuffled
        # "7 7 7 7") would poison training with a mislabeled pair.
        if neg == positive_transcript:
            neg = self._random_other(positive_transcript, candidates)
        return neg
