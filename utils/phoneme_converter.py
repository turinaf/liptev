class PhonemeConverter:
    def __init__(self):
        # A full vocabulary would map english phonemes. This represents characters/phonemes dict.
        self.vocab = {"<pad>": 0, "b": 1, "i": 2, "n": 3, "l": 4, "u": 5, "a": 6, "t": 7, "f": 8, "w": 9, "o": 10}
        
    def text_to_phonemes(self, text):
        return [self.vocab.get(c, 0) for c in text.replace(" ", "")]