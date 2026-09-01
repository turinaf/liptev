import os
import string

os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")

import nltk
from nltk.corpus import cmudict
from g2p_en import G2p

# Avoid eager downloads during import: these resources are only needed for
# dictionary-backed phoneme lookups and should be fetched lazily when requested.
try:
    d = cmudict.dict()
except LookupError:
    d = {}


def word_to_phoneme(word):
    word = word.lower().strip(string.punctuation)
    
    if word in d:
        return d[word][0]
    else:
        phonemes = []
        for char in word:
            phonemes.append(char)
        return phonemes
g2p = G2p()

# print(word_to_phoneme("Thi is a sentence"))
print(g2p("Speech"))

