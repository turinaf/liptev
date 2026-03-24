import random

class NegativeSampler:
    def __init__(self, all_transcripts):
        self.all_transcripts = all_transcripts
        
    def sample(self, positive_transcript):
        strategy = random.choice(['wrong', 'shuffled', 'phoneme_similar', 'random'])
        
        if strategy == 'wrong' or strategy == 'random':
            neg = random.choice(self.all_transcripts)
            while neg == positive_transcript:
                neg = random.choice(self.all_transcripts)
            return neg
            
        elif strategy == 'shuffled':
            words = positive_transcript.split()
            random.shuffle(words)
            return " ".join(words)
            
        elif strategy == 'phoneme_similar':
            # Dummy phoneme-similar replace
            words = positive_transcript.split()
            if len(words) > 0:
                words[1] = "red" if words[1] in ["blue", "green"] else "blue"
            return " ".join(words)
            
        return positive_transcript