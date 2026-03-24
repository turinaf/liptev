import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=50, feature_dim=256):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, feature_dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, batch_first=True),
            num_layers=2
        )

    def forward(self, phoneme_seqs):
        # phoneme_seqs: B x L
        emb = self.embedding(phoneme_seqs) # B x L x D
        text_feats = self.transformer(emb) # B x L x D
        return text_feats