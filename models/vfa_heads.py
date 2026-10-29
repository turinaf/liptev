"""Forced-alignment and per-token verification heads (VFA method).

frame/boundary mirror Visual-Forced-Alignment's frame_linear/bound_linear:
per-frame token logits over [silence + vocab] and a per-frame token-transition
logit, both reading the shared lip-encoder features. WordVerificationHead
scores each token from its span-pooled lip features concatenated with the
text-encoder embedding — the per-token analogue of the global classifier, so
"which digit was wrong" becomes answerable.
"""

import torch
import torch.nn as nn


def pool_tokens(lip_feats, span_onehot):
    """Masked-mean pool lip features (B x T x D) over each token's span
    frames given a B x T x L one-hot. Empty spans give zero vectors."""
    counts = span_onehot.sum(dim=1).clamp(min=1.0)             # B x L
    return torch.bmm(span_onehot.transpose(1, 2), lip_feats) / counts.unsqueeze(-1)


class VFAHeads(nn.Module):
    def __init__(self, vocab_size, feature_dim, hidden=128):
        super().__init__()
        # Class 0 is silence (the <pad> id never appears in transcripts);
        # classes 1..V-1 are the model's token ids.
        self.frame_head = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(), nn.Linear(hidden, vocab_size)
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def frame(self, lip_feats):
        return self.frame_head(lip_feats)                      # B x T x V

    def boundary(self, lip_feats):
        return self.boundary_head(lip_feats).squeeze(-1)       # B x T


class WordVerificationHead(nn.Module):
    def __init__(self, feature_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, lip_token_pooled, text_feats, token_mask):
        logits = self.net(torch.cat([lip_token_pooled, text_feats], dim=-1)).squeeze(-1)
        return logits * token_mask.float()   # invalid tokens read as logit 0
