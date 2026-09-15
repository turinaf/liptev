import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAlignment(nn.Module):
    """Scores how well two modality feature sequences correspond.

    mode selects the scoring path:
      - "sim":  cosine-similarity matrix only (parameter-free interaction)
      - "attn": cross-attention only (text queries attend over lip frames;
                per-token score = cosine between text_feats and attn_out)
      - "both": average of the two raw scores before mapping to [0, 1]
    """

    VALID_MODES = ("sim", "attn", "both")

    def __init__(self, feature_dim=256, mode="sim"):
        super(CrossModalAlignment, self).__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        self.mode = mode
        # Cross attention block
        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)

    def forward(self, lip_feats, text_feats, lip_padding_mask=None, text_padding_mask=None):
        # lip_feats: B x T x D
        # text_feats: B x L x D

        if text_padding_mask is not None:
            valid_word_mask = (~text_padding_mask).float()
            denom = valid_word_mask.sum(dim=-1).clamp(min=1.0)
        else:
            valid_word_mask, denom = None, None

        def aggregate(word_scores):
            # Word-level scores -> per-sample raw score in [-1, 1]
            if valid_word_mask is not None:
                return (word_scores * valid_word_mask).sum(dim=-1) / denom
            return word_scores.mean(dim=-1)

        sim_raw = None
        attn_raw = None
        sim_matrix = None
        word_scores = None
        attn_weights = None
        attn_out = None

        if self.mode in ("sim", "both"):
            # Cosine similarity for word-level similarities
            lip_norm = F.normalize(lip_feats, dim=-1)
            text_norm = F.normalize(text_feats, dim=-1)
            sim_matrix = torch.bmm(lip_norm, text_norm.transpose(1, 2))  # B x T x L

            if lip_padding_mask is not None:
                sim_matrix = sim_matrix.masked_fill(lip_padding_mask.unsqueeze(-1), -1e9)
            if text_padding_mask is not None:
                sim_matrix = sim_matrix.masked_fill(text_padding_mask.unsqueeze(1), -1e9)

            # Word-level score derived from max similarities across lip frames
            word_scores, _ = sim_matrix.max(dim=1)  # B x L
            sim_raw = aggregate(word_scores)

        if self.mode in ("attn", "both"):
            attn_out, attn_weights = self.cross_attention(
                query=text_feats,
                key=lip_feats,
                value=lip_feats,
                key_padding_mask=lip_padding_mask
            )
            # Attention-weighted features vs. the queries that produced them
            attn_word_scores = F.cosine_similarity(text_feats, attn_out, dim=-1)  # B x L
            if text_padding_mask is not None:
                attn_word_scores = attn_word_scores * valid_word_mask
            attn_raw = aggregate(attn_word_scores)

        if self.mode == "sim":
            alignment_raw = sim_raw
        elif self.mode == "attn":
            alignment_raw = attn_raw
        else:
            alignment_raw = (sim_raw + attn_raw) / 2.0

        # Map cosine-derived value from [-1, 1] into [0, 1].
        alignment_score = ((alignment_raw + 1.0) / 2.0).clamp(0.0, 1.0)

        out = {
            "alignment_score": alignment_score,
            "sim_matrix": sim_matrix,
            "word_scores": word_scores,
            "attn_weights": attn_weights,
            "aligned_features": attn_out
        }
        if self.mode == "attn":
            out["word_scores"] = attn_word_scores
        return out