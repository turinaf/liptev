import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAlignment(nn.Module):
    def __init__(self, feature_dim=256):
        super(CrossModalAlignment, self).__init__()
        # Cross attention block
        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)

    def forward(self, lip_feats, text_feats):
        # lip_feats: B x T x D
        # text_feats: B x L x D
        
        # Cosine similarity for word-level similarities 
        lip_norm = F.normalize(lip_feats, dim=-1)
        text_norm = F.normalize(text_feats, dim=-1)
        sim_matrix = torch.bmm(lip_norm, text_norm.transpose(1, 2)) # B x T x L
        
        attn_out, attn_weights = self.cross_attention(query=text_feats, key=lip_feats, value=lip_feats)
        
        # Word-level score derived from max similarities across lip frames
        word_scores, _ = sim_matrix.max(dim=1) # B x L
        
        # Sentence score / overall alignment score
        alignment_score = word_scores.mean(dim=-1) # B
        
        return {
            "alignment_score": alignment_score,
            "word_scores": word_scores,
            "sim_matrix": sim_matrix,
            "attn_weights": attn_weights,
            "aligned_features": attn_out
        }