import torch
import torch.nn as nn
from .lip_encoder import LipEncoder
from .text_encoder import TextEncoder
from .cross_modal_alignment import CrossModalAlignment

class LipTextVerificationModel(nn.Module):
    def __init__(self, vocab_size=50, feature_dim=256):
        super(LipTextVerificationModel, self).__init__()
        self.lip_encoder = LipEncoder(feature_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, feature_dim=feature_dim)
        self.alignment = CrossModalAlignment(feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, video_frames, phoneme_seqs):
        lip_feats = self.lip_encoder(video_frames)
        text_feats = self.text_encoder(phoneme_seqs)
        
        # Only does verification/alignment, no reading or visual speech gen.
        align_out = self.alignment(lip_feats, text_feats)
        
        lip_pool = lip_feats.mean(dim=1)
        text_pool = text_feats.mean(dim=1)
        # B x 1 alignment score feature
        align_score = align_out["alignment_score"].unsqueeze(-1)
        
        combined = torch.cat([lip_pool, text_pool, align_score], dim=-1)
        # B x 1 Verification Output 0 - 1
        consistency_score = self.classifier(combined)
        
        return consistency_score, align_out
