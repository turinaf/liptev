import torch
import torch.nn as nn
import torchvision.models as models

class LipEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super(LipEncoder, self).__init__()
        # 3D CNN approx: ResNet for spatial, followed by Transformer for temporal
        self.spatial_encoder = models.resnet18(pretrained=False)
        self.spatial_encoder.fc = nn.Linear(self.spatial_encoder.fc.in_features, feature_dim)
        
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, batch_first=True),
            num_layers=2
        )

    def forward(self, x):
        # x: B x T x C x H x W
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        spatial_feats = self.spatial_encoder(x) # (B*T) x D
        spatial_feats = spatial_feats.view(B, T, -1) # B x T x D
        
        temporal_feats = self.temporal_encoder(spatial_feats) # B x T x D
        return temporal_feats