import torch
import torch.nn as nn
import torchaudio

class AudioEncoder(nn.Module):
    def __init__(self, feature_dim=256, sample_rate=16000):
        super(AudioEncoder, self).__init__()
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=80
        )
        
        # assuming MFCC or Mel-spectrogram input, simple linear projection + transformer
        self.proj = nn.Linear(80, feature_dim) # example audio feature size 80
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, batch_first=True),
            num_layers=2
        )

    def forward(self, audio_waveform):
        # audio_waveform: B x T_audio or B x 1 x T_audio
        mel_spec = self.mel_transform(audio_waveform)
        # Apply log to avoid log(0) issues
        log_mel_spec = torch.log(mel_spec + 1e-6)
        
        # log_mel_spec: B x (1 x) 80 x T_mel
        # If input has a channel dimension, remove it
        if log_mel_spec.dim() == 4:
            log_mel_spec = log_mel_spec.squeeze(1)
            
        # Transpose: B x T_mel x 80
        audio_features = log_mel_spec.transpose(1, 2)
        
        # audio_features: B x T2 x 80
        proj_feats = self.proj(audio_features)
        audio_feats = self.transformer(proj_feats)
        return audio_feats
