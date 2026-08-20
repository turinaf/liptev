import torch
import torch.nn as nn
import torch.nn.functional as F
from .lip_encoder import LipEncoder, masked_mean
from .text_encoder import TextEncoder
from .cross_modal_alignment import CrossModalAlignment
from .audio_encoder import AudioEncoder

class LipTextVerificationModel(nn.Module):
    def __init__(self, vocab_size=50, feature_dim=256, lip_backbone_pretrained=True):
        super(LipTextVerificationModel, self).__init__()
        self.vocab_size = vocab_size
        self.lip_encoder = LipEncoder(feature_dim, backbone_pretrained=lip_backbone_pretrained)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, feature_dim=feature_dim)
        self.audio_encoder = AudioEncoder(feature_dim)

        self.alignment_lip_text = CrossModalAlignment(feature_dim)
        self.alignment_audio_text = CrossModalAlignment(feature_dim)
        self.alignment_lip_audio = CrossModalAlignment(feature_dim)

        # CTC predictor: per-frame token logits over lip features. blank is
        # appended after the vocab (indices 0..vocab_size-1 are real tokens),
        # so the head emits vocab_size + 1 classes. The lip encoder preserves
        # the frame count, so input_lengths = video lengths.
        self.ctc_head = nn.Linear(feature_dim, vocab_size + 1)

        # feature_dim*3 (pooled lip/text/audio) + 3 alignment scores + 1
        # per-token CTC log-likelihood.
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 3 + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _ctc_terms(self, lip_feats, phoneme_seqs, video_padding_mask, phoneme_padding_mask):
        """Per-sample CTC NLL normalized per target token.

        Computed for every pair (for negatives the log-likelihood is naturally
        low — that is what makes the classifier alignment-aware); the training
        loop masks it to positive pairs before using it as a loss. Per-token
        normalization keeps the classifier feature bounded (~[-10, 0])
        regardless of transcript length.
        """
        B, T, _ = lip_feats.shape
        log_probs = F.log_softmax(self.ctc_head(lip_feats), dim=-1)  # B x T x (V+1)

        input_lengths = (~video_padding_mask).sum(dim=1) if video_padding_mask is not None \
            else torch.full((B,), T, dtype=torch.long, device=lip_feats.device)
        target_lengths = (~phoneme_padding_mask).sum(dim=1) if phoneme_padding_mask is not None \
            else torch.full((B,), phoneme_seqs.shape[1], dtype=torch.long, device=lip_feats.device)

        nll = torch.zeros(B, device=lip_feats.device)
        valid = input_lengths >= target_lengths
        valid &= target_lengths > 0
        if valid.any():
            nll[valid] = F.ctc_loss(
                log_probs[valid].transpose(0, 1),        # T x B x (V+1)
                phoneme_seqs[valid],
                input_lengths[valid],
                target_lengths[valid],
                blank=self.vocab_size,
                reduction='none',
                zero_infinity=True
            )
        return nll / target_lengths.clamp(min=1), valid

    @staticmethod
    def _mel_domain_mask(audio_padding_mask, T_mel, hop_length):
        """Convert a sample-domain audio padding mask (True = padded) to the
        mel-frame domain: mel frame t is valid iff its window starts inside
        the real audio (t * hop < audio_len)."""
        audio_lens = (~audio_padding_mask).sum(dim=1).clamp(min=1)
        positions = torch.arange(T_mel, device=audio_padding_mask.device)
        return positions.unsqueeze(0) * hop_length >= audio_lens.unsqueeze(1)

    def forward(self, video_frames, phoneme_seqs, audio_features=None, video_padding_mask=None,
                phoneme_padding_mask=None, audio_padding_mask=None):
        lip_feats = self.lip_encoder(video_frames, padding_mask=video_padding_mask)
        text_feats = self.text_encoder(phoneme_seqs, padding_mask=phoneme_padding_mask)

        ctc_nll_per_token, ctc_valid = self._ctc_terms(
            lip_feats, phoneme_seqs, video_padding_mask, phoneme_padding_mask
        )
        # Classifier feature: per-token log-likelihood (negative, bounded).
        ctc_logp_per_token = -ctc_nll_per_token.unsqueeze(-1)

        if audio_features is not None:
            if audio_padding_mask is not None:
                # Mel length is deterministic (torchaudio MelSpectrogram uses
                # center=True): T_mel = samples // hop + 1.
                hop = self.audio_encoder.hop_length()
                T_mel = audio_features.shape[-1] // hop + 1
                audio_mel_mask = self._mel_domain_mask(audio_padding_mask, T_mel, hop)
            else:
                audio_mel_mask = None
            audio_feats = self.audio_encoder(audio_features, padding_mask=audio_mel_mask)

            # Alignments
            align_lt = self.alignment_lip_text(
                lip_feats,
                text_feats,
                lip_padding_mask=video_padding_mask,
                text_padding_mask=phoneme_padding_mask
            )
            align_at = self.alignment_audio_text(
                audio_feats,
                text_feats,
                lip_padding_mask=audio_mel_mask,
                text_padding_mask=phoneme_padding_mask
            )
            align_la = self.alignment_lip_audio(
                lip_feats,
                audio_feats,
                lip_padding_mask=video_padding_mask,
                text_padding_mask=audio_mel_mask
            )

            lip_pool = masked_mean(lip_feats, video_padding_mask)
            text_pool = masked_mean(text_feats, phoneme_padding_mask)
            audio_pool = masked_mean(audio_feats, audio_mel_mask)

            score_lt = align_lt["alignment_score"].unsqueeze(-1)
            score_at = align_at["alignment_score"].unsqueeze(-1)
            score_la = align_la["alignment_score"].unsqueeze(-1)

            # Combine features
            combined = torch.cat([lip_pool, text_pool, audio_pool, score_lt, score_at, score_la, ctc_logp_per_token], dim=-1)

            consistency_score = self.classifier(combined)

            # Weighted alignment score
            final_align_score = 0.4 * align_lt["alignment_score"] + 0.4 * align_at["alignment_score"] + 0.2 * align_la["alignment_score"]
            align_out = {
                "alignment_score": final_align_score,
                "lip_text": align_lt,
                "audio_text": align_at,
                "lip_audio": align_la
            }
        else:
            align_lt = self.alignment_lip_text(
                lip_feats,
                text_feats,
                lip_padding_mask=video_padding_mask,
                text_padding_mask=phoneme_padding_mask
            )

            lip_pool = masked_mean(lip_feats, video_padding_mask)
            text_pool = masked_mean(text_feats, phoneme_padding_mask)
            audio_pool = torch.zeros_like(lip_pool)

            score_lt = align_lt["alignment_score"].unsqueeze(-1)
            score_at = torch.zeros_like(score_lt)
            score_la = torch.zeros_like(score_lt)

            combined = torch.cat([lip_pool, text_pool, audio_pool, score_lt, score_at, score_la, ctc_logp_per_token], dim=-1)
            consistency_score = self.classifier(combined)

            align_out = align_lt

        align_out["ctc_nll_per_token"] = ctc_nll_per_token
        align_out["ctc_valid"] = ctc_valid

        return consistency_score, align_out
