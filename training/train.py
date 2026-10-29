from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.alignment import frame_token_targets

def alignment_kl_loss(sim_matrix, alignment_targets, frame_valid, temperature=1.0):
    """Frame->token CE against boundary targets.

    sim_matrix: B x T x L cosine similarities (padded positions already set
    to -1e9 by CrossModalAlignment); alignment_targets: B x T x L one-hot
    rows; frame_valid: B x T bool — speech frames of positive pairs that have
    boundary data, on non-padded video positions. Per-sample frame means are
    averaged over samples with targets so partial .lab coverage doesn't bias
    the loss.
    """
    log_probs = F.log_softmax(sim_matrix / temperature, dim=-1)
    frame_ce = -(alignment_targets * log_probs).sum(dim=-1)  # B x T

    per_sample = (frame_ce * frame_valid).sum(dim=1)
    counts = frame_valid.sum(dim=1)
    supervised = counts > 0
    if not supervised.any():
        return torch.zeros((), device=sim_matrix.device)
    return (per_sample[supervised] / counts[supervised]).mean()

def vfa_frame_ce_loss(frame_logits, frame_token_ids, frame_valid):
    """Frame-level token CE for the VFA posterior head. frame_valid gates the
    same speech/positive/has-boundary frames as the sim-matrix CE."""
    if frame_valid is not None and frame_valid.any():
        ids = frame_token_ids.clone()
        ids[~frame_valid] = -100
        return F.cross_entropy(frame_logits.reshape(-1, frame_logits.shape[-1]), ids.reshape(-1),
                               ignore_index=-100)
    return torch.zeros((), device=frame_logits.device)


def vfa_boundary_loss(bound_logits, boundary_targets, frame_valid):
    """BCE on per-frame token-transition logits, supervised frames only."""
    if frame_valid is not None and frame_valid.any():
        return F.binary_cross_entropy_with_logits(
            bound_logits[frame_valid], boundary_targets[frame_valid]
        )
    return torch.zeros((), device=bound_logits.device)


def word_verification_loss(word_logits, word_labels, word_valid):
    """Per-token verification BCE (on raw logits, same rationale as the global
    classifier). Supervised only where the collate emitted a token label and
    the pair had usable spans."""
    if word_valid is not None and word_valid.any():
        return F.binary_cross_entropy_with_logits(
            word_logits[word_valid], word_labels[word_valid]
        )
    return torch.zeros((), device=word_logits.device)


def train_one_epoch(model, dataloader, optimizer, device, epoch, writer=None,
                    lambda_ctc=1.0, lambda_align=1.0, alignment_temperature=1.0,
                    lambda_frame_ce=1.0, lambda_boundary=1.0, lambda_word_verif=1.0):
    model.train()
    # BCE on the raw logit: BCELoss on the post-sigmoid probability gets
    # dead gradients once the sigmoid saturates to exactly 0.0/1.0.
    bce_loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} Training")
    for batch_idx, batch in pbar:
        video = batch['video'].to(device)
        phonemes = batch['phonemes'].to(device)
        phoneme_padding_mask = batch.get('phoneme_padding_mask')
        if phoneme_padding_mask is not None:
            phoneme_padding_mask = phoneme_padding_mask.to(device)

        video_padding_mask = batch.get('video_padding_mask')
        if video_padding_mask is not None:
            video_padding_mask = video_padding_mask.to(device)

        labels = batch['label'].to(device) # 1 For match, 0 for negative samples
        positive_mask = labels == 1

        alignment_targets = batch.get('alignment_targets')
        if alignment_targets is not None:
            alignment_targets = alignment_targets.to(device)
        speech_mask = batch.get('speech_mask')
        if speech_mask is not None:
            speech_mask = speech_mask.to(device)
        has_align_target = batch.get('has_align_target')
        if has_align_target is not None:
            has_align_target = has_align_target.to(device)
        word_labels = batch.get('word_labels')
        if word_labels is not None:
            word_labels = word_labels.to(device)
        word_label_mask = batch.get('word_label_mask')
        if word_label_mask is not None:
            word_label_mask = word_label_mask.to(device)
        word_spans = batch.get('word_spans')
        if word_spans is not None:
            word_spans = word_spans.to(device)

        if 'audio' in batch:
            audio = batch['audio'].to(device)
            audio_padding_mask = batch.get('audio_padding_mask')
            if audio_padding_mask is not None:
                audio_padding_mask = audio_padding_mask.to(device)
        else:
            audio = None
            audio_padding_mask = None

        optimizer.zero_grad()
        consistency_score, align_out = model(
            video,
            phonemes,
            audio,
            video_padding_mask=video_padding_mask,
            phoneme_padding_mask=phoneme_padding_mask,
            audio_padding_mask=audio_padding_mask,
            word_spans=word_spans
        )

        # Binary Cross Entropy (maximize verification score) on raw logits
        loss_bce = bce_loss_fn(align_out['consistency_logit'].view(-1), labels)

        # Contrastive Logic via Margin objective
        align_score = align_out['alignment_score']
        # Maximize match score, minimize mismatch score
        loss_contrastive = torch.mean(
            labels * (1.0 - align_score) +
            (1.0 - labels) * torch.clamp(align_score - 0.5, min=0.0) # margin is implicit here
        )

        # CTC boundary supervision — positive pairs only: negatives pair the
        # video with a mismatched transcript, and aligning those would
        # contradict the verification objective.
        ctc_valid = align_out['ctc_valid'] & positive_mask
        if ctc_valid.any():
            loss_ctc = align_out['ctc_nll_per_token'][ctc_valid].mean()
        else:
            loss_ctc = torch.zeros((), device=device)

        # Frame-level alignment supervision — positive pairs with boundary
        # data, on speech frames only. sim_matrix lives under 'lip_text' when
        # audio is enabled and at the top level otherwise. Unavailable in
        # alignment mode "attn" (no cosine matrix is computed there).
        sim_source = align_out.get('lip_text', align_out)
        if alignment_targets is not None and speech_mask is not None and has_align_target is not None \
                and sim_source.get('sim_matrix') is not None:
            frame_valid = speech_mask & has_align_target.unsqueeze(1).bool() & positive_mask.unsqueeze(1)
            if video_padding_mask is not None:
                frame_valid = frame_valid & ~video_padding_mask
            loss_align = alignment_kl_loss(
                sim_source['sim_matrix'], alignment_targets, frame_valid,
                temperature=alignment_temperature
            )
        else:
            loss_align = torch.zeros((), device=device)

        # VFA supervision — same valid frames as the sim-matrix CE, plus the
        # per-token head on positives and single-substitution negatives.
        loss_frame_ce = torch.zeros((), device=device)
        loss_boundary = torch.zeros((), device=device)
        loss_word = torch.zeros((), device=device)
        if 'vfa_frame_logits' in align_out and alignment_targets is not None \
                and speech_mask is not None and has_align_target is not None:
            frame_valid = speech_mask & has_align_target.unsqueeze(1).bool() & positive_mask.unsqueeze(1)
            if video_padding_mask is not None:
                frame_valid = frame_valid & ~video_padding_mask
            if frame_valid.any():
                frame_ids, boundary_targets = frame_token_targets(alignment_targets, phonemes)
                loss_frame_ce = vfa_frame_ce_loss(align_out['vfa_frame_logits'], frame_ids, frame_valid)
                loss_boundary = vfa_boundary_loss(align_out['vfa_bound_logits'], boundary_targets, frame_valid)
        if 'word_verif_logits' in align_out and word_labels is not None and word_label_mask is not None:
            word_valid = word_label_mask & align_out['word_token_mask']
            loss_word = word_verification_loss(align_out['word_verif_logits'], word_labels, word_valid)

        loss = (loss_bce + loss_contrastive + lambda_ctc * loss_ctc + lambda_align * loss_align
                + lambda_frame_ce * loss_frame_ce + lambda_boundary * loss_boundary
                + lambda_word_verif * loss_word)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log to tensorboard step by step
        if writer is not None:
            global_step = (epoch - 1) * len(dataloader) + batch_idx
            writer.add_scalar('Loss/train_step_bce', loss_bce.item(), global_step)
            writer.add_scalar('Loss/train_step_contrastive', loss_contrastive.item(), global_step)
            writer.add_scalar('Loss/train_step_ctc', loss_ctc.item(), global_step)
            writer.add_scalar('Loss/train_step_align', loss_align.item(), global_step)
            writer.add_scalar('Loss/train_step_frame_ce', loss_frame_ce.item(), global_step)
            writer.add_scalar('Loss/train_step_boundary', loss_boundary.item(), global_step)
            writer.add_scalar('Loss/train_step_word_verif', loss_word.item(), global_step)
            writer.add_scalar('Loss/train_step_total', loss.item(), global_step)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "ctc": f"{loss_ctc.item():.4f}",
                          "align": f"{loss_align.item():.4f}", "word": f"{loss_word.item():.4f}"})

    return total_loss / len(dataloader)
