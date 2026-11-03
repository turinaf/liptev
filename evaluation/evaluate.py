import torch
import torch.distributed as dist
from utils.viterbi import forced_token_spans, spans_to_onehot

def forced_word_scores(model, align_out, batch, boundary_threshold):
    """Score inference-decoded spans, never ground-truth pooling spans."""
    tokens = batch["phonemes"]
    B, L = tokens.shape
    T = batch["video"].shape[1]
    weights = torch.zeros(B, T, L)
    usable = torch.zeros(B, L, dtype=torch.bool)
    frame_logits = align_out["vfa_frame_logits"].detach().cpu()
    bounds = torch.sigmoid(align_out["vfa_bound_logits"]).detach().cpu().numpy()
    for row in range(B):
        length = int((~batch["video_padding_mask"][row]).sum())
        ids = tokens[row][~batch["phoneme_padding_mask"][row]].tolist()
        if not ids:
            continue
        spans, fallback = forced_token_spans(
            frame_logits[row, :length], ids, bounds[row, :length], boundary_threshold)
        weights[row] = spans_to_onehot(spans, T, L)
        if not fallback:
            usable[row] = weights[row].sum(dim=0) > 0
    scorer = model.module if hasattr(model, "module") else model
    logits, _ = scorer.score_word_spans(
        align_out["lip_feats"], align_out["text_feats"],
        weights.to(align_out["lip_feats"].device))
    # Exclude words with no annotated geometry and fallback alignments.
    annotated = batch["word_spans"].sum(dim=1) > 0
    valid = batch["word_label_mask"] & annotated & usable
    return torch.sigmoid(logits).detach().cpu()[valid], batch["word_labels"][valid]


def evaluate_model(model, dataloader, device, boundary_threshold=0.8, word_metrics=False):
    model.eval()
    all_scores, all_aligns, all_labels = [], [], []
    word_scores, word_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            phonemes = batch['phonemes'].to(device)
            labels = batch['label'].to(device)

            audio = batch.get('audio')
            if audio is not None:
                audio = audio.to(device)

            audio_padding_mask = batch.get('audio_padding_mask')
            if audio_padding_mask is not None:
                audio_padding_mask = audio_padding_mask.to(device)

            phoneme_padding_mask = batch.get('phoneme_padding_mask')
            if phoneme_padding_mask is not None:
                phoneme_padding_mask = phoneme_padding_mask.to(device)

            video_padding_mask = batch.get('video_padding_mask')
            if video_padding_mask is not None:
                video_padding_mask = video_padding_mask.to(device)

            consistency_score, align_out = model(
                video,
                phonemes,
                audio,
                video_padding_mask=video_padding_mask,
                phoneme_padding_mask=phoneme_padding_mask,
                audio_padding_mask=audio_padding_mask
            )
            
            all_scores.append(consistency_score.view(-1))
            all_aligns.append(align_out["alignment_score"])
            all_labels.append(labels)
            if word_metrics:
                if "vfa_frame_logits" not in align_out:
                    raise ValueError("Word metrics require a checkpoint with VFA heads")
                scores, targets = forced_word_scores(model, align_out, batch, boundary_threshold)
                word_scores.append(scores)
                word_labels.append(targets)
            
    all_scores = torch.cat(all_scores)
    all_aligns = torch.cat(all_aligns)
    all_labels = torch.cat(all_labels)
    token_scores = torch.cat(word_scores) if word_scores else torch.empty(0)
    token_labels = torch.cat(word_labels) if word_labels else torch.empty(0)
    if dist.is_available() and dist.is_initialized():
        local = (all_scores.cpu(), all_aligns.cpu(), all_labels.cpu(), token_scores, token_labels)
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        all_scores = torch.cat([part[0] for part in gathered])
        all_aligns = torch.cat([part[1] for part in gathered])
        all_labels = torch.cat([part[2] for part in gathered])
        token_scores = torch.cat([part[3] for part in gathered])
        token_labels = torch.cat([part[4] for part in gathered])
    
    from evaluation.metrics import compute_metrics
    metrics = compute_metrics(all_labels, all_scores, all_aligns)
    if word_metrics:
        from evaluation.metrics import compute_word_metrics
        metrics.update(compute_word_metrics(token_labels, token_scores))
        print(
            f"Words (forced spans) - ROC AUC: {metrics['word_roc_auc']:.4f}, "
            f"EER: {metrics['word_eer']:.4f}, "
            f"positive/negative: {metrics['word_positive_count']}/{metrics['word_negative_count']}"
        )
    
    print(
        f"Verify - ROC AUC: {metrics['roc_auc']:.4f}, Acc@0.5: {metrics['verification_accuracy']:.4f}, "
        f"EER: {metrics['verification_eer']:.4f}, Acc@EER: {metrics['verification_accuracy_at_eer']:.4f}"
    )
    print(
        f"Align  - ROC AUC: {metrics['alignment_roc_auc']:.4f}, EER: {metrics['alignment_eer']:.4f}, "
        f"Acc@EER: {metrics['alignment_accuracy']:.4f}"
    )
    return metrics
