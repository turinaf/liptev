from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import json
import os
import time

import numpy as np

def equal_error_rate(labels, scores):
    """EER and the threshold at which it occurs.

    Sweeps the ROC thresholds and picks the one where FAR (== FPR) is
    closest to FRR (== 1 - TPR); the EER is their mean at that point, so
    constant scores give chance level 0.5 and perfectly inverted scores
    give 1.0. Degenerate single-label inputs also return 0.5 (nothing is
    measurable), mirroring the 0.0 convention used for AUC.
    """
    labels_np = labels.cpu().numpy() if hasattr(labels, "cpu") else np.asarray(labels)
    scores_np = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)

    if len(set(labels_np)) < 2:
        return 0.5, float("nan")

    fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])

def compute_metrics(labels, consistency_scores, align_scores):
    labels_np = labels.cpu().numpy()
    cons_np = consistency_scores.cpu().numpy()
    align_np = align_scores.cpu().numpy()

    # Decisions (verify if >= 0.5)
    preds = (cons_np >= 0.5).astype(int)

    two_classes = len(set(labels_np)) > 1

    auc = roc_auc_score(labels_np, cons_np) if two_classes else 0.0
    acc = accuracy_score(labels_np, preds)

    # Threshold-free EER for the verification score, plus the accuracy at
    # the EER-optimal threshold: a fixed 0.5 cut is meaningless when the
    # score distribution is not calibrated to it (acc@0.5 stays near
    # chance while AUC is high).
    cons_eer, cons_eer_thr = equal_error_rate(labels_np, cons_np)
    if two_classes:
        acc_at_eer = accuracy_score(labels_np, (cons_np >= cons_eer_thr).astype(int))
    else:
        acc_at_eer = 0.0

    # The alignment score is a cosine-derived feature, not a calibrated
    # classifier: its [0,1] mapping puts raw cosine 0 at 0.5, and deep
    # features almost never sit below that. Judge it threshold-free (AUC,
    # EER) and only report accuracy at its own EER threshold.
    align_eer, align_eer_thr = equal_error_rate(labels_np, align_np)
    align_auc = roc_auc_score(labels_np, align_np) if two_classes else 0.0
    if two_classes:
        align_acc = accuracy_score(labels_np, (align_np >= align_eer_thr).astype(int))
    else:
        align_acc = 0.0

    return {
        "roc_auc": auc,
        "verification_accuracy": acc,
        "verification_eer": cons_eer,
        "verification_eer_threshold": cons_eer_thr,
        "verification_accuracy_at_eer": acc_at_eer,
        "alignment_roc_auc": align_auc,
        "alignment_eer": align_eer,
        "alignment_eer_threshold": align_eer_thr,
        "alignment_accuracy": align_acc,
        # Mean separation between class score means (the old elementwise
        # pos-neg subtraction crashed on unbalanced label counts).
        "score_margin": float(abs(cons_np[labels_np == 1].mean() - cons_np[labels_np == 0].mean())) if two_classes else 0.0
    }


def _jsonable(value):
    """NaN -> None so json.dump emits valid JSON (bare NaN is not parseable)."""
    return None if value != value else float(value)


def save_eer_thresholds(metrics, meta, path):
    """Persist the EER calibration so single-pair inference can reuse it.

    EER/Acc@EER are set-level metrics; a lone (video, transcript) trial has
    no distribution to sweep. Evaluate over a labeled set once, store the
    EER-optimal cuts here, and every later single-pair run decides
    MATCH/MISMATCH against them instead of the uncalibrated raw score.
    """
    payload = dict(meta)
    payload["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["verification"] = {
        "eer": _jsonable(metrics["verification_eer"]),
        "threshold": _jsonable(metrics["verification_eer_threshold"]),
        "accuracy_at_eer": _jsonable(metrics["verification_accuracy_at_eer"]),
        "roc_auc": _jsonable(metrics["roc_auc"]),
    }
    payload["alignment"] = {
        "eer": _jsonable(metrics["alignment_eer"]),
        "threshold": _jsonable(metrics["alignment_eer_threshold"]),
        "accuracy_at_eer": _jsonable(metrics["alignment_accuracy"]),
        "roc_auc": _jsonable(metrics["alignment_roc_auc"]),
    }

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_eer_thresholds(path):
    """Read an eer_thresholds.json, or None if absent/unparseable."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
