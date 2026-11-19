"""Read-only GRID split audit; saves analysis outputs, never changes training data."""

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLITS = ("train", "val", "test")


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def distribution_distance(left, right):
    keys = sorted(set(left) | set(right))
    a = np.array([left[k] for k in keys], dtype=float)
    b = np.array([right[k] for k in keys], dtype=float)
    a /= a.sum()
    b /= b.sum()
    return {
        "total_variation": float(np.abs(a - b).sum() / 2),
        "jensen_shannon_distance_base2": float(jensenshannon(a, b, base=2)),
    }


def read_transcript(annotation):
    return " ".join(
        line.split()[2]
        for line in annotation.read_text().splitlines()
        if len(line.split()) == 3 and line.split()[2] not in ("sil", "sp")
    )


def inventory(root, split_dir=None):
    rows = []
    summary = {}
    for split in SPLITS:
        manifest = (root if split_dir is None else split_dir) / f"{split}_split.json"
        paths = json.loads(manifest.read_text())
        missing, empty = [], []
        for path in paths:
            p = Path(path)
            speaker = p.parent.parent.name
            video = root / speaker / "video" / p.name
            annotation = root / speaker / "align" / (p.stem + ".align")
            if not video.is_file() or not annotation.is_file():
                missing.append(str(video))
                continue
            transcript = read_transcript(annotation)
            if not transcript:
                empty.append(str(video))
                continue
            rows.append({
                "split": split, "speaker": speaker, "video": str(video),
                "clip_id": f"{speaker}/{p.name}", "transcript": transcript,
            })
        part = [r for r in rows if r["split"] == split]
        summary[split] = {
            "manifest_sha256": digest(manifest), "manifest_entries": len(paths),
            "usable_entries": len(part), "missing_video_or_annotation": missing,
            "empty_transcripts": empty,
            "duplicate_clip_ids_within_split": len(part) - len({r["clip_id"] for r in part}),
            "speaker_counts": dict(sorted(Counter(r["speaker"] for r in part).items())),
            "unique_transcripts": len({r["transcript"] for r in part}),
        }
        print(f"Inventory {split}: {len(part)} usable clips", flush=True)
    overlap = {}
    for left, right in combinations(SPLITS, 2):
        a = [r for r in rows if r["split"] == left]
        b = [r for r in rows if r["split"] == right]
        a_speakers = {r["speaker"] for r in a}
        a_text = {r["transcript"] for r in a}
        overlap[f"{left}_vs_{right}"] = {
            "shared_clip_ids": len({r["clip_id"] for r in a} & {r["clip_id"] for r in b}),
            "shared_speakers": sorted(a_speakers & {r["speaker"] for r in b}),
            f"{right}_fraction_speakers_seen_in_{left}": float(np.mean([
                r["speaker"] in a_speakers for r in b])),
            f"{right}_fraction_transcripts_seen_in_{left}": float(np.mean([
                r["transcript"] in a_text for r in b])),
            "speaker_distribution_distance": distribution_distance(
                Counter(r["speaker"] for r in a), Counter(r["speaker"] for r in b)),
            "token_distribution_distance": distribution_distance(
                Counter(w for r in a for w in r["transcript"].split()),
                Counter(w for r in b for w in r["transcript"].split())),
        }
    return rows, {"splits": summary, "overlap": overlap}


def reuse_inventory(root, inventory_path, split_dir=None):
    report = json.loads(inventory_path.read_text())
    rows = []
    for split in SPLITS:
        manifest = (root if split_dir is None else split_dir) / f"{split}_split.json"
        summary = report["splits"][split]
        if digest(manifest) != summary["manifest_sha256"]:
            raise ValueError(f"{split} manifest has changed; rerun without --reuse-inventory")
        if summary["missing_video_or_annotation"] or summary["empty_transcripts"]:
            raise ValueError("Cannot reuse an inventory with missing or empty samples")
        for path in json.loads(manifest.read_text()):
            p = Path(path)
            speaker = p.parent.parent.name
            rows.append({
                "split": split, "speaker": speaker,
                "video": str(root / speaker / "video" / p.name),
                "clip_id": f"{speaker}/{p.name}",
            })
    print("Reused full inventory after checking split manifest hashes", flush=True)
    return rows, report


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No decoded frames: {path}")
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid frame rate for {path}: {fps}")
    return np.stack(frames), fps


def descriptors(frames):
    sampled = frames[np.linspace(0, len(frames) - 1, 12).astype(int)]
    gray = np.stack([
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (16, 16))
        for f in sampled
    ]).astype(float) / 255
    return np.concatenate([
        gray.mean(axis=0).ravel(), gray.std(axis=0).ravel(),
        sampled.mean(axis=(0, 1, 2)) / 255,
        sampled.std(axis=(0, 1, 2)) / 255,
        [np.abs(np.diff(gray, axis=0)).mean()],
    ])


def video_tensor(frames, max_frames):
    true_len = min(len(frames), max_frames)
    frames = frames[:max_frames]
    if len(frames) < max_frames:
        frames = np.concatenate([
            frames, np.repeat(frames[-1:], max_frames - len(frames), axis=0)])
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
    tensor = tensor.float().unsqueeze(0) / 127.5 - 1
    mask = torch.arange(max_frames).unsqueeze(0) >= true_len
    return tensor, mask


def cluster_analysis(features, rows, feature_kind):
    split = np.array([r["split"] for r in rows])
    speakers = np.array([r["speaker"] for r in rows])
    train = split == "train"
    if feature_kind == "visual":
        scaler = StandardScaler().fit(features[train])
        x = scaler.transform(features)
    else:
        x = normalize(features)
    pca = PCA(n_components=0.95, svd_solver="full").fit(x[train])
    x = pca.transform(x)
    report = {
        "samples": len(rows), "raw_dimensions": features.shape[1],
        "train_only_pca_dimensions": x.shape[1],
        "train_only_pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "plotted_pc1_pc2_explained_variance": float(pca.explained_variance_ratio_[:2].sum()),
        "clusters": {},
    }
    rng = np.random.default_rng(42)
    for k in (16, 33):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(x[train])
        labels = km.predict(x)
        null_nmi = [
            normalized_mutual_info_score(rng.permutation(speakers), labels)
            for _ in range(100)
        ]
        report["clusters"][str(k)] = {
            "silhouette": float(silhouette_score(x, labels)),
            "nmi_speaker": float(normalized_mutual_info_score(speakers, labels)),
            "nmi_speaker_permuted_median": float(np.median(null_nmi)),
            "nmi_speaker_permuted_p95": float(np.quantile(null_nmi, 0.95)),
            "nmi_split": float(normalized_mutual_info_score(split, labels)),
            "counts_by_split": {
                s: np.bincount(labels[split == s], minlength=k).tolist() for s in SPLITS},
        }
    nn = NearestNeighbors(n_neighbors=1).fit(x[train])
    report["nearest_train"] = {}
    for s in ("val", "test"):
        select = split == s
        distances, indices = nn.kneighbors(x[select])
        report["nearest_train"][s] = {
            "fraction_same_speaker": float(np.mean(
                speakers[train][indices[:, 0]] == speakers[select])),
            "median_distance": float(np.median(distances)),
            "same_speaker_uniform_baseline": 1 / len(set(speakers)),
        }
    # Each fold fits its own preprocessing; no PCA fitted outside the fold.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    split_prediction = cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)),
        features, split, cv=cv, scoring="balanced_accuracy", n_jobs=1)
    report["split_prediction_balanced_accuracy_5fold"] = split_prediction.tolist()
    report["split_prediction_chance"] = 1 / 3
    return report, x[:, :2], labels


def quantiles(values):
    return dict(zip(("min", "p05", "median", "p95", "max"),
                    map(float, np.quantile(values, [0, .05, .5, .95, 1]))))


def plot_timing_stress(rows, threshold, output):
    baseline = {
        row["clip_id"]: row["verification_score"]
        for row in rows if row["variant"] == "baseline"
    }
    variants = [
        ("simulate_30fps_no_resampling", "30 fps duplicated"),
        ("simulate_60fps_no_resampling", "60 fps duplicated"),
        ("prepend_1s_first_frame_then_truncate", "1 s lead-in"),
        ("horizontal_flip", "Horizontal flip"),
    ]
    fig, axes = plt.subplots(1, len(variants), figsize=(15, 5), sharey=True)
    for ax, (variant, title) in zip(axes, variants):
        for row in rows:
            if row["variant"] == variant:
                ax.plot((0, 1), (baseline[row["clip_id"]], row["verification_score"]),
                        marker="o", alpha=0.6, markersize=4)
        ax.axhline(threshold, color="black", linestyle="--")
        ax.set_yscale("log")
        ax.set_xticks((0, 1), ("Baseline", "Modified"))
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Verification score (log scale)")
    fig.suptitle("Synthetic perturbations of 10 matched GRID test clips; dashed line = validation threshold")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, help="Override the configured manifest directory")
    parser.add_argument("--checkpoint", type=Path, help="Override the configured best checkpoint")
    parser.add_argument("--samples-per-speaker-split", type=int, default=6)
    parser.add_argument("--reuse-inventory", action="store_true",
                        help="Reuse an existing full inventory after checking manifest hashes")
    args = parser.parse_args()
    if args.samples_per_speaker_split < 1:
        parser.error("--samples-per-speaker-split must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo))
    from models.verification_model import LipTextVerificationModel
    from models.lip_encoder import masked_mean
    from utils.checkpoint import checkpoint_tokenizer

    start = time.monotonic()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    config = json.loads(args.config.read_text())
    if config["dataset"] != "grid" or config.get("use_audio", False):
        raise ValueError("This audit implements video-only GRID analysis")
    root = Path(config["grid_processed_data_dir"])
    split_dir = args.split_dir or root
    if not split_dir.is_absolute():
        split_dir = args.repo / split_dir
    checkpoint_path = args.checkpoint or Path(config["output_dir"]) / "best_lip_verification_model.pth"
    checkpoint_hash = digest(checkpoint_path)
    calibration = json.loads((checkpoint_path.parent / "eer_thresholds.json").read_text())
    if calibration.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Calibration hash does not match checkpoint")
    threshold = calibration["verification"]["threshold"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tokenizer = checkpoint_tokenizer(checkpoint, "grid")
    model = LipTextVerificationModel(
        len(tokenizer.vocab), lip_backbone_pretrained=False,
        alignment_mode=checkpoint["alignment_mode"],
        dropout=checkpoint.get("dropout", 0),
        use_vfa=checkpoint.get("vfa_heads", False)).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    inventory_path = args.output / "inventory.json"
    if args.reuse_inventory:
        rows, report = reuse_inventory(root, inventory_path, split_dir)
    else:
        rows, report = inventory(root, split_dir)
        inventory_path.write_text(json.dumps(report, indent=2))
    speakers = sorted({r["speaker"] for r in rows})
    rng = random.Random(42)
    selected = []
    for split in SPLITS:
        for speaker in speakers:
            group = sorted(
                (r for r in rows if r["speaker"] == speaker and r["split"] == split),
                key=lambda r: r["clip_id"])
            selected.extend(rng.sample(group, min(len(group), args.samples_per_speaker_split)))
    visual, embeddings = [], []
    perturbations = []
    stress_speakers = set(random.Random(7).sample(speakers, 10))
    stressed = set()
    max_frames = config["max_frames"]
    with torch.inference_mode():
        for i, row in enumerate(selected):
            frames, fps = read_frames(row["video"])
            if "transcript" not in row:
                video_path = Path(row["video"])
                annotation = video_path.parent.parent / "align" / f"{video_path.stem}.align"
                row["transcript"] = read_transcript(annotation)
                if not row["transcript"]:
                    raise ValueError(f"Empty transcript: {annotation}")
            visual.append(descriptors(frames))
            video, mask = video_tensor(frames, max_frames)
            tokens = torch.tensor([tokenizer.encode(row["transcript"])])
            score, out = model(
                video, tokens, video_padding_mask=mask,
                phoneme_padding_mask=tokens.eq(0))
            embeddings.append(masked_mean(out["lip_feats"], mask)[0].numpy())
            row.update({
                "fps": float(fps), "frames": len(frames),
                "duration_s": float(len(frames) / fps),
                "width": frames.shape[2], "height": frames.shape[1],
                "verification_score": float(score.item()),
                "consistency_logit": float(out["consistency_logit"].item()),
                "ctc_nll_per_token": float(out["ctc_nll_per_token"].item()),
                "alignment_score": float(out["alignment_score"].item()),
            })
            if row["split"] == "test" and row["speaker"] in stress_speakers and row["speaker"] not in stressed:
                stressed.add(row["speaker"])
                # Synthetic repeated-frame upsampling isolates unhandled FPS
                # with identical content; it is not a real phone-video benchmark.
                variants = {
                    "baseline": frames,
                    "simulate_30fps_no_resampling": frames[
                        np.floor(np.arange(round(len(frames) * 30 / fps)) * fps / 30).astype(int).clip(0, len(frames) - 1)],
                    "simulate_60fps_no_resampling": frames[
                        np.floor(np.arange(round(len(frames) * 60 / fps)) * fps / 60).astype(int).clip(0, len(frames) - 1)],
                    "prepend_1s_first_frame_then_truncate": np.concatenate([
                        np.repeat(frames[:1], round(fps), axis=0), frames]),
                    "horizontal_flip": frames[:, :, ::-1],
                }
                for name, changed in variants.items():
                    if name == "baseline":
                        value = row["verification_score"]
                    else:
                        v, m = video_tensor(changed, max_frames)
                        sc, _ = model(v, tokens, video_padding_mask=m, phoneme_padding_mask=tokens.eq(0))
                        value = float(sc.item())
                    perturbations.append({
                        "clip_id": row["clip_id"], "variant": name,
                        "verification_score": value, "match": value >= threshold})
            if (i + 1) % 66 == 0:
                print(f"Extracted {i + 1}/{len(selected)} clips ({time.monotonic() - start:.1f}s)", flush=True)
    visual = np.stack(visual)
    embeddings = np.stack(embeddings)
    np.savez_compressed(args.output / "features.npz", visual=visual, embeddings=embeddings)
    report["method"] = {
        "dataset": "GRID", "seed": 42, "config": str(args.config),
        "root": str(root), "checkpoint": str(checkpoint_path),
        "split_dir": str(split_dir),
        "checkpoint_sha256": checkpoint_hash,
        "reused_inventory": args.reuse_inventory,
        "calibration": calibration,
        "samples_per_speaker_per_split": args.samples_per_speaker_split,
        "features": {
            "visual": "12 evenly spaced RGB frames: 16x16 mean/std grayscale, RGB mean/std, motion",
            "learned": "masked mean of full trained lip encoder output on first max_frames (training normalization)",
        },
        "fitting": "Standardization (visual), L2 normalization (learned), PCA(95%) and KMeans fitted on train only.",
        "limits": [
            "Sampled clustering is exploratory, not proof of shortcut learning.",
            "No new speaker is held out from the existing checkpoint's training.",
            "Positive-only sample scores measure false rejections, not AUC/EER or false accepts.",
            "No external user recording is available; synthetic changes do not identify its failure cause.",
            "Overlap checks identify shared manifest clip IDs, not byte/perceptual duplicates.",
            "When reusing inventory, manifest hashes are checked but all video and annotation contents are not revalidated.",
        ],
    }
    report["positive_sample_scores"] = {}
    for split in SPLITS:
        part = [r for r in selected if r["split"] == split]
        scores = [r["verification_score"] for r in part]
        report["positive_sample_scores"][split] = {
            "n": len(part), "score_quantiles": quantiles(scores),
            "false_rejection_rate_at_validation_threshold": float(np.mean(np.array(scores) < threshold)),
            "ctc_nll_quantiles": quantiles([r["ctc_nll_per_token"] for r in part]),
            "fps_counts": dict(Counter(r["fps"] for r in part)),
            "frame_count_counts": dict(Counter(r["frames"] for r in part)),
            "resolution_counts": dict(Counter(f"{r['width']}x{r['height']}" for r in part)),
        }
    report["timing_stress_test"] = {}
    for variant in sorted({p["variant"] for p in perturbations}):
        part = [p for p in perturbations if p["variant"] == variant]
        report["timing_stress_test"][variant] = {
            "n": len(part),
            "false_rejection_rate": float(np.mean([not p["match"] for p in part])),
            "score_quantiles": quantiles([p["verification_score"] for p in part]),
        }
    report["timing_stress_test_rows"] = perturbations
    plot_timing_stress(perturbations, threshold, args.output / "timing_sensitivity.png")
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    heatmap, heat_axes = plt.subplots(1, 2, figsize=(17, 11))
    split_labels = np.array([r["split"] for r in selected])
    speaker_labels = np.array([speakers.index(r["speaker"]) for r in selected])
    for index, (name, features) in enumerate((("visual", visual), ("learned", embeddings))):
        result, xy, labels = cluster_analysis(features, selected, name)
        report[f"{name}_clustering"] = result
        for row, point, label in zip(selected, xy, labels):
            row[f"{name}_pc1"] = float(point[0])
            row[f"{name}_pc2"] = float(point[1])
            row[f"{name}_cluster_33"] = int(label)
        for s, color in zip(SPLITS, ("tab:blue", "tab:orange", "tab:green")):
            sel = split_labels == s
            axes[index, 0].scatter(xy[sel, 0], xy[sel, 1], s=10, alpha=.55, c=color, label=s)
        axes[index, 0].legend()
        axes[index, 0].set_title(f"{name}: train-fitted PCA, colored by split")
        axes[index, 1].scatter(xy[:, 0], xy[:, 1], c=speaker_labels, cmap="gist_ncar", s=10, alpha=.7)
        axes[index, 1].set_title(f"{name}: same projection, colored by speaker")
        axes[index, 2].scatter(xy[:, 0], xy[:, 1], c=labels, cmap="gist_ncar", s=10, alpha=.7)
        axes[index, 2].set_title(f"{name}: 33 train-fitted KMeans clusters")
        counts = np.zeros((len(speakers), 33))
        for speaker, label in zip(speaker_labels, labels):
            counts[speaker, label] += 1
        counts /= counts.sum(axis=1, keepdims=True)
        image = heat_axes[index].imshow(counts, aspect="auto", interpolation="nearest", cmap="viridis")
        heat_axes[index].set_title(f"{name}: fraction of each speaker's samples")
        heat_axes[index].set_xlabel("KMeans cluster ID")
        heat_axes[index].set_yticks(range(len(speakers)), speakers)
        heat_axes[index].set_xticks(range(0, 33, 2))
        heatmap.colorbar(image, ax=heat_axes[index], label="Fraction of speaker's samples")
    for ax in axes.flat:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle("GRID split audit: train-fitted PCA projection; clusters use all retained PCA dimensions")
    fig.tight_layout()
    fig.savefig(args.output / "clustering.png", dpi=170)
    plt.close(fig)
    heatmap.suptitle("Speaker-to-cluster distribution across train, validation and test samples")
    heatmap.tight_layout()
    heatmap.savefig(args.output / "speaker_cluster_heatmap.png", dpi=170)
    plt.close(heatmap)
    with open(args.output / "sample_scores.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    report["runtime_seconds"] = time.monotonic() - start
    (args.output / "split_cluster_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({
        "overlap": report["overlap"],
        "positive_sample_scores": report["positive_sample_scores"],
        "timing_stress_test": report["timing_stress_test"],
        "visual_clustering": report["visual_clustering"],
        "learned_clustering": report["learned_clustering"],
        "runtime_seconds": report["runtime_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
