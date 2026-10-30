# Lip-to-Text Verification

Given a silent video of a speaker and a transcript, score how well the lip movements match the text (0–1). This is a verification system, not lip reading: it does not generate text, and the score measures visual-text consistency rather than a hard match/mismatch decision on its own.

Two datasets are supported:

- **GRID** (English): word transcripts phonemized with `g2p_en` (39 ARPABET phonemes), boundaries from `.align` files.
- **Digit** (Chinese): digit sequences (0–9 plus `!` interjections), boundaries from `.lab` files.

## Model

A 3D-conv + ResNet18 lip encoder and a phoneme/digit embedding + transformer text encoder feed a cross-modal alignment module (cosine similarity or cross-attention, see `alignment_mode`) and a classifier that emits the global verification score. A CTC head over lip frames adds an alignment-aware feature.

On top of that, VFA-style heads provide per-token output: a per-frame token-posterior head and a boundary head drive a Viterbi forced aligner (port of the method from `Visual-Forced-Alignment`), giving each word/digit a time span. A per-token verification head, trained on positives and single-substitution negatives, scores whether each token is actually being articulated. Details and diagrams: `architecture.md`.

## Usage

### Preprocessing

```bash
python pipeline/data_preparation.py --config config.grid.json
```

Crops the lip region with MediaPipe (96x96, 25 fps), exports wavs, and writes train/val/test splits. On the cluster, use `scripts/slurm/preprocess.sbatch`.

### Training

```bash
torchrun --nproc_per_node=2 main.py --config config.grid.json
```

Config files: `config.grid.json` / `config.digit.json` (checkpoints and logs go to `output_dir`; per-dataset lambdas for the CTC, alignment, frame-CE, boundary, and per-token losses are in the same file). On the cluster:

```bash
mkdir -p logs
sbatch scripts/slurm/train.sbatch                       # grid, 1 GPU
sbatch --gres=gpu:2 scripts/slurm/train.sbatch          # grid, 2 GPUs (DDP, one process per GPU)
sbatch --export=ALL,DATASET=digit scripts/slurm/train.sbatch
```

### Evaluation

```bash
python test.py --config config.grid.json
```

Reports AUC / accuracy / EER on the hold-out set and writes `eer_thresholds.json` next to the checkpoints, which inference auto-discovers for MATCH/MISMATCH decisions.

### Inference

```bash
python inference.py --video clip.mp4 --text "bin blue at f two now" --dataset grid
python inference.py --video clip.mp4 --text "1 2 3" --dataset digit --json report.json
```

Prints global verification and alignment scores with decisions at the EER-optimal thresholds, plus a per-token table (timestamps, alignment score, verification score). With `--json`:

```json
{
  "global": {"verification_score": 0.9384, "alignment_score": 0.9127, "verdict": "MATCH"},
  "tokens": [
    {"token": "1", "start_s": 0.32, "end_s": 0.68, "alignment_score": 0.9102,
     "verification_score": 0.9741, "verdict": "MATCH", "low_confidence": false},
    {"token": "3", "start_s": 1.05, "end_s": 1.44, "alignment_score": 0.8955,
     "verification_score": 0.2214, "verdict": "MISMATCH", "low_confidence": false}
  ]
}
```

Per-token `alignment_score` is non-semantic (frame posterior averaged with the cross-modal word score); `verification_score` is semantic (word-head sigmoid). GRID rows are per-phoneme, digit rows per-digit. Checkpoints trained before the VFA heads load with the per-token output disabled (`"tokens": null`).

## Training data

Negatives are generated per positive by the negative sampler: random or shuffled transcripts, single-word/digit substitutions, and lexically similar hard negatives. Single-substitution negatives also carry per-token labels, which is what teaches the per-token head to localize the wrong token.

## Repository layout

- `models/` — encoders, cross-modal alignment, verification model, VFA heads
- `dataset/` — GRID/digit loaders, lip-crop preprocessing
- `training/`, `evaluation/` — training loop and losses, metrics
- `pipeline/` — data preparation
- `utils/` — phoneme conversion, negative sampling, Viterbi aligner, checkpoint/config helpers
- `scripts/slurm/` — train / test / preprocess sbatch scripts
