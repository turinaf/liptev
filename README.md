# Lip-to-Text Verification

Given a silent video of a speaker and a transcript, score how well the lip movements match the text (0–1). This is a verification system, not lip reading: it does not generate text, and the score measures visual-text consistency rather than a hard match/mismatch decision on its own.

Two datasets are supported:

- **GRID** (English): fixed vocabulary of 51 words plus `<pad>`/`<unk>`, with whole-word boundaries from `.align` files. Legacy phoneme checkpoints remain readable.
- **Digit** (Chinese): digit sequences (0–9 plus `!` interjections), boundaries from `.lab` files.

## Model

A 3D-conv + ResNet18 lip encoder and a word/digit embedding + transformer text encoder feed a cross-modal alignment module (cosine similarity or cross-attention, see `alignment_mode`) and a classifier that emits the global verification score. A CTC head over lip frames adds an alignment-aware feature.

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

GRID configs (`config.json` and `config.grid.json`) use `token_level: "word"` throughout: one text-encoder ID per word, word CTC targets, whole-word frame/boundary targets, and word verification labels. The vocabulary is deterministic and shared with negative sampling (GRID letters exclude `w`; adverbs are `again`, `now`, `please`, `soon`). Out-of-vocabulary GRID words raise an error instead of silently training unknown tokens.

GRID trains for **at most 30 epochs**, with AdamW weight decay `0.01` and feature dropout `0.2` on lip/text encoder outputs (disabled at evaluation). Validation negatives use a private fixed seed (`validation_seed: 42`). The lowest validation verification EER selects `best_lip_verification_model.pth`; `early_stopping_patience: 5` stops after five consecutive epochs without improvement (`0` disables stopping). DDP gathers validation scores across ranks before best-model selection/stopping. These controls reduce overfitting risk but do not establish that the dataset is overfitting; compare train loss and validation EER over epochs.

Word models start fresh and write to `liptev-exp/grid-word`, separate from phoneme experiments. Do not resume a phoneme checkpoint as a word model. Checkpoints record the dataset, token level, exact vocabulary and dropout; evaluation/inference use checkpoint tokenization rather than the current config. Digit configs keep their existing tokenization, optimizer and epoch settings. The existing clip-random splits are unchanged; speaker-held-out splits are needed if the goal is unseen-speaker generalization.

### Evaluation

```bash
python test.py --config config.grid.json
```

Reports AUC / accuracy / EER on the hold-out set and writes `eer_thresholds.json` next to the checkpoints, which inference auto-discovers for MATCH/MISMATCH decisions.

### Inference

```bash
python inference.py --config config.grid.json --video clip.mp4 --text "bin blue at f two now"
python inference.py --config config.digit.json --video clip.mp4 --text "1 2 3" --json report.json
```

Evaluation and inference prefer the best checkpoint, then the latest epoch if no best checkpoint exists. Run evaluation again after word training to create fresh EER thresholds; never reuse phoneme-model calibration.

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

GRID console and JSON rows are **per-word**; digit rows remain per-digit. Word-trained checkpoints align and verify each word directly: `alignment_score` combines the word's frame posterior and cross-modal score; `verification_score` is the word-head sigmoid over the whole-word span. Existing decisions use the global verification EER threshold, not a separately calibrated per-word threshold.

For legacy phoneme checkpoints (no token-level stamp), inference still groups phoneme rows into words: timestamps span first to last phoneme, scores are the lowest constituent scores, and any phoneme MISMATCH makes the word MISMATCH. Missing/truncated phonemes make the whole legacy word unavailable. Missing word spans also produce null timestamps/scores and `low_confidence: true`; uniform-alignment fallbacks carry `low_confidence: true`. Checkpoints trained before the VFA heads load with per-token output disabled (`"tokens": null`).

## Training data

Negatives are generated per positive by the negative sampler: random or shuffled transcripts, single-word/digit substitutions, and lexically similar hard negatives. Single-substitution negatives also carry per-token labels, which is what teaches the per-token head to localize the wrong token.

## Repository layout

- `models/` — encoders, cross-modal alignment, verification model, VFA heads
- `dataset/` — GRID/digit loaders, lip-crop preprocessing
- `training/`, `evaluation/` — training loop and losses, metrics
- `pipeline/` — data preparation
- `utils/` — phoneme conversion, negative sampling, Viterbi aligner, checkpoint/config helpers
- `scripts/slurm/` — train / test / preprocess sbatch scripts
