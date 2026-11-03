# Lip-to-Text Verification

Given a silent video of a speaker and a transcript, score how well the lip movements match the text (0–1). This is a verification system, not lip reading: it does not generate text, and the score measures visual-text consistency rather than a hard match/mismatch decision on its own.

Two datasets are supported:

- **GRID** (English): fixed vocabulary of 51 words plus `<pad>`/`<unk>`, with whole-word boundaries from `.align` files. Legacy phoneme checkpoints remain readable.
- **Digit** (Chinese): digit sequences (0–9 plus `!` interjections), boundaries from `.lab` files.

## Model

A 3D-conv + ResNet18 lip encoder and a word/digit embedding + transformer text encoder feed a cross-modal alignment module (cosine similarity or cross-attention, see `alignment_mode`) and a classifier that emits the global verification score. A CTC head over lip frames adds an alignment-aware feature.

On top of that, VFA-style heads provide per-token output: a per-frame token-posterior head and a boundary head drive a Viterbi forced aligner (port of the method from `Visual-Forced-Alignment`), giving each word/digit a time span. A per-token verification head, trained on positives and controlled substitution negatives, scores whether each token is actually being articulated. Details and diagrams: `architecture.md`.

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

GRID trains for **at most 30 epochs**, with AdamW weight decay `0.01` and feature dropout `0.2` on lip/text encoder outputs (disabled at evaluation). Validation negatives use a private fixed seed (`validation_seed: 42`). With `balanced_word_supervision: true`, the word loss is `0.5 * mean(positive BCE) + 0.5 * mean(negative BCE)`, so common matching words cannot swamp rare wrong words. If only one class is present in a batch, its mean is used. Raw word-label counts are logged to TensorBoard.

The best checkpoint minimizes the **mean of global and word validation EER**, with word scores evaluated on inference-style Viterbi spans, not ground-truth spans. `early_stopping_patience: 5` stops after five consecutive epochs without improvement (`0` disables stopping). Validation logs include word AUC/EER and positive/negative counts; DDP gathers both global and word scores across ranks before selection/stopping. Each best checkpoint also gets separate global, alignment and word thresholds calibrated on validation, bound to its weights by SHA-256. `supervise_silence: true` teaches the frame head silence class 0 outside annotated speech on real positive frames; padding, negatives and samples without alignments remain excluded. Boundary targets remain token-to-token transitions.

Word models start fresh and write to `liptev-exp/grid-word`, separate from phoneme experiments. **A new run reuses this directory and overwrites matching checkpoint/calibration filenames; back it up first if needed.** Retraining is required for balanced supervision and silence targets to affect predictions. Do not resume a phoneme checkpoint as a word model. Checkpoints record the dataset, token level, exact vocabulary and dropout; evaluation/inference use checkpoint tokenization rather than the current config. Digit configs keep their existing sampling, losses, tokenization, optimizer and epoch settings. The existing clip-random splits are unchanged; speaker-held-out splits are needed if the goal is unseen-speaker generalization.

### Evaluation

```bash
python test.py --config config.grid.json
```

Calibrates thresholds on the **validation split**, writes `eer_thresholds.json` next to the selected checkpoint, then reports AUC / accuracy / EER on the held-out test split without using test scores to write thresholds. Reported test EER and Acc@EER are descriptive threshold-sweep metrics, not accuracy at the fixed validation threshold. Word AUC/EER use the same forced decoder as inference. Fallback alignments, missing spans and words without annotated geometry are excluded from word calibration. Calibration fails explicitly if either word-label class is absent.

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

GRID console and JSON rows are **per-word**; digit rows remain per-digit. Word-trained checkpoints align and verify each word directly: `alignment_score` combines the word's frame posterior and cross-modal score; `verification_score` is the word-head sigmoid over the whole-word span. **Word verdicts use only the independent `word_verification` threshold**, never the global classifier threshold. Old calibration files without that block produce uncalibrated word verdicts, while global decisions still work. Re-run evaluation to calibrate an existing word checkpoint without retraining; this fixes decisions, not the underlying raw scores. Word calibration requires matching checkpoint weights and decoder settings; stale weight hashes disable all verdicts.

For legacy phoneme checkpoints (no token-level stamp), inference still groups phoneme rows into words: timestamps span first to last phoneme and scores are the lowest constituent scores; word verdicts remain uncalibrated without independent token calibration. Missing/truncated phonemes make the whole legacy word unavailable. Missing word spans also produce null timestamps/scores and `low_confidence: true`; uniform-alignment fallbacks carry `low_confidence: true` and an uncalibrated verdict. Checkpoints trained before the VFA heads load with per-token output disabled (`"tokens": null`).

## Training data

Each clip still creates two pairs (positive and negative), so batch memory is unchanged. For balanced GRID word training, 75% of negatives are controlled replacements: half replace one word and half replace 2 through all words. A quarter of controlled draws allow cross-grammar replacements from the full GRID vocabulary, covering transcripts such as a color in the command slot. Every replaced position is labeled 0; unchanged positions are 1, using the original word boundaries. The remaining 25% use existing random/shuffled/hard strategies. Uncontrolled multi-word negatives remain masked for word supervision because their position-wise timing is not established; single substitutions retain their existing word labels. Digit and legacy phoneme sampling are unchanged.

## Repository layout

- `models/` — encoders, cross-modal alignment, verification model, VFA heads
- `dataset/` — GRID/digit loaders, lip-crop preprocessing
- `training/`, `evaluation/` — training loop and losses, metrics
- `pipeline/` — data preparation
- `utils/` — phoneme conversion, negative sampling, Viterbi aligner, checkpoint/config helpers
- `scripts/slurm/` — train / test / preprocess sbatch scripts
