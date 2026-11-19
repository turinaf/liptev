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

Crops the lip region with MediaPipe (96x96), exports wavs, and writes train/val/test splits. GRID uses 25 fps. Digit smartphone recordings have variable source frame rates and resolutions: preprocessing currently labels the output container as 25 fps without temporally resampling; per-video source FPS in `meta.json` remains authoritative for digit timing. On the cluster, use `scripts/slurm/preprocess.sbatch`.

GRID splits are **speaker-disjoint across train, validation and test**. The splitter sorts speaker IDs, shuffles them with a private seed of 42, then assigns approximately 80/10/10% of speakers (33 speakers give 27/3/3). All clips from a speaker stay together. Manifests live alongside the processed data in `grid_processed_data_dir`, shared by every repository copy using that root. To regenerate only the manifests:

```bash
python create_splits.py --config config.grid.json
```

The preprocessing entry point uses the same splitter. Before replacing changed manifests, it archives the existing manifests and summary under `split_backups/<UTC timestamp>/` in the processed root. Relative manifest entries keep the splits portable; `split_summary.json` records speaker membership and counts.

Digit uses `digit_processed_data_dir` (default configuration: `./data/digit`) with layout `subset_01` through `subset_05`, each containing speaker folders with `audio/`, `video/`, `text/` and `lab/`. The policy is:

- Count processed videos in every nonempty subset and hold out the **smallest subset** (ties use the lexically first name).
- Keep all other subsets entirely in training.
- Shuffle the smallest subset's sorted speakers with seed 42, assign half (rounded down) to test and the remainder to validation. This balances speaker counts, not necessarily clip counts.
- Speaker identity is **subset-local**: identically named speaker folders in different subsets represent different people. No speaker identity crosses splits. At least two nonempty subsets and two speakers in the smallest subset are required.

On the digit server, generate these shared manifests without reprocessing videos:

```bash
python digit_create_splits.py --config config.json
```

This command explicitly selects digit and reads `digit_processed_data_dir`, even if the config's current dataset is GRID. For **training**, use the full digit-compatible settings from `config.digit.json` as that server's `config.json`, retain the correct data paths, and choose a writable, fresh `output_dir`. Changing only `"dataset"` in a GRID training config is not sufficient. Digit data is absent on this server, so its real subset counts have not been measured here.

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

Word models now start fresh and write to `liptev-exp/grid-word-speaker-disjoint`, separate from earlier clip-random and phoneme experiments. **Retrain from fresh initialization: the old `grid-word` checkpoint saw every speaker, including the new validation/test speakers, and cannot provide an unseen-speaker evaluation or be used as its fine-tuning starting point.** Until retraining produces a checkpoint, default GRID inference/evaluation has no weights in the new output directory. A later run reuses this directory and overwrites matching checkpoint/calibration filenames; back it up first if needed. Checkpoints record the dataset, token level, exact vocabulary and dropout; evaluation/inference use checkpoint tokenization rather than the current config. Digit also uses a fresh `digit-speaker-disjoint` output directory for its new split; its existing negative sampling, losses, tokenization, optimizer and epoch settings remain unchanged.

#### On-the-fly video augmentation

The supplied GRID and digit configs enable `"train_augmentation": true`; set it to `false` to disable it (omitted flags default to disabled). Only training clips are augmented; validation, test and inference remain unchanged. The same random transformation parameters are shared across every frame of a clip:

- Horizontal flip (50%).
- Mild rotation (up to 5 degrees), translation (up to 4%) and scale (0.9–1.1), applied together with 50% probability.
- Brightness/contrast (20%) and saturation (15%) jitter.
- Mild Gaussian blur (20% probability).
- Downsample/upsample quality degradation (35% probability; scale 0.5–1.0).

Augmentation preserves frame count/order, source FPS metadata and alignment labels; it does **not** change speed, drop frames or assume digit clips have GRID's fixed timing/resolution. Digit `.lab` seconds still map to frames using each clip's source FPS, with the existing metadata/duration/config fallback policy. These perturbations target appearance and capture-quality variation, but cannot create missing information or guarantee robustness. Retraining and held-out device/speaker evaluation are needed to measure any improvement. See the [augmentation preview](analysis/grid_generalization/augmentation_preview.png).

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

Prints the global verification decision at its EER-optimal threshold and a diagnostic global alignment score without a verdict, plus a per-token table (timestamps, alignment score, verification score). All verdicts refer to verification, not alignment; JSON retains both raw global scores. With `--json`:

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

### Recordings from arbitrary devices

Speaker-disjoint evaluation addresses speaker overlap, **not studio-to-device domain shift**. The current GRID inference path still crops with fixed source-pixel padding and keeps only the first 75 frames without resampling. At 60 fps that is 1.25 seconds, not 3 seconds. Normalizing RGB values or changing the video's FPS label does not fix those problems.

A robust recording pipeline needs:

- **Timestamp-based temporal normalization:** decode real presentation timestamps (including variable-frame-rate inputs), resample to 25 fps without changing playback speed, and preserve the mapping to source timestamps.
- **Scale-consistent spatial preprocessing:** correct orientation, align/stabilize the face, and crop lips with a margin relative to face/lip scale rather than a fixed 12 source pixels. Use the same crop convention in training and inference, and validate/retrain when changing it. Merely forcing every frame to 96x96 can distort the face or remove the context the model learned.
- **Explicit duration handling:** for the current model, capture the complete short phrase within a three-second window and trim excess silence. Do not squeeze an arbitrarily long recording into 75 frames or silently discard its ending. Supporting a phrase anywhere in a longer recording needs windowed phrase-search semantics and calibration for multiple windows; verifying a whole long transcript needs variable-length training, appropriate data, and an explicit compute/duration limit. Raising `max_frames` alone is not sufficient.
- **Quality checks and realistic data:** report unusable input for missing/poorly tracked faces, severe occlusion or incomplete speech instead of treating every recording as a decidable mismatch. Train with realistic crop/scale, lighting, blur/compression, pose and speaking-rate variations, updating timing labels consistently. Add representative non-studio recordings from multiple devices and speakers; GRID's fixed six-word vocabulary/grammar is not unrestricted speech.
- **Deployment validation:** keep real-world validation and test subjects separate, select/calibrate only on validation, and report false accepts and false rejects at that fixed threshold on untouched speakers/devices.

The spatial/photometric training augmentation above is implemented. Timestamp resampling, scale-normalized lip cropping, duration handling and input-quality rejection remain recommendations, not implemented fixes. In particular, do not impose GRID's 75-frame/25-fps assumption on smartphone digit recordings; proper variable-frame-rate support needs presentation timestamps and consistently transformed timing labels. See [the investigation](analysis/grid_generalization/README.md) for controlled crop/FPS failures and [the current split visualization](analysis/grid_generalization/speaker_split.png).

## Training data

Each clip still creates two pairs (positive and negative), so batch memory is unchanged. For balanced GRID word training, 75% of negatives are controlled replacements: half replace one word and half replace 2 through all words. A quarter of controlled draws allow cross-grammar replacements from the full GRID vocabulary, covering transcripts such as a color in the command slot. Every replaced position is labeled 0; unchanged positions are 1, using the original word boundaries. The remaining 25% use existing random/shuffled/hard strategies. Uncontrolled multi-word negatives remain masked for word supervision because their position-wise timing is not established; single substitutions retain their existing word labels. Digit and legacy phoneme sampling are unchanged.

## Repository layout

- `models/` — encoders, cross-modal alignment, verification model, VFA heads
- `dataset/` — GRID/digit loaders, lip-crop preprocessing
- `training/`, `evaluation/` — training loop and losses, metrics
- `pipeline/` — data preparation
- `utils/` — phoneme conversion, negative sampling, Viterbi aligner, checkpoint/config helpers
- `scripts/slurm/` — train / test / preprocess sbatch scripts
