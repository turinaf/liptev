# GRID generalization investigation

## Current split: speaker-disjoint

The shared GRID manifests in `grid_processed_data_dir` now use a
speaker-disjoint split, common to every repository using that processed root.
All 32,984 clips are retained:

| Split | Speakers | Clips |
| --- | ---: | ---: |
| Train | 27 | 26,984 |
| Validation | 3 | 3,000 |
| Test | 3 | 3,000 |

There is **zero pairwise speaker overlap**. Validation speakers are
`s18_processed`, `s5_processed`, `s9_processed`; test speakers are
`s11_processed`, `s17_processed`, `s32_processed`.
Videos and annotations are unchanged. The original overlapping manifests were
backed up, hash-verified against the historical inventory, then replaced.
Backup directory:
`/mnt/lustre/ifm-lustre-prod-us-west1-a/av-data/GRID_processed/split_backups/20261119T064016418347Z`.
All three actual dataset loaders were checked: 26,984/3,000/3,000 usable clips.
See [the new split visualization](speaker_split.png) and
[manifest hashes and verification](speaker_disjoint_split.json).

Training must start fresh in `grid-word-speaker-disjoint`: the old model saw
all 33 speakers. The results below, including `inventory.json`, describe
the **historical clip-random split and old checkpoint**, not the new split.
No new speaker-held-out model has been trained or evaluated yet.

## Training augmentation and smartphone recordings

Both datasets now support training-only, clip-consistent horizontal flip,
mild affine jitter, brightness/contrast/saturation changes, blur and
downsample/upsample degradation. The supplied configs enable
`train_augmentation`; validation/test/inference are untouched. Inspect the
[augmentation preview](augmentation_preview.png), showing identical frame
positions from one GRID training clip with three random seeds. Reproduce with:

```bash
python analysis/grid_generalization/preview_augmentation.py --config config.grid.json
```

This is a qualitative preview, not evidence of improved verification accuracy.
Digit recordings are smartphone videos with variable source FPS and resolution,
not GRID-style studio clips. Augmentation changes neither time nor labels;
digit source-FPS metadata and `.lab` conversion are preserved. Correct
timestamp resampling and scale-normalized cropping remain separate work.

Digit splitting now holds out the smallest subset by processed-video count,
divides its speakers between validation and test, and trains on the other
subsets. Speaker IDs are subset-local. Real digit counts cannot be audited
here because that dataset is on another server. See the
[main usage documentation](../../README.md) for commands and configuration.

## Historical audit

This was a read-only audit of the original GRID split manifests and
`grid-word` checkpoint. No recordings from the user were available, so these
results identify demonstrated weaknesses and plausible causes, not the
specific cause of that recording's failure.

## What to inspect

- [Clustering projection](clustering.png): 594 speaker-balanced clips, colored
  by split, speaker and 33 train-fitted KMeans clusters.
- [Speaker/cluster heatmap](speaker_cluster_heatmap.png): distribution of
  each speaker's sampled clips across those clusters.
- [Crop-scale sensitivity](crop_scale_sensitivity.png): the same five raw
  test clips, with face detection repeated at each resolution.
- [Timing sensitivity](timing_sensitivity.png): ten paired, synthetic
  perturbations of matched test clips.
- [Full split and clustering results](split_cluster_audit.json),
  [crop experiment](crop_scale_control.json), and [sample scores and cluster
  assignments](sample_scores.csv). Rows in `features.npz` correspond to CSV
  rows in the same order.

## Findings

### The existing test set is speaker-dependent

The manifests contain 26,388 training clips, 3,298 validation clips and 3,298
test clips. All **33 of 33** validation and test speakers also occur in
training. No clip IDs or exact transcripts are shared between the splits; this
is not evidence of duplicate-clip leakage. The split is valid for new clips
from known speakers but cannot estimate performance on a new speaker. The
train/test token-frequency total-variation distance is about 0.0187, indicating
similar overall vocabulary proportions, not proof of fairness.

In the sampled clustering analysis, 194/198 test clips had a nearest training
clip from the same speaker using simple visual descriptors; a uniform
speaker-balanced baseline is 1/33. The 33-cluster speaker normalized mutual
information (NMI) was 0.936 versus a permutation 95th percentile of 0.258.
Appearance, lighting and capture setup are confounded with speaker identity,
so this does **not** show which visual characteristic causes the grouping.

The trained lip-encoder mean embeddings showed a weaker association: 13/198
test clips had a same-speaker nearest training neighbor, and 33-cluster NMI
was 0.283 versus a permutation 95th percentile of 0.230. This does not prove
the network relies on speaker identity: mean pooling and these clusters do
not capture everything the classifier uses. The 2D plots show only about 46%
of visual and 42% of learned training variance; KMeans used all retained PCA
dimensions, not just the two plotted dimensions.

At the checkpoint's validation-derived verification threshold of **0.01718**,
196/198 sampled positive test clips passed when evaluated from their processed
dataset crops. This small, positive-only sample does not measure false
accepts, EER, or unseen-speaker accuracy. The threshold and model were both
selected using validation speakers also seen in training.

### Crop size is a demonstrated failure mode

[`crop_lip_region`](../../dataset/preprocessing.py) adds a fixed **12 source
pixels** around the lip landmarks, then resizes the crop to 96x96. Consequently,
changing source resolution or face size changes the amount of surrounding
context in the model input. In the crop experiment, the same raw clip and
transcript were evaluated after 1x, 2x and 4x image upscaling. MediaPipe
redetected the face separately at every resolution; no detection failed in
these five clips.

At 4x, fixed 12px padding caused **4/5** matched clips to fall below the
validation threshold. Scaling the padding to 48px restored **5/5** above the
threshold. One score changed from 0.99924 to 0.0000000425 with fixed padding
and recovered to 0.99923 with scaled padding. The accompanying CTC loss
feature also rose substantially in that case. These synthetic tests establish
scale sensitivity, not a measured failure rate on phone videos; scaling by the
image resolution alone may not correct different face framing.

The normal `inference.py` command was also run end to end on one raw GRID
clip and an OpenCV-resized, MP4-re-encoded 4x copy, both 75 frames at 25 fps
with identical text. It reported **0.9992 / MATCH** for the original and
**0.0000 / MISMATCH** for the resized copy; the latter's unrounded JSON score
was `1.0284e-08`. This separately reproduces the reported symptom in the
actual inference path.

### Timing and input quality also matter

GRID training clips are generally 75 frames at 25 fps. Inference takes the
first `max_frames` without converting the input to 25 fps. Thus 75 frames
cover 3 seconds at 25 fps but only 1.25 seconds at 60 fps. In synthetic
frame-duplication tests, all 10 baseline pairs passed and all 10 simulated
60-fps pairs failed; 30-fps duplication failed 0/10. Adding a one-second
static lead-in before truncation caused 2/10 to fail. These are small,
controlled stress tests, not measurements on real phone recordings.

Face-detection failures before the first successful crop discard frames, and
later failures repeat the previous crop. One sampled test clip had only
7 processed frames while its raw source had 74 (`s8_processed/sgai1p.mp4`);
despite the six-word transcript, its score was 0.2887, above the 0.01718
threshold. This illustrates the need to reject or flag incomplete inputs as
well as low scores; it does not establish why that processed clip was
shortened.

The global verification value is a sigmoid score printed to four decimal
places, not a calibrated probability. `0.0000` can be a small nonzero number;
the JSON report retains more precision. For this checkpoint it is well below
the validation threshold, but that threshold has not been validated on new
speakers or cameras. The global classifier also receives a CTC
log-likelihood feature, which can change sharply with crop and timing.

## Practical next steps

1. Retest with the same six-word GRID-style phrase and exact transcript. Use a
   well-lit, frontal, unobstructed face, keep the full utterance within three
   seconds, and convert the clip to 25 fps before inference. Trim long
   lead-in/trailing silence without cutting spoken words. Check the detected
   mouth crops and compare framing with raw GRID clips; changing frame rate
   alone will not solve crop-scale or speaker differences.
2. Measure face-detection dropouts, input frame rate/duration, crop geometry
   and truncation explicitly; reject incomplete recordings rather than issuing
   a confident verdict. Test a crop margin normalized to face/lip scale and
   25-fps resampling against the current checkpoint before adopting either.
3. Use **speaker-disjoint** train, validation and test manifests (now installed).
   Train a new model on train speakers only, select the checkpoint and
   calibrate on validation speakers, and evaluate once on untouched test
   speakers. Preserve the backed-up historical manifests and old checkpoints.
   Repeat across speaker folds if feasible.
4. Evaluate matched and realistic mismatched phrases on multiple new speakers,
   cameras, frame rates and lighting conditions. Include errors at the fixed
   validation threshold, not just a threshold swept on the test set. Only after
   establishing that baseline, measure the effect of augmentation and consider
   diverse data or domain-specific fine-tuning. Recalibration alone cannot fix
   bad raw scores.

For example, if `ffmpeg` is available, this converts a three-second section
to 25 fps; replace `0` with the start time that contains the complete phrase:

```bash
ffmpeg -ss 0 -i recording.mp4 -t 3 -vf fps=25 -an \
  -c:v libx264 -pix_fmt yuv420p recording_25fps.mp4
python inference.py --config config.grid.json --video recording_25fps.mp4 \
  --text "lay blue with b two please" --json recording_scores.json
```

Use the phrase actually spoken. Resampling does not fix a bad crop, and
cutting away part of the phrase makes the comparison invalid.

## Reproduce

To reproduce the historical audit, explicitly select the preserved original
manifests and checkpoint. From the repository root:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python analysis/grid_generalization/audit_grid_generalization.py \
  --repo "$PWD" --config "$PWD/config.grid.json" \
  --split-dir /mnt/lustre/ifm-lustre-prod-us-west1-a/av-data/GRID_processed/split_backups/20261119T064016418347Z \
  --checkpoint /mnt/lustre/ifm-lustre-prod-us-west1-a/av-data/liptev-exp/grid-word/best_lip_verification_model.pth \
  --output "$PWD/analysis/grid_generalization"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python analysis/grid_generalization/crop_scale_probe.py \
  --repo "$PWD" --config "$PWD/config.grid.json" \
  --split-dir /mnt/lustre/ifm-lustre-prod-us-west1-a/av-data/GRID_processed/split_backups/20261119T064016418347Z \
  --checkpoint /mnt/lustre/ifm-lustre-prod-us-west1-a/av-data/liptev-exp/grid-word/best_lip_verification_model.pth \
  --output "$PWD/analysis/grid_generalization"
```

The first command scans all manifests and annotations, then deterministically
samples six clips per speaker per split (seed 42). `--reuse-inventory` skips
the full scan after verifying manifest hashes; it does not revalidate every
video and annotation. Visual descriptors summarize 12 frames; learned
descriptors are masked means of the full lip encoder output. Scaling/PCA and
KMeans are fitted on training samples only; nearest-neighbor and cluster
results are exploratory. Each command overwrites its output files.

The audited checkpoint SHA-256 is
`7ed31125c258dde47a6f258a9487b46354f8271d8c50c76a393bb37ee8e4b894`.
The original run used Python 3.14.8, PyTorch 2.14.1+cu126, OpenCV 5.0.0,
NumPy 2.5.4 and scikit-learn 1.9.1 on CPU. Exact floating-point results can
vary with library and hardware versions.
