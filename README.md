# Lip-to-Text Verification Model

This repository contains a PyTorch implementation for a Cross-Modal Lip-to-Text Verification system using the GRID dataset. The goal of this model is to take a sequence of lip movement frames and a text transcript, then compute an alignment-based consistency score (0 to 1). 

**Important:** This system strictly performs alignment and verification. It **does not** perform lip reading or visual speech recognition (generating text from video).

## 📂 Project Structure

```text
.
├── dataset/
│   ├── grid_loader.py          # GRID Dataset PyTorch Datasets
│   └── preprocessing.py        # Video frame extraction & lip cropping (96x96)
├── models/
│   ├── lip_encoder.py          # Spatial-temporal video feature extractor
│   ├── text_encoder.py         # Phoneme embedding and Transformer encoder
│   ├── cross_modal_alignment.py# Cross-attention and alignment scoring
│   └── verification_model.py   # Final consistency linear classifier
├── training/
│   └── train.py                # Training loop, contrastive logic, and BCE loss
├── evaluation/
│   ├── metrics.py              # Precision, ROC AUC, Accuracy, margin
│   └── evaluate.py             # Validation loops and inferences
├── utils/
│   ├── phoneme_converter.py    # Maps English transcripts to dummy/phoneme vocab
│   └── negative_sampler.py     # Generates bad transcripts for contrastive training
├── main.py                     # Demo inference & training script
└── README.md                   # This file
```

---

## ⚙️ Pipeline Overview

### 1. Data Preprocessing
The `GRIDDataset` loads speaker videos and alignment files.
- **Cropping & Sizing:** Videos are cropped around the lip region and resized to `96x96`.
- **Temporal extraction:** Max frames are capped (e.g., 75) and transformed into a `.to_tensor()` format producing tensors of shape `(T x C x H x W)`.
- **Transcript Parsing:** Transcript strings are converted into integer phoneme sequences using the `PhonemeConverter`.

### 2. Deep Encoders
- **Lip Encoder:** A pseudo-3D CNN. Extracts spatial features per frame leveraging a ResNet baseline, followed by temporal sequencing using a Transformer Encoder mapping it to a feature matrix of `T x D`.
- **Text Encoder:** A standard Embedding layer maps phonemes to vectors, running through a standard multi-layer Transformer Encoder resulting in `L x D` outputs.

### 3. Verification & Alignment Engine
- **Cross-Modal Attention:** The features pass through a Cross-Attention module where Phonemes attend to Lip frames, outputting pooled attention weights and similarity matrices.
- **Verification Score:** The aligned sequences alongside pooled lip/text features are concatenated and passed through a Multi-Layer Perceptron (MLP) with a Sigmoid activation, yielding the final consistency probability.

### 4. Training & Negative Sampling
The loss is formulated to maximize the alignment & consistency score for true lip/text pairs and penalize incorrect ones.
- **Negative Sampler:** At run-time, to construct negative batches without manual annotation, we augment transcripts via:
  - Completely random wrong transcripts.
  - Temporally *shuffled* transcripts.
  - *Phoneme-similar* variants (e.g., substituting visually similar words).
- **Losses:** We combine **Binary Cross Entropy (BCE)** for the binary verification node, and a **Contrastive Margin Loss** explicitly dictating the internal cross-modal alignment scores to cleanly separate matches from mismatches.

### 5. Evaluation
Our evaluation harness computes:
- **ROC AUC:** Area under the ROC curve indicating how well the model separates positive from negative pairs.
- **Verification Accuracy:** Total binary accuracy thresholded at 0.5.
- **Alignment Accuracy:** Proxy metric dictating whether the alignment score was reliably >0.5.
- **Score Margin:** The absolute distance between the average valid score and average invalid score.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have PyTorch, Torchvision, Scikit-Learn, and Numpy installed:
```bash
pip install torch torchvision scikit-learn numpy
```

### Running the End-to-End Pipeline
To run a test with the dummy dataloader (simulating the GRID dataset interactions, training, and testing inference outputs), simply run:
```bash
python main.py
```

### Example Inference Output
When you run `main.py`, the system output looks like this:

```text
Evaluating...
ROC AUC: 1.0000, Verify Acc: 1.0000, Align Acc: 1.0000

Inference Output Demo:
Transcript: bin blue at f two now
Ground Truth Label: 1.0
Alignment Score: 0.8123
Final Consistency Score: 0.9412
Decision: Match
------------------------------
Transcript: lay green by g zero please
Ground Truth Label: 0.0
Alignment Score: 0.1235
Final Consistency Score: 0.0319
Decision: Mismatch
------------------------------
```

## 🛠 Integrating Real GRID Data
Currently, `preprocessing.py` utilizes dummy zero-arrays to test shape permutations. To train on the actual dataset:
1. Download the [GRID audiovisual sentence corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/).
2. Mount to your directory structure.
3. Replace the `extract_lip_frames()` in `dataset/preprocessing.py` with cv2/dlib/mediapipe scripts (like dlib's 68-point face landmarks) to crop lip boundaries appropriately from the `.mpg` video files and return actual pixel arrays.
