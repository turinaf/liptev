# Lip-to-Text Verification Model

This repository contains a PyTorch implementation for a Cross-Modal Lip-to-Text Verification system using the GRID dataset. The goal of this model is to take a sequence of lip movement frames and a text transcript, then compute an alignment-based consistency score (0 to 1). 

**Important:** This system strictly measures the synchronization and alignment quality between lip movements and transcripts. It **does not** classify a definitive "match/mismatch", nor does it perform lip reading or visual speech recognition (generating text from video).

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
├── pipeline/
│   └── data_preparation.py     # End-to-end dataset preprocessing
├── utils/
│   ├── phoneme_converter.py    # Maps English transcripts to integer phoneme vocabulary
│   └── negative_sampler.py     # Generates bad transcripts for contrastive training
├── main.py                     # Training orchestrator & evaluation suite
├── inference.py                # External inference execution script 
└── README.md                   # This file
```

---

## ⚙️ Pipeline Overview

### 1. Data Preprocessing
The `GRIDDataset` loads speaker videos and alignment files.
- **Cropping & Sizing:** Videos are actively tracked utilizing MediaPipe's FaceLandmarker, dynamically cropped tightly around the lips, and resized uniformly to `96x96`.
- **Temporal extraction:** Video lengths are capped/padded to standard frame limits (e.g., 75 bounds) converting into unified PyTorch tensors of shape `(T x C x H x W)`.
- **Transcript Parsing:** Transcript strings are mapped directly into integer phoneme arrays utilizing the `PhonemeConverter`.

### 2. Deep Encoders
- **Lip Encoder:** A pseudo-3D CNN. Extracts spatial features incrementally per frame utilizing a basic ResNet backbone, subsequently mapping the timeline contextally using a Transformer Encoder.
- **Text Encoder:** Processes exact phonemes through a dense Embedding layer and passes them through an equivalent multi-layer Transformer Encoder.

### 3. Verification & Alignment Engine
- **Cross-Modal Attention:** Video sequence vectors and Text vectors merge through a localized Cross-Attention module. Phonemes iteratively scan Lip frames, yielding an empirical temporal alignment map.
- **Verification Score:** The aligned output matrices are pooled, concatenated, and routed through a localized Multi-Layer Perceptron (MLP) mapping dynamically to an alignment consistency probability utilizing Sigmoid activations.

### 4. Training & Negative Sampling
The loss relies on pushing correct sequences closer and forcing improper texts further away.
- **Negative Sampler:** To teach the model what a "mismatch" looks like, we generate a negative sample for every true pair during training by dynamically augmenting the text. Instead of relying purely on completely distinct sentences, we randomly select from 3 strategic distortions:
  - **`wrong` / `random`**: Selects a completely random unrelated sentence from the dataset pool. 
    *(Example: "bin blue at f two now" ➔ "lay red with c nine please")*
  - **`shuffled`**: Randomizes the word order of the original transcript to force learning sequential temporal positioning. 
    *(Example: "bin blue at f two now" ➔ "now at blue f bin two")*
  - **`phoneme_similar`**: Selectively replaces a specific word with another valid GRID vocabulary word (such as swapping a color target). This ensures localized visual scrutiny prevents cheating via contextual sentence lengths. 
    *(Example: "bin blue at f two now" ➔ "bin red at f two now")*
- **Losses:** We combine **Binary Cross Entropy (BCE)** against a hard certainty boundary, supplemented by a **Contrastive Margin Loss** optimizing explicit interval distance mapping matches away from mismatch traps.

### 5. Evaluation
Our runtime evaluation tracks:
- **ROC AUC:** Area under the ROC curve tracking absolute classification separation.
- **Alignment Accuracy:** Total accuracy thresholding exact temporal metric alignment limits strictly passing 50% probabilities.
- **Score Margin:** The objective separation distance mapped between Valid Ground Truth and Synthesized Invalid metrics.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have all necessary data and vision pip packages installed:
```bash
pip install torch torchvision torchaudio scikit-learn numpy mediapipe opencv-python tqdm
```

### 1. Preparing the GRID Dataset
The pipeline natively processes real media from the GRID audiovisual dataset utilizing MediaPipe to track user bounds properly. **You must prepare the data before training the model.**

1. Download the [GRID audiovisual sentence corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/).
2. Extract the speaker folders into `data/raw/` (or strictly alter `raw_data_dir` via `config.json`).
3. Point your terminal at the pipeline directory and execute the setup loop dynamically ripping frames, scaling boundaries, and extracting audio parameters seamlessly:

```bash
python pipeline/data_preparation.py
```

This procedure actively constructs scaled `96x96` mp4 chunked archives, synchronized raw wav exports, and creates structured JSON manifest files (`train_split.json`, `val_split.json`, `test_split.json`).

### 2. Running the End-to-End Pipeline
Once the `processed` data directory is populated, you can run the full pipeline encompassing training splits, batch evaluations, and validation outputs simply by running:
```bash
python main.py
```

### 3. Example Inference Output
When evaluating specific subsets or running `./inference.py`, the system outputs clean probability targets estimating structural similarities:

```text
Evaluating Model on Hold-out Test Set...
Final Test Result -> AUC: 0.9421 | Align Acc: 0.9100

Inference Output Demo:
Transcript: bin blue at f two now
Alignment Score: 0.8923
------------------------------
Transcript: lay green by g zero please
Alignment Score: 0.1235
------------------------------
```
