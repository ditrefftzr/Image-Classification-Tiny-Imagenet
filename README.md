# Tiny ImageNet-200 Image Classification

EfficientNet-B4 fine-tuned on the [Tiny ImageNet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip) dataset, achieving **60.83% validation accuracy** (baseline: 58%).

## Overview

- **Model**: EfficientNet-B4 pre-trained on ImageNet, fine-tuned for 200-class classification
- **Dataset**: Tiny ImageNet-200 — 100,000 training images, 10,000 validation images, 64x64 px
- **Training strategy**: Base training (8 epochs) followed by optional 3-phase progressive fine-tuning
- **Output**: Submission CSV compatible with competition format

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (CPU works but is slow)

Install dependencies:

```bash
pip install -r requirements.txt
```

> **GPU vs CPU**: The `requirements.txt` installs the CUDA 12.8 build of PyTorch by default. If you don't have a compatible GPU, comment out the `+cu128` lines and uncomment the CPU-only alternatives.

For the CUDA build, you may need to add the PyTorch index URL:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

## Usage

Open and run `tinyimagenetcompetition.ipynb` cell by cell in Jupyter.

The notebook will:
1. **Download the dataset automatically** (~237 MB zip) into a `dataset/` folder on first run
2. Build CSV path mappings (`train_data.csv`, `val_data.csv`, `test_data.csv`)
3. Train EfficientNet-B4 for 8 epochs
4. Optionally apply 3-phase progressive fine-tuning
5. Generate a submission file: `submission_<model>_<timestamp>.csv`

## Training Details

| Setting | Value |
|---|---|
| Optimizer | AdamW (lr=0.0008, weight decay=0.01) |
| Loss | CrossEntropyLoss (label smoothing=0.05) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Batch size | 32 |
| Base training epochs | 8 |

**Augmentations (training):** RandomHorizontalFlip, RandomRotation (8°), ColorJitter

**Fine-tuning phases** (optional, run after base training):

| Phase | Layers unfrozen | Epochs |
|---|---|---|
| 1 | Classifier only | 3 |
| 2 | Classifier + last features | 4 |
| 3 | All layers | 5 |

## Results

| Epoch | Validation Accuracy |
|---|---|
| 1 | 47.20% |
| 4 | 58.17% (baseline) |
| 8 | **60.83%** |

## Project Structure

```
.
├── tinyimagenetcompetition.ipynb  # Main notebook
├── requirements.txt
├── train_data.csv                 # Path mappings (generated on first run)
├── val_data.csv
├── test_data.csv
├── checkpoints/                   # Saved model weights (generated during training)
└── dataset/                       # Downloaded dataset (gitignored)
```

## Dataset

Downloaded automatically from the Stanford CS231n course website on first run. No manual steps needed.

If the automatic download fails, download `tiny-imagenet-200.zip` manually from:
`http://cs231n.stanford.edu/tiny-imagenet-200.zip`

and extract it into a `dataset/` folder at the project root.
