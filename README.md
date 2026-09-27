# Open Source Urban Sprawl & Green Cover Tracker

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

An open-source deep learning framework for tracking urban sprawl, deforestation, and green cover dynamics from multi-temporal satellite imagery using PyTorch and U-Net semantic segmentation.

---

## 📌 Features

- 🛰️ **Semantic Segmentation**: U-Net architecture custom-tailored for multi-class satellite land cover classification.
- 🌳 **Land Cover Classes**:
  - `Built-up (Urban)` (Reddish-brown)
  - `Vegetation (Green Cover)` (Forest Green)
  - `Water` (Deep Blue)
- 📊 **Change Detection**: Quantify percentage changes in urban area and green cover across multi-year timeline datasets.
- 📈 **Automated Reports**: Generates color-coded segmentation masks, training loss curves, and detailed statistical reports.

---

## 📁 Repository Structure

```
├── ml/
│   ├── dataset.py        # Custom PyTorch Dataset with augmentations
│   ├── model.py          # U-Net PyTorch architecture definition
│   ├── train.py          # Model training pipeline & loss logging
│   ├── predict.py        # Inference, color-mask generation & change analysis
│   ├── requirements.txt  # Python dependencies
│   ├── loss_curve.png    # Training loss visualization curve
│   └── data/             # Training and validation satellite datasets
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/theharshkumarpal/Open-Source-Urban-Sprawl-Green-Cover-Tracker.git
cd Open-Source-Urban-Sprawl-Green-Cover-Tracker
pip install -r ml/requirements.txt
```

### 2. Training the Model

Train the U-Net model on your satellite imagery dataset:

```bash
python ml/train.py --data_dir ml/data --epochs 25 --batch_size 4 --lr 1e-4
```

### 3. Running Inference & Change Detection

Run segmentation on satellite imagery and perform temporal change detection analysis:

```bash
python ml/predict.py \
  --image ml/data/val/images/sample_2020.png \
  --image_compare ml/data/val/images/sample_2026.png \
  --model_path ml/checkpoints/best_model.pth \
  --output ml/output_mask_2020.png \
  --output_compare ml/output_mask_2026.png
```

---

## 📊 Outputs & Visualizations

The pipeline produces:
1. **Segmentation Mask (`.png`)**: RGB mapped classification mask.
2. **Change Analysis Report (`.txt`)**: Quantified comparison report of urban sprawl vs green cover change.
3. **Loss Curve (`loss_curve.png`)**: Training and validation cross-entropy loss metrics over epochs.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.
