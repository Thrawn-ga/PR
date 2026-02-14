# Seg_UKAN – Retinal Image Segmentation and Classification using U-KAN

This repository implements a multi-task deep learning framework for:

- 🩸 Retinal Vessel Segmentation
- 🧠 Retinal Disease Classification

using a U-KAN-based architecture and the FIVES dataset.

---

## 📌 Overview

Retinal image analysis plays a critical role in detecting and monitoring:

- Diabetic Retinopathy
- Glaucoma
- Vascular abnormalities
- Other retinal conditions

This project performs:

1. Pixel-wise vessel segmentation  
2. Image-level classification  

using a unified deep learning architecture.

---

## 🧠 Model Architecture

The framework includes:

- U-shaped encoder–decoder structure
- KAN-based feature transformation layers
- Skip connections for multi-scale feature learning
- Classification head for disease prediction

### Tasks Supported

| Task | Description |
|------|------------|
| Segmentation | Predict vessel mask for each pixel |
| Classification | Predict disease/quality category for image |

---

## 📂 Project Structure

Seg_UKAN/
│
├── archs.py
├── dataset.py
├── train.py
├── val.py
├── test_eval.py
├── config.py
├── utils.py
├── losses.py
├── metrics.py
├── prepare_fives.py
├── requirements.txt
├── environment.yml
├── README.md
│
├── datasets/ (not included)
└── outputs/ (generated during training)

yaml
Copy code

---

## 📊 Dataset

This project uses the **FIVES dataset**.

The dataset is NOT included in this repository.

### 📥 Download Instructions

Download the FIVES dataset manually and place it in:

datasets/FIVES/

vbnet
Copy code

Expected structure:

datasets/
└── FIVES/
├── train/
│ ├── images/
│ └── masks/
└── test/
├── images/
└── masks/

yaml
Copy code

Classification labels are derived from dataset metadata or image naming conventions.

---

## ⚙️ Installation

### Using pip

```bash
pip install -r requirements.txt
Using Conda
bash
Copy code
conda env create -f environment.yml
conda activate seg_ukan
🚀 Training
To train segmentation + classification:

bash
Copy code
python train.py
📈 Validation
bash
Copy code
python val.py
🧪 Evaluation
bash
Copy code
python test_eval.py
📊 Metrics
Segmentation Metrics
IoU (Intersection over Union)

Dice Score

Pixel Accuracy

Confusion Matrix

Classification Metrics
Accuracy

Precision

Recall

F1-score

📦 Outputs
Training results are saved in:

Copy code
outputs/
Contains:

Model checkpoints (.pth)

Loss curves

IoU curves

Confusion matrices

Classification reports

This directory is excluded from version control.

🛠 Requirements
Python 3.10+

PyTorch

torchvision

numpy

matplotlib

scikit-learn

See requirements.txt for full dependency list.

🧑‍💻 Author
Taraka Ram Paladugu
