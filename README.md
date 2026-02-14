# Seg_UKAN  
### Multi-Task Retinal Vessel Segmentation and Disease Classification using U-KAN

A deep learning framework for simultaneous:

- 🩸 Retinal vessel segmentation  
- 🧠 Retinal disease classification  

Built using a U-KAN based architecture and evaluated on the FIVES dataset.

---

## 📌 Overview

Retinal image analysis is critical for early diagnosis of:

- Diabetic Retinopathy
- Glaucoma
- Age-related abnormalities
- Vascular disorders

This project implements a **multi-task learning framework** that performs:

1. Pixel-wise vessel segmentation  
2. Image-level disease classification  

within a unified model architecture.

---

## 🧠 Architecture

The model is based on a U-shaped encoder–decoder structure enhanced with:

- KAN-based feature transformation layers  
- Multi-scale skip connections  
- Dedicated classification head  
- Shared feature encoder  

### Multi-Task Design

| Task | Output |
|------|--------|
| Segmentation | Binary vessel mask |
| Classification | 4-class disease prediction (N, D, G, A) |

---

## 📊 Experimental Results

### 🔹 Classification Performance

- **Validation Accuracy:** ~85%
- **Test Accuracy:** ~75–76%
- Balanced performance across 4 classes

Confusion matrix shows strong diagonal dominance with minor confusion between similar pathological classes.

---

### 🔹 Segmentation Performance

- **Train IoU:** ~0.56  
- **Validation IoU:** ~0.54  

Stable convergence with minimal overfitting.

---

### 🔹 Training Stability

- Smooth decreasing loss curves  
- Validation closely tracks training performance  
- No significant divergence between tasks  

---

## 📈 Example Metrics

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Segmentation Metrics
- IoU (Intersection over Union)
- Dice Score
- Pixel Accuracy

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
├── losses.py
├── metrics.py
├── utils.py
├── prepare_fives.py
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md

---

## 📊 Dataset

This project uses the **FIVES retinal dataset**.

The dataset is NOT included in this repository.

### 📥 Setup

Download the dataset manually and place it inside:

datasets/FIVES/

vbnet
Copy code

Expected directory structure:

datasets/
└── FIVES/
├── train/
│ ├── images/
│ └── masks/
└── test/
├── images/
└── masks/

---

## ⚙️ Installation

### Option 1 – pip

```bash
pip install -r requirements.txt
Option 2 – Conda
bash
Copy code
conda env create -f environment.yml
conda activate seg_ukan
🚀 Training
bash
Copy code
python train.py
🧪 Evaluation
bash
Copy code
python test_eval.py
📦 Outputs
Training outputs are stored in:

Copy code
outputs/
Includes:

Model checkpoints (.pth)

Loss curves

IoU curves

Accuracy curves

Confusion matrices

This directory is excluded from version control.

🛠 Technical Details
Python 3.10+

PyTorch

torchvision

NumPy

scikit-learn

Matplotlib

🔬 Future Improvements
Class-balanced loss functions

Focal loss for classification

Advanced augmentation

Deep supervision for segmentation

Cross-dataset evaluation

👤 Author
Taraka Ram Paladugu

📜 License
MIT License

