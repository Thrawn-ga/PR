# Seg_UKAN – Retinal Vessel Segmentation using U-KAN

This repository implements a segmentation framework based on U-KAN for retinal vessel segmentation using the FIVES dataset.

---

## 📌 Overview

Retinal vessel segmentation is a critical task in medical image analysis for diagnosing diseases such as:

- Diabetic Retinopathy
- Glaucoma
- Hypertension-related retinopathy

This project uses a U-KAN-based architecture to perform pixel-wise segmentation of retinal vessels.

---

## 🧠 Model Architecture

The model combines:

- U-shaped encoder–decoder structure
- KAN-based feature transformation
- Skip connections for multi-scale feature fusion

---

## 📂 Project Structure

Seg_UKAN/
│
├── archs.py
├── dataset.py
├── train.py
├── val.py
├── config.py
├── utils.py
├── losses.py
├── metrics.py
├── prepare_fives.py
├── requirements.txt
├── environment.yml
├── .gitignore
│
└── datasets/ (not included in repo)
└── outputs/ (generated during training)

yaml
Copy code

---

## 📊 Dataset

This project uses the **FIVES dataset** for retinal vessel segmentation.

The dataset is NOT included in this repository.

### 📥 Download Instructions

Download FIVES dataset manually and place it in:

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

---

## ⚙️ Installation

### Option 1 – Using pip

```bash
pip install -r requirements.txt
Option 2 – Using Conda
bash
Copy code
conda env create -f environment.yml
conda activate seg_ukan
🚀 Training
To train the model:

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
The model supports:

IoU (Intersection over Union)

Dice Score

Confusion Matrix

Pixel Accuracy

📦 Outputs
Training outputs are stored in:

Copy code
outputs/
This folder contains:

Model checkpoints (.pth)

Accuracy curves

Loss curves

Confusion matrices

This folder is excluded from version control.

🛠 Requirements
Python 3.10+

PyTorch

torchvision

numpy

matplotlib

scikit-learn

(See requirements.txt for full list.)

🧑‍💻 Author
Taraka Ram Paladugu

📜 License
This project is released under the MIT License.
