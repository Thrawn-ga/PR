import os
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import confusion_matrix

import archs
from dataset import FIVESDataset
from albumentations import Resize, Normalize
from albumentations.core.composition import Compose

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Paths
data_dir = "datasets/FIVES"
test_img_dir = os.path.join(data_dir, "train/images")
test_mask_dir = os.path.join(data_dir, "train/masks")

# Get all image IDs
all_paths = sorted(glob(os.path.join(test_img_dir, "*.png")))
all_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_paths]

# Transform
transform = Compose([
    Resize(256, 256),
    Normalize(),
])

# Dataset + Loader
test_dataset = FIVESDataset(
    all_ids,
    test_img_dir,
    test_mask_dir,
    transform=transform
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)

# Load model
model = archs.UKAN(num_classes=1, cls_classes=4).to(device)
model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
model.eval()

# Collect predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for images, masks, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        _, cls_out = model(images)
        _, preds = torch.max(cls_out, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["N","D","G","A"],
            yticklabels=["N","D","G","A"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

print("Confusion matrix saved successfully.")
