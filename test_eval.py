import os
import argparse
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

import archs
from dataset import FIVESDataset
from albumentations import Resize, Normalize
from albumentations.core.composition import Compose


# ---------------------------
# Arguments
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='datasets/FIVES')
    parser.add_argument('--model_path', default='outputs/best_model.pth')
    parser.add_argument('--batch_size', default=4, type=int)
    return parser.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------------------
    # Test Dataset
    # ---------------------------
    test_img_dir = os.path.join(args.data_dir, "test/images")
    test_mask_dir = os.path.join(args.data_dir, "test/masks")

    all_paths = sorted(glob(os.path.join(test_img_dir, "*.png")))
    test_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_paths]

    print(f"Test samples: {len(test_ids)}")

    test_transform = Compose([
        Resize(256, 256),
        Normalize()
    ])

    test_dataset = FIVESDataset(
        test_ids,
        test_img_dir,
        test_mask_dir,
        transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # ---------------------------
    # Load Model
    # ---------------------------
    model = archs.UKAN(num_classes=1, cls_classes=4).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # ---------------------------
    # Evaluation
    # ---------------------------
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks, labels in tqdm(test_loader):

            images = images.to(device)
            labels = labels.to(device)

            _, cls_out = model(images)
            _, preds = torch.max(cls_out, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ---------------------------
    # Overall Accuracy
    # ---------------------------
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n🔥 TEST ACCURACY: {accuracy:.2f}%")

    # ---------------------------
    # Classification Report
    # ---------------------------
    print("\n📊 Classification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=["Normal", "Diabetic", "Glaucoma", "Age"]
    ))

    # ---------------------------
    # Confusion Matrix
    # ---------------------------
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["N", "D", "G", "A"],
        yticklabels=["N", "D", "G", "A"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Test Confusion Matrix")
    plt.savefig("test_confusion_matrix.png")
    plt.show()

    print("Test evaluation complete.")


if __name__ == "__main__":
    main()
