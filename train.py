import os
import argparse
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from albumentations import (
    RandomRotate90,
    HorizontalFlip,
    VerticalFlip,
    Affine,
    Resize,
    Normalize
)
from albumentations.core.composition import Compose

import archs
import losses
from dataset import FIVESDataset
from metrics import iou_score


# -------------------------------------------------
# Arguments
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=70, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--data_dir', default='datasets/FIVES')
    parser.add_argument('--output_dir', default='outputs')
    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------------------------------
    # DATASET SPLIT
    # -------------------------------------------------
    train_img_dir = os.path.join(args.data_dir, "train/images")
    train_mask_dir = os.path.join(args.data_dir, "train/masks")

    all_paths = sorted(glob(os.path.join(train_img_dir, "*.png")))
    all_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_paths]

    train_ids, val_ids = train_test_split(
        all_ids, test_size=0.2, random_state=42
    )

    print(f"Train samples: {len(train_ids)}")
    print(f"Val samples:   {len(val_ids)}")

    # -------------------------------------------------
    # AUGMENTATIONS
    # -------------------------------------------------
    train_transform = Compose([
        RandomRotate90(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Affine(scale=(0.9, 1.1),
               translate_percent=(0.05, 0.05),
               rotate=(-15, 15),
               p=0.5),
        Resize(256, 256),
        Normalize()
    ])

    val_transform = Compose([
        Resize(256, 256),
        Normalize()
    ])

    train_dataset = FIVESDataset(train_ids, train_img_dir, train_mask_dir, transform=train_transform)
    val_dataset = FIVESDataset(val_ids, train_img_dir, train_mask_dir, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size
    )

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------
    model = archs.UKAN(num_classes=1, cls_classes=4).to(device)

    seg_criterion = losses.BCEDiceLoss().to(device)

    class_weights = torch.tensor([1.6, 1.0, 1.1, 1.0]).to(device)
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -------------------------------------------------
    # Early Stopping Setup
    # -------------------------------------------------
    best_score = 0
    patience = 10
    early_stop_counter = 0

    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    train_accs, val_accs = [], []

    # -------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------
    for epoch in range(args.epochs):

        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0
        train_iou = 0
        correct = 0
        total = 0

        for images, masks, labels in tqdm(train_loader):

            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            seg_out, cls_out = model(images)

            seg_loss = seg_criterion(seg_out, masks)
            cls_loss = cls_criterion(cls_out, labels)

            loss = seg_loss + 1.2 * cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iou, _, _ = iou_score(seg_out, masks)

            _, preds = torch.max(cls_out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_loss += loss.item()
            train_iou += iou

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_acc = 100 * correct / total

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0
        val_iou = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, masks, labels in val_loader:

                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                seg_out, cls_out = model(images)

                seg_loss = seg_criterion(seg_out, masks)
                cls_loss = cls_criterion(cls_out, labels)

                loss = seg_loss + 1.2 * cls_loss

                iou, _, _ = iou_score(seg_out, masks)

                _, preds = torch.max(cls_out, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                val_loss += loss.item()
                val_iou += iou

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_acc = 100 * correct / total

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Acc: {val_acc:.2f}%")

        # Combined Score
        combined_score = (val_iou + val_acc/100) / 2

        # ---------------- SAVE + EARLY STOP ----------------
        if combined_score > best_score:
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_model.pth"))
            best_score = combined_score
            early_stop_counter = 0
            print("Saved best model.")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    print("\nTraining Complete.")


if __name__ == "__main__":
    main()
