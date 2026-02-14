import os
import cv2
import matplotlib.pyplot as plt

output_dir = "outputs"

files = [
    "loss_curve.png",
    "iou_curve.png",
    "acc_curve.png",
    "confusion_matrix.png"
]

for f in files:
    path = os.path.join(output_dir, f)

    if not os.path.exists(path):
        print(f"{f} not found.")
        continue

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6,5))
    plt.imshow(img)
    plt.title(f)
    plt.axis("off")
    plt.show()

print("Done.")
