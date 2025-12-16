import cv2
import numpy as np
from pathlib import Path
import random

def destroy_img(img_path):
    # READ IMAGE
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    # Strong blur (OOD simulation)
    img = cv2.GaussianBlur(img, (51, 51), 0)

    # Darken / wash out
    img = cv2.convertScaleAbs(
        img,
        alpha=random.uniform(0.2, 0.6),
        beta=random.randint(-50, 50)
    )

    return img


classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


# Images for training
train_dir = Path("../train")
unknown_dir = train_dir / "unknown"
unknown_dir.mkdir(exist_ok=True)
TARGET_UNKNOWN = 360
cnt = 0
random.seed(42)

while cnt < TARGET_UNKNOWN:
    cls = random.choice(classes)
    img_path = random.choice(list((train_dir / cls).glob("*.jpg")))

    new_img = destroy_img(img_path)
    if new_img is None:
        continue

    cnt += 1
    out_path = unknown_dir / f"unknown_{cnt}.jpg"
    cv2.imwrite(str(out_path), new_img)

print(f"Created {cnt} training unknown images")



# Images for testing
test_dir = Path("../test")
unknown_dir = test_dir / "unknown"
unknown_dir.mkdir(exist_ok=True)
TARGET_UNKNOWN = 30
cnt = 0
random.seed(42)

while cnt < TARGET_UNKNOWN:
    cls = random.choice(classes)
    img_path = random.choice(list((test_dir / cls).glob("*.jpg")))

    new_img = destroy_img(img_path)
    if new_img is None:
        continue

    cnt += 1
    out_path = unknown_dir / f"unknown_{cnt}.jpg"
    cv2.imwrite(str(out_path), new_img)

print(f"Created {cnt} testing unknown images")

