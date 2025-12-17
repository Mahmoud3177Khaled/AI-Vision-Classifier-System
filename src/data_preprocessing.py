import os
import random
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import albumentations as A

# ======================================================
# Reproducibility
# ======================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ======================================================
# Global configuration
# ======================================================
IMG_THRESHOLD = 600
SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}

DATASET_DIR = Path("../dataset")
TRAIN_DIR = Path("../train")
TEST_DIR = Path("../test")

# ======================================================
# Per-class augmentation strength
# ======================================================
CLASS_STRENGTH = {
    "cardboard": 1.0,
    "glass": 0.8,     # preserve fragile edges
    "metal": 1.1,     # tolerate more noise
    "paper": 0.9,
    "plastic": 1.0,
    "trash": 1.3      # most diverse → strongest augmentation
}

# ======================================================
# Utility functions
# ======================================================

def is_valid_image(path: Path) -> bool:
    """Checks whether an image can be safely opened."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_clean_images(folder: Path):
    """Loads valid image paths from a folder."""
    images = []
    for file in folder.iterdir():
        if file.suffix.lower() in SUPPORTED_EXT and is_valid_image(file):
            images.append(file)
    return images


def read_image(path: Path):
    """Reads image as RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def save_augmented_image(
    image: np.ndarray,
    save_dir: Path,
    class_name: str,
    phase: int,
    source_name: str,
    index: int
):
    """Saves augmented image with traceable filename."""
    filename = (
        f"{class_name}_p{phase}_"
        f"{source_name}_aug_{index}.jpg"
    )
    save_path = save_dir / filename

    cv2.imwrite(
        str(save_path),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )


def split_original_images(clean_images, class_name: str):
    """
    Splits already-cleaned original images into train/test.
    """
    images = clean_images.copy()
    random.shuffle(images)

    if class_name == "trash":
        test_size = min(35, len(images) // 2)
    else:
        test_size = math.ceil(0.2 * len(images))

    test_images = images[:test_size]
    train_images = images[test_size:]

    # Create directories
    train_class_dir = TRAIN_DIR / class_name
    test_class_dir = TEST_DIR / class_name
    train_class_dir.mkdir(parents=True, exist_ok=True)
    test_class_dir.mkdir(parents=True, exist_ok=True)

    # Copy originals
    for img in train_images:
        shutil.copy(img, train_class_dir / img.name)

    for img in test_images:
        shutil.copy(img, test_class_dir / img.name)

    return train_images


# ======================================================
# Phase-based transformations
# ======================================================

def phase1_transform(h, w, strength):
    """Geometric + lighting robustness."""
    return A.Compose([
        A.Rotate(limit=int(15 * strength), p=0.7),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2 * strength,
            contrast_limit=0.15 * strength,
            p=0.8
        ),
        A.RandomResizedCrop(
            size=(h, w),
            scale=(0.80, 0.95),
            ratio=(0.75, 1.33),
            p=1.0
        )
    ])


def phase2_transform(strength):
    """Sensor noise + motion blur + color-safe lighting."""

    max_kernel = int(5 * strength)
    if max_kernel % 2 == 0:
        max_kernel += 1

    return A.Compose([
        # --- COLOR-SAFE LIGHTING ---
        A.HueSaturationValue(
            hue_shift_limit=0,              # NO hue changes
            sat_shift_limit=0,              # NO saturation changes
            val_shift_limit=int(20 * strength),  # brightness only
            p=0.7
        ),

        # --- CAMERA EFFECTS ---
        A.MotionBlur(
            blur_limit=(3, max_kernel),
            p=0.5
        ),

        # --- SENSOR NOISE ---
        A.GaussNoise(
            std_range=(0.01 * strength, 0.05 * strength),
            p=0.4
        ),
    ])


def phase3_transform(h, w, strength):
    return A.Compose([
        A.RandomResizedCrop(
            size=(h, w),
            scale=(0.85, 1.0),
            ratio=(0.95, 1.05),
            p=1.0
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.05 * strength,
            contrast_limit=0.05 * strength,
            p=0.5
        ),
    ])


# ======================================================
# Core augmentation logic
# ======================================================

def augment_class(class_name: str, train_images):
    train_class_dir = TRAIN_DIR / class_name
    strength = CLASS_STRENGTH[class_name]

    # current_count includes ONLY original training images
    # augmented images are added directly to disk
    current_count = len(train_images)
    needed = IMG_THRESHOLD - current_count

    if needed <= 0:
        return

    per_phase = math.ceil(needed / 3)
    aug_count = 0

    for phase in range(1, 4):
        for _ in range(per_phase):
            if aug_count >= needed:
                return

            src = random.choice(train_images)
            img = read_image(src)
            h, w = img.shape[:2]

            if phase == 1:
                transform = phase1_transform(h, w, strength)
            elif phase == 2:
                transform = phase2_transform(strength)
            else:
                transform = phase3_transform(h, w, strength)

            augmented = transform(image=img)["image"]

            save_augmented_image(
                augmented,
                train_class_dir,
                class_name,
                phase,
                src.stem,
                aug_count
            )

            aug_count += 1


# ======================================================
# Entry point
# ======================================================
def main():
    if TRAIN_DIR.exists():
        shutil.rmtree(TRAIN_DIR)
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    TRAIN_DIR.mkdir()
    TEST_DIR.mkdir()

    for class_folder in DATASET_DIR.iterdir():
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name

        if class_name not in CLASS_STRENGTH:
            print(f"Skipping unknown class: {class_name}")
            continue

        print(f"\nProcessing {class_name}")

        clean_images = load_clean_images(class_folder)

        if len(clean_images) == 0:
            print(f"No valid images found for {class_name}, skipping.")
            continue

        train_images = split_original_images(clean_images, class_name)
        augment_class(class_name, train_images)

        print(f"{class_name}: training set augmented to {IMG_THRESHOLD}")


if __name__ == "__main__":
    main()

