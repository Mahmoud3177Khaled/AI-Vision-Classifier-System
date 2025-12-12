import os
import random
import shutil


import cv2
import numpy as np
import pandas as pd
import albumentations as A
from PIL import Image
from pathlib import Path
import math
from pathlib import Path

from matplotlib import pyplot as plt

#################### Data Initialization ###########################

dataset_dir = Path("../dataset")

cardboard = []
glass = []
metal = []
paper = []
plastic = []
trash = []

# Dictionary to map folder names to lists
folder_to_list = {
    "cardboard": cardboard,
    "glass": glass,
    "metal": metal,
    "paper": paper,
    "plastic": plastic,
    "trash": trash
}

# Iterate over all folders in the dataset
for folder in dataset_dir.iterdir():
    if folder.is_dir() and folder.name in folder_to_list:
        # Iterate over all files in the folder (images)
        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                folder_to_list[folder.name].append(file)


print("Number of Images Before Removing Corrupted Images")
print("Cardboard images:", len(cardboard))
print("glass images:", len(glass))
print("metal images:", len(metal))
print("paper images:", len(paper))
print("plastic images:", len(plastic))
print("trash images:", len(trash))



########### Dropping Corrupted Pictures ######################
def drop_corrupted_images(img_paths):
    res=[]
    for path in img_paths:
        if(os.stat(path).st_size!=0):
            res.append(path)
    return res


final_cardboard = drop_corrupted_images(cardboard)
final_glass = drop_corrupted_images(glass)
final_metal = drop_corrupted_images(metal)
final_paper = drop_corrupted_images(paper)
final_plastic = drop_corrupted_images(plastic)
final_trash = drop_corrupted_images(trash)

print("Number of Images After Removing Corrupted Images")
print("Cardboard images:", len(final_cardboard))
print("glass images:", len(final_glass))
print("metal images:", len(final_metal))
print("paper images:", len(final_paper))
print("plastic images:", len(final_plastic))
print("trash images:", len(final_trash))

########### randomizing & Split For Test and Train ###########

def random_split(img_paths, percentage_train, folder_name):
    # Split sizes
    train_size = math.ceil(percentage_train * len(img_paths))

    # Randomly sample train images
    train_data = random.sample(img_paths, train_size)
    test_data = [img for img in img_paths if img not in train_data]

    # Paths
    train_path = Path(f"../train/{folder_name}")
    test_path = Path(f"../test/{folder_name}")

    # Remove existing folders
    if train_path.exists():
        shutil.rmtree(train_path)
    if test_path.exists():
        shutil.rmtree(test_path)

    # Create empty folders
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # Copy images
    for img in train_data:
        shutil.copy(img, train_path / Path(img).name)

    for img in test_data:
        shutil.copy(img, test_path / Path(img).name)

    return train_data, test_data

train_data_path = "../train"
test_data_path = "../test"

# Make sure parent folders exist
Path(train_data_path).mkdir(exist_ok=True, parents=True)
Path(test_data_path).mkdir(exist_ok=True, parents=True)

cardboard_train, cardboard_test = random_split(final_cardboard, 0.8, "cardboard")
glass_train, glass_test = random_split(final_glass, 0.8, "glass")
metal_train, metal_test = random_split(final_metal, 0.8, "metal")
paper_train, paper_test = random_split(final_paper, 0.8, "paper")
plastic_train, plastic_test = random_split(final_plastic, 0.8, "plastic")
trash_train, trash_test = random_split(final_trash, 0.8, "trash")


print("Number of Training Samples")
print("Cardboard images:", len(cardboard_train))
print("glass images:", len(glass_train))
print("metal images:", len(metal_train))
print("paper images:", len(paper_train))
print("plastic images:", len(plastic_train))
print("trash images:", len(trash_train))

# ########### Implementing 3 Phases of Augmentation ############

img_threshold = 500

def get_image_size_cv2(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]  # shape returns (height, width, channels)
    return width, height

# Phase 1 — Geometric & lighting diversity
def phase1_transform(original_height, original_width):
    return A.Compose([
        A.Rotate(limit=15, p=0.7),  # ±15 degrees
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.15, p=0.8),
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(
            size=(original_height, original_width),  # required in latest Albumentations
            scale=(0.8, 0.9),                        # crop 80–90% of area
            ratio=(0.75, 1.33),                       # default aspect ratio
            p=1.0
        )
    ])

def phase2_transform(original_height, original_width):
    return A.Compose([
        A.ColorJitter(brightness=0.2,contrast=0.15, hue=0.1,p=0.68),
        A.GaussNoise(std_range=(0.1,0.4)),
        A.MotionBlur(blur_limit=(3, 7)),

    ])

def phase3_transform(original_height, original_width):
    return A.Compose([
        A.RandomResizedCrop(
            size=(original_height, original_width),  # required in latest Albumentations
            scale=(0.8, 0.9),  # crop 80–90% of area
            ratio=(0.75, 1.33),  # default aspect ratio
            p=1.0
        ),
        A.ColorJitter(brightness=0.2, contrast=0.15, hue=0.1, p=0.8)
    ])


import random
import math
from PIL import Image
import numpy as np


def augment_image(image_paths, img_threshold):
    remaining_images = img_threshold - len(image_paths)
    augmented_images = []

    if remaining_images <= 0:
        return []  # already at or above threshold

    # Divide remaining images across 3 phases
    no_images_per_transformation = math.ceil(remaining_images / 3)

    for phase in range(1, 4):
        for _ in range(no_images_per_transformation):
            img_path = random.choice(image_paths)  # allow repetition
            img = np.array(Image.open(img_path))

            if phase == 1:
                transform = phase1_transform(img.shape[0], img.shape[1])
            elif phase == 2:
                transform = phase2_transform(img.shape[0], img.shape[1])
            else:
                transform = phase3_transform(img.shape[0], img.shape[1])

            aug_img = transform(image=img)['image']
            augmented_images.append(aug_img)

            if len(augmented_images) >= remaining_images:
                break  # stop if we reached threshold

        if len(augmented_images) >= remaining_images:
            break

    return augmented_images


aug_cardboard = augment_image(cardboard_train, img_threshold)
print("Successfully augmented cardboard images\n length=",len(aug_cardboard))
aug_paper = augment_image(paper_train, img_threshold)
print("Successfully augmented paper images\n length=",len(aug_paper))
aug_glass = augment_image(glass_train, img_threshold)
print("Successfully augmented glass images\n length=",len(aug_glass))
aug_metal = augment_image(metal_train, img_threshold)
print("Successfully augmented metal images\n length=",len(aug_metal))
aug_plastic = augment_image(plastic_train, img_threshold)
print("Successfully augmented plastic images\n length=",len(aug_plastic))
aug_trash = augment_image(trash_train, img_threshold)
print("Successfully augmented trash images\n length=",len(aug_trash))

def addToFolder(image_list, folder_name):
    # Make sure the folder exists
    os.makedirs(folder_name, exist_ok=True)

    for i, img in enumerate(image_list):
        image_name = f"{i}.jpg"  
        save_path = os.path.join(folder_name, image_name)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)

# Base folder for your train dataset
train_data_path = "../train"

# Add images to each class folder
addToFolder(aug_cardboard, os.path.join(train_data_path, "cardboard"))
print("Successfully added augmented cardboard images")

addToFolder(aug_paper, os.path.join(train_data_path, "paper"))
print("Successfully added augmented paper images")

addToFolder(aug_glass, os.path.join(train_data_path, "glass"))
print("Successfully added augmented glass images")

addToFolder(aug_metal, os.path.join(train_data_path, "metal"))
print("Successfully added augmented metal images")

addToFolder(aug_plastic, os.path.join(train_data_path, "plastic"))
print("Successfully added augmented plastic images")

addToFolder(aug_trash, os.path.join(train_data_path, "trash"))
print("Successfully added augmented trash images")

