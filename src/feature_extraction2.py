import cv2
import numpy as np
import os
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Define class mappings (ID to folder name)
class_map = {
    0: 'glass',
    1: 'paper',
    2: 'cardboard',
    3: 'plastic',
    4: 'metal',
    5: 'trash'
}

# Load a pretrained ResNet50 as feature extractor (removes the final FC layer)
# This gives a 2048-dimensional feature vector per image
device = torch.device("cpu")
print("Using device:", device)

model = models.resnet50(pretrained=True)  # or weights=models.ResNet50_Weights.IMAGENET1K_V1
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Up to avgpool
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()  # Set to evaluation mode (important for BatchNorm/Dropout)

# Preprocessing pipeline (matches what ResNet expects)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract CNN features (ResNet50) from an image path
def extract_cnn_features(image_path):
    try:
        # Load with PIL (better compatibility with transforms)
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not load image {image_path}: {e}")
    
    # Apply preprocessing
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    input_batch = input_batch.to(device)
    
    # Extract features
    with torch.no_grad():
        features = feature_extractor(input_batch)
    
    # Flatten to 2048-dim vector
    features = features.flatten().cpu().numpy()
    return features

# Function to extract CNN features (ResNet50) from an image
def extract_cnn_features(image):
    # Apply preprocessing
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    input_batch = input_batch.to(device)
    
    # Extract features
    with torch.no_grad():
        features = feature_extractor(input_batch)
    
    # Flatten to 2048-dim vector
    features = features.flatten().cpu().numpy()
    return features


# Function to extract features for all images in a folder and save as .npy
def process_folder(folder_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    features_list = []
    labels_list = []
    
    for class_id, class_name in class_map.items():
        class_dir = folder_path / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory {class_dir} not found.")
            continue
        
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    features = extract_cnn_features(img_file)
                    features_list.append(features)
                    labels_list.append(class_id)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
    
    if features_list:
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        np.save(output_dir / 'features.npy', features_array)
        np.save(output_dir / 'labels.npy', labels_array)
        print(f"Extracted CNN features for {len(features_list)} images. Feature shape: {features_array.shape[1]}")
    else:
        print("No images processed.")

# Example usage
BASE_DIR = Path(__file__).resolve().parent.parent

train_folder = BASE_DIR / "train"
train_output_dir = BASE_DIR / "features" / "train"
process_folder(train_folder, train_output_dir)

test_folder = BASE_DIR / "test"
test_output_dir = BASE_DIR / "features" / "test"
process_folder(test_folder, test_output_dir)
