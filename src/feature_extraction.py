import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
import os
from pathlib import Path

# Define class mappings (ID to folder name)
class_map = {
    0: 'glass',
    1: 'paper',
    2: 'cardboard',
    3: 'plastic',
    4: 'metal',
    5: 'trash'
}

# Function to extract HOG features
def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), block_norm='L2-Hys')
    return features

# Function to extract Color Histogram features (HSV space, 8x4x4 bins)
def extract_color_hist(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Function to extract LBP features (uniform, radius=3, points=24)
def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 26), range=(0, 25))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Combined feature extraction function
def extract_features(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.resize(image, (128, 128))  # Resize for consistency
    
    hog_features = extract_hog(image)
    color_features = extract_color_hist(image)
    lbp_features = extract_lbp(image)
    
    # Concatenate all features
    combined = np.concatenate([hog_features, color_features, lbp_features])
    return combined

# Function to extract features for all images in a folder and save as .npy
def process_folder(folder_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    features_list = []
    labels_list = []
    
    for class_id, class_name in class_map.items():
        class_dir = folder_path / class_name
        if not class_dir.exists():
            continue
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    features = extract_features(img_file)
                    features_list.append(features)
                    labels_list.append(class_id)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
    
    # Save as numpy arrays
    np.save(output_dir / 'features.npy', np.array(features_list))
    np.save(output_dir / 'labels.npy', np.array(labels_list))
    print(f"Extracted features for {len(features_list)} images. Shape: {features_list[0].shape if features_list else 'N/A'}")

# Example: Process the test folder
BASE_DIR = Path(__file__).resolve().parent.parent

test_folder = BASE_DIR / "train"
output_dir = BASE_DIR / "features" / "train"
process_folder(test_folder, output_dir)

test_folder = BASE_DIR / "test"
output_dir = BASE_DIR / "features" / "test"
process_folder(test_folder, output_dir)
