import random
import math
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import albumentations as A

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib

from scipy.stats import loguniform
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from pca import *
from sklearn.svm import SVC


def predict(dataFilePath, bestModelPath):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    IMG_THRESHOLD = 600
    SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}

    TRAIN_DIR = Path("../train")
    TEST_DIR = Path("../test")

    CLASS_STRENGTH = {
        "cardboard": 1.0,
        "glass": 0.8,     # preserve fragile edges
        "metal": 1.1,     # tolerate more noise
        "paper": 0.9,
        "plastic": 1.0,
        "trash": 1.3      # most diverse → strongest augmentation
    }

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

    if TRAIN_DIR.exists():
        shutil.rmtree(TRAIN_DIR)
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    TRAIN_DIR.mkdir()
    TEST_DIR.mkdir()

    for class_folder in dataFilePath.iterdir():
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

    class_map = {
        0: 'glass',
        1: 'paper',
        2: 'cardboard',
        3: 'plastic',
        4: 'metal',
        5: 'trash'
    }

    device = torch.device("cpu")
    print("Using device:", device)

    model = models.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_cnn_features(image_path):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not load image {image_path}: {e}")

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)

        with torch.no_grad():
            features = feature_extractor(input_batch)

        features = features.flatten().cpu().numpy()
        return features

    def extract_cnn_features(image):
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)

        with torch.no_grad():
            features = feature_extractor(input_batch)

        features = features.flatten().cpu().numpy()
        return features

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

    #PCA and scaling
    BASE_DIR = Path(__file__).resolve().parent.parent

    train_folder = BASE_DIR / "train"
    train_output_dir = BASE_DIR / "features" / "train"
    process_folder(train_folder, train_output_dir)

    test_folder = BASE_DIR / "test"
    test_output_dir = BASE_DIR / "features" / "test"
    process_folder(test_folder, test_output_dir)

    def get_data(features_dir):
        try:
            X_train = np.load(features_dir / "train" / "features.npy")
            y_train = np.load(features_dir / "train" / "labels.npy")
            X_test = np.load(features_dir / "test" / "features.npy")
            y_test = np.load(features_dir / "test" / "labels.npy")
        except Exception as e:
            print(f"Error: Failed to load data!: {e}")

        print("Raw Features: ")
        print("X train:", X_train.shape)  # (N, D)
        print("Y train:", y_train.shape)  # (N,)
        print("X test:", X_test.shape)  # (N, D)
        print("Y test:", y_test.shape)  # (N,)
        # If very close to 0 → features are nearly identical
        print("Average std:", np.std(X_train, axis=0).mean())

        for label in range(6):
            idx = y_train == label
            plt.scatter(
                X_train[idx, 0],  # first feature
                X_train[idx, 1],  # second feature
                label=f'Class {label}', alpha=0.7, s=50)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Scatter plot of raw training features')
        plt.legend()
        plt.show()

        return X_train, y_train, X_test, y_test

    # Scale features
    def scale(scaler, X_train, X_test):
        # Scale
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test

    def apply_PCA(scaler, pca, X_train, X_test, y_train):
        # Scaling
        X_train, X_test = scale(scaler, X_train, X_test)
        
        # Apply PCA
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        print("\nAfter Scaling and PCA:")
        print("X train:", X_train.shape)  # (N, D)
        print("X test:", X_test.shape)  # (N, D)

        # If very close to 0 → features are nearly identical
        print("Average std:", np.std(X_train, axis=0).mean())

        scatter = plt.scatter(
            X_train[:, 0],  # 1st principal component
            X_train[:, 1],  # 2nd principal component
            c=y_train, cmap='tab10', alpha=0.7, s=50)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA scatter plot of training data')
        plt.colorbar(scatter, label='Class label')
        plt.show()
        return X_train, X_test

    features_dir = Path(__file__).resolve().parent.parent / "features"
    model_dir = Path(__file__).resolve().parent.parent / "classifier"

    scaler = StandardScaler()
    pca = PCA(n_components=0.98)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "scaler.pkl")
    joblib.dump(pca, model_dir / "PCA.pkl")
    print("Scaler saved to ../classifier/scaler.pkl")
    print("PCA saved to ../classifier/PCA.pkl")

    #SVM Model
    X_train, y_train, X_test, y_test = get_data(features_dir)
    scaler = joblib.load(model_dir / "scaler.pkl")
    X_train, X_test = scale(scaler, X_train, X_test)

    svm_model = SVC(
        C=10,
        gamma='scale',
        kernel='rbf',
        class_weight='balanced',
        decision_function_shape='ovr',
        probability=True
    )


    def trainSVM(model, x_train_, y_train_):
        save_dir = Path("../classifier")
        save_dir.mkdir(parents=True, exist_ok=True)  # Ensure folder exists
        model.fit(x_train_, y_train_)
        joblib.dump(model, save_dir / "svm_model.pkl")  # Will overwrite if exists
        joblib.dump(scaler, save_dir / "svm_scaler.pkl")
        print("\nModel saved to ../classifier/svm_model.pkl")
        print("SVM Scaler saved to ../classifier/svm_scaler.pkl")

    def getAccuracy( X, y,svm_model_dir=Path("../classifier")):
        svm_model = joblib.load(svm_model_dir / "svm_model.pkl")
        acc = svm_model.score(X, y)
        print("\nAccuracy of svm:", acc, "\n")
        return acc

    def predictSVM(sample, threshold=0.6,svm_model_dir=Path("../classifier")):
        classes = ["glass", "paper", "cardboard", "plastic", "metal", "trash", "unknown"]

        svm_model= joblib.load(svm_model_dir / "svm_model.pkl")
        probs = svm_model.predict_proba(sample)
        pred_class_index = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)
        pred_class_names = [classes[i] for i in pred_class_index]
        final_preds = ["unknown" if mp < threshold else name
                    for name, mp in zip(pred_class_names, max_probs)]

        # print("Predicted class:", final_preds)
        # print("Class probabilities:", probs)
        return final_preds


    trainSVM(svm_model,X_train,y_train)
    save_dir = Path("../classifier")
    getAccuracy(X_test,y_test)

    threshold = 0.6

    # print("Actual:",y_test)
    predictSVM(X_test,threshold)
