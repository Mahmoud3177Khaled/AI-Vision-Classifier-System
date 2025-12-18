import cv2
import numpy as np
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

class_map = {
    0: 'glass',
    1: 'paper',
    2: 'cardboard',
    3: 'plastic',
    4: 'metal',
    5: 'trash'
}



device= torch.device("cpu")
model= models.resnet50(pretrained=True)
extractor= torch.nn.Sequential(*list(model.children())[:-1])
extractor= extractor.to(device)
extractor.eval()
preprocess= transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#----------------------------------------------------------------

def extract_cnn_features(path):
    try:
        image = Image.open(path).convert("RGB")
    except Exception as exep:
        raise ValueError(f"Could Not Load Image {path}: {exep}")

    tensor = preprocess(image)
    batch = tensor.unsqueeze(0)
    batch = batch.to(device)

    with torch.no_grad():
        features = extractor(batch)

    features = features.flatten().cpu().numpy()
    return features
#used in live app
def extract_cnn_features_live(image):
    tensor = preprocess(image)
    batch = tensor.unsqueeze(0)
    batch = batch.to(device)
    with torch.no_grad():
        features = extractor(batch)

    features = features.flatten().cpu().numpy()
    return features

def process_folder(folder_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    features_list = []
    labels_list = []
    for id, name in class_map.items():
        class_dir = folder_path / name
        if not class_dir.exists():
            print(f"Warning: Class Directory {class_dir} not Found.")
            continue
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    features = extract_cnn_features(img_file)
                    features_list.append(features)
                    labels_list.append(id)
                except Exception as ex:
                    print(f"Error Processing {img_file}: {ex}")
    if features_list:
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        np.save(output_dir / 'features.npy', features_array)
        np.save(output_dir / 'labels.npy', labels_array)
        print(f"Extracted CNN features for {len(features_list)} images. Feature shape: {features_array.shape[1]}")
    else:
        print("No images processed.")

if __name__ == "__main__":
    baseDir = Path(__file__).resolve().parent.parent

    train_folder = baseDir / "train"
    train_output_dir = baseDir / "features" / "train"
    process_folder(train_folder, train_output_dir)

    test_folder = baseDir / "test"
    test_output_dir = baseDir / "features" / "test"
    process_folder(test_folder, test_output_dir)
