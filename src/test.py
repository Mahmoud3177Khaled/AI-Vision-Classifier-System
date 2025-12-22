from pathlib import Path
import numpy as np
import joblib
from feature_extraction2 import extract_cnn_features

import cv2
import time
import joblib
import numpy as np
from pathlib import Path
from PIL import Image

import os
import random
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import albumentations as A

import cv2
import numpy as np
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
import re
from train_svm import *
import csv


def extract_number(path: Path):
    match = re.search(r'\((\d+)\)', path.stem)
    return int(match.group(1)) if match else float('inf')

def predict(dataFilePath, bestModelPath = Path(__file__).resolve().parent.parent / "classifier"):
    #feature extraction
    csv_path = dataFilePath / "predictions.csv"
    features_list = []
    image_names = []
    image_paths = sorted(
        (p for p in dataFilePath.iterdir()
         if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]),
        key=extract_number
    )

    for img in image_paths:
        try:
            features = extract_cnn_features(img)
            features_list.append(features)
            image_names.append(img.name)
            # print(img.name)
        except Exception as ex:
            print(f"Error Processing {img}: {ex}")
    
    X = np.array(features_list)

    #load model
    svm = joblib.load(bestModelPath / "svm_model.pkl")
    print("Model classes:", svm.classes_)
    scaler = joblib.load(bestModelPath / "svm_scaler.pkl")

    X = scaler.transform(X)
    y_pred = svm.predict(X)
    print("Predicted classes:", y_pred)
    # l = predictSVM(X)
    # for i in range(len(l)):
    #     print(img.name,l[i])

    preds = predictSVM(X)

    for name, pred in zip(image_names, preds):
        print(name, pred)

    return y_pred


p = Path("D:\Year 4\Machine Learning\Project\AI-Vision-Classifier-System\set-B")

predict(p)