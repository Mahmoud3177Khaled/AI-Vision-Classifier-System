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



def predict(dataFilePath, bestModelPath = Path(__file__).resolve().parent.parent / "classifier"):
    #feature extraction
    features_list = []
    for img in dataFilePath.iterdir():
        if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            try:
                features = extract_cnn_features(img)
                features_list.append(features)
            except Exception as ex:
                print(f"Error Processing {img}: {ex}")
    
    X = np.array(features_list)

    #load model
    svm = joblib.load(bestModelPath / "svm_model.pkl")
    scaler = joblib.load(bestModelPath / "svm_scaler.pkl")

    X = scaler.transform(X)
    y_pred = svm.predict(X)
    print("Predicted classes:", y_pred)

    return y_pred