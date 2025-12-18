from pathlib import Path
import numpy as np
import joblib
from feature_extraction2 import *
from pca import *

def predict(dataFilePath, bestModelPath = Path(__file__).resolve().parent.parent / "classifier"):
    #load data
    baseDir = Path(__file__).resolve().parent.parent

    #feature extraction
    train_output_dir = baseDir / "features"
    process_folder(dataFilePath, train_output_dir)

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
    pca = joblib.load(bestModelPath / "svm_PCA.pkl")

    X = scaler.transform(X)
    X = pca.transform(X)
    y_pred = svm.predict(X)
    print("Predicted classes:", y_pred)

    return y_pred