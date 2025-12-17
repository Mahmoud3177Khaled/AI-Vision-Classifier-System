import numpy as np
from sklearn.neighbors import KNeighborsClassifier as Knn
import matplotlib.pyplot as plt
import joblib
from pca import *

# Compute accuracies over a range of Ks to get best K
def get_best_k(X_train, y_train, X_test, y_test):
    k_values = range(1, 21, 2)  # get odd K
    accuracies = []
    for k in k_values:
        knn = Knn(n_neighbors=k, algorithm="auto", weights="distance",
                    metric="cosine", n_jobs=-1)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        accuracies.append(acc)

    # plt.plot(k_values, accuracies)
    # plt.xlabel("k")
    # plt.ylabel("Accuracy")
    # plt.show()

    # Best K is K of max accuracy (if multiple get farther for larger k)
    accuracies = np.array(accuracies)
    best_k = k_values[np.where(accuracies == accuracies.max())[0][-1]]
    return best_k


def run_knn(features_dir, model_dir, scaler, pca):
    # Data Preprocessing
    X_train, y_train, X_test, y_test = get_data(features_dir)
    X_train, X_test = apply_PCA(scaler, pca, X_train, X_test, y_train)
    k_neighbors = get_best_k(X_train, y_train, X_test, y_test)

    #Fit to model
    knn = Knn(n_neighbors=k_neighbors, algorithm="auto", weights="distance",
                metric="cosine", n_jobs=-1)#use all CPU cores
    knn.fit(X_train, y_train)

    # Compute Accuracy
    print("\nAccuracy of knn:", knn.score(X_test, y_test), ",with K =", k_neighbors)

    # Save model
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(knn, model_dir / "knn_model.pkl")
    joblib.dump(scaler, model_dir / "knn_scaler.pkl")
    joblib.dump(pca, model_dir / "knn_PCA.pkl")
    print("\nModel saved to ../classifier/knn_model.pkl")
    print("\nKnn Scaler saved to ../classifier/knn_scaler.pkl")
    print("\nKnn PCA saved to ../classifier/knn_PCA.pkl")

    return X_test

def predict_knn(X_test, model_dir, threshold = 0.6):
    classes = ["glass", "paper", "cardboard", "plastic", "metal", "trash", "unknown"]

    knn = joblib.load(model_dir / "knn_model.pkl")
    probs = knn.predict_proba(X_test)
    max_probs = probs.max(axis=1)

    distances, _ = knn.kneighbors(X_test, n_neighbors=1)
    min_dist = distances[:, 0]
    y_pred = knn.predict(X_test)

    unknown_mask = (max_probs < threshold) | (min_dist > threshold)
    y_pred[unknown_mask] = 6    #unknown class id
    pred_class_names = [classes[i] for i in y_pred]
    num_unknown = np.sum(unknown_mask)

    print("\nNumber of samples classified as unknown: ", num_unknown)
    print("Predicted classes:", pred_class_names)


features_dir = Path(__file__).resolve().parent.parent / "features"
model_dir = Path(__file__).resolve().parent.parent / "classifier"

scaler = joblib.load(model_dir / "scaler.pkl")
pca = joblib.load(model_dir / "PCA.pkl")
X_test = run_knn(features_dir, model_dir, scaler, pca)
predict_knn(X_test, model_dir, 0.6)
