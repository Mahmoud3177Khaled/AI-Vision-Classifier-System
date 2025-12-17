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
    print("\nModel saved to ../classifier/knn_model.pkl")
    joblib.dump(knn, model_dir / "knn_model.pkl")


scaler = joblib.load(model_dir / "scaler.pkl")
pca = joblib.load(model_dir / "PCA.pkl")
run_knn(features_dir, model_dir, scaler, pca)
