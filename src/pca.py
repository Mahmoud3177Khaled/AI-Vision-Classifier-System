# Run only once to avoid randomness of Scaler and PCA =========
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib

# Retrieve features and labels of train and test data
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
if __name__ == "__main__":

    scaler = StandardScaler()
    pca = PCA(n_components=0.98)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "scaler.pkl")
    joblib.dump(pca, model_dir / "PCA.pkl")
    print("Scaler saved to ../classifier/scaler.pkl")
    print("PCA saved to ../classifier/PCA.pkl")

