from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from pca import *
import numpy as np
from sklearn.svm import SVC

# Load preprocessing objects

X_train, y_train, X_test, y_test = get_data(features_dir)
scaler = joblib.load(model_dir / "scaler.pkl")
pca = joblib.load(model_dir / "PCA.pkl")


svm = SVC(
    C=10,
    gamma='scale',
    kernel='rbf',
    class_weight='balanced',
    decision_function_shape='ovr'
)

svm.fit(X_train, y_train)

# Evaluate on test set
test_acc = svm.score(X_test, y_test)
print("Test accuracy:", test_acc)
