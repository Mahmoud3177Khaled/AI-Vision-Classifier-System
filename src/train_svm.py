from scipy.stats import loguniform
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from pca import *
import numpy as np
from sklearn.svm import SVC

# Load preprocessing objects

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
    save_dir = Path(__file__).resolve().parent.parent / "classifier"
    save_dir.mkdir(parents=True, exist_ok=True)  # Ensure folder exists
    model.fit(x_train_, y_train_)
    joblib.dump(model, save_dir / "svm_model.pkl")  # Will overwrite if exists
    joblib.dump(scaler, save_dir / "svm_scaler.pkl")
    print("\nModel saved to ../classifier/svm_model.pkl")
    print("SVM Scaler saved to ../classifier/svm_scaler.pkl")

def getAccuracy( X, y,svm_model_dir=Path(__file__).resolve().parent.parent / "classifier"):
    svm_model = joblib.load(svm_model_dir / "svm_model.pkl")
    acc = svm_model.score(X, y)
    print("\nAccuracy of svm:", acc, "\n")
    return acc

def predictSVM(sample, threshold=0.6,svm_model_dir=Path(__file__).resolve().parent.parent / "classifier"):
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




# if __name__ == "__main__":
#
#     trainSVM(svm_model,X_train,y_train)
#     save_dir = Path(__file__).resolve().parent.parent / "classifier"
#     getAccuracy(X_test,y_test)
#
#     threshold = 0.6
#
#     # print("Actual:",y_test)
#     predictSVM( X_test,threshold)
