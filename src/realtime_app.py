import cv2
import time
import joblib
import numpy as np
from pathlib import Path

from feature_extraction import extract_hog, extract_color_hist, extract_lbp

from pca import get_data

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load trained pipeline
BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "classifier"
FEATURES_DIR = BASE / "features"

knn = joblib.load(MODEL_DIR / "knn_model.pkl")
scaler = StandardScaler()
pca = PCA(n_components=knn.n_features_in_)


# fit scaler + PCA once using training data
X_train, _, _, _ = get_data(FEATURES_DIR)
X_train = scaler.fit_transform(X_train)
pca.fit(X_train)


class_map = {
    0: 'glass',
    1: 'paper',
    2: 'cardboard',
    3: 'plastic',
    4: 'metal',
    5: 'trash',
    6: 'unknown'
}

# ui draw helper
def draw_label(frame, text):
    h, w, _ = frame.shape
    box_w, box_h = 320, 50
    x1 = (w - box_w) // 2
    y1 = h - box_h - 20
    x2 = x1 + box_w
    y2 = y1 + box_h

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        frame, text,
        (x1 + 15, y1 + 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (0, 255, 0), 2
    )

# main loop
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available")
        return

    last_infer_time = time.time()
    prediction_text = "Detecting..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # run every 1 second
        if time.time() - last_infer_time >= 1:
            resized = cv2.resize(frame, (128, 128))

            features = np.concatenate([
                extract_hog(resized),
                extract_color_hist(resized),
                extract_lbp(resized)
            ])

            features = scaler.transform([features])
            features = pca.transform(features)
            pred = knn.predict(features)[0]

            prediction_text = f"Prediction: {class_map[pred]}"
            last_infer_time = time.time()

        draw_label(frame, prediction_text)
        cv2.imshow("Live Waste Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
