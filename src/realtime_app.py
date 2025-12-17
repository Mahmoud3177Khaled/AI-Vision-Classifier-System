import cv2
import time
import joblib
import numpy as np
from pathlib import Path

# Import the exact feature extractors used during training
from feature_extraction import extract_hog, extract_color_hist, extract_lbp

# ----------------------------- Paths -----------------------------
BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "classifier"

# ----------------------------- Load Model Pipeline -----------------------------
try:
    knn = joblib.load(MODEL_DIR / "knn_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    pca = joblib.load(MODEL_DIR / "PCA.pkl")
    print("Model, scaler, and PCA loaded successfully.")
except Exception as e:
    raise FileNotFoundError(f"Failed to load model files: {e}")

# ----------------------------- Class Mapping (includes unknown) -----------------------------
class_map = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'trash',
    6: 'unknown'
}

# ----------------------------- Preprocessing Helper (exact match with training) -----------------------------
def preprocess_frame(frame, target_size=160):
    """
    Matches resize_with_aspect_ratio_and_center_crop from feature_extraction.py
    """
    # Convert BGR (OpenCV) → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w = rgb_frame.shape[:2]
    scale = target_size / min(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center crop to target_size x target_size
    start_x = (new_w - target_size) // 2
    start_y = (new_h - target_size) // 2
    cropped = resized[start_y:start_y + target_size, start_x:start_x + target_size]

    return cropped

# ----------------------------- UI Helper -----------------------------
def draw_label(frame, text, confidence=None):
    h, w, _ = frame.shape
    box_w, box_h = 420, 70
    x1 = (w - box_w) // 2
    y1 = h - box_h - 30
    x2 = x1 + box_w
    y2 = y1 + box_h

    # Background
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # Text
    label_text = text
    if confidence is not None:
        label_text += f" ({confidence:.1%})"

    cv2.putText(
        frame, label_text,
        (x1 + 20, y1 + 45),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7, (0, 255, 0), 2
    )

# ----------------------------- Main Loop -----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Optimize camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_infer_time = 0
    current_prediction = "Initializing..."
    current_confidence = None

    print("Live Waste Classification Started")
    print("Place object in view • Updates every second • Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Mirror view
        frame = cv2.flip(frame, 1)

        current_time = time.time()
        if current_time - last_infer_time >= 1.0:
            try:
                # Preprocess exactly like training
                processed_img = preprocess_frame(frame, target_size=160)

                # Extract features
                features = np.concatenate([
                    extract_hog(processed_img),
                    extract_color_hist(processed_img),
                    extract_lbp(processed_img)
                ]).reshape(1, -1)

                # Transform using fitted scaler & PCA
                features_scaled = scaler.transform(features)
                features_pca = pca.transform(features_scaled)

                # Predict
                pred_label = int(knn.predict(features_pca)[0])

                # Compute confidence via distance-weighted voting
                distances, indices = knn.kneighbors(features_pca)
                weights = 1 / (distances[0] + 1e-5)
                votes = np.zeros(7)  # 7 classes including unknown
                for idx, weight in zip(indices[0], weights):
                    neighbor_label = int(knn._y[idx])
                    votes[neighbor_label] += weight

                confidence = votes[pred_label] / votes.sum()

                # Optional: Force "unknown" if confidence is too low
                if confidence < 0.1:  # Adjustable threshold
                    current_prediction = "unknown"
                    current_confidence = None
                else:
                    current_prediction = class_map.get(pred_label, f"Class {pred_label}")
                    current_confidence = confidence

                last_infer_time = current_time

            except Exception as e:
                current_prediction = "Error"
                current_confidence = None
                print(f"Inference error: {e}")

        # Display prediction
        display_text = f"Prediction: {current_prediction}"
        draw_label(frame, display_text, current_confidence)

        # Instructions overlay
        cv2.putText(frame, "Hold object steady in frame",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Live Waste Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()