# realtime_app.py
import cv2
import time
import joblib
import numpy as np
from pathlib import Path

# Import the exact feature extraction functions from feature_extraction2.py
# (do NOT copy-paste the model/preprocess code)
from feature_extraction2 import extract_cnn_features, preprocess
from feature_extraction2 import feature_extractor, device  # already built and on correct device

# ----------------------------- Paths -----------------------------
BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "classifier"

# ----------------------------- Load SVM Pipeline -----------------------------
try:
    svm = joblib.load(MODEL_DIR / "svm_model.pkl")
    scaler = joblib.load(MODEL_DIR / "svm_scaler.pkl")
    print("SVM model and scaler loaded successfully.")
except Exception as e:
    raise FileNotFoundError(f"Failed to load SVM model files: {e}")

# ----------------------------- Class Mapping -----------------------------
class_map = {
    0: 'glass',
    1: 'paper',
    2: 'cardboard',
    3: 'plastic',
    4: 'metal',
    5: 'trash'
}

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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_infer_time = 0
    current_prediction = "Initializing..."
    current_confidence = None

    print("Live Waste Classification (ResNet50 + SVM) Started")
    print("Hold object steady • Updates every second • Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)  # mirror

        current_time = time.time()
        if current_time - last_infer_time >= 1.0:
            try:
                # Convert OpenCV frame (BGR) → RGB → PIL (exactly what the preprocess expects)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Feature extraction using the shared ResNet50 extractor
                features = extract_cnn_features(pil_image)  # 2048-dim vector
                features = features.reshape(1, -1)

                # Apply the same scaler that was fitted on training features
                features_scaled = scaler.transform(features)

                # SVM prediction + confidence
                probs = svm.predict_proba(features_scaled)[0]
                pred_idx = int(np.argmax(probs))
                max_prob = float(np.max(probs))

                # Unknown rejection (same threshold used in training)
                if max_prob < 0.6:
                    current_prediction = "unknown"
                    current_confidence = None
                else:
                    current_prediction = class_map.get(pred_idx, f"Class {pred_idx}")
                    current_confidence = max_prob

                last_infer_time = current_time

            except Exception as e:
                current_prediction = "Error"
                current_confidence = None
                print(f"Inference error: {e}")

        # Display result
        display_text = f"Prediction: {current_prediction}"
        draw_label(frame, display_text, current_confidence)

        cv2.putText(frame, "Hold object steady in frame",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Live Waste Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # PIL is only needed here for converting frames
    from PIL import Image
    main()