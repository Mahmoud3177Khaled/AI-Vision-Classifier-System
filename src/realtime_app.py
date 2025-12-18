import cv2
import time
import joblib
import numpy as np
from pathlib import Path

from feature_extraction2 import extract_cnn_features, preprocess
from feature_extraction2 import feature_extractor, device  # already built and on correct device

# paths we will use
CLASSIFIERS = Path(__file__).resolve().parent.parent / "classifier"

# load prefitted models
try:
    svm = joblib.load(CLASSIFIERS / "svm_model.pkl")
    scaler = joblib.load(CLASSIFIERS / "svm_scaler.pkl")
    print("SVM model and scaler loaded successfully.")
except Exception as e:
    raise FileNotFoundError(f"Failed to load SVM model files: {e}")

class_map = {
    0: 'glass',
    1: 'paper',
    2: 'cardboard',
    3: 'plastic',
    4: 'metal',
    5: 'trash'
}

# bottom label box drawer
def draw_label_box(frame, text, confidence=None):
    h, w, _ = frame.shape
    box_w, box_h = 420, 70
    x1 = (w - box_w) // 2
    y1 = h - box_h - 30
    x2 = x1 + box_w
    y2 = y1 + box_h

    # black background
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    # green border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # green label
    label_text = text
    if confidence is not None:
        label_text += f" ({confidence:.1%})"

    # print
    cv2.putText(
        frame, label_text,
        (x1 + 20, y1 + 45),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7, (0, 255, 0), 2
    )

def main():

    # try to open cam
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        print("Error: Could not open webcam.")
        return

    # set img size
    vid.set(cv2.vid_PROP_FRAME_WIDTH, 640)
    vid.set(cv2.vid_PROP_FRAME_HEIGHT, 480)

    # inits and logs
    last_time = 0
    prediction = "Initializing..."
    confidence = None
    print("Live Waste Classification (ResNet50 + SVM) Started")
    print("Hold object steady • Updates every second • Press 'q' to quit")

    # video stream reading loop
    while True:

        # raad the frame
        ok, frame = vid.read()
        if not ok:
            print("Failed to grab frame.")
            break

        # mirror is more comfortable :)
        frame = cv2.flip(frame, 1)

        # track curr time to know when to use another frame
        cur_time = time.time()
        if cur_time - last_time >= 1.0:
            try:
                # convert frame (BGR → RGB → PIL) as the preprocess needs it
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # apply feature extraction
                features = extract_cnn_features(pil_image)
                features = features.reshape(1, -1)

                # apply the prefitted scaler
                features_scaled = scaler.transform(features)

                # apply SVM prediction and get confidence from max probability
                probs = svm.predict_proba(features_scaled)[0]
                prediction_index = int(np.argmax(probs))
                prob = float(np.max(probs))

                # handle uncertian classifications
                if prob < 0.6:
                    prediction = "unknown"
                    confidence = None
                else:
                    prediction = class_map.get(prediction_index, f"Class {prediction_index}")
                    confidence = prob

                last_time = cur_time

            except Exception as e:
                prediction = "Error"
                confidence = None
                print(f"Inference error: {e}")


        # display the results in the box
        label = f"Prediction: {prediction}"
        draw_label_box(frame, label, confidence)

        cv2.putText(frame, "Hold object steady in frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Live Waste Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    from PIL import Image
    main()