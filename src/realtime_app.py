import cv2
import time
from datetime import datetime

def main():

    #try to open cam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Camera started. Capturing a photo every 10 seconds...")

    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame.")
            break

        cv2.imshow("Live Camera Feed", frame)

        #take image evert 10 seconds
        current_time = time.time()
        if current_time - last_capture_time >= 10:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.jpg"

            #save to folder (placeholder for future model integration)
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")

            last_capture_time = current_time

        #take output from Model and display it on the image (placeholder for future model integration)

        #press q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
