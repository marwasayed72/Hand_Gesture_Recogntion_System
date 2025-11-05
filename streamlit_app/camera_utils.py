import cv2
import numpy as np

def preprocess_frame(frame, img_size=64, grayscale=True):
    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (img_size, img_size))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=-1)
    else:
        frame = cv2.resize(frame, (img_size, img_size))
        frame = frame / 255.0
    return frame.astype("float32")

def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to capture frame.")
        return None, None  # بدل return None بس

    # ROI
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess
    processed = preprocess_frame(roi, img_size=64, grayscale=True)
    processed = np.expand_dims(processed, axis=0)

    return roi, processed
