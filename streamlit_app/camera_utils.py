import cv2
import numpy as np

def preprocess_frame(frame, img_size=64, grayscale=True):
    """Preprocess the frame for model input"""
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
    """Try to capture a frame from available cameras (0 or 1)"""
    for cam_index in [0, 1]:
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        cap.release()

        if ret:
            # Define ROI (Region of Interest)
            x1, y1, x2, y2 = 100, 100, 300, 300
            roi = frame[y1:y2, x1:x2]

            # Preprocess for model
            processed = preprocess_frame(roi, img_size=64, grayscale=True)
            processed = np.expand_dims(processed, axis=0)
            return roi, processed
        else:
            print(f"❌ Failed to read frame from camera {cam_index}.")

    print("❌ No available camera found. Make sure a webcam is connected and accessible.")
    return None, None
