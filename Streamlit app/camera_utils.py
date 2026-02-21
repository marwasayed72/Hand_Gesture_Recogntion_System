import cv2
import numpy as np

def preprocess_frame(frame, img_size=64, grayscale=True):
    if grayscale:
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (img_size, img_size))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=(0, -1))
    else:
        frame = cv2.resize(frame, (img_size, img_size))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)
    return frame.astype("float32")