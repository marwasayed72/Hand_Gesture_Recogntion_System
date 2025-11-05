import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os


model_path = os.path.join(os.path.dirname(__file__), "best.h5")
model = load_model(model_path)
#model = load_model("best.h5")


labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
labels.append('del')
labels.append('nothing')
labels.append('space')


def predict_gesture(frame):
    img = cv2.resize(frame, (64, 64)) # you should put the size you trained the model with 
    
    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    
    prediction = model.predict(img)

    class_index = np.argmax(prediction)
    predicted_label = labels[class_index]


    return predicted_label

