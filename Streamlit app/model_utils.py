import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# تحديد مسار الموديل بشكل ديناميكي لضمان عمله على السيرفر
model_path = os.path.join(os.path.dirname(__file__), "best.h5")
model = load_model(model_path)

# تجهيز قائمة الحروف (Labels)
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
labels.extend(['del', 'nothing', 'space'])

def predict_gesture(processed_frame):
    """
    بتاخد الصورة بعد المعالجة (Preprocessed) وبترجع الحرف المتوقع
    """
    try:
        # التوقع باستخدام الموديل
        prediction = model.predict(processed_frame, verbose=0) # verbose=0 عشان ميطبعش Log كتير في الكونسول
        
        # الحصول على الحرف صاحب أعلى احتمالية
        class_index = np.argmax(prediction)
        predicted_label = labels[class_index]
        
        return predicted_label
    except Exception as e:
        return f"Error: {str(e)}"