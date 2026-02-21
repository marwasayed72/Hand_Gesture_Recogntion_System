import streamlit as st
from PIL import Image
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp

# استدعاء الدوال من ملفاتك
from model_utils import predict_gesture
from camera_utils import preprocess_frame 

# ------------------ MediaPipe Setup (للـ Live Video) ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ------------------ Page Config ------------------
st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🤖", layout="centered")

# ------------------ Style (نفس الستايل بتاعك مع تعديل بسيط للـ Video) ------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f7f9fc; color: #e5ebf1; }
[data-testid="stVerticalBlock"] {
    width: 100%; background-color: #3C467B; padding: 40px; border-radius: 20px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
h1, h2, h3 { color: white !important; text-align: center; font-family: 'Arial Black'; }
.stButton>button {
    background-color: white; color: black; border-radius: 10px; height: 3em; width: 100%; font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
st.markdown("""
<div style="text-align:center;">
    <h2 style="color:white; font-size:40px;">🧠 Hand Gesture Recognition AI</h2>
    <p style="color:white; font-size:20px;">Recognize ASL gestures from Images or Live Webcam</p>
</div>
""", unsafe_allow_html=True)

# ------------------ Sidebar / Mode Selection ------------------
option = st.sidebar.selectbox("Choose Input Mode", ("📁 Upload Image", "🎥 Live Webcam"))

# ------------------ Logic for Live Webcam ------------------
if option == "🎥 Live Webcam":
    st.subheader("Live Real-time Recognition")
    
    class ASLTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = img.shape
                    # تحديد المربع حول اليد
                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w) - 20
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h) - 20
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w) + 20
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h) + 20
                    
                    roi = img[max(0, y_min):min(h, y_max), max(0, x_min):min(w, x_max)]
                    
                    if roi.size != 0:
                        # المعالجة (تعديل grayscale=True لو الموديل بتاعك أبيض وأسود)
                        processed = preprocess_frame(roi, img_size=64, grayscale=True)
                        label = predict_gesture(processed)
                        
                        # الرسم على الشاشة
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(img, f"Result: {label}", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return img

    webrtc_streamer(
        key="hand-gesture",
        video_transformer_factory=ASLTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

# ------------------ Logic for Upload Image ------------------
else:
    st.subheader("Upload an Image")
    uploaded_image = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Gesture"):
            # تجهيز الصورة
            frame = np.array(image)
            processed = preprocess_frame(frame, img_size=64, grayscale=True)
            
            try:
                predicted_gesture = predict_gesture(processed)
                st.success(f"Predicted Gesture: {predicted_gesture}")
            except Exception as e:
                st.error(f"❌ Error: {e}")