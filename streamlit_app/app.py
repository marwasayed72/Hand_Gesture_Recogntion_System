import streamlit as st
from PIL import Image
import cv2
import numpy as np
from camera_utils import capture_frame
from model_utils import predict_gesture

# ------------------ Page Config ------------------
st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🤖", layout="centered")

# ------------------ Style ------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f7f9fc; color: #e5ebf1; }
[data-testid="stVerticalBlock"] {
    width: 120%; background-color: #3C467B; padding: 60px; border-radius: 20px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
h1 { color: white !important; text-align: center; font-family: 'Arial Black'; }
.stButton>button {
    background-color: white; color: black; border-radius: 10px; height: 3em; width: 100%; font-size: 16px;
}
.stButton>button:hover { background-color: #155d8a; }
.main-title { color: #1f77b4; text-align: center; font-size: 40px; font-weight: bold; text-shadow: 2px 2px 5px #ccc; }
.sub-text { color: #555; text-align: center; font-size: 18px; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
st.markdown("""
<div style="text-align:center;">
    <h2 style="color:white; font-size:40px;">🧠 Welcome to the Hand Gesture Recognition App</h2>
   <p style="color:white; font-size:25px;text-align:left;">This AI-powered model can recognize various hand gestures in real-time.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ------------------ Input Type ------------------
#st.markdown("<h3 style='color:white; font-size:25px; text-align:left;'>Select Input Type:</h3>", unsafe_allow_html=True)
#option = st.radio("", ("📁 Upload Image"))

# ------------------ Upload Image ------------------
#if option == "📁 Upload Image":
st.subheader("Upload an Image")
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.success("Image uploaded successfully!")
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Gesture"):
        st.write("🧠 Loading model and predicting...")
        # Convert PIL image to numpy array
        frame = np.array(image)

        # Convert to grayscale if necessary
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif len(frame.shape) == 2:
            pass  # already grayscale

        # Predict gesture
        try:
            predicted_gesture = predict_gesture(frame)
            st.success(f"Predicted Gesture: {predicted_gesture}")
        except Exception as e:
            st.error(f"❌ Error predicting gesture: {e}")
else:
    st.warning("Please upload an image first.")

st.markdown("---")














