import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="Solar Panel Defect Detection", page_icon="??", layout="centered")

st.title("?? Solar Panel Defect Detection Platform")
st.write("Upload a solar panel image to automatically detect defects using YOLOv8.")

# Load the trained model
@st.cache_resource
def load_model():
    model = YOLO('best_solar_model.pt')
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose a solar panel image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, caption="Original Image", use_column_width=True)
    
    with st.spinner("Analyzing panel for defects..."):
        results = model(img_array)
        res_im = results[0].plot()
        res_im_rgb = cv2.cvtColor(res_im, cv2.COLOR_BGR2RGB)
        
    with col2:
        st.subheader("Detection Result")
        st.image(res_im_rgb, caption="Processed Image", use_column_width=True)
        
    # Inspection Report
    st.subheader("?? Inspection Report")
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.success("? Panel is completely Non-Defective!")
    else:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0]) * 100
            cls_name = model.names[cls_id]
            st.warning(f"?? **Defect detected:** {cls_name} | **Confidence:** {conf:.2f}%")
