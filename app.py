import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

st.set_page_config(page_title="License Plate Detection", layout="wide")

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 24px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .uploadedFile {
        border: 2px dashed #4CAF50;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    .css-1v3fvcr { padding: 2rem !important; }
    .centered { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    return YOLO("runs/detect/license_plate_detection/weights/best.pt")

model = load_model()

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

st.markdown("<h1 class='centered'>🚗 License Plate Detection App</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='centered'>Upload an image to detect license plates using YOLOv8.</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    with st.spinner("🔍 Detecting license plates... Please wait."):
        results = model(image)
    
    for result in results:
        img_with_boxes = result.plot()
    
    result_image = Image.fromarray(img_with_boxes)
    st.image(result_image, caption="✅ Detected Image", use_column_width=True)
    
    st.markdown("<h2 class='centered'>📊 Detection Results:</h2>", unsafe_allow_html=True)
    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf >= conf_threshold:
                st.markdown(f"<h4 class='centered'>🔹 Confidence Score: {round(conf * 100, 2)}%</h4>", unsafe_allow_html=True)
    
    result_image.save("output_image.jpg")
    with open("output_image.jpg", "rb") as file:
        st.download_button(label="⬇️ Download Processed Image", data=file, file_name="detected_license_plate.jpg", mime="image/jpeg")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h6 class='centered'>Developed with ❤️ using YOLOv8 and Streamlit</h6>", unsafe_allow_html=True)
