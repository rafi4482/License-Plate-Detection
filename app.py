import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

model = YOLO("runs/detect/license_plate_detection/weights/best.pt")

st.title("License Plate Detection App")
st.write("Upload an image and detect license plates using YOLOv8.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)

    results = model(image)

    for result in results:
        img_with_boxes = result.plot()  

    result_image = Image.fromarray(img_with_boxes)

    st.image(result_image, caption="Detected Image", use_container_width=True)

    result_image.save("output_image.jpg")
    with open("output_image.jpg", "rb") as file:
        st.download_button(label="Download Processed Image", data=file, file_name="output_image.jpg", mime="image/jpeg")
