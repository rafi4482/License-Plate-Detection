# License Plate Detection App

This project implements a license plate detection application using YOLOv8 and Streamlit.  It allows users to upload an image, and the application will detect and highlight license plates within the image.  The user can then download the processed image with the detected license plates.

## Introduction

This application leverages the power of YOLOv8 (You Only Look Once version 8), a state-of-the-art object detection model, to accurately identify license plates in images.  It provides a user-friendly interface built with Streamlit, making it easy for anyone to use without requiring coding knowledge.

## Features

* **Image Upload:** Users can upload images in JPG, JPEG, or PNG format.
* **License Plate Detection:**  The YOLOv8 model detects license plates within the uploaded image.
* **Confidence Threshold:** A slider allows users to adjust the confidence threshold for detection, controlling the sensitivity of the model.
* **Visual Output:** Detected license plates are highlighted within the image.
* **Download Processed Image:** Users can download the processed image with the detected license plates.
* **Streamlit Interface:**  A clean and intuitive web interface makes the application easy to use.

