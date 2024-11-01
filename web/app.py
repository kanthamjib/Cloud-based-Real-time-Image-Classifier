"""
app.py
------

This module creates a web interface using Streamlit for users to upload images
and see real-time predictions from the trained CNN model hosted on an API.

Author: MiniJibli
Date: 2024-10-31
Project: Cloud-based Real-time Image Classifier

Usage:
    Run the app with:
        $ streamlit run app.py
"""

import streamlit as st
import requests
from PIL import Image
import io

# URL -> API inferencing (check API "running" status)
API_URL = "http://localhost:5001/predict"

# Web Header
st.title("Fashion MNIST Image Classifier")
st.write("Upload an image and get the predicted clothing item class.")

# Image uploading
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # send the image to API when "the Prediction process start!"
    if st.button("Predict"):
        # convert image. file to byte, then move to API
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()

        # sending to API
        response = requests.post(API_URL, files={"file": image_bytes})

        # display the result from API
        if response.status_code == 200:
            result = response.json()
            st.write(f"Predicted Class: **{result['predicted_class']}**")
        else:
            st.write("Error: Unable to get a prediction.")
