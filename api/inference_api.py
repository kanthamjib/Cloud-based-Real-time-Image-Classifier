"""
inference_api.py
-----------------

This module defines an API for real-time image classification using a trained CNN model.
It accepts an image as input, preprocesses it, and returns the predicted class.

Author: MiniJibli
Date: 2024-10-31
Project: Cloud-based Real-time Image Classifier

Endpoints:
- POST /predict: Accepts an image file and returns the predicted class.

Usage:
    Run the API server:
        $ python3 inference_api.py
"""

#import sys
#sys.path.append('../')
import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
from utils.helper_functions import preprocess_image, decode_predictions

# create Flask app
app = Flask(__name__)

import os
# Call for the trained Model.
model_path = os.path.join(os.path.dirname(__file__), '../fashion_mnist_cnn_model.h5')
model = tf.keras.models.load_model(model_path)

# define the class's name of Fashion MNIST
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
			"Shirt", "Sneaker", "Bag", "Ankle boot"]

# Route for home
@app.route("/")
def home():
    return "Welcome to the Fashion MNIST Image Classifier API! Use /predict to classify an image."

@app.route("/predict", methods=["POST"])
def predict():
    """Predicts the class of the uploaded image file and returns the result."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    # Load and preprocess the image
    image = np.array(tf.image.decode_image(file.read(), channels=1))
    processed_image = preprocess_image(image)

    # Prediction
    prediction = model.predict(processed_image)
    predicted_class = decode_predictions(prediction, class_names)

    # return the predicted result
    return jsonify({"predicted_class": predicted_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
