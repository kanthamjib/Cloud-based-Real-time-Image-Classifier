"""
helper_functions.py
-------------------

This module contains helper functions for preprocessing images and decoding
predictions for the image classification project.

Author: MiniJibli
Date: 2024-10-31
Project: Cloud-based Real-time Image Classifier

Functions:
- preprocess_image: Preprocess an image to prepare it for model prediction.
- decode_predictions: Decode the predicted labels into human-readable class names.

Usage:
    from helper_functions import preprocess_image, decode_predictions

    # Preprocess a single image
    processed_image = preprocess_image(image)

    # Decode predictions
    class_name = decode_predictions(prediction, class_names)
"""

import numpy as np
import tensorflow as tf

def preprocess_image(image, target_size=(28, 28)):
    """
    Resize and normalize an image to prepare it for model prediction.

    Args:
        image (np.array): The input image to preprocess.
        target_size (tuple): The desired image size (height, width).

    Returns:
        np.array: The preprocessed image ready for model prediction.
    """
    # Resize image
    image = tf.image.resize(image, target_size)
    # Normalize image to [0, 1]
    image = image / 255.0
    # Expand dimensions to (1, height, width, channels)
    image = np.expand_dims(image, axis=0)

    return image

def decode_predictions(prediction, class_names):
    """
    Decode the model's prediction into a human-readable class name.

    Args:
        prediction (np.array): The prediction output from the model.
        class_names (list): List of class names for decoding.

    Returns:
        str: The name of the predicted class.
    """
    # Get the index of the highest probability
    predicted_index = np.argmax(prediction, axis=1)[0]
    # Return the corresponding class name
    return class_names[predicted_index]
