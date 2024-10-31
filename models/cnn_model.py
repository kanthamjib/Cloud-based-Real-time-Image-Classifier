"""
cnn_model.py
------------

This module defines a Convolutional Neural Network (CNN) model for image classification
using the Fashion MNIST dataset. It includes functions to build, compile, and train the model.

Author: MiniJibli
Date: 2024-10-31
Project: Cloud-based Real-time Image Classifier

Functions:
- build_cnn_model: Define and compile the CNN model architecture.
- train_model: Train the CNN model on the Fashion MNIST dataset.
- save_model: Save the trained model to a file.

Usage:
    from cnn_model import build_cnn_model, train_model, save_model

    # Build and compile the model
    model = build_cnn_model()

    # Train the model
    history = train_model(model, train_generator, test_generator, epochs=10)

    # Save the trained model
    save_model(model, "fashion_mnist_cnn_model.h5")
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    # create CNN (Convolutional Neural Network (CNN)) with the Sequential API
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(model, train_generator, test_generator, epochs=10):
    # Train the model with the provided data generators
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=epochs
    )
    return history

def save_model(model, filename="fashion_mnist_cnn_model.h5"):
    # Save the trained model to a "filename.h5"
    model.save(filename)
    print(f"Model saved to {filename}")
