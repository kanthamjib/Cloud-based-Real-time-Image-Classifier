"""
dataset.py
-----------------

This module is responsible for loading and preprocessing the dataset
for the image classification model. It includes functions to load data,
normalize, and augment images for training.

Author: MiniJibli
Date: 2024-10-31
Project: Cloud-based Real-time Image Classifier

Functions:
- load_and_preprocess_data: Load and normalize Fashion MNIST dataset.
- get_data_generators: Create data generators with augmentation for training and testing.

Usage:
    from dataset_loader import load_and_preprocess_data, get_data_generators

    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Get data generators
    train_gen, test_gen = get_data_generators(x_train, y_train, x_test, y_test)
"""

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

def load_and_preprocess_data():
    # Loading dataset from "Fashion MNIST"
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Reshape data to 4D (batch, height, width, channels)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')

    # Data Normalize into [0, 1] pixel interval
    x_train /= 255.0
    x_test /= 255.0

    # transform label into one-hot encoding format
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

def get_data_generators(x_train, y_train, x_test, y_test):
    # Setting the data augmentation (random the possible case) for train set
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()  # ไม่ต้องทำ augmentation สำหรับ test set

    # create generators
    train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
    test_generator = test_datagen.flow(x_test, y_test, batch_size=32)

    return train_generator, test_generator
