"""
model_evaluation.py
-------------------

This module provides functions to evaluate the performance of a trained CNN model
on the test dataset. It includes functions to calculate accuracy, loss,
and visualize the confusion matrix.

Author: MiniJibli
Date: 2024-10-31
Project: Cloud-based Real-time Image Classifier

Functions:
- evaluate_model: Evaluate the model on test data and print loss and accuracy.
- plot_confusion_matrix: Plot the confusion matrix for the test data.

Usage:
    from model_evaluation import evaluate_model, plot_confusion_matrix

    # Evaluate the model
    evaluate_model(model, test_generator)

    # Plot confusion matrix
    plot_confusion_matrix(model, x_test, y_test_labels)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, test_generator):
    # Model evaluation by "test generator"
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

def plot_confusion_matrix(model, x_test, y_test_labels, class_names):
    # The prediction for the test set
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test_labels, axis=1)

    # create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # display the confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
