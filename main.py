"""
main.py
-------

This script integrates all the components of the project, including data loading,
model building, training, evaluation, and starting the API and web app.

Author: MiniJibli
Date: 2024-10-31
Project: Cloud-based Real-time Image Classifier

Usage:
    Run the entire project pipeline:
        $ python3 main.py
"""

from data.dataset import load_and_preprocess_data, get_data_generators
from models.cnn_model import build_cnn_model, train_model, save_model
from models.model_evaluation import evaluate_model, plot_confusion_matrix
import subprocess

# Step 1: Load and Preprocess Data
print("Loading and preprocessing data...")
x_train, y_train, x_test, y_test = load_and_preprocess_data()
train_generator, test_generator = get_data_generators(x_train, y_train, x_test, y_test)

# Step 2: Build and Train Model
print("Building and training model...")
model = build_cnn_model()
history = train_model(model, train_generator, test_generator, epochs=10)

# Step 3: Evaluate Model
print("Evaluating model performance...")
evaluate_model(model, test_generator)
plot_confusion_matrix(model, x_test, y_test, class_names=[
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
])

# Step 4: Save Model
print("Saving the trained model...")
save_model(model, "fashion_mnist_cnn_model.h5")

# Step 5: Start API and Web App
print("Starting the API and web app...")

# Run the API in a separate process
subprocess.Popen(["python3", "api/inference_api.py"])

# Start the Streamlit web interface
subprocess.Popen(["streamlit", "run", "web/app.py"])
