import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from model import ANN

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Preprocess the data
X = X / 255.0  # Normalize pixel values between 0 and 1

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoded vectors
num_classes = 10
y_train_encoded = pd.get_dummies(y_train).T.values

# Create and train the model
input_size = X_train.shape[1]
output_size = num_classes
learning_rate = 0.1
num_hidden_layers = 2
num_nodes_hidden = 64

model = ANN(input_size, output_size, learning_rate, num_hidden_layers, num_nodes_hidden)
model.train(X_train.T, y_train_encoded, num_iterations=1000)

# Predict labels for test set
y_pred = model.predict(X_test.T)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
