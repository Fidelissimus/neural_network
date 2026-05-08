# Neural Network code without Torch or TensorFlow, just numpy.. (and json (built-in package in python) to load/save the NN)

A lightweight, educational, and easy-to-use neural network library built from the ground up using only **NumPy**. This project is designed to provide a clear and concise implementation of the core components of a deep learning framework, making it an excellent tool for learning and experimentation.

## 🌟 Features

  * **Modular Architecture:** Easily build complex models by stacking layers.
  * **Rich Layer Selection:** Includes **Dense**, **Dropout**, and **Batch Normalization** layers.
  * **Variety of Activation Functions:** `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`, and `ELU`.
  * **Modern Optimizers:** `SGD` (with momentum), `Adam`, and `RMSprop`.
  * **Common Loss Functions:** `Mean Squared Error`, `Binary Cross-Entropy`, and `Categorical Cross-Entropy`.
  * **Model Persistence:** Save your trained models to a JSON file and load them back for inference.
  * **Training Utilities:** Includes functions for data splitting, one-hot encoding, and plotting training history.

## ⚙️ Installation

The library relies on a few common Python packages. You can install them using pip:

```bash
pip install numpy matplotlib
```

After installing the dependencies, you can clone this repository or download the source code to use the `neural_network` module in your project.

## 🚀 Getting Started: A Quick Example

Here's how to build, train, and evaluate a neural network for a simple classification task in just a few steps.

### 1\. Prepare Your Data

First, you need data. For this example, let's create some sample data and split it into training and testing sets.

```python
import numpy as np
from neural_network.utils import train_test_split, one_hot_encode

# Create dummy data: 100 samples, 10 features
X = np.random.rand(100, 10)
# Create dummy labels for 3 classes
y_labels = np.random.randint(0, 3, 100)
y = one_hot_encode(y_labels, 3)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### 2\. Build the Model

Define the architecture of your neural network by creating a list of layers.

```python
from neural_network import NeuralNetwork, Dense, Dropout
from neural_network.activations import LeakyReLU

# Define the layers
layers = [
    Dense(input_size=10, output_size=64, activation=LeakyReLU(alpha=0.1)),
    Dropout(0.1),
    Dense(input_size=64, output_size=32, activation='relu'),
    Dense(input_size=32, output_size=3, activation='softmax') # 3 output classes
]

# Create the model
model = NeuralNetwork(layers)
```

### 3\. Compile the Model

Configure the learning process by specifying the optimizer, loss function, and learning rate.

```python
model.compile(optimizer='adam', loss='categoricalcrossentropy', learning_rate=0.001)

# You can print a summary of the model
model.summary()
```

### 4\. Train the Model

Fit the model to your training data using the `.train()` method.

```python
history = model.train(X_train, y_train, epochs=100, batch_size=16, 
                      validation_data=(X_test, y_test), verbose=True)
```

### 5\. Evaluate and Predict

Check your model's performance on the test set and make predictions on new data.

```python
# Evaluate performance
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Make a prediction
sample_prediction = model.predict(X_test[0:1])
print(f"Sample Prediction: {np.argmax(sample_prediction)}")
```

### 6\. Save and Load

Save your trained model's architecture and weights for later use.

```python
# Save the model to a file
model.save('my_model.json')

# Load the model from the file
loaded_model = NeuralNetwork.load('my_model.json')

# Verify the loaded model
loss, accuracy = loaded_model.evaluate(X_test, y_test)
print(f"Loaded Model Test Accuracy: {accuracy:.4f}")
```

## 📈 Example: Spiral Classification

This library is capable of solving non-linear problems. The `spiral_classification_example.py` script demonstrates how to train a deep neural network to classify points belonging to one of several intertwined spirals.

To run the example:

```bash
python spiral_classification_example.py
```

This script will:

1.  Generate a spiral dataset with 4 classes.
2.  Build a deep model with `Dense`, `BatchNorm`, and `Dropout` layers.
3.  Train the model for 1000 epochs using the Adam optimizer.
4.  Save the trained model to `spiral_model.json`.
5.  Display plots for training/validation history and the final decision boundary.

You can then use `load_spiral_classification_example.py` to see how to load and use the saved model for inference.

## 📂 Project Structure

The project is organized into logical modules:

  * `neuralnetwork.py`: The main `NeuralNetwork` class that orchestrates the model's training, evaluation, and persistence.
  * `layers.py`: Contains the layer implementations (`Dense`, `Dropout`, `BatchNorm`).
  * `activations.py`: Defines all activation functions (`ReLU`, `Softmax`, etc.).
  * `optimizers.py`: Implements the optimization algorithms (`SGD`, `Adam`, `RMSprop`).
  * `losses.py`: Contains the loss function implementations (`MSE`, `CrossEntropy`, etc.).
  * `utils.py`: Provides helper functions for data processing, plotting, and model persistence.
  * `*_example.py`: Example scripts demonstrating the library's usage.

## 🤝 Contributing

Contributions are welcome\! If you have ideas for new features, bug fixes, or improvements, feel free to open an issue or submit a pull request. Areas for future development include:

  * Adding more layer types (e.g., `Convolutional`, `RNN`).
  * Implementing more regularization techniques.
  * Expanding the suite of optimizers and loss functions.
  * Improving the `summary()` method to correctly infer and display output shapes.
  * etc.
