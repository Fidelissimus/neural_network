import numpy as np

class Loss:
    """Base class for loss functions"""
    def forward(self, y_pred, y_true):
        raise NotImplementedError
        
    def backward(self, y_pred, y_true):
        raise NotImplementedError

class MSE(Loss):
    """Mean Squared Error loss"""
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
        
    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size

class BinaryCrossEntropy(Loss):
    """Binary Cross Entropy loss"""
    def forward(self, y_pred, y_true):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
    def backward(self, y_pred, y_true):
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.size

class CategoricalCrossEntropy(Loss):
    """Categorical Cross Entropy loss"""
    def forward(self, y_pred, y_true):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
    def backward(self, y_pred, y_true):
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return (y_pred - y_true) / y_true.size

# Loss function registry
LOSSES = {
    'mse': MSE,
    'binarycrossentropy': BinaryCrossEntropy,
    'categoricalcrossentropy': CategoricalCrossEntropy
}