import numpy as np

class Activation:
    """Base class for activation functions"""
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, x):
        raise NotImplementedError
        
    def backward(self, doutput):
        raise NotImplementedError
        
    def __call__(self, x):
        return self.forward(x)

class ReLU(Activation):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
        
    def backward(self, doutput):
        return doutput * (self.input > 0)

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x):
        self.input = x
        return np.where(x > 0, x, self.alpha * x)
        
    def backward(self, doutput):
        return doutput * np.where(self.input > 0, 1, self.alpha)

class Sigmoid(Activation):
    def forward(self, x):
        self.input = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output
        
    def backward(self, doutput):
        return doutput * self.output * (1 - self.output)

class Tanh(Activation):
    def forward(self, x):
        self.input = x
        self.output = np.tanh(x)
        return self.output
        
    def backward(self, doutput):
        return doutput * (1 - self.output ** 2)

class Softmax(Activation):
    def forward(self, x):
        self.input = x
        # Numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output
        
    def backward(self, doutput):
        # This is a simplified version assuming doutput is the gradient from the loss
        return doutput  # For use with cross-entropy loss

class ELU(Activation):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x):
        self.input = x
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        
    def backward(self, doutput):
        return doutput * np.where(self.input > 0, 1, self.alpha * np.exp(self.input))

# Activation function registry
ACTIVATIONS = {
    'relu': ReLU,
    'leakyrelu': LeakyReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'softmax': Softmax,
    'elu': ELU
}