import numpy as np
import json
from .activations import ACTIVATIONS

class Layer:
    """Base class for all layers"""
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input):
        raise NotImplementedError
        
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError
        
    def get_parameters(self):
        return {}
        
    def set_parameters(self, parameters):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size, activation='relu'):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        
        # Handle activation function
        if isinstance(activation, str):
            self.activation = ACTIVATIONS[activation]()
        else:
            self.activation = activation
            
    def forward(self, input):
        self.input = input
        self.z = np.dot(input, self.weights) + self.biases
        self.output = self.activation(self.z)
        return self.output
        
    def backward(self, output_gradient, learning_rate):
        # Calculate gradient of activation
        activation_gradient = self.activation.backward(output_gradient)
        
        # Calculate gradients
        weights_gradient = np.dot(self.input.T, activation_gradient)
        input_gradient = np.dot(activation_gradient, self.weights.T)
        
        # Update parameters
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(activation_gradient, axis=0, keepdims=True)
        
        return input_gradient
        
    def get_parameters(self):
        return {
            'weights': self.weights.tolist(),
            'biases': self.biases.tolist(),
            'activation': self.activation.__class__.__name__.lower()
        }
        
    def set_parameters(self, parameters):
        self.weights = np.array(parameters['weights'])
        self.biases = np.array(parameters['biases'])

class Dropout(Layer):
    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate
        self.mask = None
        
    def forward(self, input, training=True):
        self.input = input
        if not training:
            return input
            
        self.mask = np.random.binomial(1, 1 - self.rate, size=input.shape) / (1 - self.rate)
        return input * self.mask
        
    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.mask

class BatchNorm(Layer):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
    def forward(self, input, training=True):
        self.input = input
        
        if training:
            self.mean = np.mean(input, axis=0, keepdims=True)
            self.var = np.var(input, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
            
            # Normalize
            self.x_centered = input - self.mean
            self.std = np.sqrt(self.var + self.eps)
            self.x_norm = self.x_centered / self.std
        else:
            # Use running statistics during inference
            self.x_centered = input - self.running_mean
            self.x_norm = self.x_centered / np.sqrt(self.running_var + self.eps)
            
        self.output = self.gamma * self.x_norm + self.beta
        return self.output
        
    def backward(self, output_gradient, learning_rate):
        m = self.input.shape[0]
        
        # Calculate gradients
        dgamma = np.sum(output_gradient * self.x_norm, axis=0, keepdims=True)
        dbeta = np.sum(output_gradient, axis=0, keepdims=True)
        
        dx_norm = output_gradient * self.gamma
        dvar = np.sum(dx_norm * self.x_centered * -0.5 * (self.var + self.eps) ** (-1.5), axis=0, keepdims=True)
        dmean = np.sum(dx_norm * -1 / self.std, axis=0, keepdims=True) + dvar * np.mean(-2 * self.x_centered, axis=0, keepdims=True)
        
        input_gradient = dx_norm / self.std + dvar * 2 * self.x_centered / m + dmean / m
        
        # Update parameters
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return input_gradient
        
    def get_parameters(self):
        return {
            'gamma': self.gamma.tolist(),
            'beta': self.beta.tolist(),
            'running_mean': self.running_mean.tolist(),
            'running_var': self.running_var.tolist()
        }
        
    def set_parameters(self, parameters):
        self.gamma = np.array(parameters['gamma'])
        self.beta = np.array(parameters['beta'])
        self.running_mean = np.array(parameters['running_mean'])
        self.running_var = np.array(parameters['running_var'])