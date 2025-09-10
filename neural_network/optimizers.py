import numpy as np

class Optimizer:
    """Base class for optimizers"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update(self, layer):
        raise NotImplementedError

class SGD(Optimizer):
    """Stochastic Gradient Descent"""
    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}
        
    def update(self, layer):
        # Initialize velocities if not present
        if id(layer) not in self.velocities:
            self.velocities[id(layer)] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
            
        v = self.velocities[id(layer)]
        
        # Update weights with momentum
        v['weights'] = self.momentum * v['weights'] + (1 - self.momentum) * np.dot(layer.input.T, layer.delta)
        v['biases'] = self.momentum * v['biases'] + (1 - self.momentum) * np.sum(layer.delta, axis=0, keepdims=True)
        
        # Apply updates
        layer.weights -= self.learning_rate * v['weights']
        layer.biases -= self.learning_rate * v['biases']

class Adam(Optimizer):
    """Adam optimizer"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment vector
        self.v = {}  # Second moment vector
        self.t = 0   # Time step
        
    def update(self, layer):
        # Initialize moment vectors if not present
        if id(layer) not in self.m:
            self.m[id(layer)] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
            self.v[id(layer)] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
            
        self.t += 1
        
        # Calculate gradients
        grad_w = np.dot(layer.input.T, layer.delta)
        grad_b = np.sum(layer.delta, axis=0, keepdims=True)
        
        # Update biased first moment estimate
        self.m[id(layer)]['weights'] = self.beta1 * self.m[id(layer)]['weights'] + (1 - self.beta1) * grad_w
        self.m[id(layer)]['biases'] = self.beta1 * self.m[id(layer)]['biases'] + (1 - self.beta1) * grad_b
        
        # Update biased second raw moment estimate
        self.v[id(layer)]['weights'] = self.beta2 * self.v[id(layer)]['weights'] + (1 - self.beta2) * (grad_w ** 2)
        self.v[id(layer)]['biases'] = self.beta2 * self.v[id(layer)]['biases'] + (1 - self.beta2) * (grad_b ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat_w = self.m[id(layer)]['weights'] / (1 - self.beta1 ** self.t)
        m_hat_b = self.m[id(layer)]['biases'] / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat_w = self.v[id(layer)]['weights'] / (1 - self.beta2 ** self.t)
        v_hat_b = self.v[id(layer)]['biases'] / (1 - self.beta2 ** self.t)
        
        # Update parameters
        layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

class RMSprop(Optimizer):
    """RMSprop optimizer"""
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.cache = {}
        
    def update(self, layer):
        # Initialize cache if not present
        if id(layer) not in self.cache:
            self.cache[id(layer)] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
            
        # Calculate gradients
        grad_w = np.dot(layer.input.T, layer.delta)
        grad_b = np.sum(layer.delta, axis=0, keepdims=True)
        
        # Update cache
        self.cache[id(layer)]['weights'] = self.beta * self.cache[id(layer)]['weights'] + (1 - self.beta) * (grad_w ** 2)
        self.cache[id(layer)]['biases'] = self.beta * self.cache[id(layer)]['biases'] + (1 - self.beta) * (grad_b ** 2)
        
        # Update parameters
        layer.weights -= self.learning_rate * grad_w / (np.sqrt(self.cache[id(layer)]['weights']) + self.epsilon)
        layer.biases -= self.learning_rate * grad_b / (np.sqrt(self.cache[id(layer)]['biases']) + self.epsilon)

# Optimizer registry
OPTIMIZERS = {
    'sgd': SGD,
    'adam': Adam,
    'rmsprop': RMSprop
}