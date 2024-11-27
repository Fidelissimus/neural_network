import json
import numpy as np

#np.random.seed(0)

class LayerDense:
    def __init__(self, input_size, output_size, activation_function):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.output = None
        self.delta = None

    def forward(self, input):
        self.input = input
        z = np.dot(input, self.weights) + self.biases
        self.output = self.activation_function(z)
        return self.output

    def backward(self, delta):
        if self.activation_function.__name__ == 'relu':
            d_activation = self.activation_function(self.output, derivative=True)
        else:
            d_activation = self.activation_function(self.output, derivative=True)
        
        self.delta = delta * d_activation
        return np.dot(self.delta, self.weights.T)

    def update(self, learning_rate):
        self.weights -= learning_rate * np.dot(self.input.T, self.delta)
        self.biases -= learning_rate * np.sum(self.delta, axis=0, keepdims=True)

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def get_output(self):
        return self.output

    def get_delta(self):
        return self.delta

    def to_dict(self):
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation_function_name': self.activation_function.__name__,
            'weights': self.weights.tolist(),
            'biases': self.biases.tolist()
        }

    @staticmethod
    def from_dict(data, activation_functions):
        input_size = data['input_size']
        output_size = data['output_size']
        activation_function_name = data['activation_function_name']
        weights = np.array(data['weights'])
        biases = np.array(data['biases'])
        
        activation_function = activation_functions[activation_function_name]
        
        layer = LayerDense(input_size, output_size, activation_function)
        layer.weights = weights
        layer.biases = biases
        return layer


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.input = None
        self.output = None
        self.delta = None

    def forward(self, input):
        self.input = input
        for layer in self.layers:
            self.input = layer.forward(self.input)
        self.output = self.input
        return self.output

    def backward(self, delta):
        self.delta = delta
        for layer in reversed(self.layers):
            self.delta = layer.backward(self.delta)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.get_weights())
        return weights

    def get_biases(self):
        biases = []
        for layer in self.layers:
            biases.append(layer.get_biases())
        return biases

    def get_output(self):
        return self.output

    def get_delta(self):
        return self.delta

    def save(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)

    @staticmethod
    def load(file_path, activation_functions):
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        layers = [LayerDense.from_dict(layer_data, activation_functions) for layer_data in data['layers']]
        return NeuralNetwork(layers)

    def to_dict(self):
        return {
            'layers': [layer.to_dict() for layer in self.layers]
        }


#####
## Activation functions
#####

def relu(x, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.maximum(x, 0)

def sigmoid(x, derivative=False):
    if derivative:
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)
    return 1 / (1 + np.exp(-x))

# this function for -1 to 1 output range
def tanh(x, derivative=False):
    if derivative:
        return 1 - np.tanh(x) ** 2
    return np.tanh(x)


# Map activation function names to the functions themselves, made it for json so i can reload
activation_functions = {
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh
}
###########