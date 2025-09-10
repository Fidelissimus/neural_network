import numpy as np
import json
from typing import List, Dict, Any, Callable, Union

from .layers import Dense, Dropout, BatchNorm
from .activations import ACTIVATIONS
from .optimizers import OPTIMIZERS
from .losses import LOSSES
from .utils import plot_training_history, save_history

class NeuralNetwork:
    def __init__(self, layers: List = None):
        self.layers = layers if layers else []
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.optimizer = None
        self.loss_fn = None
        
    def add(self, layer):
        self.layers.append(layer)
        
    def compile(self, loss: str = 'mse', optimizer: str = 'sgd', 
                learning_rate: float = 0.001, **optimizer_kwargs):
        """
        Configure the model for training
        
        Args:
            loss: Name of loss function
            optimizer: Name of optimizer
            learning_rate: Learning rate
            optimizer_kwargs: Additional optimizer parameters
        """
        self.loss_fn = LOSSES[loss]()
        
        # Initialize optimizer
        if optimizer in OPTIMIZERS:
            self.optimizer = OPTIMIZERS[optimizer](learning_rate=learning_rate, **optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the network
        
        Args:
            x: Input data
            training: Whether in training mode (affects dropout, batchnorm)
            
        Returns:
            Network output
        """
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNorm)):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        return x
        
    def backward(self, output_gradient: np.ndarray):
        """
        Backward pass through the network
        
        Args:
            output_gradient: Gradient of loss with respect to output
        """
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                output_gradient = layer.backward(output_gradient, self.optimizer.learning_rate)
                
    def update(self):
        """Update parameters using the optimizer"""
        for layer in self.layers:
            if hasattr(layer, 'delta') and hasattr(layer, 'input'):
                self.optimizer.update(layer)
                
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              epochs: int, batch_size: int = 32, 
              validation_data: tuple = None, verbose: bool = True,
              callbacks: List[Callable] = None) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            x_train: Training data
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size
            validation_data: Tuple of (x_val, y_val) for validation
            verbose: Whether to print progress
            callbacks: List of callback functions
            
        Returns:
            Training history
        """
        n_samples = x_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            correct_predictions = 0
            
            for i in range(0, n_samples, batch_size):
                # Get batch
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                output = self.forward(x_batch, training=True)
                
                # Calculate loss
                loss = self.loss_fn.forward(output, y_batch)
                epoch_loss += loss * x_batch.shape[0]
                
                # Calculate accuracy
                if y_batch.shape[1] == 1:  # Binary classification
                    predictions = (output > 0.5).astype(int)
                    correct_predictions += np.sum(predictions == y_batch)
                else:  # Regression or multi-class
                    if output.shape[1] > 1:  # Multi-class classification
                        predictions = np.argmax(output, axis=1)
                        true_labels = np.argmax(y_batch, axis=1)
                        correct_predictions += np.sum(predictions == true_labels)
                    else:  # Regression
                        # For regression, we don't calculate accuracy
                        pass
                
                # Backward pass
                error = self.loss_fn.backward(output, y_batch)
                self.backward(error)
                self.update()
            
            # Calculate metrics
            train_loss = epoch_loss / n_samples
            
            if y_train.shape[1] == 1 or y_train.shape[1] > 1:
                train_acc = correct_predictions / n_samples
            else:
                train_acc = 0  # Not applicable for regression
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            val_loss, val_acc = 0, 0
            if validation_data:
                val_loss, val_acc = self.evaluate(*validation_data)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            # Call callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch)
            
            if verbose and epoch % 100 == 0:
                msg = f"Epoch {epoch}: loss={train_loss:.4f}, acc={train_acc:.4f}"
                if validation_data:
                    msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                print(msg)
                
        return self.history
                
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """
        Evaluate the model on given data
        
        Args:
            x: Input data
            y: True labels
            
        Returns:
            loss, accuracy
        """
        output = self.forward(x, training=False)
        loss = self.loss_fn.forward(output, y)
        
        if y.shape[1] == 1:  # Binary classification
            predictions = (output > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
        elif y.shape[1] > 1:  # Multi-class classification
            predictions = np.argmax(output, axis=1)
            true_labels = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == true_labels)
        else:  # Regression
            accuracy = 0  # Not applicable
        
        return loss, accuracy
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input data
        
        Args:
            x: Input data
            
        Returns:
            Predictions
        """
        return self.forward(x, training=False)
        
    def save(self, file_path: str):
        """
        Save model to file
        
        Args:
            file_path: Path to save file
        """
        data = {
            'layers': [],
            'config': {
                'loss': self.loss_fn.__class__.__name__.lower(),
                'optimizer': self.optimizer.__class__.__name__.lower(),
                'learning_rate': self.optimizer.learning_rate
            },
            'history': self.history
        }
        
        for layer in self.layers:
            layer_data = {
                'type': layer.__class__.__name__,
                'parameters': layer.get_parameters()
            }
            data['layers'].append(layer_data)
            
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
            
    @classmethod
    def load(cls, file_path: str):
        """
        Load model from file
        
        Args:
            file_path: Path to model file
            
        Returns:
            Loaded model
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        network = cls()
        network.loss_fn = LOSSES[data['config']['loss']]()
        
        # Initialize optimizer
        optimizer_name = data['config']['optimizer']
        learning_rate = data['config']['learning_rate']
        network.optimizer = OPTIMIZERS[optimizer_name](learning_rate=learning_rate)
        
        network.history = data['history']
        
        # Recreate layers
        for layer_data in data['layers']:
            layer_type = layer_data['type']
            parameters = layer_data['parameters']
            
            if layer_type == 'Dense':
                activation_name = parameters.get('activation', 'relu')
                layer = Dense(1, 1, activation_name)  # Placeholder dimensions
                layer.set_parameters(parameters)
            elif layer_type == 'BatchNorm':
                layer = BatchNorm(1)  # Placeholder dimensions
                layer.set_parameters(parameters)
            elif layer_type == 'Dropout':
                rate = parameters.get('rate', 0.5)
                layer = Dropout(rate)
            # Add other layer types as needed (don't forget to help with the project in github when you do so)
            
            network.add(layer)
            
        return network
        
    def summary(self):
        """Print model summary"""
        print("Model Summary:")
        print("=" * 50)
        print(f"{'Layer (type)':<20} {'Output Shape':<20} {'Param #'}")
        print("=" * 50)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            # Get output shape,,, this part needs work
            if True:
                output_shape = "(?, ?)"
            
            # Get parameter count
            params = 0
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                params = layer.weights.size + layer.biases.size
            elif hasattr(layer, 'gamma') and hasattr(layer, 'beta'):
                params = layer.gamma.size + layer.beta.size
            
            total_params += params
            
            print(f"{layer.__class__.__name__:<20} {output_shape:<20} {params}")
        
        print("=" * 50)
        print(f"Total params: {total_params}")
        print("=" * 50)