import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                    random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and testing sets
    
    Args:
        X: Input features
        y: Target values
        test_size: Proportion of dataset to include in test split
        random_state: Seed for random number generator
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Made for GNN, but I didn't write GNN type layer class yet, so this function is an orphan
def train_test_split_indices(n_samples, test_size=0.2, random_state=None):
    """
    Split indices into training and testing sets
    
    Args:
        n_samples: Total number of samples
        test_size: Proportion of dataset to include in test split
        random_state: Seed for random number generator
        
    Returns:
        train_indices, test_indices
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_size))
    
    return indices[:split_idx], indices[split_idx:]


def one_hot_encode(y: np.ndarray, num_classes: int = None) -> np.ndarray:
    """
    Convert class labels to one-hot encoded vectors
    
    Args:
        y: Class labels (1D array)
        num_classes: Number of classes (if None, inferred from y)
        
    Returns:
        One-hot encoded matrix
    """
    if num_classes is None:
        num_classes = len(np.unique(y))
        
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        
    return np.eye(num_classes)[y.astype(int)].reshape(y.shape[0], num_classes)

def normalize(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Normalize data to have zero mean and unit variance
    
    Args:
        X: Input data
        axis: Axis along which to compute mean and std
        
    Returns:
        Normalized data
    """
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    return (X - mean) / (std + 1e-8)

def minmax_scale(X: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Scale features to a given range
    
    Args:
        X: Input data
        feature_range: Desired range of transformed data
        
    Returns:
        Scaled data
    """
    min_val, max_val = feature_range
    X_min = np.min(X, axis=0, keepdims=True)
    X_max = np.max(X, axis=0, keepdims=True)
    
    X_std = (X - X_min) / (X_max - X_min + 1e-8)
    return X_std * (max_val - min_val) + min_val

def plot_training_history(history: Dict[str, List[float]], 
                         metrics: List[str] = ['loss', 'acc'],
                         figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Plot training history
    
    Args:
        history: Dictionary containing training history
        metrics: List of metrics to plot
        figsize: Figure size
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if f'train_{metric}' in history:
            axes[i].plot(history[f'train_{metric}'], label=f'Train {metric}')
        if f'val_{metric}' in history:
            axes[i].plot(history[f'val_{metric}'], label=f'Validation {metric}')
        
        axes[i].set_title(f'{metric.capitalize()} over Epochs')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_history(history: Dict[str, List[float]], file_path: str) -> None:
    """
    Save training history to a JSON file
    
    Args:
        history: Training history
        file_path: Path to save file
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_history = {k: [float(x) for x in v] for k, v in history.items()}
    
    with open(file_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)

def load_history(file_path: str) -> Dict[str, List[float]]:
    """
    Load training history from a JSON file
    
    Args:
        file_path: Path to history file
        
    Returns:
        Training history
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def learning_rate_scheduler(initial_lr: float, decay_type: str = 'exponential', 
                           decay_rate: float = 0.1, decay_steps: int = 1000) -> callable:
    """
    Create a learning rate scheduler function
    
    Args:
        initial_lr: Initial learning rate
        decay_type: Type of decay ('exponential', 'step', 'time_based')
        decay_rate: Rate of decay
        decay_steps: Number of steps between decays (for step decay)
        
    Returns:
        Function that takes epoch number and returns learning rate
    """
    if decay_type == 'exponential':
        def scheduler(epoch):
            return initial_lr * np.exp(-decay_rate * epoch)
    
    elif decay_type == 'step':
        def scheduler(epoch):
            return initial_lr * decay_rate ** (epoch // decay_steps)
    
    elif decay_type == 'time_based':
        def scheduler(epoch):
            return initial_lr / (1 + decay_rate * epoch)
    
    else:
        def scheduler(epoch):
            return initial_lr
    
    return scheduler