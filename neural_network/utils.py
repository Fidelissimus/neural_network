import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.2,
                     random_state: Optional[int] = None
                     ) -> Tuple[np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]:
    """
    Split arrays into random train and test subsets.

    Args:
        X:            Feature matrix, shape (N, ...).
        y:            Target array, shape (N, ...).
        test_size:    Proportion of the dataset to use as test set. Default 0.2.
        random_state: Seed for reproducibility. Default None.

    Returns:
        x_train, x_test, y_train, y_test
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test    = max(1, int(n_samples * test_size))

    indices       = np.random.permutation(n_samples)
    test_indices  = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def train_test_split_indices(n_samples: int, test_size: float = 0.2,
                             random_state: Optional[int] = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate train and test index arrays for a dataset of `n_samples` samples.

    Useful for graph-structured data (e.g. GNNs) where you cannot simply
    slice the array along the first axis.

    Args:
        n_samples:    Total number of samples.
        test_size:    Fraction to use as test. Default 0.2.
        random_state: Seed for reproducibility.

    Returns:
        train_indices, test_indices (both as 1-D integer arrays)
    """
    if random_state is not None:
        np.random.seed(random_state)

    indices   = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1.0 - test_size))
    return indices[:split_idx], indices[split_idx:]


# ---------------------------------------------------------------------------
# Encoding and preprocessing
# ---------------------------------------------------------------------------

def one_hot_encode(y: np.ndarray,
                   num_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert integer class labels to a one-hot encoded matrix.

    Args:
        y:           1-D array of integer class labels.
        num_classes: Number of classes. If None, inferred as max(y) + 1.

    Returns:
        One-hot matrix of shape (N, num_classes).
    """
    y = np.asarray(y, dtype=int).flatten()
    if num_classes is None:
        num_classes = int(y.max()) + 1
    return np.eye(num_classes, dtype=float)[y]


def normalize(X: np.ndarray, axis: int = 0,
              eps: float = 1e-8) -> np.ndarray:
    """
    Standardise data to zero mean and unit variance (Z-score normalisation).

    Args:
        X:    Input data array.
        axis: Axis along which mean and std are computed. Default 0 (per feature).
        eps:  Small constant added to std to avoid division by zero.

    Returns:
        Normalised array with the same shape as X.
    """
    mean = np.mean(X, axis=axis, keepdims=True)
    std  = np.std(X,  axis=axis, keepdims=True)
    return (X - mean) / (std + eps)


def minmax_scale(X: np.ndarray,
                 feature_range: Tuple[float, float] = (0.0, 1.0),
                 eps: float = 1e-8) -> np.ndarray:
    """
    Scale features to a specified range using min-max normalisation.

    Args:
        X:             Input data array.
        feature_range: (min, max) of the output range. Default (0, 1).
        eps:           Small constant to avoid division by zero.

    Returns:
        Scaled array with values in `feature_range`.
    """
    lo, hi = feature_range
    x_min = np.min(X, axis=0, keepdims=True)
    x_max = np.max(X, axis=0, keepdims=True)
    x_std = (X - x_min) / (x_max - x_min + eps)
    return x_std * (hi - lo) + lo


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_history(history: Dict[str, List[float]],
                          metrics: List[str] = ('loss', 'acc'),
                          figsize: Tuple[int, int] = (12, 4),
                          save_path: Optional[str] = None) -> None:
    """
    Plot training (and optionally validation) curves for the requested metrics.

    Args:
        history:   Dict returned by NeuralNetwork.train(), containing keys
                   like 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        metrics:   Which metrics to plot. Each metric produces one subplot.
                   Default ('loss', 'acc').
        figsize:   Overall figure size as (width, height) in inches.
        save_path: If provided, saves the figure to this path instead of
                   showing it interactively.
    """
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        train_key = f'train_{metric}'
        val_key   = f'val_{metric}'

        if train_key in history and history[train_key]:
            ax.plot(history[train_key], label=f'Train {metric}', linewidth=1.5)
        if val_key in history and history[val_key]:
            ax.plot(history[val_key],   label=f'Val {metric}',   linewidth=1.5,
                    linestyle='--')

        ax.set_title(f'{metric.capitalize()} over epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# History I/O
# ---------------------------------------------------------------------------

def save_history(history: Dict[str, List[float]], file_path: str) -> None:
    """
    Serialise a training history dict to a JSON file.

    All values are converted to plain Python floats so json.dump works
    regardless of whether the history contains numpy scalars.

    Args:
        history:   Dict with string keys and lists of numeric values.
        file_path: Destination path.
    """
    serialisable = {k: [float(x) for x in v] for k, v in history.items()}
    with open(file_path, 'w') as f:
        json.dump(serialisable, f, indent=4)


def load_history(file_path: str) -> Dict[str, List[float]]:
    """
    Load a training history dict that was previously saved with save_history.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Dict with string keys and lists of floats.
    """
    with open(file_path, 'r') as f:
        return json.load(f)
