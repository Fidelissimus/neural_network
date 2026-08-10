import numpy as np


class Loss:
    """
    Base class for all loss functions.

    A loss function measures the discrepancy between the network's predictions
    and the ground-truth targets. It provides:
        forward  – compute the scalar loss value.
        backward – compute dL/d(y_pred), the gradient fed back into the network.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Return the scalar loss."""
        raise NotImplementedError

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Return dL/d(y_pred)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Regression losses
# ---------------------------------------------------------------------------

class MSE(Loss):
    """
    Mean Squared Error: L = mean((y_pred - y_true)^2).

    The standard loss for regression tasks. Heavily penalises large errors
    because of the square term.

    Gradient: dL/d(y_pred) = 2 * (y_pred - y_true) / N
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2.0 * (y_pred - y_true) / y_true.size


class MAE(Loss):
    """
    Mean Absolute Error: L = mean(|y_pred - y_true|).

    More robust than MSE to outliers because errors are not squared.
    The gradient is not defined at 0; we use a subgradient of 0 there.

    Gradient: dL/d(y_pred) = sign(y_pred - y_true) / N
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(np.abs(y_pred - y_true)))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.sign(y_pred - y_true) / y_true.size


class Huber(Loss):
    """
    Huber loss (smooth L1): combines the best of MSE and MAE.

    Behaves like MSE for small errors (|e| <= delta) and like MAE for large
    errors, giving robustness to outliers while remaining differentiable.

    L = 0.5 * e^2                         if |e| <= delta
    L = delta * (|e| - 0.5 * delta)       otherwise

    Args:
        delta: Threshold separating the quadratic and linear regions. Default 1.0.
    """

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        e = y_pred - y_true
        loss = np.where(np.abs(e) <= self.delta,
                        0.5 * e ** 2,
                        self.delta * (np.abs(e) - 0.5 * self.delta))
        return float(np.mean(loss))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        e = y_pred - y_true
        grad = np.where(np.abs(e) <= self.delta,
                        e,
                        self.delta * np.sign(e))
        return grad / y_true.size

    def __repr__(self) -> str:
        return f"Huber(delta={self.delta})"


# ---------------------------------------------------------------------------
# Classification losses
# ---------------------------------------------------------------------------

class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy: L = -mean(y * log(p) + (1-y) * log(1-p)).

    Used with sigmoid output for binary classification tasks.
    Predictions are clipped to (eps, 1-eps) to avoid log(0).

    Gradient: dL/d(p) = (p - y) / (p * (1-p)) / N
    """

    _EPS = 1e-12

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        p = np.clip(y_pred, self._EPS, 1.0 - self._EPS)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, self._EPS, 1.0 - self._EPS)
        return (p - y_true) / (p * (1.0 - p)) / y_true.size


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy: L = -mean(sum(y * log(p), axis=1)).

    Used with softmax output for multi-class classification.
    When paired with Softmax, the combined gradient simplifies to:
        dL/d(z) = (p - y) / N
    which is what the backward method returns (pass-through from Softmax).

    Predictions are clipped to avoid log(0).
    """

    _EPS = 1e-12

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Sum over the last axis (classes), not a hardcoded axis=1, so this
        # is correct whether y_pred is (N, num_classes) or a per-timestep
        # (N, T, num_classes) sequence output -- see the same reasoning in
        # Softmax.forward.
        p = np.clip(y_pred, self._EPS, 1.0 - self._EPS)
        return float(-np.mean(np.sum(y_true * np.log(p), axis=-1)))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Simplified gradient for the Softmax + CCE combination.
        # Softmax.backward passes this straight through.
        return (y_pred - y_true) / y_true.shape[0]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LOSSES = {
    'mse':                    MSE,
    'mae':                    MAE,
    'huber':                  Huber,
    'binarycrossentropy':     BinaryCrossEntropy,
    'categoricalcrossentropy': CategoricalCrossEntropy,
}
