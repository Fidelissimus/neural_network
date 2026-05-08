import numpy as np


class Activation:
    """
    Base class for all activation functions.
    
    Each activation stores its input (and sometimes output) during the forward
    pass so that the backward pass can compute the local gradient without
    repeating work.
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute and return the activation output."""
        raise NotImplementedError

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        """
        Given the upstream gradient dL/d(output), return dL/d(input).
        """
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Piecewise-linear family
# ---------------------------------------------------------------------------

class ReLU(Activation):
    """
    Rectified Linear Unit: f(x) = max(0, x).

    Gradient is 1 where x > 0, 0 elsewhere.
    Simple and fast; the default choice for hidden layers in most networks.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0.0, x)

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        return doutput * (self.input > 0).astype(float)


class LeakyReLU(Activation):
    """
    Leaky ReLU: f(x) = x if x > 0 else alpha * x.

    Fixes the 'dying ReLU' problem by allowing a small, non-zero gradient
    when the unit is not active.

    Args:
        alpha: Slope for negative inputs. Default 0.01.
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        return doutput * np.where(self.input > 0, 1.0, self.alpha)

    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


class PReLU(Activation):
    """
    Parametric ReLU: like LeakyReLU but alpha is a learned parameter.

    alpha is updated during backward in the same step as weight gradients.
    The returned gradient is with respect to the layer input; alpha's own
    gradient is stored in self.dalpha for the layer/optimizer to consume.

    Args:
        alpha: Initial value for the learnable slope. Default 0.25.
    """

    def __init__(self, alpha: float = 0.25):
        super().__init__()
        self.alpha = alpha
        self.dalpha = 0.0  # accumulated gradient for alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        mask = (self.input <= 0)
        self.dalpha = float(np.sum(doutput * self.input * mask))
        return doutput * np.where(self.input > 0, 1.0, self.alpha)

    def __repr__(self) -> str:
        return f"PReLU(alpha={self.alpha:.4f})"


class ELU(Activation):
    """
    Exponential Linear Unit: f(x) = x if x > 0 else alpha * (exp(x) - 1).

    Smooth for negative inputs; drives mean activations closer to zero.

    Args:
        alpha: Scale for the negative saturation region. Default 1.0.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1.0))

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        grad = np.where(self.input > 0, 1.0, self.alpha * np.exp(self.input))
        return doutput * grad

    def __repr__(self) -> str:
        return f"ELU(alpha={self.alpha})"


# ---------------------------------------------------------------------------
# Smooth / probabilistic family
# ---------------------------------------------------------------------------

class Sigmoid(Activation):
    """
    Logistic sigmoid: f(x) = 1 / (1 + exp(-x)).

    Output is in (0, 1). Commonly used in binary classification output layers.
    Gradient: sigma(x) * (1 - sigma(x)).
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        # Numerically stable: avoid exp overflow for large negative x
        self.output = np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )
        return self.output

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        return doutput * self.output * (1.0 - self.output)


class Tanh(Activation):
    """
    Hyperbolic tangent: f(x) = tanh(x).

    Output is in (-1, 1); zero-centred, which can help training compared to
    sigmoid. Gradient: 1 - tanh(x)^2.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.tanh(x)
        return self.output

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        return doutput * (1.0 - self.output ** 2)


class Softmax(Activation):
    """
    Softmax: f(x)_i = exp(x_i) / sum_j(exp(x_j)).

    Converts a vector of raw scores into a probability distribution.
    Used exclusively in multi-class classification output layers.

    The backward pass returns the upstream gradient unchanged because
    the full Jacobian is folded into CategoricalCrossEntropy.backward,
    which simplifies the combined gradient to (y_pred - y_true) / N.
    Using Softmax with any other loss requires computing the full Jacobian.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        # Gradient is passed straight through; the real work is in the loss.
        return doutput


class GELU(Activation):
    """
    Gaussian Error Linear Unit: f(x) = x * Phi(x).

    Uses the tanh approximation popularised by BERT / GPT:
        f(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    GELU is the default activation in most modern transformer architectures.
    """

    _SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
    _COEFF = 0.044715

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self._tanh_arg = self._SQRT_2_OVER_PI * (x + self._COEFF * x ** 3)
        self._tanh_val = np.tanh(self._tanh_arg)
        self.output = 0.5 * x * (1.0 + self._tanh_val)
        return self.output

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        x = self.input
        sech2 = 1.0 - self._tanh_val ** 2
        dtanh_arg = self._SQRT_2_OVER_PI * (1.0 + 3.0 * self._COEFF * x ** 2)
        dgelu = 0.5 * (1.0 + self._tanh_val) + 0.5 * x * sech2 * dtanh_arg
        return doutput * dgelu


class Swish(Activation):
    """
    Swish (SiLU): f(x) = x * sigmoid(x).

    Self-gated activation found via neural architecture search.
    Smooth and non-monotonic; often outperforms ReLU on deeper networks.
    Gradient: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    """

    def __init__(self):
        super().__init__()
        self._sigmoid = Sigmoid()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self._sig = self._sigmoid.forward(x)
        self.output = x * self._sig
        return self.output

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        grad = self._sig * (1.0 + self.input * (1.0 - self._sig))
        return doutput * grad


class Linear(Activation):
    """
    Identity / linear activation: f(x) = x.

    Used in regression output layers where the output is unbounded.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        return doutput


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ACTIVATIONS = {
    'relu':      ReLU,
    'leakyrelu': LeakyReLU,
    'prelu':     PReLU,
    'elu':       ELU,
    'sigmoid':   Sigmoid,
    'tanh':      Tanh,
    'softmax':   Softmax,
    'gelu':      GELU,
    'swish':     Swish,
    'linear':    Linear,
    'none':      Linear,   # alias for output layers with no activation
}
