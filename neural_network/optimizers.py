import numpy as np


class Optimizer:
    """
    Base class for all parameter optimizers.

    Optimizers receive a layer object and update its weights and biases using
    the gradients stored in layer.dweights and layer.dbiases (computed during
    the backward pass and stored by the layer itself, not re-computed here).
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, layer) -> None:
        """Apply one parameter update step to the given layer."""
        raise NotImplementedError

    def update_raw(self, param_id: int, param: np.ndarray,
                   grad: np.ndarray) -> np.ndarray:
        """
        Apply the optimizer update rule to a single (param, grad) pair and
        return the updated parameter.

        This lower-level method is used by recurrent and attention layers that
        manage multiple weight matrices independently.  Subclasses must override
        this alongside update().

        Args:
            param_id: A unique integer identifier for this parameter tensor
                      (used to maintain per-parameter optimizer state).
            param:    Current value of the parameter array.
            grad:     Gradient of the loss w.r.t. param.

        Returns:
            Updated parameter array.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.learning_rate})"


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.

    Without momentum (momentum=0.0) this is vanilla gradient descent:
        w -= lr * dw

    With momentum it accumulates a velocity vector that dampens oscillations
    and accelerates convergence in the relevant direction:
        v = momentum * v + (1 - momentum) * dw
        w -= lr * v

    Args:
        learning_rate: Step size. Default 0.01.
        momentum: Momentum coefficient in [0, 1). 0 means no momentum.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self._velocities: dict = {}

    def update(self, layer) -> None:
        lid = id(layer)
        if lid not in self._velocities:
            self._velocities[lid] = {
                'weights': np.zeros_like(layer.weights),
                'biases':  np.zeros_like(layer.biases),
            }
        v = self._velocities[lid]
        v['weights'] = self.momentum * v['weights'] + (1.0 - self.momentum) * layer.dweights
        v['biases']  = self.momentum * v['biases']  + (1.0 - self.momentum) * layer.dbiases
        layer.weights -= self.learning_rate * v['weights']
        layer.biases  -= self.learning_rate * v['biases']

    def update_raw(self, param_id: int, param: np.ndarray,
                   grad: np.ndarray) -> np.ndarray:
        if param_id not in self._velocities:
            self._velocities[param_id] = np.zeros_like(param)
        v = self._velocities[param_id]
        v[:] = self.momentum * v + (1.0 - self.momentum) * grad
        return param - self.learning_rate * v

    def __repr__(self) -> str:
        return f"SGD(lr={self.learning_rate}, momentum={self.momentum})"


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Maintains per-parameter first and second moment estimates of the gradient,
    with bias correction in the early steps.

    Update rule:
        m = beta1 * m + (1 - beta1) * g          # biased 1st moment
        v = beta2 * v + (1 - beta2) * g^2        # biased 2nd moment
        m_hat = m / (1 - beta1^t)                 # bias-corrected
        v_hat = v / (1 - beta2^t)
        w -= lr * m_hat / (sqrt(v_hat) + epsilon)

    Args:
        learning_rate: Step size. Default 0.001.
        beta1: Decay rate for the first moment. Default 0.9.
        beta2: Decay rate for the second moment. Default 0.999.
        epsilon: Small constant for numerical stability. Default 1e-8.
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m: dict = {}   # first moment
        self._v: dict = {}   # second moment
        self._t: int = 0     # global time step

    def update(self, layer) -> None:
        lid = id(layer)
        if lid not in self._m:
            self._m[lid] = {
                'weights': np.zeros_like(layer.weights),
                'biases':  np.zeros_like(layer.biases),
            }
            self._v[lid] = {
                'weights': np.zeros_like(layer.weights),
                'biases':  np.zeros_like(layer.biases),
            }

        self._t += 1

        for param in ('weights', 'biases'):
            grad = layer.dweights if param == 'weights' else layer.dbiases

            self._m[lid][param] = self.beta1 * self._m[lid][param] + (1.0 - self.beta1) * grad
            self._v[lid][param] = self.beta2 * self._v[lid][param] + (1.0 - self.beta2) * grad ** 2

            m_hat = self._m[lid][param] / (1.0 - self.beta1 ** self._t)
            v_hat = self._v[lid][param] / (1.0 - self.beta2 ** self._t)

            if param == 'weights':
                layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            else:
                layer.biases  -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def update_raw(self, param_id, param: np.ndarray,
                   grad: np.ndarray) -> np.ndarray:
        """
        param_id should be a stable hashable key (string or int) that
        uniquely identifies this parameter across training steps.
        """
        if param_id not in self._m:
            self._m[param_id] = np.zeros_like(param)
            self._v[param_id] = np.zeros_like(param)
        elif self._m[param_id].shape != param.shape:
            # Shape changed (shouldn't happen, but guard against it)
            self._m[param_id] = np.zeros_like(param)
            self._v[param_id] = np.zeros_like(param)
        # Use a per-key time step so each parameter has its own bias correction
        t_key = f'__t_{param_id}'
        self._m[t_key] = self._m.get(t_key, 0) + 1
        t = self._m[t_key]
        self._m[param_id] = self.beta1 * self._m[param_id] + (1.0 - self.beta1) * grad
        self._v[param_id] = self.beta2 * self._v[param_id] + (1.0 - self.beta2) * grad ** 2
        m_hat = self._m[param_id] / (1.0 - self.beta1 ** t)
        v_hat = self._v[param_id] / (1.0 - self.beta2 ** t)
        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def __repr__(self) -> str:
        return (f"Adam(lr={self.learning_rate}, beta1={self.beta1}, "
                f"beta2={self.beta2}, eps={self.epsilon})")


class RMSprop(Optimizer):
    """
    RMSprop optimizer.

    Maintains a moving average of squared gradients to normalise the gradient,
    which helps deal with non-stationary objectives and varying gradient scales.

    Update rule:
        cache = beta * cache + (1 - beta) * g^2
        w -= lr * g / (sqrt(cache) + epsilon)

    Args:
        learning_rate: Step size. Default 0.001.
        beta: Decay rate for the squared gradient. Default 0.9.
        epsilon: Small constant for numerical stability. Default 1e-8.
    """

    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9,
                 epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self._cache: dict = {}

    def update(self, layer) -> None:
        lid = id(layer)
        if lid not in self._cache:
            self._cache[lid] = {
                'weights': np.zeros_like(layer.weights),
                'biases':  np.zeros_like(layer.biases),
            }

        for param in ('weights', 'biases'):
            grad = layer.dweights if param == 'weights' else layer.dbiases

            self._cache[lid][param] = (self.beta * self._cache[lid][param]
                                       + (1.0 - self.beta) * grad ** 2)

            update = self.learning_rate * grad / (np.sqrt(self._cache[lid][param]) + self.epsilon)

            if param == 'weights':
                layer.weights -= update
            else:
                layer.biases  -= update

    def update_raw(self, param_id: int, param: np.ndarray,
                   grad: np.ndarray) -> np.ndarray:
        if param_id not in self._cache:
            self._cache[param_id] = np.zeros_like(param)
        self._cache[param_id] = self.beta * self._cache[param_id] + (1.0 - self.beta) * grad ** 2
        return param - self.learning_rate * grad / (np.sqrt(self._cache[param_id]) + self.epsilon)

    def __repr__(self) -> str:
        return f"RMSprop(lr={self.learning_rate}, beta={self.beta})"


class Adagrad(Optimizer):
    """
    Adagrad (Adaptive Gradient) optimizer.

    Accumulates the sum of squared gradients for each parameter and scales the
    learning rate accordingly. Parameters that receive large gradients get
    smaller effective learning rates; sparse parameters get larger updates.

    Update rule:
        G += g^2                            # accumulated squared gradients
        w -= lr * g / (sqrt(G) + epsilon)

    Note: the monotonically increasing G means the learning rate shrinks to
    zero over time, which can cause premature stopping on long training runs.

    Args:
        learning_rate: Step size. Default 0.01.
        epsilon: Small constant for numerical stability. Default 1e-8.
    """

    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self._G: dict = {}

    def update(self, layer) -> None:
        lid = id(layer)
        if lid not in self._G:
            self._G[lid] = {
                'weights': np.zeros_like(layer.weights),
                'biases':  np.zeros_like(layer.biases),
            }

        for param in ('weights', 'biases'):
            grad = layer.dweights if param == 'weights' else layer.dbiases

            self._G[lid][param] += grad ** 2
            update = self.learning_rate * grad / (np.sqrt(self._G[lid][param]) + self.epsilon)

            if param == 'weights':
                layer.weights -= update
            else:
                layer.biases  -= update

    def update_raw(self, param_id: int, param: np.ndarray,
                   grad: np.ndarray) -> np.ndarray:
        if param_id not in self._G:
            self._G[param_id] = np.zeros_like(param)
        self._G[param_id] += grad ** 2
        return param - self.learning_rate * grad / (np.sqrt(self._G[param_id]) + self.epsilon)

    def __repr__(self) -> str:
        return f"Adagrad(lr={self.learning_rate})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OPTIMIZERS = {
    'sgd':      SGD,
    'adam':     Adam,
    'rmsprop':  RMSprop,
    'adagrad':  Adagrad,
}
