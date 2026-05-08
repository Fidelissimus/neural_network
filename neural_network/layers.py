import numpy as np
from typing import Optional, Tuple
from .activations import ACTIVATIONS, Activation


class Layer:
    """
    Base class for all layers.

    Every layer must implement forward and backward.  Trainable layers
    (those with weights and biases) must also expose:
        dweights   – gradient of the loss w.r.t. weights (set by backward)
        dbiases    – gradient of the loss w.r.t. biases  (set by backward)

    The optimizer reads these attributes to apply the update; the layer does
    NOT update its own parameters inside backward.  This separation of
    concerns makes it possible to swap optimizers freely without touching
    any layer code.
    """

    def __init__(self):
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None
        # Indicates whether this layer has learnable parameters that an
        # optimizer should update.
        self.trainable: bool = False

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Compute the layer output from input x."""
        raise NotImplementedError

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        """
        Given dL/d(output), compute and store parameter gradients and
        return dL/d(input) for the layer below.
        """
        raise NotImplementedError

    def get_parameters(self) -> dict:
        """Return a JSON-serialisable dict of all persistent parameters."""
        return {}

    def set_parameters(self, parameters: dict) -> None:
        """Restore parameters from a dict produced by get_parameters."""
        pass

    def output_shape(self, input_shape: tuple) -> tuple:
        """Return the output shape given the input shape (batch dim excluded)."""
        raise NotImplementedError

    def param_count(self) -> int:
        """Return the total number of trainable scalar parameters."""
        return 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Dense (fully-connected) layer
# ---------------------------------------------------------------------------

class Dense(Layer):
    """
    Fully-connected layer: output = activation(input @ W + b).

    Weight initialisation uses He (Kaiming) initialisation scaled by
    sqrt(2 / fan_in), which is appropriate for ReLU-family activations.
    For sigmoid / tanh, consider scaling by sqrt(1 / fan_in) (Xavier).

    Args:
        input_size:  Number of input features.
        output_size: Number of output neurons.
        activation:  Activation name (str) or an Activation instance.
                     Defaults to 'relu'.
    """

    def __init__(self, input_size: int, output_size: int,
                 activation: str | Activation = 'relu'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True

        # He initialisation
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases  = np.zeros((1, output_size))

        # Gradient placeholders (populated by backward)
        self.dweights: Optional[np.ndarray] = None
        self.dbiases:  Optional[np.ndarray] = None

        # Resolve activation
        if isinstance(activation, str):
            key = activation.lower()
            if key not in ACTIVATIONS:
                raise ValueError(
                    f"Unknown activation '{activation}'. "
                    f"Available: {list(ACTIVATIONS.keys())}"
                )
            self.activation: Activation = ACTIVATIONS[key]()
        elif isinstance(activation, Activation):
            self.activation = activation
        else:
            raise TypeError(
                f"activation must be a str or Activation instance, "
                f"got {type(activation).__name__}"
            )

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = x
        self.z     = x @ self.weights + self.biases   # pre-activation
        self.output = self.activation.forward(self.z)
        return self.output

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        # Backprop through the activation: dL/dz
        dz = self.activation.backward(doutput)

        # Gradients w.r.t. parameters (stored for the optimizer)
        self.dweights = self.input.T @ dz
        self.dbiases  = np.sum(dz, axis=0, keepdims=True)

        # Gradient w.r.t. input (passed to the layer below)
        return dz @ self.weights.T

    def get_parameters(self) -> dict:
        return {
            'weights':    self.weights.tolist(),
            'biases':     self.biases.tolist(),
            'activation': self.activation.__class__.__name__.lower(),
            'input_size': self.input_size,
            'output_size': self.output_size,
        }

    def set_parameters(self, parameters: dict) -> None:
        self.weights = np.array(parameters['weights'])
        self.biases  = np.array(parameters['biases'])

    def output_shape(self, input_shape: tuple) -> tuple:
        return (self.output_size,)

    def param_count(self) -> int:
        return self.weights.size + self.biases.size

    def __repr__(self) -> str:
        return (f"Dense({self.input_size} -> {self.output_size}, "
                f"activation={self.activation})")


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

class Dropout(Layer):
    """
    Inverted dropout regularisation layer.

    During training, each neuron is independently zeroed with probability
    `rate`, and the surviving activations are scaled up by 1/(1-rate) so
    that the expected output magnitude is unchanged at inference time.
    During inference the layer is a no-op (returns input unchanged).

    Args:
        rate: Fraction of neurons to drop. Must be in [0, 1). Default 0.5.
    """

    def __init__(self, rate: float = 0.5):
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate}.")
        self.rate = rate
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = x
        if not training or self.rate == 0.0:
            return x
        # Inverted dropout: scale at train time, nothing to do at test time
        keep_prob = 1.0 - self.rate
        self._mask = (np.random.rand(*x.shape) < keep_prob) / keep_prob
        return x * self._mask

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        if self._mask is None:
            # Was in inference mode during forward; gradient passes unchanged
            return doutput
        return doutput * self._mask

    def get_parameters(self) -> dict:
        return {'rate': self.rate}

    def output_shape(self, input_shape: tuple) -> tuple:
        return input_shape

    def __repr__(self) -> str:
        return f"Dropout(rate={self.rate})"


# ---------------------------------------------------------------------------
# Batch Normalisation
# ---------------------------------------------------------------------------

class BatchNorm(Layer):
    """
    Batch Normalisation layer.

    Normalises each feature across the batch to zero mean and unit variance,
    then applies learned scale (gamma) and shift (beta) parameters.
    Running statistics are maintained for use at inference time.

    During training:
        mu  = mean(x, axis=0)
        var = var(x,  axis=0)
        x_hat = (x - mu) / sqrt(var + eps)
        out = gamma * x_hat + beta

    During inference, the exponential running mean and variance accumulated
    during training are used instead of the batch statistics.

    The optimizer updates gamma and beta via the standard .weights / .biases
    interface; both are exposed as properties that alias gamma and beta.

    Args:
        num_features: Number of input features (= output size of the previous layer).
        momentum:     EMA coefficient for running stats. Default 0.9.
        eps:          Small constant for numerical stability. Default 1e-5.
    """

    def __init__(self, num_features: int, momentum: float = 0.9,
                 eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.trainable = True

        self.gamma = np.ones((1, num_features))
        self.beta  = np.zeros((1, num_features))

        # Gradient placeholders (populated by backward, read by optimizer)
        self.dweights: Optional[np.ndarray] = None   # alias for dgamma
        self.dbiases:  Optional[np.ndarray] = None   # alias for dbeta

        # Running statistics (updated only during training forward passes)
        self.running_mean = np.zeros((1, num_features))
        self.running_var  = np.ones((1,  num_features))

        # Intermediate values needed in backward
        self._x_hat:      Optional[np.ndarray] = None
        self._std:        Optional[np.ndarray] = None
        self._x_centered: Optional[np.ndarray] = None
        self._var:        Optional[np.ndarray] = None

    # Expose gamma/beta as weights/biases so the generic optimizer interface works.
    @property
    def weights(self) -> np.ndarray:
        """Alias so the optimizer can update gamma via the standard interface."""
        return self.gamma

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        self.gamma = value

    @property
    def biases(self) -> np.ndarray:
        """Alias so the optimizer can update beta via the standard interface."""
        return self.beta

    @biases.setter
    def biases(self, value: np.ndarray) -> None:
        self.beta = value

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = x

        if training:
            mean = np.mean(x, axis=0, keepdims=True)
            var  = np.var(x,  axis=0, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            self.running_var  = self.momentum * self.running_var  + (1.0 - self.momentum) * var

            self._x_centered = x - mean
            self._var        = var
            self._std        = np.sqrt(var + self.eps)
            self._x_hat      = self._x_centered / self._std
        else:
            self._x_centered = x - self.running_mean
            self._std        = np.sqrt(self.running_var + self.eps)
            self._x_hat      = self._x_centered / self._std

        self.output = self.gamma * self._x_hat + self.beta
        return self.output

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        m = self.input.shape[0]

        # Gradients for gamma and beta
        dgamma = np.sum(doutput * self._x_hat,    axis=0, keepdims=True)
        dbeta  = np.sum(doutput,                  axis=0, keepdims=True)

        # Stored so the optimizer can update gamma and beta
        self.dweights = dgamma
        self.dbiases  = dbeta

        # Gradient w.r.t. x_hat
        dx_hat = doutput * self.gamma

        # Gradient w.r.t. variance
        dvar = np.sum(
            dx_hat * self._x_centered * (-0.5) * (self._var + self.eps) ** (-1.5),
            axis=0, keepdims=True
        )

        # Gradient w.r.t. mean
        dmean = (np.sum(dx_hat * (-1.0 / self._std), axis=0, keepdims=True)
                 + dvar * np.mean(-2.0 * self._x_centered, axis=0, keepdims=True))

        # Gradient w.r.t. input
        dinput = (dx_hat / self._std
                  + dvar * 2.0 * self._x_centered / m
                  + dmean / m)
        return dinput

    def get_parameters(self) -> dict:
        return {
            'gamma':        self.gamma.tolist(),
            'beta':         self.beta.tolist(),
            'running_mean': self.running_mean.tolist(),
            'running_var':  self.running_var.tolist(),
            'num_features': self.num_features,
            'momentum':     self.momentum,
            'eps':          self.eps,
        }

    def set_parameters(self, parameters: dict) -> None:
        self.gamma        = np.array(parameters['gamma'])
        self.beta         = np.array(parameters['beta'])
        self.running_mean = np.array(parameters['running_mean'])
        self.running_var  = np.array(parameters['running_var'])

    def output_shape(self, input_shape: tuple) -> tuple:
        return input_shape

    def param_count(self) -> int:
        return self.gamma.size + self.beta.size

    def __repr__(self) -> str:
        return f"BatchNorm({self.num_features}, momentum={self.momentum})"


class Flatten(Layer):
    """
    Flatten layer.

    Collapses all dimensions except the batch dimension into a single vector.
    Useful as a bridge between Conv2D layers and Dense layers.

    Example:
        input  shape: (batch, 8, 8, 16)
        output shape: (batch, 1024)
    """

    def __init__(self):
        super().__init__()
        self._input_shape: Optional[tuple] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = x
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        return doutput.reshape(self._input_shape)

    def output_shape(self, input_shape: tuple) -> tuple:
        return (int(np.prod(input_shape)),)

    def __repr__(self) -> str:
        return "Flatten()"


# ---------------------------------------------------------------------------
# Conv2D
# ---------------------------------------------------------------------------

class Conv2D(Layer):
    """
    2-D Convolutional layer.

    Applies a bank of learnable filters to a 4-D input tensor of shape
    (batch, height, width, in_channels) and produces an output of shape
    (batch, out_height, out_width, out_channels).

    Convolution is implemented as a matrix multiplication using the
    im2col transformation, which converts each receptive field into a
    column so that the full forward pass reduces to a single GEMM call.
    The corresponding col2im inverse is used in the backward pass.

    Only 'valid' and 'same' padding modes are supported.

    Args:
        in_channels:   Number of input feature maps.
        out_channels:  Number of filters (output feature maps).
        kernel_size:   Height and width of each filter (square kernel).
        stride:        Step size of the sliding window. Default 1.
        padding:       'valid' (no padding) or 'same' (output same spatial
                       size as input when stride=1). Default 'valid'.
        activation:    Activation name or instance. Default 'relu'.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1,
                 padding: str = 'valid',
                 activation: str | Activation = 'relu'):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding.lower()
        self.trainable    = True

        if self.padding not in ('valid', 'same'):
            raise ValueError("padding must be 'valid' or 'same'.")

        # He initialisation for filters: shape (kH, kW, C_in, C_out)
        fan_in = kernel_size * kernel_size * in_channels
        self.weights = (np.random.randn(kernel_size, kernel_size, in_channels, out_channels)
                        * np.sqrt(2.0 / fan_in))
        self.biases  = np.zeros((1, out_channels))

        self.dweights: Optional[np.ndarray] = None
        self.dbiases:  Optional[np.ndarray] = None

        if isinstance(activation, str):
            self.activation: Activation = ACTIVATIONS[activation.lower()]()
        else:
            self.activation = activation

        # Cached im2col output for use in backward
        self._col:    Optional[np.ndarray] = None
        self._padded: Optional[np.ndarray] = None
        self._pad_h:  int = 0
        self._pad_w:  int = 0

    # ------------------------------------------------------------------
    # im2col / col2im helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _im2col(x: np.ndarray, kH: int, kW: int,
                stride: int) -> np.ndarray:
        """
        Convert a padded image tensor into a column matrix.

        Args:
            x:      (N, H, W, C)
            kH, kW: kernel height and width
            stride: convolution stride

        Returns:
            col: (N, out_H, out_W, kH * kW * C)
        """
        N, H, W, C = x.shape
        out_H = (H - kH) // stride + 1
        out_W = (W - kW) // stride + 1

        col = np.zeros((N, out_H, out_W, kH * kW * C))

        for i in range(out_H):
            for j in range(out_W):
                row_start = i * stride
                col_start = j * stride
                patch = x[:, row_start:row_start + kH,
                           col_start:col_start + kW, :]   # (N, kH, kW, C)
                col[:, i, j, :] = patch.reshape(N, -1)

        return col

    @staticmethod
    def _col2im(col: np.ndarray, x_shape: tuple, kH: int, kW: int,
                stride: int) -> np.ndarray:
        """
        Inverse of _im2col: scatter column values back into an image tensor.

        Args:
            col:     (N, out_H, out_W, kH * kW * C)
            x_shape: shape of the original padded image (N, H, W, C)
            kH, kW:  kernel height and width
            stride:  convolution stride

        Returns:
            x: (N, H, W, C) accumulated gradient
        """
        N, H, W, C = x_shape
        out_H = (H - kH) // stride + 1
        out_W = (W - kW) // stride + 1

        x = np.zeros(x_shape)

        for i in range(out_H):
            for j in range(out_W):
                row_start = i * stride
                col_start = j * stride
                patch = col[:, i, j, :].reshape(N, kH, kW, C)
                x[:, row_start:row_start + kH,
                   col_start:col_start + kW, :] += patch

        return x

    # ------------------------------------------------------------------
    # Padding helpers
    # ------------------------------------------------------------------

    def _apply_padding(self, x: np.ndarray) -> Tuple[np.ndarray, int, int]:
        if self.padding == 'valid':
            return x, 0, 0
        # 'same': output size = ceil(in_size / stride)
        _, H, W, _ = x.shape
        pad_h = max((H - 1) * self.stride + self.kernel_size - H, 0)
        pad_w = max((W - 1) * self.stride + self.kernel_size - W, 0)
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left
        x_padded = np.pad(x, ((0, 0), (pad_top, pad_bottom),
                               (pad_left, pad_right), (0, 0)),
                          mode='constant')
        return x_padded, pad_h // 2, pad_w // 2

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Args:
            x: (N, H, W, C_in)

        Returns:
            out: (N, out_H, out_W, C_out)
        """
        self.input = x
        kH = kW = self.kernel_size

        x_padded, self._pad_h, self._pad_w = self._apply_padding(x)
        self._padded = x_padded

        N, pH, pW, C = x_padded.shape
        out_H = (pH - kH) // self.stride + 1
        out_W = (pW - kW) // self.stride + 1

        # im2col: (N, out_H, out_W, kH*kW*C_in)
        col = self._im2col(x_padded, kH, kW, self.stride)
        self._col = col

        # Flatten filters: (kH*kW*C_in, C_out)
        W_flat = self.weights.reshape(-1, self.out_channels)

        # Matrix multiply: (N, out_H, out_W, C_out)
        z = col @ W_flat + self.biases
        self.output = self.activation.forward(z)
        return self.output

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        """
        Args:
            doutput: (N, out_H, out_W, C_out)

        Returns:
            dinput: (N, H, W, C_in)
        """
        kH = kW = self.kernel_size
        N   = self.input.shape[0]

        # Backprop through activation: (N, out_H, out_W, C_out)
        dz = self.activation.backward(doutput)

        W_flat = self.weights.reshape(-1, self.out_channels)   # (kH*kW*C, C_out)

        # Gradient w.r.t. weights: col^T @ dz summed over (N, out_H, out_W)
        # col is (N, out_H, out_W, kH*kW*C); treat first 3 dims as batch
        col_flat = self._col.reshape(-1, kH * kW * self.in_channels)  # (N*oH*oW, kH*kW*C)
        dz_flat  = dz.reshape(-1, self.out_channels)                   # (N*oH*oW, C_out)

        dW_flat = col_flat.T @ dz_flat                           # (kH*kW*C, C_out)
        self.dweights = dW_flat.reshape(self.weights.shape)
        self.dbiases  = np.sum(dz, axis=(0, 1, 2), keepdims=False).reshape(1, -1)

        # Gradient w.r.t. col: (N*oH*oW, kH*kW*C)
        dcol_flat = dz_flat @ W_flat.T
        dcol = dcol_flat.reshape(self._col.shape)   # (N, oH, oW, kH*kW*C)

        # col2im to get gradient w.r.t. padded input
        dx_padded = self._col2im(dcol, self._padded.shape, kH, kW, self.stride)

        # Remove padding
        if self._pad_h == 0 and self._pad_w == 0:
            return dx_padded
        return dx_padded[:, self._pad_h: dx_padded.shape[1] - self._pad_h,
                           self._pad_w: dx_padded.shape[2] - self._pad_w, :]

    def get_parameters(self) -> dict:
        return {
            'weights':      self.weights.tolist(),
            'biases':       self.biases.tolist(),
            'in_channels':  self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size':  self.kernel_size,
            'stride':       self.stride,
            'padding':      self.padding,
            'activation':   self.activation.__class__.__name__.lower(),
        }

    def set_parameters(self, parameters: dict) -> None:
        self.weights = np.array(parameters['weights'])
        self.biases  = np.array(parameters['biases'])

    def output_shape(self, input_shape: tuple) -> tuple:
        H, W, _ = input_shape
        if self.padding == 'valid':
            out_H = (H - self.kernel_size) // self.stride + 1
            out_W = (W - self.kernel_size) // self.stride + 1
        else:
            out_H = int(np.ceil(H / self.stride))
            out_W = int(np.ceil(W / self.stride))
        return (out_H, out_W, self.out_channels)

    def param_count(self) -> int:
        return self.weights.size + self.biases.size

    def __repr__(self) -> str:
        return (f"Conv2D({self.in_channels} -> {self.out_channels}, "
                f"kernel={self.kernel_size}, stride={self.stride}, "
                f"padding='{self.padding}', activation={self.activation})")


# ===========================================================================
# Normalisation
# ===========================================================================

class LayerNorm(Layer):
    """
    Layer Normalisation (Ba et al., 2016).

    Unlike BatchNorm, which normalises across the batch dimension, LayerNorm
    normalises across the *feature* dimension for each sample independently.
    This makes it suitable for variable-length sequences and small batch sizes,
    and it is the standard normalisation layer in transformers and RNNs.

    Given input x of shape (..., features):
        mu    = mean(x, axis=-1, keepdims=True)
        sigma = std(x,  axis=-1, keepdims=True)
        x_hat = (x - mu) / (sigma + eps)
        out   = gamma * x_hat + beta

    gamma and beta are learned per-feature scale and shift parameters,
    initialised to 1 and 0 respectively.

    The layer works on any rank of input (2-D batch, 3-D sequence, etc.);
    normalisation always happens over the last axis.

    Args:
        num_features: Size of the last (feature) dimension.
        eps:          Small constant for numerical stability. Default 1e-5.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.trainable = True

        self.gamma = np.ones((num_features,))
        self.beta  = np.zeros((num_features,))

        self.dweights: Optional[np.ndarray] = None
        self.dbiases:  Optional[np.ndarray] = None

        self._x_hat:      Optional[np.ndarray] = None
        self._std:        Optional[np.ndarray] = None
        self._x_centered: Optional[np.ndarray] = None

    # Expose gamma/beta through the standard optimizer interface.
    @property
    def weights(self) -> np.ndarray:
        return self.gamma

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        self.gamma = value

    @property
    def biases(self) -> np.ndarray:
        return self.beta

    @biases.setter
    def biases(self, value: np.ndarray) -> None:
        self.beta = value

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = x
        mean = np.mean(x, axis=-1, keepdims=True)
        var  = np.var(x,  axis=-1, keepdims=True)
        self._x_centered = x - mean
        self._std        = np.sqrt(var + self.eps)
        self._x_hat      = self._x_centered / self._std
        self.output = self.gamma * self._x_hat + self.beta
        return self.output

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        # Number of features in the last axis
        n = self.input.shape[-1]

        # Gradients for gamma and beta (sum over all axes except features)
        # doutput shape: same as input, e.g. (N, F) or (N, T, F)
        reduce_axes = tuple(range(doutput.ndim - 1))
        self.dweights = np.sum(doutput * self._x_hat, axis=reduce_axes)
        self.dbiases  = np.sum(doutput,               axis=reduce_axes)

        dx_hat = doutput * self.gamma

        # Gradient of variance and mean (same derivation as BatchNorm but
        # over the last axis)
        dvar  = np.sum(dx_hat * self._x_centered * (-0.5)
                       * (self._std ** 2 - self.eps) ** (-0.75),
                       axis=-1, keepdims=True)
        dmean = (np.sum(dx_hat * (-1.0 / self._std), axis=-1, keepdims=True)
                 + dvar * np.mean(-2.0 * self._x_centered, axis=-1, keepdims=True))

        dinput = dx_hat / self._std + dvar * 2.0 * self._x_centered / n + dmean / n
        return dinput

    def get_parameters(self) -> dict:
        return {
            'gamma':        self.gamma.tolist(),
            'beta':         self.beta.tolist(),
            'num_features': self.num_features,
            'eps':          self.eps,
        }

    def set_parameters(self, parameters: dict) -> None:
        self.gamma = np.array(parameters['gamma'])
        self.beta  = np.array(parameters['beta'])

    def output_shape(self, input_shape: tuple) -> tuple:
        return input_shape

    def param_count(self) -> int:
        return self.gamma.size + self.beta.size

    def __repr__(self) -> str:
        return f"LayerNorm({self.num_features}, eps={self.eps})"


# ===========================================================================
# Pooling
# ===========================================================================

class MaxPool2D(Layer):
    """
    2-D Max Pooling layer.

    Reduces spatial dimensions by taking the maximum value inside each
    non-overlapping (stride == pool_size) pooling window.

    Input / output shapes follow the Conv2D convention: (N, H, W, C).
    The pool window is always square.  Only 'valid' padding is supported
    (no padding is applied before pooling).

    Backward pass uses a mask that routes gradients only to the positions
    that contained the maximum value during the forward pass.

    Args:
        pool_size: Side length of the square pooling window. Default 2.
        stride:    Step between consecutive windows. Default equals pool_size
                   (non-overlapping, standard max pooling).
    """

    def __init__(self, pool_size: int = 2, stride: Optional[int] = None):
        super().__init__()
        self.pool_size = pool_size
        self.stride    = stride if stride is not None else pool_size
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Args:
            x: (N, H, W, C)

        Returns:
            out: (N, out_H, out_W, C)
        """
        self.input = x
        N, H, W, C = x.shape
        p, s = self.pool_size, self.stride
        out_H = (H - p) // s + 1
        out_W = (W - p) // s + 1

        out  = np.zeros((N, out_H, out_W, C))
        # Store max-position mask for backward (same shape as input)
        self._mask = np.zeros_like(x, dtype=bool)

        for i in range(out_H):
            for j in range(out_W):
                rs, cs = i * s, j * s
                patch = x[:, rs:rs + p, cs:cs + p, :]        # (N, p, p, C)
                max_vals = np.max(patch, axis=(1, 2), keepdims=True)  # (N,1,1,C)
                out[:, i, j, :] = max_vals[:, 0, 0, :]
                # Mark the first occurrence of the max in the patch
                eq = (patch == max_vals)
                # Only credit the first max per window to avoid gradient splitting
                # when multiple elements share the maximum value.
                first = np.zeros_like(eq, dtype=bool)
                for n in range(N):
                    for c in range(C):
                        flat_idx = np.argmax(patch[n, :, :, c])
                        r_idx, c_idx = divmod(flat_idx, p)
                        first[n, r_idx, c_idx, c] = True
                self._mask[:, rs:rs + p, cs:cs + p, :] |= first

        self.output = out
        return out

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        """
        Args:
            doutput: (N, out_H, out_W, C)

        Returns:
            dinput: (N, H, W, C)
        """
        N, H, W, C = self.input.shape
        p, s = self.pool_size, self.stride
        out_H = (H - p) // s + 1
        out_W = (W - p) // s + 1
        dinput = np.zeros_like(self.input)

        for i in range(out_H):
            for j in range(out_W):
                rs, cs = i * s, j * s
                # doutput[:, i, j, :] has shape (N, C); broadcast over patch
                d = doutput[:, i, j, :][:, np.newaxis, np.newaxis, :]  # (N,1,1,C)
                dinput[:, rs:rs + p, cs:cs + p, :] += (
                    self._mask[:, rs:rs + p, cs:cs + p, :] * d
                )
        return dinput

    def output_shape(self, input_shape: tuple) -> tuple:
        H, W, C = input_shape
        out_H = (H - self.pool_size) // self.stride + 1
        out_W = (W - self.pool_size) // self.stride + 1
        return (out_H, out_W, C)

    def param_count(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"MaxPool2D(pool_size={self.pool_size}, stride={self.stride})"


class AvgPool2D(Layer):
    """
    2-D Average Pooling layer.

    Reduces spatial dimensions by computing the mean value inside each
    non-overlapping pooling window.  Simpler than MaxPool2D: the backward
    pass distributes the gradient uniformly across all positions in the window.

    Args:
        pool_size: Side length of the square pooling window. Default 2.
        stride:    Step between consecutive windows. Default equals pool_size.
    """

    def __init__(self, pool_size: int = 2, stride: Optional[int] = None):
        super().__init__()
        self.pool_size = pool_size
        self.stride    = stride if stride is not None else pool_size

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = x
        N, H, W, C = x.shape
        p, s = self.pool_size, self.stride
        out_H = (H - p) // s + 1
        out_W = (W - p) // s + 1
        out = np.zeros((N, out_H, out_W, C))

        for i in range(out_H):
            for j in range(out_W):
                rs, cs = i * s, j * s
                out[:, i, j, :] = np.mean(
                    x[:, rs:rs + p, cs:cs + p, :], axis=(1, 2)
                )
        self.output = out
        return out

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        N, H, W, C = self.input.shape
        p, s = self.pool_size, self.stride
        out_H = (H - p) // s + 1
        out_W = (W - p) // s + 1
        dinput = np.zeros_like(self.input)

        for i in range(out_H):
            for j in range(out_W):
                rs, cs = i * s, j * s
                # Distribute gradient evenly over the pool window
                d = doutput[:, i, j, :][:, np.newaxis, np.newaxis, :] / (p * p)
                dinput[:, rs:rs + p, cs:cs + p, :] += d
        return dinput

    def output_shape(self, input_shape: tuple) -> tuple:
        H, W, C = input_shape
        out_H = (H - self.pool_size) // self.stride + 1
        out_W = (W - self.pool_size) // self.stride + 1
        return (out_H, out_W, C)

    def param_count(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"AvgPool2D(pool_size={self.pool_size}, stride={self.stride})"


# ===========================================================================
# Embedding
# ===========================================================================

class Embedding(Layer):
    """
    Learnable token embedding table.

    Maps integer token indices (shape (N, T)) to dense vectors
    (shape (N, T, embed_dim)) by performing a simple lookup in a weight
    matrix of shape (vocab_size, embed_dim).

    This is the standard first layer for NLP models. During the backward
    pass the embedding rows are updated with the gradients that flow back
    from subsequent layers; each row is updated only for the token indices
    that appeared in the batch (sparse update).

    The embedding matrix is initialised with small random values from
    N(0, 0.01).

    Args:
        vocab_size: Total number of distinct tokens (size of the vocabulary).
        embed_dim:  Dimensionality of each token embedding vector.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.trainable  = True

        # Small random initialisation
        self.weights = np.random.randn(vocab_size, embed_dim) * 0.01
        self.biases  = np.zeros((1, embed_dim))  # unused; present for optimizer API

        self.dweights: Optional[np.ndarray] = None
        self.dbiases:  Optional[np.ndarray] = None

        self._indices: Optional[np.ndarray] = None  # saved for backward

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Args:
            x: Integer index array of shape (N, T) or (N,).

        Returns:
            Embedding vectors of shape (N, T, embed_dim) or (N, embed_dim).
        """
        self.input    = x
        self._indices = x.astype(int)
        self.output   = self.weights[self._indices]
        return self.output

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        """
        Args:
            doutput: Gradient of the same shape as the forward output.

        Returns:
            Zero gradient of the same shape as the input (indices are not
            differentiable; gradients flow only into the embedding table).
        """
        # Accumulate embedding gradients (sparse: only touched rows get updates)
        self.dweights = np.zeros_like(self.weights)
        np.add.at(self.dweights, self._indices, doutput)
        self.dbiases = np.zeros_like(self.biases)  # never used but keeps API consistent
        return np.zeros_like(self.input, dtype=float)

    def get_parameters(self) -> dict:
        return {
            'weights':    self.weights.tolist(),
            'vocab_size': self.vocab_size,
            'embed_dim':  self.embed_dim,
        }

    def set_parameters(self, parameters: dict) -> None:
        self.weights = np.array(parameters['weights'])

    def output_shape(self, input_shape: tuple) -> tuple:
        # input_shape = (T,) for a sequence of length T
        return input_shape + (self.embed_dim,)

    def param_count(self) -> int:
        return self.weights.size

    def __repr__(self) -> str:
        return f"Embedding(vocab={self.vocab_size}, dim={self.embed_dim})"


# ===========================================================================
# Recurrent layers
# ===========================================================================

class SimpleRNN(Layer):
    """
    Elman Simple Recurrent Network layer.

    Processes a sequence of shape (N, T, input_size) step-by-step, maintaining
    a hidden state h_t that is passed from one timestep to the next:

        h_t = activation(x_t @ W_x + h_{t-1} @ W_h + b)

    The hidden state at every timestep is returned by default
    (return_sequences=True), giving output shape (N, T, hidden_size).
    With return_sequences=False only h_T (the last timestep) is returned,
    shape (N, hidden_size).

    Backpropagation Through Time (BPTT) is used for the backward pass.
    Gradients are accumulated over all timesteps and clipping is recommended
    for long sequences to avoid explosion.

    Weight initialisation uses orthogonal initialisation for the recurrent
    matrix W_h (helps preserve gradient norms) and He initialisation for W_x.

    Args:
        input_size:       Number of input features per timestep.
        hidden_size:      Dimensionality of the hidden state.
        activation:       Activation applied to the hidden state. Default 'tanh'.
        return_sequences: If True, return outputs for every timestep.
                          If False, return only the last hidden state. Default True.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 activation: str = 'tanh',
                 return_sequences: bool = True):
        super().__init__()
        self.input_size       = input_size
        self.hidden_size      = hidden_size
        self.return_sequences = return_sequences
        self.trainable        = True

        # Input-to-hidden weights and hidden-to-hidden (recurrent) weights
        self.W_x = np.random.randn(input_size,   hidden_size) * np.sqrt(2.0 / input_size)
        self.W_h = _orthogonal_init(hidden_size, hidden_size)
        self.b   = np.zeros((1, hidden_size))

        # Fused dweights / dbiases for the optimizer (W_x stacked with W_h)
        self.dweights: Optional[np.ndarray] = None  # dW_x
        self.dbiases:  Optional[np.ndarray] = None  # db
        self.dW_h:     Optional[np.ndarray] = None  # separate because shape differs

        # Expose W_x through the standard interface; W_h updated manually
        self._weights_is_Wx = True

        # Resolve activation
        act_key = activation.lower()
        if act_key not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'.")
        self._act_proto = act_key
        # One activation instance per timestep, allocated in forward
        self._acts: list = []

        self._h:      Optional[np.ndarray] = None   # (N, T+1, hidden_size)
        self._inputs: Optional[np.ndarray] = None   # cached input tensor

    @property
    def weights(self) -> np.ndarray:
        return self.W_x

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        self.W_x = value

    @property
    def biases(self) -> np.ndarray:
        return self.b

    @biases.setter
    def biases(self, value: np.ndarray) -> None:
        self.b = value

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Args:
            x: (N, T, input_size)

        Returns:
            (N, T, hidden_size) if return_sequences else (N, hidden_size)
        """
        self.input = x
        N, T, _ = x.shape
        self._inputs = x

        # h[0] is the initial (zero) hidden state
        h = np.zeros((N, T + 1, self.hidden_size))
        outputs = np.zeros((N, T, self.hidden_size))

        # Allocate fresh activation instances for each timestep
        self._acts = [ACTIVATIONS[self._act_proto]() for _ in range(T)]

        for t in range(T):
            z = x[:, t, :] @ self.W_x + h[:, t, :] @ self.W_h + self.b
            h[:, t + 1, :] = self._acts[t].forward(z)
            outputs[:, t, :] = h[:, t + 1, :]

        self._h = h
        self.output = outputs if self.return_sequences else outputs[:, -1, :]
        return self.output

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        """
        BPTT backward pass.

        Args:
            doutput: gradient of shape (N, T, hidden_size) if return_sequences
                     else (N, hidden_size).

        Returns:
            dinput: (N, T, input_size)
        """
        N, T, _ = self._inputs.shape

        if not self.return_sequences:
            # Expand: only the last timestep received a gradient
            dout_full = np.zeros((N, T, self.hidden_size))
            dout_full[:, -1, :] = doutput
        else:
            dout_full = doutput

        dW_x = np.zeros_like(self.W_x)
        dW_h = np.zeros_like(self.W_h)
        db   = np.zeros_like(self.b)
        dx   = np.zeros_like(self._inputs)
        dh_next = np.zeros((N, self.hidden_size))

        for t in reversed(range(T)):
            dh = dout_full[:, t, :] + dh_next
            # Backprop through activation
            dz = self._acts[t].backward(dh)

            dW_x += self._inputs[:, t, :].T @ dz
            dW_h += self._h[:, t, :].T @ dz
            db   += np.sum(dz, axis=0, keepdims=True)
            dx[:, t, :] = dz @ self.W_x.T
            dh_next = dz @ self.W_h.T

        self.dweights = dW_x
        self.dW_h     = dW_h
        self.dbiases  = db
        return dx

    def _extra_update(self, optimizer) -> None:
        """Update W_h via the low-level update_raw interface."""
        if self.dW_h is not None:
            self.W_h = optimizer.update_raw(f'{id(self)}_W_h', self.W_h, self.dW_h)

    def get_parameters(self) -> dict:
        return {
            'W_x':              self.W_x.tolist(),
            'W_h':              self.W_h.tolist(),
            'b':                self.b.tolist(),
            'input_size':       self.input_size,
            'hidden_size':      self.hidden_size,
            'activation':       self._act_proto,
            'return_sequences': self.return_sequences,
        }

    def set_parameters(self, parameters: dict) -> None:
        self.W_x = np.array(parameters['W_x'])
        self.W_h = np.array(parameters['W_h'])
        self.b   = np.array(parameters['b'])

    def output_shape(self, input_shape: tuple) -> tuple:
        T = input_shape[0]
        if self.return_sequences:
            return (T, self.hidden_size)
        return (self.hidden_size,)

    def param_count(self) -> int:
        return self.W_x.size + self.W_h.size + self.b.size

    def __repr__(self) -> str:
        return (f"SimpleRNN({self.input_size} -> {self.hidden_size}, "
                f"act='{self._act_proto}', return_seq={self.return_sequences})")


class GRU(Layer):
    """
    Gated Recurrent Unit (Cho et al., 2014).

    An improved recurrent layer that uses reset and update gates to control
    information flow across timesteps, largely solving the vanishing gradient
    problem that plagues SimpleRNN on long sequences.  GRU is lighter than
    LSTM (fewer parameters, no cell state) while achieving comparable quality
    on most tasks.

    Equations (all operations element-wise unless stated):
        z_t = sigmoid(x_t @ W_xz + h_{t-1} @ W_hz + b_z)    # update gate
        r_t = sigmoid(x_t @ W_xr + h_{t-1} @ W_hr + b_r)    # reset gate
        n_t = tanh(x_t @ W_xn + (r_t * h_{t-1}) @ W_hn + b_n)  # candidate
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}

    Args:
        input_size:       Number of features per input timestep.
        hidden_size:      Dimensionality of the hidden state.
        return_sequences: Return output at every timestep (True) or only the
                          last timestep (False). Default True.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 return_sequences: bool = True):
        super().__init__()
        self.input_size       = input_size
        self.hidden_size      = hidden_size
        self.return_sequences = return_sequences
        self.trainable        = True

        H, I = hidden_size, input_size

        # --- Update gate weights ---
        self.W_xz = np.random.randn(I, H) * np.sqrt(2.0 / I)
        self.W_hz = _orthogonal_init(H, H)
        self.b_z  = np.zeros((1, H))

        # --- Reset gate weights ---
        self.W_xr = np.random.randn(I, H) * np.sqrt(2.0 / I)
        self.W_hr = _orthogonal_init(H, H)
        self.b_r  = np.zeros((1, H))

        # --- Candidate hidden state weights ---
        self.W_xn = np.random.randn(I, H) * np.sqrt(2.0 / I)
        self.W_hn = _orthogonal_init(H, H)
        self.b_n  = np.zeros((1, H))

        # All gradients stored here; the optimizer is called once per gate
        # group from NeuralNetwork._rnn_extra_update.
        self.dweights: Optional[np.ndarray] = None   # dW_xz (exposed to optimizer)
        self.dbiases:  Optional[np.ndarray] = None   # db_z

        self._all_grads: dict = {}
        self._cache:     dict = {}

    # Standard interface routes to the update-gate input weights; all other
    # weight matrices are updated via _extra_update.
    @property
    def weights(self) -> np.ndarray:
        return self.W_xz

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        self.W_xz = value

    @property
    def biases(self) -> np.ndarray:
        return self.b_z

    @biases.setter
    def biases(self, value: np.ndarray) -> None:
        self.b_z = value

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        N, T, _ = x.shape
        H = self.hidden_size
        self.input = x

        _sig = lambda a: 1.0 / (1.0 + np.exp(-np.clip(a, -30, 30)))

        h = np.zeros((N, T + 1, H))
        cache_z = np.zeros((N, T, H))
        cache_r = np.zeros((N, T, H))
        cache_n = np.zeros((N, T, H))

        for t in range(T):
            ht = h[:, t, :]
            z = _sig(x[:, t, :] @ self.W_xz + ht @ self.W_hz + self.b_z)
            r = _sig(x[:, t, :] @ self.W_xr + ht @ self.W_hr + self.b_r)
            n = np.tanh(x[:, t, :] @ self.W_xn + (r * ht) @ self.W_hn + self.b_n)
            h[:, t + 1, :] = (1.0 - z) * n + z * ht
            cache_z[:, t, :] = z
            cache_r[:, t, :] = r
            cache_n[:, t, :] = n

        self._cache = {'x': x, 'h': h, 'z': cache_z, 'r': cache_r, 'n': cache_n}
        outputs = h[:, 1:, :]  # (N, T, H)
        self.output = outputs if self.return_sequences else outputs[:, -1, :]
        return self.output

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        x = self._cache['x']
        h = self._cache['h']
        z = self._cache['z']
        r = self._cache['r']
        n = self._cache['n']
        N, T, _ = x.shape
        H = self.hidden_size

        if not self.return_sequences:
            dout_full = np.zeros((N, T, H))
            dout_full[:, -1, :] = doutput
        else:
            dout_full = doutput

        # Gradient accumulators
        dW_xz = np.zeros_like(self.W_xz); dW_hz = np.zeros_like(self.W_hz); db_z = np.zeros((1, H))
        dW_xr = np.zeros_like(self.W_xr); dW_hr = np.zeros_like(self.W_hr); db_r = np.zeros((1, H))
        dW_xn = np.zeros_like(self.W_xn); dW_hn = np.zeros_like(self.W_hn); db_n = np.zeros((1, H))
        dx   = np.zeros_like(x)
        dh_t = np.zeros((N, H))

        for t in reversed(range(T)):
            xt = x[:, t, :]
            ht = h[:, t, :]     # h_{t-1}
            zt, rt, nt = z[:, t, :], r[:, t, :], n[:, t, :]

            dh = dout_full[:, t, :] + dh_t

            # Gradient through h_t = (1-z)*n + z*h_{t-1}
            dz = dh * (ht - nt)   # (N, H)
            dn = dh * (1.0 - zt)
            dh_prev_from_h = dh * zt

            # Through n = tanh(...)
            dn_pre = dn * (1.0 - nt ** 2)
            dW_xn += xt.T @ dn_pre
            drht   = dn_pre @ self.W_hn.T       # gradient into (r * h_{t-1})
            dW_hn += (rt * ht).T @ dn_pre
            db_n  += np.sum(dn_pre, axis=0, keepdims=True)
            dx[:, t, :] += dn_pre @ self.W_xn.T
            dr = drht * ht
            dh_prev_from_n = drht * rt

            # Through z = sigmoid(...)
            dz_pre = dz * zt * (1.0 - zt)
            dW_xz += xt.T @ dz_pre
            dW_hz += ht.T @ dz_pre
            db_z  += np.sum(dz_pre, axis=0, keepdims=True)
            dx[:, t, :] += dz_pre @ self.W_xz.T
            dh_prev_from_z = dz_pre @ self.W_hz.T

            # Through r = sigmoid(...)
            dr_pre = dr * rt * (1.0 - rt)
            dW_xr += xt.T @ dr_pre
            dW_hr += ht.T @ dr_pre
            db_r  += np.sum(dr_pre, axis=0, keepdims=True)
            dx[:, t, :] += dr_pre @ self.W_xr.T
            dh_prev_from_r = dr_pre @ self.W_hr.T

            dh_t = dh_prev_from_h + dh_prev_from_n + dh_prev_from_z + dh_prev_from_r

        self._all_grads = dict(
            dW_xz=dW_xz, dW_hz=dW_hz, db_z=db_z,
            dW_xr=dW_xr, dW_hr=dW_hr, db_r=db_r,
            dW_xn=dW_xn, dW_hn=dW_hn, db_n=db_n,
        )
        # Expose update-gate grads via standard interface
        self.dweights = dW_xz
        self.dbiases  = db_z
        return dx

    def _extra_update(self, optimizer) -> None:
        """Update all GRU weight matrices via the low-level update_raw interface."""
        g = self._all_grads
        pairs = [
            ('W_hz', 'dW_hz'), ('W_xr', 'dW_xr'), ('W_hr', 'dW_hr'),
            ('W_xn', 'dW_xn'), ('W_hn', 'dW_hn'),
            ('b_r',  'db_r'),  ('b_n',  'db_n'),
        ]
        lid = id(self)
        for w_name, dw_name in pairs:
            W  = getattr(self, w_name)
            dW = g[dw_name]
            setattr(self, w_name, optimizer.update_raw(f'{lid}_{w_name}', W, dW))

    def get_parameters(self) -> dict:
        return {
            'W_xz': self.W_xz.tolist(), 'W_hz': self.W_hz.tolist(), 'b_z': self.b_z.tolist(),
            'W_xr': self.W_xr.tolist(), 'W_hr': self.W_hr.tolist(), 'b_r': self.b_r.tolist(),
            'W_xn': self.W_xn.tolist(), 'W_hn': self.W_hn.tolist(), 'b_n': self.b_n.tolist(),
            'input_size':       self.input_size,
            'hidden_size':      self.hidden_size,
            'return_sequences': self.return_sequences,
        }

    def set_parameters(self, parameters: dict) -> None:
        for name in ('W_xz','W_hz','b_z','W_xr','W_hr','b_r','W_xn','W_hn','b_n'):
            setattr(self, name, np.array(parameters[name]))

    def output_shape(self, input_shape: tuple) -> tuple:
        T = input_shape[0]
        return (T, self.hidden_size) if self.return_sequences else (self.hidden_size,)

    def param_count(self) -> int:
        return (self.W_xz.size + self.W_hz.size + self.b_z.size +
                self.W_xr.size + self.W_hr.size + self.b_r.size +
                self.W_xn.size + self.W_hn.size + self.b_n.size)

    def __repr__(self) -> str:
        return (f"GRU({self.input_size} -> {self.hidden_size}, "
                f"return_seq={self.return_sequences})")


class LSTM(Layer):
    """
    Long Short-Term Memory layer (Hochreiter & Schmidhuber, 1997).

    LSTM extends the RNN with a dedicated cell state c_t and three
    multiplicative gates (forget, input, output) that give the network
    explicit read/write/erase control over memory across long sequences.
    This largely solves the vanishing gradient problem.

    Gate equations (all element-wise; @ denotes matrix multiplication):
        f_t = sigmoid(x_t @ W_xf + h_{t-1} @ W_hf + b_f)   # forget gate
        i_t = sigmoid(x_t @ W_xi + h_{t-1} @ W_hi + b_i)   # input gate
        g_t = tanh(   x_t @ W_xg + h_{t-1} @ W_hg + b_g)   # cell gate (candidate)
        o_t = sigmoid(x_t @ W_xo + h_{t-1} @ W_ho + b_o)   # output gate
        c_t = f_t * c_{t-1} + i_t * g_t                     # cell state
        h_t = o_t * tanh(c_t)                                # hidden state

    The layer carries two state vectors between timesteps: h (hidden) and
    c (cell).  Both are initialised to zero for each new forward call.

    Backpropagation Through Time (BPTT) computes gradients by unrolling the
    computation graph over T timesteps.

    Args:
        input_size:       Number of features per input timestep.
        hidden_size:      Dimensionality of the hidden (and cell) state.
        return_sequences: Return output at every timestep if True, or only
                          h_T if False. Default True.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 return_sequences: bool = True):
        super().__init__()
        self.input_size       = input_size
        self.hidden_size      = hidden_size
        self.return_sequences = return_sequences
        self.trainable        = True

        H, I = hidden_size, input_size
        scale = np.sqrt(2.0 / I)

        # Forget gate
        self.W_xf = np.random.randn(I, H) * scale; self.W_hf = _orthogonal_init(H, H); self.b_f = np.ones((1, H))  # bias=1 encourages remembering at init
        # Input gate
        self.W_xi = np.random.randn(I, H) * scale; self.W_hi = _orthogonal_init(H, H); self.b_i = np.zeros((1, H))
        # Cell gate
        self.W_xg = np.random.randn(I, H) * scale; self.W_hg = _orthogonal_init(H, H); self.b_g = np.zeros((1, H))
        # Output gate
        self.W_xo = np.random.randn(I, H) * scale; self.W_ho = _orthogonal_init(H, H); self.b_o = np.zeros((1, H))

        self.dweights: Optional[np.ndarray] = None  # dW_xf (standard interface)
        self.dbiases:  Optional[np.ndarray] = None  # db_f
        self._all_grads: dict = {}
        self._cache:     dict = {}

    @property
    def weights(self) -> np.ndarray:
        return self.W_xf

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        self.W_xf = value

    @property
    def biases(self) -> np.ndarray:
        return self.b_f

    @biases.setter
    def biases(self, value: np.ndarray) -> None:
        self.b_f = value

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Args:
            x: (N, T, input_size)

        Returns:
            (N, T, hidden_size) if return_sequences else (N, hidden_size)
        """
        N, T, _ = x.shape
        H = self.hidden_size
        self.input = x

        _sig = lambda a: 1.0 / (1.0 + np.exp(-np.clip(a, -30, 30)))

        h = np.zeros((N, T + 1, H))
        c = np.zeros((N, T + 1, H))

        cf = np.zeros((N, T, H))   # forget gate outputs
        ci = np.zeros((N, T, H))   # input gate outputs
        cg = np.zeros((N, T, H))   # cell gate outputs
        co = np.zeros((N, T, H))   # output gate outputs
        ctanh = np.zeros((N, T, H))  # tanh(c_t)

        for t in range(T):
            ht = h[:, t, :]
            xt = x[:, t, :]

            f = _sig(xt @ self.W_xf + ht @ self.W_hf + self.b_f)
            i = _sig(xt @ self.W_xi + ht @ self.W_hi + self.b_i)
            g = np.tanh(xt @ self.W_xg + ht @ self.W_hg + self.b_g)
            o = _sig(xt @ self.W_xo + ht @ self.W_ho + self.b_o)
            c_new = f * c[:, t, :] + i * g
            tc    = np.tanh(c_new)
            h[:, t + 1, :] = o * tc

            cf[:, t, :] = f; ci[:, t, :] = i
            cg[:, t, :] = g; co[:, t, :] = o; ctanh[:, t, :] = tc

        self._cache = {'x': x, 'h': h, 'c': c, 'f': cf, 'i': ci,
                       'g': cg, 'o': co, 'tanh_c': ctanh}
        outputs = h[:, 1:, :]
        self.output = outputs if self.return_sequences else outputs[:, -1, :]
        return self.output

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        x     = self._cache['x']
        h     = self._cache['h']
        c     = self._cache['c']
        f_all = self._cache['f']
        i_all = self._cache['i']
        g_all = self._cache['g']
        o_all = self._cache['o']
        tc_all= self._cache['tanh_c']
        N, T, _ = x.shape
        H = self.hidden_size

        if not self.return_sequences:
            dout_full = np.zeros((N, T, H))
            dout_full[:, -1, :] = doutput
        else:
            dout_full = doutput

        # Accumulators
        dW_xf=np.zeros_like(self.W_xf); dW_hf=np.zeros_like(self.W_hf); db_f=np.zeros((1,H))
        dW_xi=np.zeros_like(self.W_xi); dW_hi=np.zeros_like(self.W_hi); db_i=np.zeros((1,H))
        dW_xg=np.zeros_like(self.W_xg); dW_hg=np.zeros_like(self.W_hg); db_g=np.zeros((1,H))
        dW_xo=np.zeros_like(self.W_xo); dW_ho=np.zeros_like(self.W_ho); db_o=np.zeros((1,H))
        dx = np.zeros_like(x)
        dh_next = np.zeros((N, H))
        dc_next = np.zeros((N, H))

        for t in reversed(range(T)):
            xt = x[:, t, :]
            ht = h[:, t, :]       # h_{t-1}
            ct = c[:, t, :]       # c_{t-1}
            f, i, g, o, tc = f_all[:,t,:], i_all[:,t,:], g_all[:,t,:], o_all[:,t,:], tc_all[:,t,:]
            c_cur = f * ct + i * g   # c_t

            dh = dout_full[:, t, :] + dh_next

            # Gradient through h_t = o * tanh(c_t)
            do = dh * tc
            dc = dh * o * (1.0 - tc ** 2) + dc_next

            # Gradient through c_t = f*c_{t-1} + i*g
            df = dc * ct
            di = dc * g
            dg = dc * i
            dc_prev = dc * f

            # Gate pre-activation gradients
            do_pre = do * o * (1.0 - o)
            df_pre = df * f * (1.0 - f)
            di_pre = di * i * (1.0 - i)
            dg_pre = dg * (1.0 - g ** 2)

            # Accumulate weight gradients
            dW_xf += xt.T @ df_pre; dW_hf += ht.T @ df_pre; db_f += np.sum(df_pre,axis=0,keepdims=True)
            dW_xi += xt.T @ di_pre; dW_hi += ht.T @ di_pre; db_i += np.sum(di_pre,axis=0,keepdims=True)
            dW_xg += xt.T @ dg_pre; dW_hg += ht.T @ dg_pre; db_g += np.sum(dg_pre,axis=0,keepdims=True)
            dW_xo += xt.T @ do_pre; dW_ho += ht.T @ do_pre; db_o += np.sum(do_pre,axis=0,keepdims=True)

            dx[:, t, :] = (df_pre @ self.W_xf.T + di_pre @ self.W_xi.T +
                           dg_pre @ self.W_xg.T + do_pre @ self.W_xo.T)
            dh_next = (df_pre @ self.W_hf.T + di_pre @ self.W_hi.T +
                       dg_pre @ self.W_hg.T + do_pre @ self.W_ho.T)
            dc_next = dc_prev

        self._all_grads = dict(
            dW_xf=dW_xf,dW_hf=dW_hf,db_f=db_f,
            dW_xi=dW_xi,dW_hi=dW_hi,db_i=db_i,
            dW_xg=dW_xg,dW_hg=dW_hg,db_g=db_g,
            dW_xo=dW_xo,dW_ho=dW_ho,db_o=db_o,
        )
        self.dweights = dW_xf
        self.dbiases  = db_f
        return dx

    def _extra_update(self, optimizer) -> None:
        """Update all LSTM weight matrices via the low-level update_raw interface."""
        g = self._all_grads
        pairs = [
            ('W_hf','dW_hf'), ('W_xi','dW_xi'), ('W_hi','dW_hi'),
            ('W_xg','dW_xg'), ('W_hg','dW_hg'),
            ('W_xo','dW_xo'), ('W_ho','dW_ho'),
            ('b_i', 'db_i'),  ('b_g', 'db_g'),  ('b_o', 'db_o'),
        ]
        lid = id(self)
        for w_name, dw_name in pairs:
            W  = getattr(self, w_name)
            dW = g[dw_name]
            setattr(self, w_name, optimizer.update_raw(f'{lid}_{w_name}', W, dW))

    def get_parameters(self) -> dict:
        params = {}
        for name in ('W_xf','W_hf','b_f','W_xi','W_hi','b_i',
                     'W_xg','W_hg','b_g','W_xo','W_ho','b_o'):
            params[name] = getattr(self, name).tolist()
        params.update({'input_size': self.input_size,
                       'hidden_size': self.hidden_size,
                       'return_sequences': self.return_sequences})
        return params

    def set_parameters(self, parameters: dict) -> None:
        for name in ('W_xf','W_hf','b_f','W_xi','W_hi','b_i',
                     'W_xg','W_hg','b_g','W_xo','W_ho','b_o'):
            setattr(self, name, np.array(parameters[name]))

    def output_shape(self, input_shape: tuple) -> tuple:
        T = input_shape[0]
        return (T, self.hidden_size) if self.return_sequences else (self.hidden_size,)

    def param_count(self) -> int:
        total = 0
        for name in ('W_xf','W_hf','b_f','W_xi','W_hi','b_i',
                     'W_xg','W_hg','b_g','W_xo','W_ho','b_o'):
            total += getattr(self, name).size
        return total

    def __repr__(self) -> str:
        return (f"LSTM({self.input_size} -> {self.hidden_size}, "
                f"return_seq={self.return_sequences})")


class MultiHeadAttention(Layer):
    """
    Multi-Head Scaled Dot-Product Attention (Vaswani et al., 2017).

    Allows the model to attend to information from different representation
    sub-spaces at different positions simultaneously.  The input is projected
    into `num_heads` query, key and value subspaces, attention is computed
    independently in each head, and the results are concatenated and projected
    back to the original dimension.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    The layer operates on sequences: input shape is (N, T, d_model).
    Self-attention is used (Q = K = V = input); cross-attention can be
    implemented by passing key/value sequences directly (not yet exposed
    in this interface, but straightforward to add).

    For simplicity this implementation uses separate W_q, W_k, W_v projection
    matrices per head and a single output projection W_o.

    Args:
        d_model:   Dimensionality of the input (and output) sequence.
        num_heads: Number of attention heads.  d_model must be divisible by num_heads.
        dropout:   Attention weight dropout rate. Default 0.0 (no dropout).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # dimension per head
        self.dropout   = dropout
        self.trainable = True

        # Per-head projection matrices: shape (num_heads, d_model, d_k)
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(num_heads, d_model, self.d_k) * scale
        self.W_k = np.random.randn(num_heads, d_model, self.d_k) * scale
        self.W_v = np.random.randn(num_heads, d_model, self.d_k) * scale
        # Output projection: (d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.b_o = np.zeros((1, d_model))

        # Standard optimizer interface routes to W_o
        self.dweights: Optional[np.ndarray] = None
        self.dbiases:  Optional[np.ndarray] = None
        self._all_grads: dict = {}
        self._cache:     dict = {}

    @property
    def weights(self) -> np.ndarray:
        return self.W_o

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        self.W_o = value

    @property
    def biases(self) -> np.ndarray:
        return self.b_o

    @biases.setter
    def biases(self, value: np.ndarray) -> None:
        self.b_o = value

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Args:
            x: (N, T, d_model)  —  query, key and value are all x (self-attention)

        Returns:
            out: (N, T, d_model)
        """
        N, T, _ = x.shape
        H, d_k  = self.num_heads, self.d_k
        self.input = x

        # Q, K, V per head: (num_heads, N, T, d_k)
        Q = np.einsum('ntd,hdk->hntk', x, self.W_q)
        K = np.einsum('ntd,hdk->hntk', x, self.W_k)
        V = np.einsum('ntd,hdk->hntk', x, self.W_v)

        # Scaled dot-product attention scores: (num_heads, N, T, T)
        scores = np.einsum('hntk,hnsk->hnts', Q, K) / np.sqrt(d_k)
        attn   = self._softmax(scores)

        # Optional attention dropout
        self._attn_mask: Optional[np.ndarray] = None
        if training and self.dropout > 0.0:
            keep = 1.0 - self.dropout
            self._attn_mask = (np.random.rand(*attn.shape) < keep) / keep
            attn = attn * self._attn_mask

        # Weighted sum of values: (num_heads, N, T, d_k)
        context = np.einsum('hnts,hnsk->hntk', attn, V)

        # Concatenate heads: (N, T, d_model)
        concat = context.transpose(1, 2, 0, 3).reshape(N, T, H * d_k)

        # Output projection
        out = concat @ self.W_o + self.b_o
        self._cache = {'x': x, 'Q': Q, 'K': K, 'V': V,
                       'scores': scores, 'attn': attn, 'context': context,
                       'concat': concat}
        self.output = out
        return out

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        """
        Args:
            doutput: (N, T, d_model)

        Returns:
            dinput: (N, T, d_model)
        """
        c = self._cache
        x, Q, K, V = c['x'], c['Q'], c['K'], c['V']
        attn, context, concat = c['attn'], c['context'], c['concat']
        N, T, _ = x.shape
        H, d_k  = self.num_heads, self.d_k

        # Gradient of output projection
        dW_o  = concat.reshape(N * T, H * d_k).T @ doutput.reshape(N * T, self.d_model)
        db_o  = np.sum(doutput, axis=(0, 1), keepdims=True).reshape(1, self.d_model)
        dconcat = doutput @ self.W_o.T   # (N, T, d_model)

        # Reshape back to (H, N, T, d_k)
        dcontext = dconcat.reshape(N, T, H, d_k).transpose(2, 0, 1, 3)

        # Gradients through attention: context = attn @ V
        dV    = np.einsum('hnts,hntk->hnsk', attn, dcontext)
        dattn = np.einsum('hntk,hnsk->hnts', dcontext, V)

        if self._attn_mask is not None:
            dattn = dattn * self._attn_mask

        # Softmax backward: d_scores = attn * (dattn - sum(dattn * attn, keepdims))
        dattn_sum = np.sum(dattn * attn, axis=-1, keepdims=True)
        dscores   = attn * (dattn - dattn_sum) / np.sqrt(d_k)

        # Gradients of Q and K
        dQ = np.einsum('hnts,hnsk->hntk', dscores, K)
        dK = np.einsum('hnts,hntk->hnsk', dscores, Q)

        # Gradients of projection matrices
        dW_q = np.einsum('ntd,hntk->hdk', x, dQ)
        dW_k = np.einsum('ntd,hntk->hdk', x, dK)
        dW_v = np.einsum('ntd,hntk->hdk', x, dV)

        # Gradient of input x
        dx  = (np.einsum('hntk,hdk->ntd', dQ, self.W_q) +
               np.einsum('hntk,hdk->ntd', dK, self.W_k) +
               np.einsum('hntk,hdk->ntd', dV, self.W_v))

        self._all_grads = dict(dW_q=dW_q, dW_k=dW_k, dW_v=dW_v)
        self.dweights = dW_o
        self.dbiases  = db_o
        return dx

    def _extra_update(self, optimizer) -> None:
        """Update W_q, W_k, W_v via the low-level update_raw interface."""
        g = self._all_grads
        lid = id(self)
        for w_name, dw_name in (('W_q','dW_q'), ('W_k','dW_k'), ('W_v','dW_v')):
            W  = getattr(self, w_name)
            dW = g[dw_name]
            setattr(self, w_name, optimizer.update_raw(f'{lid}_{w_name}', W, dW))

    def get_parameters(self) -> dict:
        return {
            'W_q':       self.W_q.tolist(),
            'W_k':       self.W_k.tolist(),
            'W_v':       self.W_v.tolist(),
            'W_o':       self.W_o.tolist(),
            'b_o':       self.b_o.tolist(),
            'd_model':   self.d_model,
            'num_heads': self.num_heads,
            'dropout':   self.dropout,
        }

    def set_parameters(self, parameters: dict) -> None:
        for name in ('W_q', 'W_k', 'W_v', 'W_o', 'b_o'):
            setattr(self, name, np.array(parameters[name]))

    def output_shape(self, input_shape: tuple) -> tuple:
        return input_shape  # (T, d_model) -> (T, d_model)

    def param_count(self) -> int:
        return (self.W_q.size + self.W_k.size + self.W_v.size +
                self.W_o.size + self.b_o.size)

    def __repr__(self) -> str:
        return (f"MultiHeadAttention(d_model={self.d_model}, "
                f"heads={self.num_heads}, dropout={self.dropout})")


# ===========================================================================
# Helper functions (private to this module)
# ===========================================================================

def _orthogonal_init(rows: int, cols: int) -> np.ndarray:
    """
    Generate an orthogonal matrix via QR decomposition.

    Orthogonal initialisation for recurrent weight matrices preserves the
    norm of vectors during the forward pass, which helps gradients flow
    further back in time compared to random initialisation.

    Args:
        rows: Number of rows.
        cols: Number of columns.

    Returns:
        An (rows, cols) orthogonal (or semi-orthogonal) matrix.
    """
    random_matrix = np.random.randn(max(rows, cols), min(rows, cols))
    Q, _ = np.linalg.qr(random_matrix)
    return Q[:rows, :cols]
