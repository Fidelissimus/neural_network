import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple

from .layers import (Layer, Dense, Dropout, BatchNorm, Conv2D, Flatten,
                     LayerNorm, MaxPool2D, AvgPool2D,
                     Embedding, PositionalEncoding, SimpleRNN, GRU, LSTM, MultiHeadAttention)
from .activations import ACTIVATIONS
from .optimizers import OPTIMIZERS
from .losses import LOSSES
from .callbacks import Callback
from .utils import plot_training_history, save_history


# Layer type -> class mapping used during model loading
_LAYER_CLASSES = {
    'Dense':              Dense,
    'Dropout':            Dropout,
    'BatchNorm':          BatchNorm,
    'Conv2D':             Conv2D,
    'Flatten':            Flatten,
    'LayerNorm':          LayerNorm,
    'MaxPool2D':          MaxPool2D,
    'AvgPool2D':          AvgPool2D,
    'Embedding':          Embedding,
    'PositionalEncoding': PositionalEncoding,
    'SimpleRNN':          SimpleRNN,
    'GRU':                GRU,
    'LSTM':               LSTM,
    'MultiHeadAttention': MultiHeadAttention,
}


class NeuralNetwork:
    """
    Sequential neural network container.

    Layers are executed in the order they were added. Any combination of
    Dense, Dropout, BatchNorm, Conv2D and Flatten layers is supported.

    The training loop is decoupled from parameter updates: backward() stores
    gradients in each layer, and the chosen optimizer then reads and applies
    them. This makes it straightforward to swap optimizers without touching
    any layer code.

    Example::

        from nn import NeuralNetwork
        from nn.layers import Dense, Dropout, BatchNorm
        from nn.callbacks import EarlyStopping, LearningRateScheduler

        model = NeuralNetwork()
        model.add(Dense(128, 64, activation='relu'))
        model.add(BatchNorm(64))
        model.add(Dropout(0.3))
        model.add(Dense(64, 10, activation='softmax'))

        model.compile(loss='categoricalcrossentropy', optimizer='adam',
                      learning_rate=1e-3)

        history = model.train(
            x_train, y_train,
            epochs=200,
            batch_size=64,
            validation_data=(x_val, y_val),
            callbacks=[EarlyStopping(patience=15)],
        )
    """

    def __init__(self, layers: Optional[List[Layer]] = None):
        self.layers: List[Layer] = layers if layers is not None else []
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss':   [],
            'train_acc':  [],
            'val_acc':    [],
        }
        self.optimizer = None
        self.loss_fn   = None
        self._gradient_clip: Optional[float] = None
        self._weight_decay: float = 0.0

    # ------------------------------------------------------------------
    # Building the model
    # ------------------------------------------------------------------

    def add(self, layer: Layer) -> 'NeuralNetwork':
        """
        Append a layer to the end of the network.

        Returns self so calls can be chained:
            model.add(Dense(64, 32)).add(Dense(32, 10, 'softmax'))
        """
        self.layers.append(layer)
        return self

    def compile(self, loss: str = 'mse', optimizer: str = 'adam',
                learning_rate: float = 0.001,
                gradient_clip: Optional[float] = None,
                weight_decay: float = 0.0,
                **optimizer_kwargs) -> None:
        """
        Configure the model for training.

        Args:
            loss:           Name of the loss function. One of:
                            'mse', 'mae', 'huber',
                            'binarycrossentropy', 'categoricalcrossentropy'.
            optimizer:      Name of the optimizer. One of:
                            'sgd', 'adam', 'rmsprop', 'adagrad'.
            learning_rate:  Initial learning rate.
            gradient_clip:  If set, all gradients are clipped to the L2-norm
                            ball of this radius before the optimizer update.
                            Useful for preventing gradient explosions in deep
                            or recurrent networks. Default None (no clipping).
            weight_decay:   L2 regularization strength. If > 0, `weight_decay
                            * W` is added to the weight gradient of every
                            regularizable layer before the optimizer step
                            (i.e. classic L2 regularization, not decoupled
                            AdamW-style decay). Biases and normalization
                            scale parameters (BatchNorm/LayerNorm gamma) are
                            never decayed. Default 0.0 (disabled).
            **optimizer_kwargs: Extra keyword arguments forwarded to the
                            optimizer constructor (e.g. momentum for SGD,
                            beta1/beta2 for Adam).
        """
        loss_key = loss.lower().replace('_', '')
        if loss_key not in LOSSES:
            raise ValueError(
                f"Unknown loss '{loss}'. Available: {list(LOSSES.keys())}"
            )
        self.loss_fn = LOSSES[loss_key]()

        opt_key = optimizer.lower()
        if opt_key not in OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                f"Available: {list(OPTIMIZERS.keys())}"
            )
        self.optimizer = OPTIMIZERS[opt_key](learning_rate=learning_rate,
                                             **optimizer_kwargs)

        self._gradient_clip = gradient_clip
        self._weight_decay  = weight_decay

    def _apply_weight_decay(self) -> None:
        """
        Add `weight_decay * W` to every regularizable layer's weight
        gradient in-place, before clipping/the optimizer step. No-op when
        weight_decay is 0 (the default).
        """
        if not self._weight_decay:
            return
        for layer in self.layers:
            if (layer.trainable and getattr(layer, 'regularizable', True)
                    and layer.dweights is not None):
                layer.dweights = layer.dweights + self._weight_decay * layer.weights

    # ------------------------------------------------------------------
    # Forward / backward / update
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Run a forward pass through all layers.

        Layers that behave differently at train vs inference time (Dropout,
        BatchNorm, Conv2D) receive the `training` flag.

        Args:
            x:        Input array. Shape depends on the first layer:
                      (N, features) for Dense, (N, H, W, C) for Conv2D.
            training: True during training; False during inference.

        Returns:
            Network output of shape (N, output_size).
        """
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, output_gradient: np.ndarray) -> None:
        """
        Run a backward pass, storing gradients in each trainable layer.

        After this call every trainable layer's .dweights and .dbiases are
        populated; call update() to apply them.

        Args:
            output_gradient: dL/d(network_output), the gradient produced by
                             the loss function's backward method.
        """
        grad = output_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _clip_gradients(self) -> None:
        """
        Clip per-layer weight and bias gradients to the configured L2-norm.

        Only called when gradient_clip was set in compile().
        """
        for layer in self.layers:
            if not layer.trainable:
                continue
            if layer.dweights is not None:
                norm = np.linalg.norm(layer.dweights)
                if norm > self._gradient_clip:
                    layer.dweights *= self._gradient_clip / norm
            if layer.dbiases is not None:
                norm = np.linalg.norm(layer.dbiases)
                if norm > self._gradient_clip:
                    layer.dbiases *= self._gradient_clip / norm

    def update(self) -> None:
        """
        Apply one optimizer step to every trainable layer.

        For standard layers (Dense, BatchNorm, LayerNorm, Embedding) the
        optimizer is called once via the standard weights/biases interface.
        Recurrent and attention layers (SimpleRNN, GRU, LSTM,
        MultiHeadAttention) expose multiple weight matrices through
        _extra_update, which is called after the standard update.
        """
        for layer in self.layers:
            if layer.trainable and layer.dweights is not None:
                self.optimizer.update(layer)
                if hasattr(layer, '_extra_update'):
                    layer._extra_update(self.optimizer)

    # ------------------------------------------------------------------
    # Accuracy helper
    # ------------------------------------------------------------------

    def _is_regression(self, y: np.ndarray) -> bool:
        """
        True if the compiled loss and target shape indicate a regression
        task (single output column, regression loss function).
        """
        return (y.shape[1] == 1
                and self.loss_fn.__class__.__name__ in ('MSE', 'MAE', 'Huber'))

    @staticmethod
    def _batch_correct(output: np.ndarray, y_batch: np.ndarray) -> int:
        """
        Count the number of correct predictions in a single batch.

        For binary classification (output has 1 column) a threshold of 0.5
        is applied. For multi-class the argmax is used. For regression
        (single output column used for non-binary tasks) returns 0.
        """
        if y_batch.shape[1] == 1:   # binary classification
            predictions = (output > 0.5).astype(int)
            return int(np.sum(predictions == y_batch))
        elif y_batch.shape[1] > 1:  # multi-class classification
            return int(np.sum(np.argmax(output, axis=1)
                              == np.argmax(y_batch, axis=1)))
        return 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              epochs: int, batch_size: int = 32,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              verbose: bool = True,
              verbose_interval: int = 1,
              callbacks: Optional[List[Callback]] = None
              ) -> Dict[str, List[float]]:
        """
        Train the network using mini-batch gradient descent.

        Args:
            x_train:          Training inputs, shape (N, ...).
            y_train:          Training targets, shape (N, output_size).
            epochs:           Number of complete passes over the training data.
            batch_size:       Number of samples per gradient update.
            validation_data:  Optional (x_val, y_val) tuple. If supplied,
                              validation metrics are computed at each epoch.
            verbose:          Print training progress. Default True.
            verbose_interval: Print progress every this many epochs. Default 1.
            callbacks:        List of Callback instances called during training.

        Returns:
            Training history dict with keys 'train_loss', 'train_acc',
            'val_loss' (if validation_data provided), 'val_acc'.
        """
        if self.loss_fn is None or self.optimizer is None:
            raise RuntimeError(
                "Model has not been compiled. Call model.compile() first."
            )

        n_samples = x_train.shape[0]
        callbacks  = callbacks or []
        is_regression = self._is_regression(y_train)

        for cb in callbacks:
            cb.on_train_begin(self)

        for epoch in range(epochs):
            # Shuffle training data each epoch
            indices    = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss    = 0.0
            epoch_correct = 0

            for start in range(0, n_samples, batch_size):
                x_batch = x_shuffled[start: start + batch_size]
                y_batch = y_shuffled[start: start + batch_size]
                actual_batch = x_batch.shape[0]

                # Forward pass
                output = self.forward(x_batch, training=True)

                # Loss and accuracy
                epoch_loss    += self.loss_fn.forward(output, y_batch) * actual_batch
                epoch_correct += self._batch_correct(output, y_batch)

                # Backward pass and optimizer update
                error = self.loss_fn.backward(output, y_batch)
                self.backward(error)

                self._apply_weight_decay()

                if self._gradient_clip is not None:
                    self._clip_gradients()

                self.update()

            train_loss = epoch_loss / n_samples
            train_acc  = 0.0 if is_regression else epoch_correct / n_samples

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation
            val_loss, val_acc = 0.0, 0.0
            if validation_data is not None:
                val_loss, val_acc = self.evaluate(*validation_data)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

            # Build logs dict for callbacks
            logs = {
                'train_loss': train_loss,
                'train_acc':  train_acc,
            }
            if validation_data is not None:
                logs['val_loss'] = val_loss
                logs['val_acc']  = val_acc

            # Fire epoch-end callbacks
            stop = False
            for cb in callbacks:
                cb.on_epoch_end(self, epoch, logs)
                if getattr(cb, 'stop_training', False):
                    stop = True

            # Verbose output
            if verbose and (epoch % verbose_interval == 0 or epoch == epochs - 1):
                msg = (f"Epoch {epoch + 1:>4}/{epochs}  "
                       f"loss={train_loss:.6f}  acc={train_acc:.4f}")
                if validation_data is not None:
                    msg += f"  val_loss={val_loss:.6f}  val_acc={val_acc:.4f}"
                print(msg)

            if stop:
                break

        for cb in callbacks:
            cb.on_train_end(self)

        return self.history

    # ------------------------------------------------------------------
    # Evaluation and prediction
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray,
                 y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the model on the given data without updating parameters.

        Args:
            x: Input data.
            y: Ground-truth targets.

        Returns:
            (loss, accuracy) — accuracy is 0.0 for regression tasks.
        """
        output = self.forward(x, training=False)
        loss   = self.loss_fn.forward(output, y)

        if self._is_regression(y):
            # A single-column regression target should never be scored as a
            # 0.5-threshold binary accuracy; report 0.0 like train() does.
            acc = 0.0
        elif y.shape[1] == 1:
            predictions = (output > 0.5).astype(int)
            acc = float(np.mean(predictions == y))
        elif y.shape[1] > 1:
            acc = float(np.mean(np.argmax(output, axis=1)
                                == np.argmax(y, axis=1)))
        else:
            acc = 0.0

        return float(loss), acc

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input data.

        Args:
            x: Input array, shape (N, ...).

        Returns:
            Network output array, shape (N, output_size).
        """
        return self.forward(x, training=False)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, file_path: str) -> None:
        """
        Serialise the full model to a JSON file.

        Saved information includes layer types and parameters, optimizer
        class and learning rate, loss function, and training history.

        Args:
            file_path: Destination path (e.g. 'model.json').
        """
        if self.loss_fn is None or self.optimizer is None:
            raise RuntimeError(
                "Model has not been compiled and cannot be saved. "
                "Call model.compile() first."
            )

        data = {
            'config': {
                'loss':          self.loss_fn.__class__.__name__.lower(),
                'optimizer':     self.optimizer.__class__.__name__.lower(),
                'learning_rate': float(self.optimizer.learning_rate),
            },
            'history': {k: [float(v) for v in vals]
                        for k, vals in self.history.items()},
            'layers': [
                {
                    'type':       layer.__class__.__name__,
                    'parameters': layer.get_parameters(),
                }
                for layer in self.layers
            ],
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, file_path: str) -> 'NeuralNetwork':
        """
        Load a model previously saved with model.save().

        Args:
            file_path: Path to the JSON file produced by save().

        Returns:
            A fully configured NeuralNetwork instance ready for inference or
            continued training.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        network = cls()

        # Restore loss, optimizer, history
        cfg = data['config']
        network.loss_fn   = LOSSES[cfg['loss']]()
        network.optimizer = OPTIMIZERS[cfg['optimizer']](
            learning_rate=cfg['learning_rate']
        )
        network.history = data['history']

        # Recreate layers
        for layer_data in data['layers']:
            layer_type = layer_data['type']
            params     = layer_data['parameters']

            if layer_type not in _LAYER_CLASSES:
                raise ValueError(
                    f"Cannot load unknown layer type '{layer_type}'. "
                    f"Known types: {list(_LAYER_CLASSES.keys())}"
                )

            layer_cls = _LAYER_CLASSES[layer_type]

            if layer_type == 'Dense':
                layer = Dense(
                    params['input_size'],
                    params['output_size'],
                    params.get('activation', 'relu')
                )
                layer.set_parameters(params)

            elif layer_type == 'BatchNorm':
                layer = BatchNorm(
                    params['num_features'],
                    momentum=params.get('momentum', 0.9),
                    eps=params.get('eps', 1e-5)
                )
                layer.set_parameters(params)

            elif layer_type == 'Dropout':
                layer = Dropout(rate=params.get('rate', 0.5))

            elif layer_type == 'Flatten':
                layer = Flatten()

            elif layer_type == 'Conv2D':
                layer = Conv2D(
                    in_channels=params['in_channels'],
                    out_channels=params['out_channels'],
                    kernel_size=params['kernel_size'],
                    stride=params.get('stride', 1),
                    padding=params.get('padding', 'valid'),
                    activation=params.get('activation', 'relu')
                )
                layer.set_parameters(params)

            elif layer_type == 'LayerNorm':
                layer = LayerNorm(params['num_features'], eps=params.get('eps', 1e-5))
                layer.set_parameters(params)

            elif layer_type == 'MaxPool2D':
                layer = MaxPool2D(pool_size=params.get('pool_size', 2),
                                  stride=params.get('stride'))

            elif layer_type == 'AvgPool2D':
                layer = AvgPool2D(pool_size=params.get('pool_size', 2),
                                  stride=params.get('stride'))

            elif layer_type == 'Embedding':
                layer = Embedding(params['vocab_size'], params['embed_dim'])
                layer.set_parameters(params)

            elif layer_type == 'PositionalEncoding':
                layer = PositionalEncoding(params['d_model'],
                                           max_len=params.get('max_len', 5000))

            elif layer_type == 'SimpleRNN':
                layer = SimpleRNN(params['input_size'], params['hidden_size'],
                                  activation=params.get('activation', 'tanh'),
                                  return_sequences=params.get('return_sequences', True))
                layer.set_parameters(params)

            elif layer_type == 'GRU':
                layer = GRU(params['input_size'], params['hidden_size'],
                            return_sequences=params.get('return_sequences', True))
                layer.set_parameters(params)

            elif layer_type == 'LSTM':
                layer = LSTM(params['input_size'], params['hidden_size'],
                             return_sequences=params.get('return_sequences', True))
                layer.set_parameters(params)

            elif layer_type == 'MultiHeadAttention':
                layer = MultiHeadAttention(params['d_model'], params['num_heads'],
                                           dropout=params.get('dropout', 0.0),
                                           causal=params.get('causal', False))
                layer.set_parameters(params)

            else:
                raise ValueError(
                    f"Cannot load unknown layer type '{layer_type}'. "
                    f"Known types: {list(_LAYER_CLASSES.keys())}"
                )

            network.add(layer)

        return network

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, input_shape: Optional[tuple] = None) -> None:
        """
        Print a human-readable summary of the network architecture.

        Args:
            input_shape: Shape of a single sample (excluding the batch
                         dimension). Required to compute output shapes.
                         Example: (784,) for flat MNIST, (28, 28, 1) for images.
                         If not provided, output shapes are shown as '?'.
        """
        width = 60
        print("=" * width)
        print(f"{'Layer':<22} {'Output shape':<18} {'Parameters':>8}")
        print("=" * width)

        current_shape = input_shape
        total_params  = 0

        for layer in self.layers:
            if current_shape is not None and hasattr(layer, 'output_shape'):
                try:
                    out_shape = layer.output_shape(current_shape)
                    shape_str = str(('N',) + out_shape)
                    current_shape = out_shape
                except Exception:
                    shape_str = '?'
                    current_shape = None
            else:
                shape_str = '?'

            n_params = layer.param_count()
            total_params += n_params

            print(f"{repr(layer)[:21]:<22} {shape_str:<18} {n_params:>8,}")

        print("=" * width)
        print(f"{'Total parameters':<40} {total_params:>8,}")
        print("=" * width)

    def __repr__(self) -> str:
        lines = ["NeuralNetwork("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}) {repr(layer)}")
        lines.append(")")
        return "\n".join(lines)
