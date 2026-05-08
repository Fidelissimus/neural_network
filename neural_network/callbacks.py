import numpy as np
from typing import Optional


class Callback:
    """
    Base class for training callbacks.

    A callback is an object with methods that the NeuralNetwork.train loop
    calls at specific moments: the start of training, the end of each epoch,
    and the end of training.  Subclass this and override the methods you need.
    """

    def on_train_begin(self, network) -> None:
        """Called once before the first epoch."""

    def on_epoch_end(self, network, epoch: int, logs: dict) -> None:
        """
        Called at the end of every epoch.

        Args:
            network: The NeuralNetwork instance being trained.
            epoch:   Zero-based epoch index.
            logs:    Dict with at least 'train_loss' and optionally 'val_loss',
                     'train_acc', 'val_acc'.
        """

    def on_train_end(self, network) -> None:
        """Called once after the last epoch (or after early stopping)."""


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.

    Args:
        monitor:   Name of the quantity to watch. One of 'val_loss',
                   'train_loss', 'val_acc', 'train_acc'. Default 'val_loss'.
        patience:  Number of epochs with no improvement before stopping.
                   Default 10.
        min_delta: Minimum change counted as an improvement. Default 1e-4.
        restore_best_weights: If True, the network's layer parameters are
                   rolled back to the epoch with the best monitored value
                   at the end of training. Default True.
        verbose:   Print a message when early stopping is triggered. Default True.
    """

    def __init__(self, monitor: str = 'val_loss', patience: int = 10,
                 min_delta: float = 1e-4, restore_best_weights: bool = True,
                 verbose: bool = True):
        self.monitor             = monitor
        self.patience            = patience
        self.min_delta           = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose             = verbose

        self._best_value:  float = np.inf
        self._wait:        int   = 0
        self._best_params: list  = []
        self._stopped_epoch: Optional[int] = None
        self.stop_training: bool = False

        # For accuracy-like metrics higher is better; for loss lower is better
        self._monitor_op = np.less if 'loss' in monitor else np.greater
        if 'loss' in monitor:
            self._best_value = np.inf
        else:
            self._best_value = -np.inf

    def on_train_begin(self, network) -> None:
        self._wait         = 0
        self._best_params  = []
        self._stopped_epoch = None
        self.stop_training = False
        if 'loss' in self.monitor:
            self._best_value = np.inf
        else:
            self._best_value = -np.inf

    def on_epoch_end(self, network, epoch: int, logs: dict) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return  # metric not available this epoch (e.g. no validation data)

        improved = self._monitor_op(current, self._best_value - self.min_delta)

        if improved:
            self._best_value = current
            self._wait = 0
            if self.restore_best_weights:
                import copy
                self._best_params = [
                    copy.deepcopy(layer.get_parameters())
                    for layer in network.layers
                ]
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self._stopped_epoch = epoch
                self.stop_training  = True

    def on_train_end(self, network) -> None:
        if self._stopped_epoch is not None and self.verbose:
            print(f"\nEarlyStopping: stopped at epoch {self._stopped_epoch}. "
                  f"Best {self.monitor} = {self._best_value:.6f}.")

        if self.restore_best_weights and self._best_params:
            for layer, params in zip(network.layers, self._best_params):
                if params:
                    layer.set_parameters(params)
            if self.verbose:
                print("EarlyStopping: restored best weights.")


class LearningRateScheduler(Callback):
    """
    Adjust the optimizer's learning rate according to a schedule function.

    The schedule function receives the current epoch (0-based) and the current
    learning rate, and should return the new learning rate.

    Built-in schedules are provided as static factory methods for convenience.

    Args:
        schedule: Callable(epoch, current_lr) -> new_lr.
        verbose:  Print the new LR each time it changes. Default False.

    Example::

        scheduler = LearningRateScheduler(
            LearningRateScheduler.exponential_decay(initial_lr=0.01, decay=0.95)
        )
    """

    def __init__(self, schedule, verbose: bool = False):
        self.schedule = schedule
        self.verbose  = verbose

    def on_epoch_end(self, network, epoch: int, logs: dict) -> None:
        old_lr = network.optimizer.learning_rate
        new_lr = self.schedule(epoch, old_lr)
        if new_lr != old_lr:
            network.optimizer.learning_rate = new_lr
            if self.verbose:
                print(f"LearningRateScheduler: epoch {epoch} LR {old_lr:.6f} -> {new_lr:.6f}")

    # ------------------------------------------------------------------
    # Built-in schedule factories
    # ------------------------------------------------------------------

    @staticmethod
    def exponential_decay(initial_lr: float, decay: float = 0.95):
        """
        LR is multiplied by `decay` every epoch.
            lr(epoch) = initial_lr * decay^epoch
        """
        def schedule(epoch: int, _lr: float) -> float:
            return initial_lr * (decay ** epoch)
        return schedule

    @staticmethod
    def step_decay(initial_lr: float, drop: float = 0.5, epochs_drop: int = 10):
        """
        LR is halved (or reduced by `drop`) every `epochs_drop` epochs.
        """
        def schedule(epoch: int, _lr: float) -> float:
            return initial_lr * (drop ** (epoch // epochs_drop))
        return schedule

    @staticmethod
    def cosine_annealing(initial_lr: float, total_epochs: int,
                         min_lr: float = 0.0):
        """
        LR follows a cosine curve from `initial_lr` down to `min_lr`.
        """
        def schedule(epoch: int, _lr: float) -> float:
            cos_val = np.cos(np.pi * epoch / total_epochs)
            return min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + cos_val)
        return schedule

    @staticmethod
    def reduce_on_plateau(factor: float = 0.5, patience: int = 5,
                          min_lr: float = 1e-6, min_delta: float = 1e-4,
                          monitor: str = 'val_loss'):
        """
        Reduce LR when the monitored metric has stopped improving.

        Returns a stateful schedule function that tracks the metric via
        the logs dict.  Unlike the other factories this one wraps a small
        stateful object so that it can maintain its own patience counter.
        """
        state = {
            'best':    np.inf if 'loss' in monitor else -np.inf,
            'wait':    0,
            'monitor': monitor,
        }
        monitor_op = np.less if 'loss' in monitor else np.greater

        def schedule(epoch: int, lr: float) -> float:
            # This schedule function is intentionally a no-op here; the actual
            # reduction happens inside a dedicated ReduceOnPlateau callback
            # that has access to the logs dict.  This factory exists as a
            # convenience reference; prefer the ReduceOnPlateau class below.
            return lr

        return schedule


class ReduceOnPlateau(Callback):
    """
    Reduce the learning rate when a metric has stopped improving.

    Args:
        monitor:   Metric to watch. Default 'val_loss'.
        factor:    Multiplicative factor by which to reduce LR. Default 0.5.
        patience:  Number of non-improving epochs before reduction. Default 5.
        min_lr:    Lower bound on the learning rate. Default 1e-6.
        min_delta: Minimum change counted as an improvement. Default 1e-4.
        verbose:   Print a message each time LR is reduced. Default True.
    """

    def __init__(self, monitor: str = 'val_loss', factor: float = 0.5,
                 patience: int = 5, min_lr: float = 1e-6,
                 min_delta: float = 1e-4, verbose: bool = True):
        self.monitor   = monitor
        self.factor    = factor
        self.patience  = patience
        self.min_lr    = min_lr
        self.min_delta = min_delta
        self.verbose   = verbose

        self._monitor_op = np.less if 'loss' in monitor else np.greater
        self._best_value: float = np.inf if 'loss' in monitor else -np.inf
        self._wait: int = 0

    def on_train_begin(self, network) -> None:
        self._best_value = np.inf if 'loss' in self.monitor else -np.inf
        self._wait = 0

    def on_epoch_end(self, network, epoch: int, logs: dict) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return

        if self._monitor_op(current, self._best_value - self.min_delta):
            self._best_value = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                old_lr = network.optimizer.learning_rate
                new_lr = max(old_lr * self.factor, self.min_lr)
                network.optimizer.learning_rate = new_lr
                self._wait = 0
                if self.verbose:
                    print(f"\nReduceOnPlateau: epoch {epoch}: "
                          f"LR {old_lr:.6f} -> {new_lr:.6f}")
