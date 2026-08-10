# neural_network

A modular, from-scratch neural network library built with **NumPy only** (plus the Python standard library for JSON serialisation). No PyTorch, no TensorFlow, no autograd engine — every forward and backward pass is written out explicitly.

It covers a genuinely broad surface for a from-scratch project: dense, convolutional, recurrent (RNN/GRU/LSTM), and attention (Transformer-style) layers, all sharing the same `forward` / `backward` / `get_parameters` interface, plus optimizers, losses, callbacks, metrics, save/load, and L2 regularization.

---

## Project goals

- Every forward and backward pass is written explicitly in NumPy. Nothing is hidden behind an autograd engine — reading `layers.py` *is* the documentation for how backprop works in each layer.
- The design is modular and follows separation of concerns: layers compute gradients, optimizers apply them, losses measure error, callbacks respond to training events.
- The codebase is meant to be readable and educational — every non-trivial formula has a docstring explaining what it is and why it works.
- Every layer's gradients are verified against numerical (finite-difference) gradient checks during development, not just derived on paper.

---

## Installation

This library has no install step beyond having NumPy available — it's a plain Python package, not published to PyPI.

```bash
pip install numpy          # the only hard dependency
pip install matplotlib     # optional, only for utils.plot_training_history
```

Then either drop the `neural_network/` folder into your project, or install it in editable mode from the repo root:

```bash
pip install -e .
```

*(this repo doesn't ship a `pyproject.toml` so the installation won't work yet)*

---

## File structure

```
neural_network/
├── __init__.py         # public API exports
├── activations.py      # activation functions + ACTIVATIONS registry
├── layers.py            # all layer types (Dense, recurrent, convolutional, attention…)
├── losses.py            # loss functions + LOSSES registry
├── optimizers.py        # optimizers + OPTIMIZERS registry
├── neuralnetwork.py      # NeuralNetwork container (train / evaluate / save / load)
├── metrics.py            # standalone evaluation metrics (accuracy, F1, R², …)
├── callbacks.py          # training callbacks (EarlyStopping, LR scheduling, ModelCheckpoint, …)
└── utils.py               # data utilities (split, normalise, plot, …)

spiral_classification_example.py     # Dense/BatchNorm/Dropout classifier on synthetic spiral data
transformer_arithmetic_example.py    # causal Transformer that learns 2-digit addition
shape_classifier_cnn_example.py      # CNN + full callback stack on a synthetic shape dataset
```

---

## Quick start

```python
import numpy as np
from neural_network import NeuralNetwork
from neural_network.layers import Dense, Dropout, BatchNorm
from neural_network.utils import train_test_split, one_hot_encode, normalize
from neural_network.callbacks import EarlyStopping
from neural_network import metrics as M

# --- data ---
X = normalize(X_raw)
y = one_hot_encode(labels, num_classes=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --- build ---
model = NeuralNetwork()
model.add(Dense(128, 64, activation='relu'))
model.add(BatchNorm(64))
model.add(Dropout(0.3))
model.add(Dense(64, 10, activation='softmax'))

# --- compile ---
model.compile(
    loss='categoricalcrossentropy',
    optimizer='adam',
    learning_rate=1e-3,
    gradient_clip=5.0,   # optional: clip gradients to this L2-norm ball
    weight_decay=1e-4,   # optional: L2 regularization on weight matrices
)

# --- train ---
history = model.train(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=True,
    verbose_interval=10,
    callbacks=[EarlyStopping(monitor='val_loss', patience=20)],
)

# --- evaluate ---
preds = model.predict(X_test)
print(f"Accuracy: {M.accuracy(preds, y_test):.4f}")
print(f"F1:       {M.f1_score(preds, y_test):.4f}")

# --- save / load ---
model.save('model.json')
model = NeuralNetwork.load('model.json')
```

---

## Layers

### Core

| Class | Description |
|---|---|
| `Dense(in, out, activation)` | Fully-connected layer. He initialisation. Supports both plain `(N, in)` input and per-timestep sequence input `(N, T, in)` (e.g. stacked after a recurrent/attention layer). |
| `Dropout(rate)` | Inverted dropout. No-op at inference time. |
| `BatchNorm(features)` | Batch normalisation. Maintains running stats for inference. Supports both `(N, features)` input **and** channels-last spatial input of any rank, e.g. `(N, H, W, features)` after `Conv2D` — statistics are computed over every axis except the last (channel) axis. |
| `LayerNorm(features)` | Layer normalisation over the feature (last) axis. Standard in transformers and RNNs. Works on any input rank. |
| `Flatten()` | Collapses all non-batch dimensions into a vector. |

### Convolutional

| Class | Description |
|---|---|
| `Conv2D(in_ch, out_ch, kernel, stride, padding, activation)` | 2-D convolution via a vectorized im2col/col2im (uses `numpy.lib.stride_tricks.sliding_window_view` and `np.add.at`, not a Python loop over output positions — see [Design notes](#design-notes) for honest performance characteristics). Padding: `'valid'` or `'same'`. |
| `MaxPool2D(pool_size, stride)` | Max pooling with a max-position mask for the backward pass. |
| `AvgPool2D(pool_size, stride)` | Average pooling. Gradient is distributed uniformly. |

### Recurrent

| Class | Description |
|---|---|
| `SimpleRNN(in, hidden, activation, return_sequences)` | Elman RNN with BPTT. Orthogonal init for the recurrent matrix. |
| `GRU(in, hidden, return_sequences)` | Gated Recurrent Unit. Update, reset, and candidate gates. |
| `LSTM(in, hidden, return_sequences)` | Long Short-Term Memory. Forget, input, cell, and output gates. Forget-bias initialised to 1. |

All recurrent layers accept input of shape `(batch, timesteps, features)` and return either `(batch, timesteps, hidden)` (`return_sequences=True`) or `(batch, hidden)` (`return_sequences=False`). They can be stacked.

### Attention

| Class | Description |
|---|---|
| `MultiHeadAttention(d_model, num_heads, dropout, causal)` | Scaled dot-product self-attention. Per-head Q/K/V projections + output projection. `causal=True` applies a lower-triangular causal mask automatically on every forward call (for autoregressive/language-model-style usage). Call `set_padding_mask(mask)` before `forward()` to additionally mask out padded key positions for one batch — the mask is consumed (cleared) after that single forward call, so it must be re-set per batch. |
| `Embedding(vocab_size, embed_dim)` | Learnable token embedding table. Integer indices → dense vectors. Sparse backward update. |
| `PositionalEncoding(d_model, max_len)` | Fixed (non-trainable) sinusoidal positional encoding (Vaswani et al., 2017). Self-attention has no inherent notion of token order, so this should be placed right after an `Embedding` layer and before any `MultiHeadAttention` layer. Backward is the identity function — nothing to learn here. |

---

## Activations

`relu`, `leakyrelu`, `prelu`, `elu`, `sigmoid`, `tanh`, `softmax`, `gelu`, `swish`, `linear`

Pass the name as a string to any layer that takes an `activation` argument, e.g. `Dense(64, 32, 'gelu')`. `softmax` normalizes over the last axis regardless of input rank, so it's safe to use as a per-timestep classification head, e.g. `(N, T, vocab_size)` language-model logits.

---

## Loss functions

| Name | Class | Use case |
|---|---|---|
| `'mse'` | `MSE` | Regression |
| `'mae'` | `MAE` | Regression (robust to outliers) |
| `'huber'` | `Huber(delta)` | Regression (combines MSE + MAE) |
| `'binarycrossentropy'` | `BinaryCrossEntropy` | Binary classification (sigmoid output) |
| `'categoricalcrossentropy'` | `CategoricalCrossEntropy` | Multi-class (softmax output). Sums over the last axis, so it's safe to use directly on flattened `(N*T, num_classes)` sequence output — see `transformer_arithmetic_example.py` for exactly this pattern. |

---

## Optimizers

| Name | Class | Notable args |
|---|---|---|
| `'sgd'` | `SGD` | `momentum` (EMA-style: the update is `w -= lr * ((1-momentum)*g + momentum*v)`-equivalent moving average of the gradient, which scales the *effective* learning rate by roughly `1/(1-momentum)` once warmed up — this is not the classic "velocity accumulator" momentum used in PyTorch's SGD, so the same `momentum` value won't behave identically if you're porting hyperparameters from another framework) |
| `'adam'` | `Adam` | `beta1`, `beta2`, `epsilon` |
| `'rmsprop'` | `RMSprop` | `beta`, `epsilon` |
| `'adagrad'` | `Adagrad` | `epsilon` |

All optimizers expose both an `update(layer)` method (used by standard layers) and an `update_raw(param_id, param, grad)` method (used internally by recurrent and attention layers that manage multiple weight matrices, and directly usable if you're composing layers by hand outside the `NeuralNetwork` container — see `transformer_arithmetic_example.py`).

### L2 weight decay

`NeuralNetwork.compile(..., weight_decay=1e-4)` adds classic L2 regularization: `weight_decay * W` is added to each regularizable layer's weight gradient before the optimizer step (not decoupled/AdamW-style decay). Biases are never decayed. Layers whose "weights" are actually normalization scale parameters (`BatchNorm.gamma`, `LayerNorm.gamma`) are automatically excluded via a `Layer.regularizable` class attribute, since decaying a norm layer's scale toward zero is not standard practice. Default is `0.0` (disabled).

---

## Callbacks

| Class | Description |
|---|---|
| `EarlyStopping(monitor, patience, min_delta, restore_best_weights)` | Stop when a monitored metric stops improving. Optionally restores the best checkpoint. Correctly handles the direction of improvement for both "lower is better" (`*_loss`) and "higher is better" (`*_acc`) metrics. |
| `LearningRateScheduler(schedule)` | Adjust LR each epoch via a callable. Built-in schedules: `exponential_decay`, `step_decay`, `cosine_annealing`. |
| `ReduceOnPlateau(monitor, factor, patience, min_lr)` | Reduce LR when a metric has plateaued. Same improvement-direction handling as `EarlyStopping`. |
| `ModelCheckpoint(filepath, monitor, save_best_only, verbose)` | Save the model to `filepath` (via `NeuralNetwork.save()`) whenever the monitored metric improves, so you can recover the best epoch even if training continues past it or is stopped early. |

---

## Metrics

All metrics in `neural_network/metrics.py` are standalone functions — pass predictions and targets directly.

**Classification:** `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix`
**Regression:** `r2_score`, `mean_absolute_percentage_error`

---

## More examples

### Sequence model (LSTM)

```python
from neural_network import NeuralNetwork
from neural_network.layers import LSTM, Dense

model = NeuralNetwork()
model.add(LSTM(input_size=16, hidden_size=64, return_sequences=True))
model.add(LSTM(input_size=64, hidden_size=32, return_sequences=False))
model.add(Dense(32, 1, activation='sigmoid'))

model.compile(loss='binarycrossentropy', optimizer='adam', learning_rate=5e-4)
# input shape: (batch, timesteps, features)
model.train(X_seq, y, epochs=100, batch_size=32)
```

### Attention / Transformer-style model

```python
from neural_network import NeuralNetwork
from neural_network.layers import Embedding, PositionalEncoding, MultiHeadAttention, LayerNorm, Flatten, Dense

model = NeuralNetwork()
model.add(Embedding(vocab_size=5000, embed_dim=64))
model.add(PositionalEncoding(d_model=64, max_len=seq_len))
model.add(MultiHeadAttention(d_model=64, num_heads=4, dropout=0.1, causal=True))
model.add(LayerNorm(64))
model.add(Flatten())
model.add(Dense(64 * seq_len, 128, activation='relu'))
model.add(Dense(128, num_classes, activation='softmax'))

model.compile(loss='categoricalcrossentropy', optimizer='adam', learning_rate=1e-3)
```

`NeuralNetwork` is a plain **sequential** container, so this pattern (single attention layer, no residual connections) is as far as the high-level API goes on its own. For a real multi-block Transformer with residual connections and pre-norm blocks, compose the layer objects by hand — every layer already exposes the same `forward`/`backward`/`get_parameters` contract needed to do this in a page of code. See `transformer_arithmetic_example.py` for a complete, working, gradient-checked example of exactly this.

### Conv net

```python
from neural_network import NeuralNetwork
from neural_network.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNorm

model = NeuralNetwork()
model.add(Conv2D(1, 16, kernel_size=3, padding='same', activation='relu'))
model.add(BatchNorm(16))   # BatchNorm after Conv2D works directly on (N, H, W, C)
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(16, 32, kernel_size=3, padding='same', activation='relu'))
model.add(BatchNorm(32))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(32 * 7 * 7, 128, activation='relu'))
model.add(Dense(128, 10, activation='softmax'))

# input shape: (batch, H, W, channels)
model.compile(loss='categoricalcrossentropy', optimizer='adam')
```

---

## Example scripts in this repo

| Script | What it demonstrates |
|---|---|
| `spiral_classification_example.py` | The basics: `Dense` + `BatchNorm` + `Dropout` on a synthetic multi-class spiral dataset, with `NeuralNetwork.train()`/`.evaluate()` and a decision-boundary plot. |
| `transformer_arithmetic_example.py` | A hand-composed, residual, causal Transformer (`Embedding` → `PositionalEncoding` → 2× [`LayerNorm`/`MultiHeadAttention(causal=True)`/`LayerNorm`/`Dense`+GELU] blocks → `Dense` softmax head) trained as a character-level language model to add two 2-digit numbers, evaluated via genuine autoregressive generation on held-out pairs it never trained on. Every parameter's gradient in this example was numerically verified against finite differences before trusting the training results — see the script's docstring for the honest, non-oversold accuracy this budget actually achieves. |
| `shape_classifier_cnn_example.py` | A real `Conv2D`/`BatchNorm`/`MaxPool2D` CNN trained on a procedurally-generated 4-class shape dataset (no external data or network access needed), using the *full* callback stack together — `EarlyStopping`, `ReduceOnPlateau`, and `ModelCheckpoint` — plus `weight_decay` and `gradient_clip`, then reloading the best checkpoint and printing a per-class accuracy breakdown and confusion matrix. |

---

## Design notes

**Gradient flow.** Each layer's `backward(doutput)` method stores computed gradients in `layer.dweights` and `layer.dbiases` but does *not* apply them. `NeuralNetwork.update()` then passes each layer to the optimizer. This separates gradient computation from parameter update, making it trivial to swap optimizers.

**Recurrent layers.** SimpleRNN, GRU, and LSTM each manage several weight matrices (e.g. LSTM has 8: W_xf, W_hf, W_xi, W_hi, …). The standard optimizer interface (`weights`/`biases`) routes to one pair; the rest are updated via `_extra_update(optimizer)`, which calls `optimizer.update_raw(stable_key, param, grad)` — a lower-level method that every optimizer implements. `MultiHeadAttention` follows the same pattern for its Q/K/V/output projections.

**Orthogonal initialisation.** Recurrent weight matrices (W_h in SimpleRNN, all H→H matrices in GRU/LSTM) are initialised via QR decomposition. This preserves vector norms during the initial forward passes and typically leads to faster convergence and more stable gradients on longer sequences compared to random Gaussian initialisation.

**Numerical stability.** Softmax subtracts the row-wise (technically: last-axis-wise) max before exponentiating. Sigmoid uses a branch-free implementation that avoids overflow for both large positive and large negative inputs. In LSTM/GRU, only the *sigmoid* gate pre-activations (forget/input/output/update/reset) are clipped to [-30, 30] before the sigmoid, to avoid `exp` overflow; the `tanh`-based candidate/cell gates are left unclipped since `tanh` saturates smoothly and can't overflow.

**Conv2D vectorization, honestly benchmarked.** `Conv2D._im2col`/`_col2im` use `sliding_window_view` and `np.add.at` instead of a Python-level loop over every output spatial position. This is a real, verified (bit-exact against the old loop-based implementation, including overlapping-window `stride < kernel_size` cases) improvement, but it is *not* a universal speedup: for channel-heavy, smaller-spatial shapes typical of deeper CNN layers it measured up to ~5x faster in benchmarking; for large-spatial, low-channel shapes (e.g. a single big early-layer feature map) the unavoidable transpose/copy needed to align axes can make it roughly break-even or even slightly slower than the loop version. If you're optimizing hot-path Conv2D performance for a specific shape, benchmark both against your actual workload rather than assuming either version wins.

**Sequential-only container.** `NeuralNetwork` applies its layers strictly one after another — there's no branching/graph support, so architectures that need skip connections (ResNets, Transformer blocks with residuals) can't be expressed purely through `model.add(...)`. Rather than add graph support to the core library, the recommended pattern (demonstrated in `transformer_arithmetic_example.py`) is to hold the layer objects directly and wire up `forward`/`backward` (including the residual addition and its gradient split) yourself — every layer's public interface is designed to support exactly this.

---

## Dependencies

- `numpy` — all numerical computation
- `json` — model serialisation (standard library)
- `matplotlib` — optional, only used by `utils.plot_training_history` (I should remove it and this awkward part)
