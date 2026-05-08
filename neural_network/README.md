# numpy-nn

A modular, from-scratch neural network library built with **NumPy only** (plus the Python standard library for JSON serialisation). No PyTorch, no TensorFlow, no dependencies beyond what ships with a standard scientific Python install.

---

## Project goals

- Every forward and backward pass is written explicitly in NumPy. Nothing is hidden behind an autograd engine.
- The design is modular and follows separation of concerns: layers compute gradients, optimizers apply them, losses measure error, callbacks respond to training events.
- The codebase is meant to be readable and educational — every formula in every layer has a docstring explaining what it is and why it works.

---

## File structure

```
nn/
├── __init__.py         # public API exports
├── activations.py      # activation functions + ACTIVATIONS registry
├── layers.py           # all layer types (Dense, recurrent, convolutional, attention…)
├── losses.py           # loss functions + LOSSES registry
├── optimizers.py       # optimizers + OPTIMIZERS registry
├── neuralnetwork.py    # NeuralNetwork container (train / evaluate / save / load)
├── metrics.py          # standalone evaluation metrics (accuracy, F1, R², …)
├── callbacks.py        # training callbacks (EarlyStopping, LR scheduling, …)
└── utils.py            # data utilities (split, normalise, plot, …)
```

---

## Layers

### Core

| Class | Description |
|---|---|
| `Dense(in, out, activation)` | Fully-connected layer. He initialisation. |
| `Dropout(rate)` | Inverted dropout. No-op at inference time. |
| `BatchNorm(features)` | Batch normalisation. Maintains running stats for inference. |
| `LayerNorm(features)` | Layer normalisation over the feature axis. Standard in transformers and RNNs. |
| `Flatten()` | Collapses all non-batch dimensions into a vector. |

### Convolutional

| Class | Description |
|---|---|
| `Conv2D(in_ch, out_ch, kernel, stride, padding, activation)` | 2-D convolution via im2col. Padding: `'valid'` or `'same'`. |
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
| `MultiHeadAttention(d_model, num_heads, dropout)` | Scaled dot-product self-attention. Per-head Q/K/V projections + output projection. |
| `Embedding(vocab_size, embed_dim)` | Learnable token embedding table. Integer indices → dense vectors. Sparse backward update. |

---

## Activations

`relu`, `leakyrelu`, `prelu`, `elu`, `sigmoid`, `tanh`, `softmax`, `gelu`, `swish`, `linear`

Pass the name as a string to any layer that takes an `activation` argument, e.g. `Dense(64, 32, 'gelu')`.

---

## Loss functions

| Name | Class | Use case |
|---|---|---|
| `'mse'` | `MSE` | Regression |
| `'mae'` | `MAE` | Regression (robust to outliers) |
| `'huber'` | `Huber(delta)` | Regression (combines MSE + MAE) |
| `'binarycrossentropy'` | `BinaryCrossEntropy` | Binary classification (sigmoid output) |
| `'categoricalcrossentropy'` | `CategoricalCrossEntropy` | Multi-class (softmax output) |

---

## Optimizers

| Name | Class | Notable args |
|---|---|---|
| `'sgd'` | `SGD` | `momentum` |
| `'adam'` | `Adam` | `beta1`, `beta2`, `epsilon` |
| `'rmsprop'` | `RMSprop` | `beta`, `epsilon` |
| `'adagrad'` | `Adagrad` | `epsilon` |

All optimizers expose both a `update(layer)` method (used by standard layers) and an `update_raw(param_id, param, grad)` method (used internally by recurrent and attention layers that manage multiple weight matrices).

---

## Callbacks

| Class | Description |
|---|---|
| `EarlyStopping(monitor, patience, min_delta, restore_best_weights)` | Stop when a monitored metric stops improving. Optionally restores the best checkpoint. |
| `LearningRateScheduler(schedule)` | Adjust LR each epoch via a callable. Built-in schedules: `exponential_decay`, `step_decay`, `cosine_annealing`. |
| `ReduceOnPlateau(monitor, factor, patience)` | Reduce LR when a metric has plateaued. |

---

## Metrics

All metrics in `nn/metrics.py` are standalone functions — pass predictions and targets directly.

**Classification:** `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix`  
**Regression:** `r2_score`, `mean_absolute_percentage_error`

---

## Quick start

```python
import numpy as np
from nn import NeuralNetwork
from nn.layers import Dense, Dropout, BatchNorm
from nn.utils import train_test_split, one_hot_encode, normalize
from nn.callbacks import EarlyStopping
from nn import metrics as M

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
    gradient_clip=5.0,   # optional L2 gradient clipping
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

### Sequence model (LSTM)

```python
from nn.layers import LSTM, Dense

model = NeuralNetwork()
model.add(LSTM(input_size=16, hidden_size=64, return_sequences=True))
model.add(LSTM(input_size=64, hidden_size=32, return_sequences=False))
model.add(Dense(32, 1, activation='sigmoid'))

model.compile(loss='binarycrossentropy', optimizer='adam', learning_rate=5e-4)
# input shape: (batch, timesteps, features)
model.train(X_seq, y, epochs=100, batch_size=32)
```

### Attention model

```python
from nn.layers import Embedding, MultiHeadAttention, LayerNorm, Flatten, Dense

model = NeuralNetwork()
model.add(Embedding(vocab_size=5000, embed_dim=64))
model.add(MultiHeadAttention(d_model=64, num_heads=4, dropout=0.1))
model.add(LayerNorm(64))
model.add(Flatten())
model.add(Dense(64 * seq_len, 128, activation='relu'))
model.add(Dense(128, num_classes, activation='softmax'))

model.compile(loss='categoricalcrossentropy', optimizer='adam', learning_rate=1e-3)
```

### Conv net

```python
from nn.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNorm

model = NeuralNetwork()
model.add(Conv2D(1, 16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(16, 32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(32 * 7 * 7, 128, activation='relu'))
model.add(Dense(128, 10, activation='softmax'))

# input shape: (batch, H, W, channels)
model.compile(loss='categoricalcrossentropy', optimizer='adam')
```

---

## Design notes

**Gradient flow.** Each layer's `backward(doutput)` method stores computed gradients in `layer.dweights` and `layer.dbiases` but does *not* apply them. `NeuralNetwork.update()` then passes each layer to the optimizer. This separates gradient computation from parameter update, making it trivial to swap optimizers.

**Recurrent layers.** SimpleRNN, GRU, and LSTM each manage several weight matrices (e.g. LSTM has 8: W_xf, W_hf, W_xi, W_hi, …). The standard optimizer interface (`weights`/`biases`) routes to one pair; the rest are updated via `_extra_update(optimizer)`, which calls `optimizer.update_raw(stable_key, param, grad)` — a lower-level method that every optimizer implements.

**Orthogonal initialisation.** Recurrent weight matrices (W_h in SimpleRNN, all H→H matrices in GRU/LSTM) are initialised via QR decomposition. This preserves vector norms during the initial forward passes and typically leads to faster convergence and more stable gradients on longer sequences compared to random Gaussian initialisation.

**Numerical stability.** Softmax subtracts the row-wise max before exponentiating. Sigmoid uses a branch-free implementation that avoids overflow for both large positive and large negative inputs. LSTM/GRU gate pre-activations are clipped to [-30, 30].

---

## Dependencies

- `numpy` — all numerical computation
- `json` — model serialisation (standard library)
- `matplotlib` — optional, only used by `utils.plot_training_history`
