"""
Convolutional shape classifier, trained end-to-end with the full callback
stack: EarlyStopping, ReduceOnPlateau, ModelCheckpoint, and L2 weight decay.
=============================================================================

Rather than another MNIST clone (which needs a network download this
environment doesn't have, and which every neural-net library ships anyway),
this generates its own dataset: 28x28 grayscale images of four hand-drawn
shape classes -- circle, square, triangle, cross -- each rendered with
randomized position, size, rotation (where applicable) and per-pixel noise.
It's a genuinely non-trivial vision task (shapes overlap in size/position
range, noise obscures edges, rotation changes a triangle's silhouette a lot)
while staying fast enough to train in well under a minute on CPU, entirely
self-contained with no external data or network access.

This example exists to show what a *complete, realistic* training setup
looks like with this library, not just a bare training loop:
    - A real conv stack: Conv2D -> BatchNorm -> MaxPool2D, twice, then a
      Dense classification head with Dropout.
    - L2 weight decay (compile(weight_decay=...)) on the conv/dense weights.
    - Gradient clipping, in case a bad batch produces an outsized gradient.
    - EarlyStopping to stop once validation loss stops improving.
    - ReduceOnPlateau to shrink the learning rate on a plateau instead of
      stopping outright.
    - ModelCheckpoint to persist the best validation checkpoint to disk as
      training progresses, independent of whichever epoch training
      eventually stops on.
    - A final per-class accuracy breakdown and a plain-text confusion
      matrix, not just one overall accuracy number.
"""
import numpy as np

from neural_network import NeuralNetwork
from neural_network.layers import Conv2D, BatchNorm, MaxPool2D, Flatten, Dense, Dropout
from neural_network.callbacks import EarlyStopping, ReduceOnPlateau, ModelCheckpoint
from neural_network.utils import one_hot_encode, train_test_split

np.random.seed(0)

IMG_SIZE = 28
CLASSES = ['circle', 'square', 'triangle', 'cross']
N_CLASSES = len(CLASSES)


# ---------------------------------------------------------------------------
# 1. Procedural shape renderer
# ---------------------------------------------------------------------------
def _blank_canvas():
    return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float64)


def _coords():
    yy, xx = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE), indexing='ij')
    return yy.astype(np.float64), xx.astype(np.float64)


_YY, _XX = _coords()


def draw_circle(cy, cx, r, thickness=2.0):
    dist = np.sqrt((_YY - cy) ** 2 + (_XX - cx) ** 2)
    return (np.abs(dist - r) < thickness).astype(np.float64)


def draw_square(cy, cx, half_size, angle, thickness=2.0):
    # Rotate coordinates into the square's local frame, then test proximity
    # to the (axis-aligned, in local frame) boundary.
    c, s = np.cos(-angle), np.sin(-angle)
    ly = c * (_YY - cy) - s * (_XX - cx)
    lx = s * (_YY - cy) + c * (_XX - cx)
    outside_dist = np.maximum(np.abs(ly), np.abs(lx)) - half_size
    return (np.abs(outside_dist) < thickness).astype(np.float64)


def _point_segment_dist(py, px, ay, ax, by, bx):
    """Vectorized distance from grid points (py, px) to segment (a -> b)."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len_sq = abx ** 2 + aby ** 2
    t = np.clip((apx * abx + apy * aby) / ab_len_sq, 0.0, 1.0)
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)


def draw_triangle(cy, cx, size, angle, thickness=2.0):
    # Equilateral triangle outline: three vertices on a circle of radius
    # `size` around (cy, cx), edges drawn as thin bands using point-to-segment
    # distance so this is a genuine outline, not a filled wedge.
    verts = []
    for k in range(3):
        a = angle + k * (2 * np.pi / 3)
        verts.append((cy + size * np.sin(a), cx + size * np.cos(a)))

    canvas = np.full((IMG_SIZE, IMG_SIZE), np.inf)
    for i in range(3):
        ay, ax = verts[i]
        by, bx = verts[(i + 1) % 3]
        d = _point_segment_dist(_YY, _XX, ay, ax, by, bx)
        canvas = np.minimum(canvas, d)
    return (canvas < thickness).astype(np.float64)


def draw_cross(cy, cx, half_size, angle, thickness=2.5):
    c, s = np.cos(-angle), np.sin(-angle)
    ly = c * (_YY - cy) - s * (_XX - cx)
    lx = s * (_YY - cy) + c * (_XX - cx)
    vertical   = (np.abs(lx) < thickness) & (np.abs(ly) < half_size)
    horizontal = (np.abs(ly) < thickness) & (np.abs(lx) < half_size)
    return (vertical | horizontal).astype(np.float64)


def generate_sample(class_idx: int, rng: np.random.RandomState):
    margin = 6
    cy = rng.uniform(margin, IMG_SIZE - margin)
    cx = rng.uniform(margin, IMG_SIZE - margin)
    size = rng.uniform(6, 10)
    angle = rng.uniform(0, 2 * np.pi)
    thickness = rng.uniform(1.5, 2.5)

    if class_idx == 0:
        img = draw_circle(cy, cx, size, thickness)
    elif class_idx == 1:
        img = draw_square(cy, cx, size, angle, thickness)
    elif class_idx == 2:
        img = draw_triangle(cy, cx, size * 1.6, angle, thickness)
    else:
        img = draw_cross(cy, cx, size, angle, thickness)

    # Mild blur (cheap 3x3 box blur via shifted-average) then pixel noise,
    # so edges aren't perfectly crisp -- more realistic, harder than a clean
    # binary mask.
    blurred = img.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            blurred += np.roll(np.roll(img, dy, axis=0), dx, axis=1)
    blurred /= 9.0

    noise = rng.normal(0, 0.08, size=(IMG_SIZE, IMG_SIZE))
    out = np.clip(blurred + noise, 0.0, 1.0)
    return out


def make_dataset(n_per_class: int, seed: int):
    rng = np.random.RandomState(seed)
    images, labels = [], []
    for class_idx in range(N_CLASSES):
        for _ in range(n_per_class):
            images.append(generate_sample(class_idx, rng))
            labels.append(class_idx)
    X = np.stack(images)[..., None]              # (N, 28, 28, 1) -- channels-last
    y = np.array(labels)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


print("Generating synthetic shape dataset...")
X_all, y_all_int = make_dataset(n_per_class=250, seed=0)   # 1000 images total
y_all = one_hot_encode(y_all_int, N_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)
print(f"Train: {X_train.shape[0]} images, Test: {X_test.shape[0]} images, "
      f"image size: {IMG_SIZE}x{IMG_SIZE}, classes: {CLASSES}")


# ---------------------------------------------------------------------------
# 2. Model: two Conv->BatchNorm->Pool blocks, then a Dense head
# ---------------------------------------------------------------------------
model = NeuralNetwork()
model.add(Conv2D(1, 16, kernel_size=3, stride=1, padding='same', activation='relu'))
model.add(BatchNorm(16))
model.add(MaxPool2D(pool_size=2, stride=2))          # 28x28 -> 14x14

model.add(Conv2D(16, 32, kernel_size=3, stride=1, padding='same', activation='relu'))
model.add(BatchNorm(32))
model.add(MaxPool2D(pool_size=2, stride=2))          # 14x14 -> 7x7

model.add(Flatten())
model.add(Dense(7 * 7 * 32, 64, 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64, N_CLASSES, 'softmax'))

model.compile(loss='categoricalcrossentropy', optimizer='adam', learning_rate=1e-3,
             weight_decay=1e-4, gradient_clip=5.0)
model.summary()

# ---------------------------------------------------------------------------
# 3. Train with the full callback stack
# ---------------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, min_delta=1e-4, verbose=True),
    ReduceOnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=True),
    ModelCheckpoint('shape_classifier_best.json', monitor='val_loss',
                    save_best_only=True, verbose=False),
]

history = model.train(X_train, y_train, epochs=20, batch_size=32,
                      validation_data=(X_test, y_test),
                      callbacks=callbacks, verbose=True, verbose_interval=2)

# ---------------------------------------------------------------------------
# 4. Reload the best checkpoint (not necessarily the last epoch) and evaluate
# ---------------------------------------------------------------------------
best_model = NeuralNetwork.load('shape_classifier_best.json')
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f"\nBest checkpoint -- test loss: {test_loss:.4f}, test accuracy: {test_acc:.1%}")

# Per-class accuracy + confusion matrix
y_pred = np.argmax(best_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
for t, p in zip(y_true, y_pred):
    confusion[t, p] += 1

print("\nPer-class accuracy:")
for i, name in enumerate(CLASSES):
    class_total = confusion[i].sum()
    class_correct = confusion[i, i]
    print(f"  {name:10s}: {class_correct}/{class_total} "
          f"({class_correct / max(class_total, 1):.1%})")

print("\nConfusion matrix (rows = true class, columns = predicted class):")
header = "             " + "".join(f"{name:>10s}" for name in CLASSES)
print(header)
for i, name in enumerate(CLASSES):
    row = "".join(f"{confusion[i, j]:>10d}" for j in range(N_CLASSES))
    print(f"  {name:10s} {row}")
