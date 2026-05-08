import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Fraction of correctly classified samples.

    Supports both binary and multi-class tasks. For multi-class, both
    y_pred and y_true are expected as probability / one-hot matrices and
    the argmax is taken along axis 1.

    Args:
        y_pred: Network output, shape (N, 1) for binary or (N, C) for multi-class.
        y_true: Ground-truth labels, same shape as y_pred.

    Returns:
        Accuracy in [0, 1].
    """
    if y_pred.shape[1] == 1:   # binary
        predictions = (y_pred > 0.5).astype(int).flatten()
        targets     = y_true.astype(int).flatten()
    else:                       # multi-class
        predictions = np.argmax(y_pred, axis=1)
        targets     = np.argmax(y_true, axis=1)

    return float(np.mean(predictions == targets))


def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray,
                     num_classes: Optional[int] = None) -> np.ndarray:
    """
    Compute the confusion matrix.

    Rows correspond to true classes; columns to predicted classes.
    Entry (i, j) counts the number of samples with true class i that were
    predicted as class j.

    Args:
        y_pred:      Network output (probabilities or one-hot).
        y_true:      Ground-truth labels (same format).
        num_classes: Number of classes. Inferred from data if not given.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    if y_pred.shape[1] == 1:
        pred_labels = (y_pred > 0.5).astype(int).flatten()
        true_labels = y_true.astype(int).flatten()
    else:
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)

    if num_classes is None:
        num_classes = int(max(true_labels.max(), pred_labels.max()) + 1)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(true_labels, pred_labels):
        cm[true, pred] += 1
    return cm


def precision(y_pred: np.ndarray, y_true: np.ndarray,
              average: str = 'macro') -> float:
    """
    Precision: TP / (TP + FP) per class, then averaged.

    Args:
        y_pred:  Network output.
        y_true:  Ground-truth labels.
        average: 'macro' (unweighted mean) or 'weighted' (weighted by support).

    Returns:
        Scalar precision value.
    """
    cm = confusion_matrix(y_pred, y_true)
    num_classes = cm.shape[0]
    col_sums = cm.sum(axis=0)
    per_class = np.where(col_sums > 0,
                         np.diag(cm) / col_sums,
                         0.0)

    if average == 'weighted':
        support = cm.sum(axis=1)
        return float(np.sum(per_class * support) / support.sum())
    return float(np.mean(per_class))


def recall(y_pred: np.ndarray, y_true: np.ndarray,
           average: str = 'macro') -> float:
    """
    Recall: TP / (TP + FN) per class, then averaged.

    Args:
        y_pred:  Network output.
        y_true:  Ground-truth labels.
        average: 'macro' or 'weighted'.

    Returns:
        Scalar recall value.
    """
    cm = confusion_matrix(y_pred, y_true)
    row_sums = cm.sum(axis=1)
    per_class = np.where(row_sums > 0,
                         np.diag(cm) / row_sums,
                         0.0)

    if average == 'weighted':
        return float(np.sum(per_class * row_sums) / row_sums.sum())
    return float(np.mean(per_class))


def f1_score(y_pred: np.ndarray, y_true: np.ndarray,
             average: str = 'macro') -> float:
    """
    F1 score: harmonic mean of precision and recall.

    F1 = 2 * precision * recall / (precision + recall)

    Args:
        y_pred:  Network output.
        y_true:  Ground-truth labels.
        average: 'macro' or 'weighted'.

    Returns:
        Scalar F1 score.
    """
    p = precision(y_pred, y_true, average)
    r = recall(y_pred,    y_true, average)
    denom = p + r
    return float(2.0 * p * r / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Coefficient of determination R².

    Measures the proportion of the variance in y_true explained by the model.
    R² = 1  means perfect prediction.
    R² = 0  means the model performs no better than predicting the mean.
    R² < 0  means the model is worse than predicting the mean.

    Args:
        y_pred: Model predictions, shape (N, 1) or (N,).
        y_true: Ground-truth targets, same shape.

    Returns:
        R² as a float.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def mean_absolute_percentage_error(y_pred: np.ndarray,
                                   y_true: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    MAPE = mean(|y_true - y_pred| / |y_true|) * 100

    Samples where y_true == 0 are excluded to avoid division by zero.

    Args:
        y_pred: Model predictions.
        y_true: Ground-truth targets.

    Returns:
        MAPE as a percentage (e.g. 5.2 means 5.2 %).
    """
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
