"""Evaluation metrics.

These are NumPy re-implementations of the TensorFlow 1 `tf.metrics` functions
used by the original version of this code (`tf.metrics.auc`,
`tf.metrics.*_at_thresholds` and `tf.metrics.mean_squared_error`), so that
results are computed the same way as in the published study.
"""
import numpy as np

KEPSILON = 1e-7


def generate_thresholds(num_thresholds, kepsilon=KEPSILON):
    thresholds = [
        (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
    ]
    return [0.0 - kepsilon] + thresholds + [1.0 - kepsilon]


def confusion_at_thresholds(labels, predictions, thresholds):
    """True/false positives/negatives at each threshold.

    A prediction is positive if it is strictly greater than the threshold.
    Returns four arrays (tp, fp, fn, tn) with one value per threshold.
    """
    labels = np.asarray(labels).reshape(-1) > 0
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    # Shape: [num_thresholds, num_predictions].
    positive = predictions[np.newaxis, :] > thresholds[:, np.newaxis]

    tp = np.sum(positive & labels, axis=1)
    fp = np.sum(positive & ~labels, axis=1)
    fn = np.sum(~positive & labels, axis=1)
    tn = np.sum(~positive & ~labels, axis=1)

    return tp, fp, fn, tn


def specificities_sensitivities(tp, fp, fn, tn, kepsilon=KEPSILON):
    specificities = tn / (tn + fp + kepsilon)
    sensitivities = tp / (tp + fn + kepsilon)
    return specificities, sensitivities


def confusion_matrix(tp, fp, fn, tn):
    """2x2 confusion matrix laid out as [[tp, fp], [fn, tn]]."""
    return np.array([[tp, fp], [fn, tn]], dtype=np.int64)


def auc(labels, predictions, num_thresholds=200, kepsilon=KEPSILON):
    """Area under the ROC curve, approximated like `tf.metrics.auc`.

    Uses a Riemann sum with the trapezoidal rule over `num_thresholds`
    linearly spaced thresholds.
    """
    thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                  for i in range(num_thresholds - 2)]
    thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

    tp, fp, fn, tn = confusion_at_thresholds(labels, predictions, thresholds)

    tpr = tp / (tp + fn + kepsilon)
    fpr = fp / (fp + tn + kepsilon)

    return float(np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + tpr[1:]) / 2.))


def brier_score(labels, predictions):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    return float(np.mean((predictions - labels) ** 2))
