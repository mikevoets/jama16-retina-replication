import csv

import numpy as np
import tensorflow as tf

import lib.metrics


def predict(model, dataset, batch_statistics=False):
    """Run the model over a dataset of (images, labels) batches.

    Returns a tuple of NumPy arrays `(predictions, labels)`, both with shape
    [num_examples, 1].

    If `batch_statistics` is true, batch normalization layers normalize with
    the statistics of each batch instead of their moving averages. This is
    how the TensorFlow 1 graphs of the published models behaved, and is needed
    to reproduce their results.
    """
    @tf.function(reduce_retracing=True)
    def predict_batch(images):
        return tf.sigmoid(model(images, training=batch_statistics))

    all_predictions = []
    all_labels = []

    for images, labels in dataset:
        all_predictions.append(predict_batch(images).numpy())
        all_labels.append(labels.numpy())

    return np.vstack(all_predictions), np.vstack(all_labels)


def report(labels, predictions, operating_threshold=0.5):
    """Print Brier score, AUC and the confusion matrix. Returns the AUC."""
    brier = lib.metrics.brier_score(labels, predictions)
    auc = lib.metrics.auc(labels, predictions)
    tp, fp, fn, tn = lib.metrics.confusion_at_thresholds(
        labels, predictions, [operating_threshold])

    print(f"Brier score: {brier:6.4}, AUC: {auc:10.8}")
    print(f"Confusion matrix at operating threshold {operating_threshold:0.3f}:")
    print(lib.metrics.confusion_matrix(tp[0], fp[0], fn[0], tn[0]))

    return auc


def write_operating_thresholds(path, labels, predictions, thresholds):
    """Write specificity and sensitivity at each threshold to a csv file."""
    tp, fp, fn, tn = lib.metrics.confusion_at_thresholds(
        labels, predictions, thresholds)
    specificities, sensitivities = \
        lib.metrics.specificities_sensitivities(tp, fp, fn, tn)

    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow(['threshold', 'specificity', 'sensitivity'])

        for idx in range(len(thresholds)):
            writer.writerow([
                "{:0.4f}".format(x) for x in [
                    thresholds[idx], specificities[idx], sensitivities[idx]]])

    return specificities, sensitivities
