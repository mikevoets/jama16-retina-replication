import numpy as np
import tensorflow as tf
import pdb
import os
import random
from glob import glob
from math import ceil

import metrics

print(f"Numpy version: {np.__version__}")
print(f"Tensorflow version: {tf.__version__}")

tf.logging.set_verbosity(tf.logging.INFO)
random.seed(432)

# Various constants.
training_records_dir = '../data/eyepacs/jama_dist_train'
validation_records_dir = '../data/eyepacs/jama_dist_validation'
test_records_dir = '../data/eyepacs/jama_dist_test'

image_dim = 299
num_channels = 3
num_workers = 8
num_labels = 1

# Maximum number of epochs. Can be stopped early.
num_epochs = 200

# Batch sizes.
training_batch_size = 1
validation_batch_size = 1

# Buffer size for image shuffling.
shuffle_buffer_size = 100  # 100 * training_batch_size

# Various hyper-parameter variables.
learning_rate = 3e-3

tf.keras.backend.set_learning_phase(False)
tf.keras.backend.set_image_data_format('channels_first')


def parse_example(example_proto):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string),
                "image/format": tf.FixedLenFeature((), tf.string),
                "image/class/label": tf.FixedLenFeature((), tf.int64),
                "image/height": tf.FixedLenFeature((), tf.int64),
                "image/width": tf.FixedLenFeature((), tf.int64)}
    parsed = tf.parse_single_example(example_proto, features)

    image = tf.image.convert_image_dtype(
        tf.image.decode_jpeg(parsed["image/format"]))
    label = tf.cast(parsed["image/class/label"], tf.int32)
    return image, label


def _tfrecord_dataset_from_folder(folder, ext='.tfrecord'):
    tfrecords = [os.path.join(folder, n)
                 for n in os.listdir(folder) if n.endswith(ext)]
    return tf.data.TFRecordDataset(tfrecords)


def _parse_example(proto):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string),
                "image/format": tf.FixedLenFeature((), tf.string),
                "image/class/label": tf.FixedLenFeature((), tf.int64),
                "image/height": tf.FixedLenFeature((), tf.int64),
                "image/width": tf.FixedLenFeature((), tf.int64)}
    parsed = tf.parse_single_example(proto, features)

    # Rescale to 1./255.
    image = tf.image.convert_image_dtype(
        tf.image.decode_jpeg(parsed["image/encoded"]), tf.float32)

    # Reshape such that channels come first.
    image = tf.reshape(image, [num_channels, image_dim, image_dim])

    label = tf.cast(parsed["image/class/label"], tf.int32)

    return {'x': image}, label


def initialize_dataset(image_dir, batch_size, num_epochs=1,
                       num_workers=None, prefetch_buffer_size=None,
                       shuffle_buffer_size=None):
    dataset = _tfrecord_dataset_from_folder(image_dir) \
                    .map(_parse_example, num_parallel_calls=num_workers)

    if prefetch_buffer_size is not None:
        dataset = dataset.prefetch(prefetch_buffer_size)

    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    return dataset


def dataset_input_fn(dataset):
    return dataset.make_one_shot_iterator().get_next()


def model_fn(features, labels, mode, params):
    # Base model InceptionV3 without top and global average pooling.
    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet', input_tensor=features['x'],
        pooling='avg')

    # Add dense layer with the same amount of neurons as labels.
    logits = tf.layers.dense(base_model.output, units=params["num_labels"])

    # Get the predictions with a sigmoid activation function.
    predictions = tf.sigmoid(logits)

    # Get the class predictions for labels.
    class_predictions = tf.round(predictions)

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.keras.backend.set_learning_phase(False)

        specs = {"mode": mode,
                 "predictions": {"Moderate DR+": class_predictions[:, 0]}}

        if params["num_labels"] == 2:
            spec["predictions"]["Severe DR+"] = class_predictions[:, 1]

        return tf.estimator.EstimatorSpec(**specs)

    tf.keras.backend.set_learning_phase(True)

    # The label classes are in a range of 0 to 4 (no DR towards
    #  proliferative DR).
    # Convert the classes to a binary label where class 0 and 1 is interpreted
    #  as 0; and class 2, 3 and 4 are interpreted as 1.
    y = tf.cast(
        tf.reshape(tf.greater_equal(labels, tf.constant(2)), [-1, 1]),
        tf.float32)

    # The optional second binary label is 0 if class is 0, 1 or 2;
    #  and 1 if higher.
    if params["num_labels"] == 2:
        second_label = tf.cast(
            tf.reshape(tf.greater_equal(labels, tf.constant(3)), [-1, 1]),
            tf.float32)

        # Add the second label to the first label.
        y = tf.reshape(tf.stack([y, second_label], axis=2), shape=[-1, 2])

    # Retrieve loss of network using cross entropy.
    loss = tf.losses.sigmoid_cross_entropy(y, logits)

    # Calculate other metrics.
    eval_metric_ops = {
        "auc": tf.metrics.auc(y, predictions, num_thresholds=2),
        "confusion_matrix": metrics.confusion_matrix(y, class_predictions),
        "rmse": tf.metrics.root_mean_squared_error(y, class_predictions)
    }

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=params["learning_rate"], momentum=params["momentum"],
        use_nesterov=params["nesterov"])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


model_params = {
    "num_labels": num_labels,
    "learning_rate": learning_rate,
    "momentum": 0.9,
    "nesterov": True,
}

nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

training_dataset = initialize_dataset(
    training_records_dir, training_batch_size, num_epochs,
    num_workers=num_workers, prefetch_buffer_size=100 * training_batch_size,
    shuffle_buffer_size=1000)

nn.train(input_fn=lambda: dataset_input_fn(training_dataset))
