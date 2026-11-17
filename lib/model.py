import os

import keras
import tensorflow as tf

IMAGE_SIZE = 299
NUM_CHANNELS = 3

LEGACY_CHECKPOINT_SUFFIXES = (".index", ".meta", ".data-00000-of-00001")


def build_model(weights="imagenet"):
    """Inception-v3 without top, global average pooling and a single logit.

    The model outputs logits; apply a sigmoid to get predictions.
    """
    # Reset Keras' layer name counters so that layers are always named
    #  `conv2d`, `conv2d_1`, ..., `batch_normalization`, ..., `dense`. This
    #  matches the variable names in the checkpoints written by the original
    #  TensorFlow 1 version of this code.
    keras.backend.clear_session()

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name="x")
    base_model = keras.applications.InceptionV3(
        include_top=False, weights=weights, pooling="avg", input_tensor=inputs)
    logits = keras.layers.Dense(1, name="dense")(base_model.output)

    return keras.Model(inputs=inputs, outputs=logits, name="inception_v3_rdr")


def is_legacy_checkpoint(path):
    return os.path.exists(f"{path}.index") and not path.endswith(".keras")


def load_legacy_checkpoint(path):
    """Load a checkpoint written by the TensorFlow 1 version of this code.

    This includes the trained models published at
    https://doi.org/10.6084/m9.figshare.8312183.

    Note that the batch normalization moving statistics in these checkpoints
    were never updated during training (they are still at their initial
    values: mean 0, variance 1). The TensorFlow 1 graph always normalized with
    the statistics of the current batch, also when evaluating, so these models
    must be run with `training=True` to reproduce their published results.
    See `lib.evaluation.predict`.
    """
    reader = tf.train.load_checkpoint(path)
    available = reader.get_variable_to_shape_map()

    model = build_model(weights=None)

    for layer in model.layers:
        for weight in layer.weights:
            key = f"{layer.name}/{weight.name}"
            if key not in available:
                raise ValueError(
                    f"Variable {key} not found in checkpoint {path}.")
            value = reader.get_tensor(key)
            if tuple(value.shape) != tuple(weight.shape):
                raise ValueError(
                    f"Shape mismatch for {key}: checkpoint has {value.shape}, "
                    f"model expects {tuple(weight.shape)}.")
            weight.assign(value)

    return model


def load_model(path):
    """Load a model saved by `train.py`, or a legacy TensorFlow 1 checkpoint.

    Returns a tuple `(model, is_legacy)`.
    """
    if is_legacy_checkpoint(path):
        return load_legacy_checkpoint(path), True

    keras_path = path if path.endswith(".keras") else f"{path}.keras"
    if not os.path.exists(keras_path):
        raise FileNotFoundError(
            f"No model found at {keras_path} or {path}.index.")

    return keras.models.load_model(keras_path, compile=False), False


def model_path_stem(path):
    """Strip model file extensions from a path, e.g. `tmp/model-1.index`."""
    for suffix in (".keras",) + LEGACY_CHECKPOINT_SUFFIXES:
        if path.endswith(suffix):
            return path[:-len(suffix)]
    return None
