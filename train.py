import argparse
import os
import random
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

import lib.dataset  # noqa: E402
import lib.evaluation  # noqa: E402
import lib.metrics  # noqa: E402
import lib.model  # noqa: E402
from lib.preprocess import rescale_min_1_to_1  # noqa: E402

print(f"Numpy version: {np.__version__}")
print(f"Tensorflow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

random.seed(432)

# Various loading and saving constants.
default_train_dir = "./data/eyepacs/bin2/train"
default_val_dir = "./data/eyepacs/bin2/validation"
default_save_model_path = "./tmp/model"
default_save_summaries_dir = "./tmp/logs"
default_save_operating_thresholds_path = "./tmp/op_pts.csv"

parser = argparse.ArgumentParser(
                    description="Trains and saves neural network for "
                                "detection of diabetic retinopathy.")
parser.add_argument("-t", "--train_dir",
                    help="path to folder that contains training tfrecords",
                    default=default_train_dir)
parser.add_argument("-v", "--val_dir",
                    help="path to folder that contains validation tfrecords",
                    default=default_val_dir)
parser.add_argument("-sm", "--save_model_path",
                    help="path to where the model should be saved "
                         "('.keras' is appended)",
                    default=default_save_model_path)
parser.add_argument("-ss", "--save_summaries_dir",
                    help="path to folder where summaries should be saved",
                    default=default_save_summaries_dir)
parser.add_argument("-so", "--save_operating_thresholds_path",
                    help="path to where operating points should be saved",
                    default=default_save_operating_thresholds_path)

args = parser.parse_args()
train_dir = str(args.train_dir)
val_dir = str(args.val_dir)
save_model_path = lib.model.model_path_stem(str(args.save_model_path)) \
                  or str(args.save_model_path)
save_summaries_dir = str(args.save_summaries_dir)
save_operating_thresholds_path = str(args.save_operating_thresholds_path)

print("""
Training images folder: {},
Validation images folder: {},
Saving model at: {}.keras,
Saving summaries at: {},
Saving operating points at: {},
""".format(train_dir, val_dir, save_model_path, save_summaries_dir,
           save_operating_thresholds_path))

for path in [save_model_path, save_operating_thresholds_path]:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

# Various constants.
num_channels = 3
normalization_fn = rescale_min_1_to_1

# Hyper-parameters for training.
learning_rate = 1e-3
# Weight decay as reported in the study: L2 regularization of the
#  convolutional and dense kernels, as in the Inception-v3 reference training
#  setup (`weight_decay * sum(w ** 2) / 2`).
weight_decay = 4e-5
# The study does not report these RMSProp settings; these are the defaults of
#  `tf.train.RMSPropOptimizer` in TensorFlow 1.
rho = 0.9
rmsprop_epsilon = 1e-10
train_batch_size = 32

# Hyper-parameters for validation.
num_epochs = 200
wait_epochs = 10
min_delta_auc = 0.01
val_batch_size = 32
num_thresholds = 200

# Define thresholds.
thresholds = lib.metrics.generate_thresholds(num_thresholds)

# Buffer size for image shuffling.
shuffle_buffer_size = 2048
prefetch_buffer_size = tf.data.AUTOTUNE

# Initialize each data set.
train_dataset = lib.dataset.initialize_dataset(
    train_dir, train_batch_size,
    prefetch_buffer_size=prefetch_buffer_size,
    shuffle_buffer_size=shuffle_buffer_size, num_channels=num_channels,
    normalization_fn=normalization_fn)

val_dataset = lib.dataset.initialize_dataset(
    val_dir, val_batch_size,
    prefetch_buffer_size=prefetch_buffer_size,
    num_channels=num_channels,
    normalization_fn=normalization_fn, augmentation=False)

# Inception-v3 with a dense layer with one output (the logit).
# Weights are pre-initialized from Inception-v3 trained on ImageNet.
model = lib.model.build_model(weights="imagenet")

optimizer = keras.optimizers.RMSprop(
    learning_rate=learning_rate, rho=rho, epsilon=rmsprop_epsilon)

# Kernels of the convolutional and dense layers, to which weight decay is
#  applied (not to biases or batch normalization parameters).
regularized_weights = [
    layer.kernel for layer in model.layers
    if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense))]


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        # Retrieve loss of network using cross entropy.
        mean_xentropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits))
        l2_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(w) for w in regularized_weights])
        loss = mean_xentropy + l2_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    squared_errors = tf.square(tf.sigmoid(logits) - labels)
    return mean_xentropy, tf.reduce_sum(squared_errors), tf.size(labels)


def print_training_status(epoch, num_epochs, batch_num, xent, i_step=None):
    def length(x): return len(str(x))

    m = []
    m.append(
        f"Epoch: {{0:>{length(num_epochs)}}}/{{1:>{length(num_epochs)}}}"
        .format(epoch, num_epochs))
    m.append(f"Batch: {batch_num:>4}, Xent: {xent:6.4}")

    if i_step is not None:
        m.append(f"Step: {i_step:>10}")

    print(", ".join(m), end="\r")


summary_writer = tf.summary.create_file_writer(
    os.path.join(save_summaries_dir, "validation"))

# Train for the specified amount of epochs.
# Can be stopped early if peak of validation auc (Area under curve)
#  is reached.
latest_peak_auc = 0
waited_epochs = 0

for epoch in range(num_epochs):
    # Track brier score for an indication on convergence.
    sum_squared_errors = 0.
    num_examples = 0

    for batch_num, (images, labels) in enumerate(train_dataset):
        # Optimize cross entropy.
        batch_xent, batch_sse, batch_size = train_step(images, labels)
        sum_squared_errors += float(batch_sse)
        num_examples += int(batch_size)

        # Print a nice training status.
        print_training_status(
            epoch, num_epochs, batch_num, float(batch_xent),
            int(optimizer.iterations.numpy()))

    train_brier = sum_squared_errors / max(num_examples, 1)
    print("\nEnd of epoch {0}! (Brier: {1:8.6})".format(epoch, train_brier))

    # Perform validation.
    val_predictions, val_labels = lib.evaluation.predict(model, val_dataset)
    val_auc = lib.evaluation.report(val_labels, val_predictions)

    with summary_writer.as_default():
        tf.summary.scalar('auc', val_auc, step=epoch)
        tf.summary.scalar('brier', train_brier, step=epoch)

    if val_auc < latest_peak_auc + min_delta_auc:
        # Stop early if peak of val auc has been reached.
        # If it is lower than the previous auc value, wait up to `wait_epochs`
        #  to see if it does not increase again.

        if wait_epochs == waited_epochs:
            print("Stopped early at epoch {0} with saved peak auc {1:10.8}"
                  .format(epoch+1, latest_peak_auc))
            break

        waited_epochs += 1
    else:
        latest_peak_auc = val_auc
        print(f"New peak auc reached: {val_auc:10.8}")

        # Save the model.
        model.save(f"{save_model_path}.keras")

        # Reset waited epochs.
        waited_epochs = 0

summary_writer.close()

# Load the best saved model.
model, _ = lib.model.load_model(save_model_path)

# Get predictions of all data of our training set, without augmentation.
train_eval_dataset = lib.dataset.initialize_dataset(
    train_dir, val_batch_size,
    prefetch_buffer_size=prefetch_buffer_size,
    num_channels=num_channels,
    normalization_fn=normalization_fn, augmentation=False)

train_predictions, train_labels = lib.evaluation.predict(
    model, train_eval_dataset)

# Write sensitivities and specificities to file.
lib.evaluation.write_operating_thresholds(
    save_operating_thresholds_path, train_labels, train_predictions,
    thresholds)

sys.exit(0)
