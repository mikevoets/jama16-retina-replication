import numpy as np
import tensorflow as tf
import pdb
import os
import random
import re
from glob import glob
from math import ceil

import metrics

print(f"Numpy version: {np.__version__}")
print(f"Tensorflow version: {tf.__version__}")

tf.logging.set_verbosity(tf.logging.INFO)
random.seed(432)

# Various loading and saving constants..
training_records_dir = '../data/eyepacs/jama_dist_train'
validation_records_dir = '../data/eyepacs/jama_dist_validation'
test_records_dir = '../data/eyepacs/jama_dist_test'

save_model_path = "./tmp/model-2-labels-1.ckpt"
save_summaries_dir = "./tmp/logs-2-labels"

# Various training and evaluation constants.
image_dim = 299
num_channels = 3
num_labels = 2
wait_epochs = 10
num_workers = 8
#num_summarize_layers = 4
#report_per_step = 100
mode = 'test'

# Maximum number of epochs. Can be stopped early.
num_epochs = 200

# Batch sizes.
training_batch_size = 32
validation_batch_size = 32
test_batch_size = 16

# Buffer size for image shuffling.
shuffle_buffer_size = 5000
prefetch_buffer_size = 100 * training_batch_size

# Various hyper-parameter variables.
learning_rate = 3e-3

# Set image datas format to channels first if GPU is available.
if tf.test.is_gpu_available():
    print("Found GPU! Using channels first as default image data format.")
    image_data_format = 'channels_first'
    image_shape = [num_channels, image_dim, image_dim]
else:
    image_data_format = 'channels_last'
    image_shape = [image_dim, image_dim, num_channels]


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

    image = tf.reshape(image, image_shape)
    label = tf.cast(parsed["image/class/label"], tf.int32)

    return image, label


def initialize_dataset(image_dir, batch_size, num_epochs=1,
                       num_workers=None, prefetch_buffer_size=None,
                       shuffle_buffer_size=None):
    # Retrieve data set from pattern.
    dataset = _tfrecord_dataset_from_folder(image_dir)
    dataset = dataset.map(_parse_example, num_parallel_calls=num_workers)

    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    if prefetch_buffer_size is not None:
        dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# Set up a session and bind it to Keras.
sess = tf.Session()
tf.keras.backend.set_session(sess)
tf.keras.backend.set_learning_phase(True)
tf.keras.backend.set_image_data_format(image_data_format)

# Initialize each data set.
training_dataset = initialize_dataset(
    training_records_dir, training_batch_size,
    num_workers=num_workers, prefetch_buffer_size=prefetch_buffer_size,
    shuffle_buffer_size=shuffle_buffer_size)

validation_dataset = initialize_dataset(
    validation_records_dir, validation_batch_size,
    num_workers=num_workers, prefetch_buffer_size=prefetch_buffer_size,
    shuffle_buffer_size=shuffle_buffer_size)

# Create an initialize iterators.
iterator = tf.data.Iterator.from_structure(
    training_dataset.output_types, training_dataset.output_shapes)

images, labels = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

if mode == 'test':
    # Evaluate saved model.
    test_dataset = initialize_dataset(
        test_records_dir, test_batch_size,
        num_workers=num_workers, prefetch_buffer_size=prefetch_buffer_size,
        shuffle_buffer_size=shuffle_buffer_size)

    test_init_op = iterator.make_initializer(test_dataset)

# Base model InceptionV3 without top and global average pooling.
base_model = tf.keras.applications.InceptionV3(
    include_top=False, weights='imagenet', input_tensor=images, pooling='avg')

# Add dense layer with the same amount of neurons as labels.
with tf.name_scope('logits'):
    logits = tf.layers.dense(base_model.output, units=num_labels)

# Get the predictions with a sigmoid activation function.
with tf.name_scope('predictions'):
    predictions = tf.sigmoid(logits)

# Get the class predictions for labels.
predictions_classes = tf.round(predictions)

# The label classes are in a range of 0 to 4 (no DR towards
#  proliferative DR).
# Convert the classes to a binary label where class 0 and 1 is interpreted
#  as 0; and class 2, 3 and 4 are interpreted as 1.
y = tf.cast(
    tf.reshape(tf.greater_equal(labels, tf.constant(2)), [-1, 1]), tf.float32)

if num_labels == 2:
    # The optional second binary label is 0 if class is 0, 1 or 2;
    #  and 1 if higher.
    second_label = tf.cast(
        tf.reshape(tf.greater_equal(labels, tf.constant(3)), [-1, 1]),
        tf.float32)

    # Add the second label to the first label.
    y = tf.reshape(tf.stack([y, second_label], axis=2), shape=[-1, 2])

# Retrieve loss of network using cross entropy.
mean_xentropy = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

# Define SGD optimizer with momentum and nesterov.
global_step = tf.Variable(0, dtype=tf.int32)

train_op = tf.train.MomentumOptimizer(
    learning_rate, momentum=0.9, use_nesterov=True) \
        .minimize(loss=mean_xentropy, global_step=global_step)


# Metrics for finding best validation set.
tp, update_tp, reset_tp = metrics.create_reset_metric(
    metrics.true_positives, scope='tp', labels=y,
    predictions=predictions_classes, num_labels=num_labels)

fp, update_fp, reset_fp = metrics.create_reset_metric(
    metrics.false_positives, scope='fp', labels=y,
    predictions=predictions_classes, num_labels=num_labels)

fn, update_fn, reset_fn = metrics.create_reset_metric(
    metrics.false_negatives, scope='fn', labels=y,
    predictions=predictions_classes, num_labels=num_labels)

tn, update_tn, reset_tn = metrics.create_reset_metric(
    metrics.true_negatives, scope='tn', labels=y,
    predictions=predictions_classes, num_labels=num_labels)

confusion_matrix = metrics.confusion_matrix(
    tp, fp, fn, tn, num_labels=num_labels)

brier, update_brier, reset_brier = metrics.create_reset_metric(
    tf.metrics.mean_squared_error, scope='brier',
    labels=y, predictions=predictions)

auc, update_auc, reset_auc = metrics.create_reset_metric(
    tf.metrics.auc, scope='auc',
    labels=y, predictions=predictions)
tf.summary.scalar('auc', auc)


# Merge all the summaries and write them out.
summaries_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(save_summaries_dir + "/train")
test_writer = tf.summary.FileWriter(save_summaries_dir + "/test")


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


def perform_test(init_op, summary_writer=None, epoch=None):
    tf.keras.backend.set_learning_phase(False)
    sess.run(init_op)

    # Reset all streaming variables.
    sess.run([reset_tp, reset_fp, reset_fn, reset_tn, reset_brier, reset_auc])

    try:
        while True:
            # Retrieve the validation set confusion metrics.
            sess.run([update_tp, update_fp, update_fn,
                      update_tn, update_brier, update_auc])

    except tf.errors.OutOfRangeError:
        pass

    # Retrieve confusion matrix and estimated roc auc score.
    test_conf_matrix, test_brier, test_auc, summaries = sess.run(
        [confusion_matrix, brier, auc, summaries_op])

    # Write summary.
    if summary_writer is not None:
        summary_writer.add_summary(summaries, epoch)

    # Print total roc auc score for validation.
    print(f"Brier score: {test_brier:6.4}, AUC: {test_auc:10.8}")

    # Print confusion matrix for each label.
    for i in range(num_labels):
        print(f"Confusion matrix for label {i+1}:")
        print(test_conf_matrix[i])


# Add ops for saving and restoring all variables.
saver = tf.train.Saver()

# Initialize session.
if mode == 'test':
    saver.restore(sess, save_model_path)
    perform_test(init_op=test_init_op)

    sess.close()
    sys.exit(0)
else:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())


# Train for the specified amount of epochs.
# Can be stopped early if peak of validation auc (Area under curve)
#  is reached.
latest_peak_auc = 0
waited_epochs = 0

for epoch in range(num_epochs):
    # Start training.
    tf.keras.backend.set_learning_phase(True)
    sess.run(training_init_op)
    batch_num = 0

    try:
        while True:
            # Optimize cross entropy.
            i_global, batch_xent, _ = sess.run(
                [global_step, mean_xentropy, train_op])

            # Print a nice training status.
            print_training_status(
                epoch, num_epochs, batch_num, batch_xent, i_global)

            # Report summaries.
            batch_num += 1

    except tf.errors.OutOfRangeError:
        print(f"\nEnd of epoch {epoch}!")

    perform_test(init_op=validation_init_op, summary_writer=train_writer,
                 epoch=epoch)

    if val_auc < latest_peak_auc:
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

        # Save the model weights.
        saver.save(sess, save_model_path)

        # Reset waited epochs.
        waited_epochs = 0

# Close the session.
sess.close()
