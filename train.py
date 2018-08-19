import numpy as np
import tensorflow as tf
import os
import random
import sys
import argparse
import csv
from glob import glob
import lib.metrics
import lib.dataset
import lib.evaluation
from lib.preprocess import rescale_min_1_to_1, rescale_0_to_1

print(f"Numpy version: {np.__version__}")
print(f"Tensorflow version: {tf.__version__}")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
                    help="path to where graph model should be saved",
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
save_model_path = str(args.save_model_path)
save_summaries_dir = str(args.save_summaries_dir)
save_operating_thresholds_path = str(args.save_operating_thresholds_path)

print("""
Training images folder: {},
Validation images folder: {},
Saving model and graph checkpoints at: {},
Saving summaries at: {},
Saving operating points at: {},
""".format(train_dir, val_dir, save_model_path, save_summaries_dir,
           save_operating_thresholds_path))

# Various constants.
num_channels = 3
num_workers = 8
normalization_fn = rescale_min_1_to_1

# Hyper-parameters for training.
learning_rate = 1e-3
decay = 4e-5
train_batch_size = 32

# Hyper-parameters for validation.
num_epochs = 200
wait_epochs = 10
min_delta_auc = 0.01
val_batch_size = 32
num_thresholds = 200
kepsilon = 1e-7

# Define thresholds.
thresholds = lib.metrics.generate_thresholds(num_thresholds, kepsilon) + [0.5]

# Buffer size for image shuffling.
shuffle_buffer_size = 2048
prefetch_buffer_size = 2 * train_batch_size

# Set image datas format to channels first if GPU is available.
if tf.test.is_gpu_available():
    print("Found GPU! Using channels first as default image data format.")
    image_data_format = 'channels_first'
else:
    image_data_format = 'channels_last'

# Set up a session and bind it to Keras.
sess = tf.Session()
tf.keras.backend.set_session(sess)
tf.keras.backend.set_learning_phase(True)
tf.keras.backend.set_image_data_format(image_data_format)

# Initialize each data set.
train_dataset = lib.dataset.initialize_dataset(
    train_dir, train_batch_size,
    num_workers=num_workers, prefetch_buffer_size=prefetch_buffer_size,
    shuffle_buffer_size=shuffle_buffer_size,
    image_data_format=image_data_format, num_channels=num_channels,
    normalization_fn=normalization_fn)

val_dataset = lib.dataset.initialize_dataset(
    val_dir, val_batch_size,
    num_workers=num_workers, prefetch_buffer_size=prefetch_buffer_size,
    shuffle_buffer_size=shuffle_buffer_size,
    image_data_format=image_data_format, num_channels=num_channels,
    normalization_fn=normalization_fn)

# Create initializable iterators.
iterator = tf.data.Iterator.from_structure(
    train_dataset.output_types, train_dataset.output_shapes)

images, labels = iterator.get_next()
x = tf.placeholder_with_default(images, images.shape, name='x')
y = tf.placeholder_with_default(labels, labels.shape, name='y')

train_init_op = iterator.make_initializer(train_dataset)
val_init_op = iterator.make_initializer(val_dataset)

# Base model InceptionV3 without top and global average pooling.
base_model = tf.keras.applications.InceptionV3(
    include_top=False, weights='imagenet', pooling='avg', input_tensor=x)

# Add dense layer with the same amount of neurons as labels.
logits = tf.layers.dense(base_model.output, units=1)

# Get the predictions with a sigmoid activation function.
predictions = tf.sigmoid(logits, name='predictions')

# Retrieve loss of network using cross entropy.
mean_xentropy = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

# Define optimizer.
global_step = tf.Variable(0, dtype=tf.int32)

train_op = tf.train.RMSPropOptimizer(
    learning_rate=learning_rate, decay=decay) \
        .minimize(loss=mean_xentropy, global_step=global_step)

# Metrics for finding best validation set.
tp, update_tp, reset_tp = lib.metrics.create_reset_metric(
    tf.metrics.true_positives_at_thresholds, scope='tp',
    labels=y, predictions=predictions, thresholds=thresholds)

fp, update_fp, reset_fp = lib.metrics.create_reset_metric(
    tf.metrics.false_positives_at_thresholds, scope='fp',
    labels=y, predictions=predictions, thresholds=thresholds)

fn, update_fn, reset_fn = lib.metrics.create_reset_metric(
    tf.metrics.false_negatives_at_thresholds, scope='fn',
    labels=y, predictions=predictions, thresholds=thresholds)

tn, update_tn, reset_tn = lib.metrics.create_reset_metric(
    tf.metrics.true_negatives_at_thresholds, scope='tn',
    labels=y, predictions=predictions, thresholds=thresholds)

confusion_matrix = lib.metrics.confusion_matrix(
    tp[-1], fp[-1], fn[-1], tn[-1], scope='confusion_matrix')

brier, update_brier, reset_brier = lib.metrics.create_reset_metric(
    tf.metrics.mean_squared_error, scope='brier',
    labels=y, predictions=predictions)

auc, update_auc, reset_auc = lib.metrics.create_reset_metric(
    tf.metrics.auc, scope='auc', labels=y, predictions=predictions)
tf.summary.scalar('auc', auc)

specificities = tf.div(tn, tn + fp + kepsilon)
sensitivities = tf.div(tp, tp + fn + kepsilon)

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


# Add ops for saving and restoring all variables.
saver = tf.train.Saver()

# Initialize variables.
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
    sess.run(train_init_op)
    batch_num = 0

    # Track brier score for an indication on convergance.
    sess.run(reset_brier)

    try:
        while True:
            # Optimize cross entropy.
            i_global, batch_xent, *_ = sess.run(
                [global_step, mean_xentropy, train_op, update_brier])

            # Print a nice training status.
            print_training_status(
                epoch, num_epochs, batch_num, batch_xent, i_global)

            # Report summaries.
            batch_num += 1

    except tf.errors.OutOfRangeError:
        # Retrieve training brier score.
        train_brier = sess.run(brier)
        print("\nEnd of epoch {0}! (Brier: {1:8.6})".format(epoch, train_brier))

    # Perform validation.
    val_auc = lib.evaluation.perform_test(
        sess=sess, init_op=val_init_op,
        summary_writer=train_writer, epoch=epoch)

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

        # Save the model weights.
        saver.save(sess, save_model_path)

        # Reset waited epochs.
        waited_epochs = 0

# Load the saved best meta graph and restore variables from that checkpoint.
saver = tf.train.import_meta_graph("{}.meta".format(save_model_path))
saver.restore(sess, save_model_path)

# Get predictions of all data of our training set.
tf.keras.backend.set_learning_phase(False)
sess.run([train_init_op, reset_tp, reset_fp, reset_fn, reset_tn])

try:
    while True:
        # Update all confusion metrics for each batch.
        sess.run([update_tp, update_fp, update_fn, update_tn])

except tf.errors.OutOfRangeError:
    pass

# Write sensitivities and specificities to file.
with open(save_operating_thresholds_path, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(['threshold', 'specificity', 'sensitivity'])

    train_specificities, train_sensitivities = \
        sess.run([specificities, sensitivities])

    for idx in range(num_thresholds):
        writer.writerow([
            "{:0.4f}".format(x) for x in [
                thresholds[idx], train_specificities[idx],
                train_sensitivities[idx]]])

# Close the session.
sess.close()
sys.exit(0)
