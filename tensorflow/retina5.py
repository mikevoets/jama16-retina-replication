import numpy as np
import tensorflow as tf
import pdb
import os
import random
from glob import glob
from math import ceil

print(f"Numpy version: {np.__version__}")
print(f"Tensorflow version: {tf.__version__}")

tf.logging.set_verbosity(tf.logging.INFO)
random.seed(432)

# Various constants.
training_images_dir = '../data/eyepacs/jama_dist/train'
validation_images_dir = '../data/eyepacs/jama_dist/val'
image_dim = 299
num_channels = 3

# Maximum number of epochs. Can be stopped early.
num_epochs = 10

# Buffer size for image shuffling.
shuffle_buffer_size = 10000

# Batch sizes.
training_batch_size = 32
validation_batch_size = 32

# Training and predicting mode.
mode = 'one_label'

# Various hyper-parameter variables.
learning_rate = 3e-4

# Other tensors.
global_step = tf.Variable(
    initial_value=0, name='global_step', trainable=False)

# Input tensors.
x = tf.placeholder(
    tf.float32, shape=(None, image_dim, image_dim, num_channels), name='x')

# Create placeholder for label classes.
y_orig_cls = tf.placeholder(tf.float32, shape=(None), name='y_orig_cls')

# Set variable according to specified mode.
#  'one_label' creates an a one-label y_true.
#  'two_labels' creates a two-label y_true.
if mode == 'one_label':
    num_labels = 1
elif mode == 'two_labels':
    num_labels = 2
else:
    TypeError('invalid mode: choose either one_label or two_labels')

# The label classes are in a range of 0 to 4 (no DR towards proliferative DR).
# Convert the classes to a binary label where class 0 and 1 is interpreted
#  as 0; and class 2, 3 and 4 are interpreted as 1.
y_true = tf.reshape(
    tf.cast(
        tf.greater_equal(y_orig_cls, tf.constant(2.0)), tf.float32,
        name='y_true'),
    shape=[-1, 1])

# The optional second binary label is 0 if class is 0, 1 or 2; and 1 if higher.
if mode == 'two_labels':
    second_label = tf.reshape(
        tf.cast(
            tf.greater_equal(y_orig_cls, tf.constant(3.0)), tf.float32),
        shape=[-1, 1])

    # Add the second label to the first label.
    y_true = tf.reshape(
        tf.stack([y_true, second_label], axis=2), shape=[-1, 2])

# Base model InceptionV3 without top and global average pooling.
base_model = tf.keras.applications.InceptionV3(
    include_top=False, weights='imagenet', input_tensor=x, pooling='avg')

# Add dense layer with the same amount of neurons as labels.
logits = tf.layers.dense(base_model.layers[-1].output, units=num_labels)

# Get the predictions with a sigmoid activation function.
y_pred = tf.sigmoid(logits, name='y_pred')

# Predicted classes for labels.
y_pred_cls = tf.round(y_pred, name='y_pred_cls')

# Retrieve loss of network.
loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits))

# Use stochastic gradient descent for optimizing.
optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=0.9, use_nesterov=True) \
                .minimize(loss, global_step)

# Calculate metrics for training.
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_cls, y_true), tf.float32))


def true_positives_each(labels, predictions):
    tp = tf.Variable(
        tf.zeros((labels.shape[-1]), dtype=tf.int64))
    tp_op = tf.assign(
        tp, tf.add(tp, tf.count_nonzero(labels * predictions, axis=0)))
    return tp, tp_op


def true_negatives_each(labels, predictions):
    tn = tf.Variable(
        tf.zeros((labels.shape[-1]), dtype=tf.int64))
    tn_op = tf.assign(
        tn, tf.add(tn, tf.count_nonzero((labels-1) * (predictions-1), axis=0)))
    return tn, tn_op


def false_positives_each(labels, predictions):
    fp = tf.Variable(
        tf.zeros((labels.shape[-1]), dtype=tf.int64))
    fp_op = tf.assign(
        fp, tf.add(fp, tf.count_nonzero(labels * (predictions-1), axis=0)))
    return fp, fp_op


def false_negatives_each(labels, predictions):
    fn = tf.Variable(
        tf.zeros((labels.shape[-1]), dtype=tf.int64))
    fn_op = tf.assign(
        fn, tf.add(fn, tf.count_nonzero((labels-1) * predictions, axis=0)))
    return fn, fn_op


def create_reset_metric(metric, scope='reset_metrics', **metric_args):
    with tf.variable_scope(scope) as scope:
        metric_op, update_op = metric(**metric_args)
        vars = tf.contrib.framework.get_variables(scope)
        reset_op = tf.variables_initializer(vars)
    return metric_op, update_op, reset_op

# Calculate metrics for validation.
mse, update_mse_op, reset_mse_op = create_reset_metric(
    tf.metrics.mean_squared_error, scope='mse',
    labels=y_true, predictions=y_pred_cls)

auc, update_auc_op, reset_auc_op = create_reset_metric(
    tf.metrics.auc, scope='auc', labels=y_true, predictions=y_pred)

tp, update_tp_op, reset_tp_op = create_reset_metric(
    true_positives_each, scope='true_positives',
    labels=y_true, predictions=y_pred_cls)

tn, update_tn_op, reset_tn_op = create_reset_metric(
    true_negatives_each, scope='true_negatives',
    labels=y_true, predictions=y_pred_cls)

fp, update_fp_op, reset_fp_op = create_reset_metric(
    false_positives_each, scope='false_positives',
    labels=y_true, predictions=y_pred_cls)

fn, update_fn_op, reset_fn_op = create_reset_metric(
    false_negatives_each, scope='false_negatives',
    labels=y_true, predictions=y_pred_cls)

# Operations for confusion matrix.
confusion_matrix = tf.reshape(
    tf.stack([tp, fp, fn, tn], axis=1), shape=[num_labels, 2, 2])


# Data batcher.
class ImageGenerator():
    def __init__(self, images_dir, batch_size, shuffle=True,
                 preprocess_py_fn=None, preprocess_tf_fn=None):
        self.steps_set_by_user = False
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.do_shuffle = shuffle
        self.preprocess_py_fn = preprocess_py_fn
        self.preprocess_tf_fn = preprocess_tf_fn

        self.classes = self._find_classes()
        self.class_dict = self._generate_class_dict()
        self.paths_to_images = self._paths_to_images()
        self.path_tensor = self._path_tensor()
        self.label_tensor = self._label_tensor()
        self.dataset = self._generate_dataset()

        self.steps = ceil(len(self) / self.batch_size)

    def __len__(self):
        if self.steps_set_by_user is True:
            return self.batch_size * self.steps
        else:
            return len(self.paths_to_images)

    def _paths_to_images(self):
        return glob(os.path.join(self.images_dir, "*/*.jpeg"))

    def _find_classes(self):
        return sorted(
            [name for name in os.listdir(self.images_dir)
             if os.path.isdir(os.path.join(self.images_dir, name))])

    def _find_label(self, filename):
        return self.class_dict[filename.split("/")[-2]]

    def _generate_class_dict(self):
        return dict(zip(self.classes, range(len(self.classes))))

    def _generate_dataset(self):
        def _read_image(filename, label):
            image_string = tf.read_file(filename)
            image = tf.image.convert_image_dtype(
                tf.image.decode_image(image_string), tf.float32)
            return image, label

        dataset = tf.data.Dataset.from_tensor_slices(
            (self.path_tensor, self.label_tensor))
        dataset = dataset.map(_read_image)

        if self.preprocess_py_fn is not None:
            dataset = dataset.map(
                lambda filename, label: tuple(tf.py_func(
                    self.preprocess_py_fn, [filename, label],
                    [tf.uint8, label.dtype])))

        if self.preprocess_tf_fn is not None:
            dataset = dataset.map(self.preprocess_tf_fn)

        if self.do_shuffle is True:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        dataset = dataset.batch(self.batch_size)
        return dataset

    def _path_tensor(self):
        return tf.constant(self.paths_to_images)

    def _label_tensor(self):
        return tf.constant(
            [self._find_label(path) for path in self.paths_to_images],
            tf.float32)

    def set_steps(self, num):
        self.steps = num
        self.steps_set_by_user = True


training_generator = ImageGenerator(
    training_images_dir, batch_size=training_batch_size)
validation_generator = ImageGenerator(
    validation_images_dir, batch_size=validation_batch_size)

training_dataset = training_generator.dataset
validation_dataset = validation_generator.dataset

print("Training: {0} images found of {1} classes."
      .format(len(training_generator), len(training_generator.classes)))
print("Validation: {0} images found of {1} classes."
      .format(len(validation_generator), len(validation_generator.classes)))

iterator = tf.data.Iterator.from_structure(
    training_dataset.output_types, training_dataset.output_shapes)

next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)


def print_training_status(epoch, num_epochs, batch, num_batches, acc, loss):
    def length(x): return len(str(x))
    end = "\r"
    m = []
    m.append(
        f"Epoch: {{0:>{length(num_epochs)}}}/{{1:>{length(num_epochs)}}}"
        .format(epoch+1, num_epochs))
    m.append(
        f"Step: {{0:>{length(num_batches)}}}/{{1:>{length(num_batches)}}}"
        .format(batch+1, num_batches))
    m.append(f"Acc: {acc:6.4}, Xent: {loss:6.4}")

    if batch == num_batches-1:
        end = ", "

    print(", ".join(m), end=end)


sess = tf.Session()
tf.keras.backend.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# Train for the specified amount of epochs.
# Can be stopped early if peak of validation auc (Area under curve)
#  is reached.
previous_auc = 0
waited_epochs = 0

for epoch in range(num_epochs):
    # Start training.
    sess.run(training_init_op)

    for step in range(training_generator.steps):
        # Retrieve a batch of training data.
        images, labels = sess.run(next_element)

        # Create a feed dictionary for the input data.
        feed_dict_training = {
            x: images, y_orig_cls: labels,
            tf.keras.backend.learning_phase(): 1}

        # Optimize loss.
        i_global, _, batch_acc, batch_loss = sess.run(
            [global_step, optimizer, accuracy, loss],
            feed_dict=feed_dict_training)

        # Print a nice training status.
        print_training_status(
            epoch, num_epochs, step, training_generator.steps,
            batch_acc, batch_loss)

    # Validation.
    sess.run(validation_init_op)

    for _ in range(validation_generator.steps):
        # Retrieve a batch of validation data.
        images, labels = sess.run(next_element)

        # Validate the current classifier against validation set.
        feed_dict_validation = {x: images,
                                y_orig_cls: labels,
                                tf.keras.backend.learning_phase(): 0}

        # Retrieve the validation set confusion metrics.
        sess.run(
            [update_tp_op, update_tn_op, update_fp_op, update_fn_op,
             update_auc_op, update_mse_op],
            feed_dict=feed_dict_validation)

    # Retrieve confusion matrix and estimated roc auc score.
    val_confusion_matrix, val_mse, val_auc = sess.run(
            [confusion_matrix, mse, auc])

    # Print total roc auc score for validation.
    print(f"Val mse: {val_mse:6.4}, Val auc: {val_auc:6.4}")

    # Print confusion matrix for each label.
    for i in range(num_labels):
        print(f"Confusion matrix for label {i+1}:")
        print(val_confusion_matrix[i])

    # Reset all streaming variables.
    sess.run(
        [reset_tp_op, reset_tn_op, reset_fp_op, reset_fn_op,
         reset_mse_op, reset_auc_op])

    if val_auc < previous_auc:
        # Stop early if peak of val auc has been reached.
        # If it is lower than the previous auc value, wait up to `wait_epochs`
        #  to see if it does not increase again.

        if wait_epochs == waited_epochs:
            print("Stopped early at epoch {0} with saved peak auc {1:6.4}"
                  .format(epoch+1, latest_peak_auc))
            break

        waited_epochs += 1
    else:
        latest_peak_auc = val_auc
        # Save the model weights.

# Close the session.
sess.close()
