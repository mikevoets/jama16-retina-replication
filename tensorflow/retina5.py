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

# Various constants.
image_dim = 299
num_channels = 3
num_epochs = 5
shuffle_buffer_size = 10000
training_batch_size = 32
validation_batch_size = 32
mode = 'two_labels'

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
optimizer = tf.train.GradientDescentOptimizer(learning_rate) \
                .minimize(loss, global_step)

# Calculate metrics for training.
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_cls, y_true), tf.float32))

# Calculate metrics for validation.
auc, auc_op = tf.metrics.auc(y_true, y_pred)

# Variables for confusion matrix and other metrics. Initialize values with zero.


def true_positives_each(predictions, labels):
    pdb.set_trace()
    tp = tf.Variable(
        tf.zeros(shape(labels.shape[?]), dtype=tf.int64), name='reset_metrics/TruePositives')
    tp_op = tf.assign(tp, tf.add(tp, tf.count_nonzero(labels * predictions, axis=0)))
    return tp, tp_op

def true_negatives_each(p


def create_reset_metric(metric, scope='reset_metrics', **metric_args):
    with tf.variable_scope(scope) as scope:
        metric_op, update_op = metric(**metric_args)
        vars = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars)
    return metric_op, update_op, reset_op


zeros_labels = tf.Constant(tf.zeros(shape=(num_labels), dtype=tf.int64))
reset_op = tf.assign(tp, 
tp = tf.Variable(tf.zeros(shape=(num_labels), dtype=tf.int64), name="tp")
tn = tf.Variable(tf.zeros(shape=(num_labels), dtype=tf.int64), name="tn")
fp = tf.Variable(tf.zeros(shape=(num_labels), dtype=tf.int64), name="fp")
fn = tf.Variable(tf.zeros(shape=(num_labels), dtype=tf.int64), name="fn")

# Operations for calculating total amount of tp, tn, fp and fn.
tp_op = tf.assign(tp, tf.add(tp, tf.count_nonzero(y_true * y_pred_cls, axis=0)))
tn_op = tf.assign(tn, tf.add(tn, tf.count_nonzero((y_true-1) * (y_pred_cls-1), axis=0)))
fp_op = tf.assign(fp, tf.add(fp, tf.count_nonzero(y_true * (y_pred_cls-1), axis=0)))
fn_op = tf.assign(fn, tf.add(fn, tf.count_nonzero((y_true-1) * y_pred_cls, axis=0)))

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
        self.labels = self._labels_tensor()
        self.filenames = self._filenames_tensor()
        self.dataset = self._generate_dataset()
        
        self.steps = ceil(len(self) / self.batch_size)

    def __len__(self):
        if self.steps_set_by_user is True:
            return self.batch_size * self.steps
        else:
            return len(self._paths_to_images())

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
            (self.filenames, self.labels))
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

    def _filenames_tensor(self):
        return tf.constant(self._paths_to_images())

    def _labels_tensor(self):
        return tf.constant(
            [self._find_label(path) for path in self._paths_to_images()],
            tf.float32)

    def set_steps(self, num):
        self.steps = num
        self.steps_set_by_user = True


training_generator = ImageGenerator(
    '../data/eyepacs/jama_dist/train', batch_size=training_batch_size)
validation_generator = ImageGenerator(
    '../data/eyepacs/jama_dist/val', batch_size=validation_batch_size)

training_dataset = training_generator.dataset
validation_dataset = validation_generator.dataset

training_generator.set_steps(20)

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
    m.append(f"Accuracy: {acc:6.4}, Loss: {loss:6.4}")
    
    if batch == num_batches-1:
        end = "\n"

    print(", ".join(m), end=end)


sess = tf.Session()
tf.keras.backend.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# Train for the specified amount of epochs.
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
        sess.run([tp_op, tn_op, fp_op, fn_op, auc_op], feed_dict=feed_dict_validation)
    
    val_confusion_matrix, val_auc = sess.run([confusion_matrix, auc])

    print(val_confusion_matrix, val_auc)
