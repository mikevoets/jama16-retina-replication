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
shuffle_buffer_size = 100
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

# Calculate metrics and streaming metrics operations.
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_cls, y_true), tf.float32))


# Data batcher.
class ImageGenerator():
    def __init__(self, images_dir, batch_size, shuffle=True,
                 preprocess_py_fn=None, preprocess_tf_fn=None):
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.do_shuffle = shuffle
        self.preprocess_py_fn = preprocess_py_fn
        self.preprocess_tf_fn = preprocess_tf_fn

        self.class_dict = self._generate_class_dict()
        self.dataset = self._generate_dataset()

    def __len__(self):
        return len(self._paths_to_images())

    def _paths_to_images(self):
        return glob(os.path.join(self.images_dir, "*/*.jpeg"))

    def _find_label(self, filename):
        return self.class_dict[os.path.basename(
            os.path.normpath(os.path.join(filename, os.pardir)))]

    def _generate_class_dict(self):
        classes = self.classes()
        return dict(zip(classes, range(len(classes))))

    def _generate_dataset(self):
        def _read_image(filename, label):
            image_string = tf.read_file(filename)
            image = tf.image.convert_image_dtype(
                tf.image.decode_image(image_string), tf.float32)
            return image, label

        dataset = tf.data.Dataset.from_tensor_slices(
            (self.filenames(), self.labels()))
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

    def classes(self):
        return sorted(
            [name for name in os.listdir(self.images_dir)
             if os.path.isdir(os.path.join(self.images_dir, name))])

    def filenames(self):
        return tf.constant(self._paths_to_images())

    def labels(self):
        return tf.constant(
            [self._find_label(path) for path in self._paths_to_images()],
            tf.float32)


training_generator = ImageGenerator(
    '../data/eyepacs/jama_dist/train', batch_size=training_batch_size)
validation_generator = ImageGenerator(
    '../data/eyepacs/jama_dist/val', batch_size=validation_batch_size,
    shuffle=False)

training_dataset = training_generator.dataset
validation_dataset = validation_generator.dataset

iterator = tf.data.Iterator.from_structure(
    training_dataset.output_types, training_dataset.output_shapes)

next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

sess = tf.Session()
tf.keras.backend.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# Train for 5 epochs.
for epoch in range(5):
    # Start training.
    sess.run(training_init_op)

    while True:
        try:
            # Retrieve a batch of training data.
            images, labels = sess.run(next_element)

            # Create a feed dictionary for the input data.
            feed_dict_training = {
                x: images, y_orig_cls: labels,
                tf.keras.backend.learning_phase(): 1}

            # Optimize loss.
            i_global, _ = sess.run(
                [global_step, optimizer], feed_dict=feed_dict_training)

            if i_global % 10 == 0:
                # Calculate the current accuracy on the training batch.
                batch_acc, batch_loss = sess.run(
                    [accuracy, loss], feed_dict=feed_dict_training)

                print("Epoch: {0:>3}, Step: {1:>6}, "
                      "Accuracy: {2:>3.1%}, Loss: {3:6.4}"
                      .format(epoch, i_global, batch_acc, batch_loss))
        except tf.errors.OutOfRangeError:
            break

    # Validation.
    sess.run(validation_init_op)
    # Retrieve a batch of validation data.
    images, labels = sess.run(next_element)
    # Validate the current classifier against validation set.
    feed_dict_validation = {x: images,
                            y_orig_cls: labels,
                            tf.keras.backend.learning_phase(): 0}

    # Retrieve the accuracy and loss on the validation set.
    validation_acc, validation_loss = sess.run(
        [accuracy, loss], feed_dict=feed_dict_validation)

    print("Epoch: {0:>3}, Validation accuracy: {1:>3.1%}, "
          "Validation loss: {2:6.4}"
          .format(epoch, validation_acc, validation_loss))
