import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
import os
import utils.images as im
import utils.convnets.layers as la
import csv
import argparse

# Development
import pdb
#

# Parsing arguments
parser = argparse.ArgumentParser(description='Retina training tool',
                                 prog='main.py')
parser.add_argument('-nt', '--num-threads',
                    help='Number of threads', nargs=1, type=int, default=[1])
parser.add_argument('-tbs', '--test-batch-size',
                    help='Test batch size', nargs=1, type=int, default=[16])
parser.add_argument('-trbs', '--train-batch-size',
                    help='Train batch size', nargs=1, type=int, default=[16])
parser.add_argument('-mad', '--min-after-dequeue',
                    help='Minimal buffer', nargs=1, type=int, default=[10])
parser.add_argument('-i', '--iterations',
                    help='Iterations', nargs=1, type=int, default=[100])
args = vars(parser.parse_args())

num_threads = args['num_threads'][0]
test_batch_size = args['test_batch_size'][0]
train_batch_size = args['train_batch_size'][0]
min_after_dequeue = args['min_after_dequeue'][0]
num_iterations = args['num_iterations'][0]

# Image locations
train_dir = './data/train/'
test_dir = './data/test/'

# Label locations
train_labels_fn = './data/trainLabels.csv'
test_labels_fn = './data/testLabels.csv'

partial_train_labels_fn = './data/trainLabels.part.csv'
partial_test_labels_fn = './data/testLabels.part.csv'

# Define image size
im_size = 256

# Convolutional Layer 1
filter_size1 = 5        # Convolution filters are 5 x 5 pixels
num_filters1 = 16       # There are 16 of these filters

# Convolutional Layer 2
filter_size2 = 5        # Convolution filters are 5 x 5 pixels
num_filters2 = 36       # There are 36 of these filters

# Fully connected Layer
fc_size = 128           # Number of neurons in fully connected layer

# Tuple with height and width of images used to reshape arrays
im_shape = (im_size, im_size)

# Multiple of height and with of images
im_size_flat = im_size * im_size

# Number of color channels for the images: rgb (3)
num_channels = 3

# Number of classes, one class for each grade scale
num_classes = 5

# Tensors for images
x = tf.placeholder(tf.float32, shape=[None, im_size_flat], name='x')
x_image = tf.reshape(x, [-1, im_size, im_size, num_channels])

# Tensors for labels
y_true = tf.placeholder(
    tf.float32, shape=[None, num_classes], name='y_true')
# Placeholder variable for the class-number (using argmax)
y_true_cls = tf.argmax(y_true, axis=1)

# Convolutional Layer 1
layer_conv1, weights_conv1 = \
    la.new_conv_layer(input=x_image,
                      num_input_channels=num_channels,
                      filter_size=filter_size1,
                      num_filters=num_filters1,
                      use_pooling=True)

# Convolutional Layer 2
layer_conv2, weights_conv2 = \
    la.new_conv_layer(input=layer_conv1,
                      num_input_channels=num_filters1,
                      filter_size=filter_size2,
                      num_filters=num_filters2,
                      use_pooling=True)

# Flatten Layer
layer_flat, num_features = la.flatten_layer(layer_conv2)

# Fully-connected Layer 1
layer_fc1 = la.new_fc_layer(input=layer_flat,
                            num_inputs=num_features,
                            num_outputs=fc_size,
                            use_relu=True)

# Fully-connected Layer 2
layer_fc2 = la.new_fc_layer(input=layer_fc1,
                            num_inputs=fc_size,
                            num_outputs=num_classes,
                            use_relu=False)

# Predicted class
# NB: The second fully-connected layer estimates how likely it is
# that the input image belongs to each of the 5 classes. However,
# these estimates are a bit rough and difficult to interpret because
# the numbers may be very small or large, so we want to normalize
# them so that each element is limited between zero and one and the
# 5 elements sum to one. This is calculated using the so-called
# softmax function and the result is stored in y_pred
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

# Cost-function optimization
# NB: Cross entropy is a performance measure used in classification.
# The cross-entropy is a continuous function that is always positive
# and if the predicted output of the model exactly matches the
# desired output then the cross-entropy equals zero. The goal of
# optimization is therefore to minimize the cross-entropy so it gets
# as close to zero as possible by changing the variables of the
# network layers
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)

# In order to use the cross-entropy to guide the optimization of
# the model's variables we need a single scalar value, so we simply
# take the average of the cross-entropy for all the image
# classifications
cost = tf.reduce_mean(cross_entropy)

# Optimization method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Performance measures
# Vector of booleans whether the predicted class equals the true class
# of each image
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# Classification accuracy by first type-casting the vector of booleans
# to floats, so that False becomes 0 and True becomes 1, and then
# calculating the average of these numbers
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Counter for total number of iterations performed so far
total_iterations = 0


def optimize():
    # Ensure we update the global variable rather than a local copy
    global total_iterations

    start_time = time.time()

    # Get a batch of training examples
    # x_batch holds a batch of images
    # y_true are the true labels of those images
    image, label = im.input_pipeline(
        partial_train_labels_fn, train_dir, train_batch_size,
        im_shape, num_classes,
        record_defaults=[[''], [0]],
        min_after_dequeue=min_after_dequeue,
        num_threads=num_threads)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(total_iterations,
                       total_iterations + num_iterations):
            try:
                x_batch, y_true_batch = sess.run([image, label])
            except tf.errors.OutOfRangeError:
                break

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph
            feed_dict_train = {x_image: x_batch,
                               y_true: y_true_batch}

            # Run the optimizer using this batch of training data
            sess.run(optimizer, feed_dict=feed_dict_train)

            # Update the total number of iterations performed
            total_iterations += num_iterations

            end_time = time.time()
            time_dif = end_time - start_time

            # Print status every 100 iterations
            if i % 100 == 0:
                # Calculate the accuracy on the training set
                acc = sess.run(accuracy, feed_dict=feed_dict_train)

                # Message for printing
                m = 'Optimization Iteration: {0:>6},' \
                    ' Training Accuracy: {1:>6.1%}, ' \
                    ' Time usage: {2}'
                print(m.format(
                    i + 1, acc,
                    timedelta(seconds=int(round(time_dif)))))

        # Safely queue coordinator and stop threads
        coord.request_stop()
        coord.join(threads)


def cls_true_labels(label_csv_path):
    return np.genfromtxt(
        label_csv_path, delimiter=',', usecols=1, dtype=np.int)


def print_test_accuracy():
    cls_true = cls_true_labels(partial_test_labels_fn)
    # Get size of test dataset
    num_test = cls_true.size

    # Array for the predicted classes which will be calculated in
    # batches and filled into this array
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches
    # We will just iterate through all the batches
    i = 0

    # Get the images from the test set between i in j
    image, label = im.input_pipeline(
        partial_test_labels_fn, test_dir, test_batch_size,
        im_shape, num_classes,
        record_defaults=[[''], [0], ['']],
        min_after_dequeue=min_after_dequeue,
        num_threads=num_threads)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while i < num_test:
            j = min(i + test_batch_size, num_test)

            print('Testing on {} / {}'.format(i, j))

            try:
                images, labels = sess.run([image, label])
            except tf.errors.OutOfRangeError:
                break

            # Create a feed dictionary with these images and labels
            feed_dict = {x_image: images,
                         y_true: labels}

            # Calculate the predicted class
            results = sess.run(y_pred_cls, feed_dict=feed_dict)

            if j == num_test:
                cls_pred[i:j] = results[0:num_test-i]
            else:
                cls_pred[i:j] = results

            # Set the start index for the next batch to the end index
            # of the current batch
            i = j

        # Safely queue coordinator and stop threads
        coord.request_stop()
        coord.join(threads)

    # Convenience variable for the true class numbers of the test set
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images
    # When summing a boolean array, False means 0 and True means 1
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test set
    accuracy = float(correct_sum) / num_test

    # Print the accuracy
    m = 'Accuracy on Test Set: {0:.1%} ({1} / {2})'
    print(m.format(accuracy, correct_sum, num_test))


def list_files_wo_extensions(dir, extension='.'):
    return [f.split(extension)[0]
            for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]


def extract_labels(image_dir,
                   labels_source,
                   labels_destination,
                   delimiter=',',
                   image_name_index=0):

    images_filenames = list_files_wo_extensions(image_dir)

    with open(labels_destination, 'wt') as w:
        with open(labels_source, 'rt') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for i, line in enumerate(reader):
                if line[image_name_index] in images_filenames:
                    w.write(delimiter.join(line) + '\n')


def preprocessing():
    extract_labels(train_dir, train_labels_fn, partial_train_labels_fn)
    extract_labels(test_dir, test_labels_fn, partial_test_labels_fn)


def postprocessing():
    os.remove(partial_train_labels_fn)
    os.remove(partial_test_labels_fn)


def print_start_info():
    print('Welcome to Retinalearn!\n'
          '=======================')
    print('Num threads:\t\t{}\n'
          'Training batch size:\t{}\n'
          'Test batch size:\t{}\n'
          'Dequeue buffer size:\t{}'
          .format(num_threads, train_batch_size,
                  test_batch_size, min_after_dequeue)
          )
    print('=======================')


def main():
    print_start_info()
    optimize()
    print_test_accuracy()


if __name__ == '__main__':
    main()
