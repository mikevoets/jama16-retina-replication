import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
import os
import utils.images as im
import utils.convnets.layers as la
import csv

# Development
import pdb
#

# Image locations
train_dir = './data/train/'
test_dir = './data/test/'

# Label locations
train_labels_fn = './data/trainLabels.csv'
test_labels_fn = './data/testLabels.csv'

partial_train_labels_fn = './data/trainLabels.part.csv'
partial_test_labels_fn = './data/testLabels.part.csv'


def test_labels():
    return np.genfromtxt(
        partial_test_labels_fn, delimiter=',', usecols=1, dtype=np.int)


def read_images(labels_path, image_dir, im_size, record_defaults=None):
    if record_defaults is None:
        record_defaults = [[''], ['0']]

    # Reading and decoding labels in csv-format
    csv_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = csv_reader.read(
        tf.train.string_input_producer([labels_path]))
    row = tf.decode_csv(
        csv_row, record_defaults=record_defaults)

    im_path = row[0]
    label = row[1]
    # Reading and decoding images in jpeg-format
    image = tf.read_file(image_dir + im_path + '.jpeg')
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.

    # Explicitily set size of image
    image = tf.image.resize_images(image, im_size)

    return image, label


def input_pipeline(labels_path,
                   image_dir,
                   batch_size,
                   im_size,
                   record_defaults=None):
    # Retrieve example image and label
    example, label = read_images(
        labels_path, image_dir, im_size, record_defaults=record_defaults)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )
    return example_batch, label_batch


def main():
    # Correct test classes
    cls_true = test_labels()

    # Get number of images in test set
    num_test = cls_true.size

    # Array for the predicted classes which will be calculated in
    # batches and filled into this array
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    image_batch, label_batch = input_pipeline(
        partial_test_labels_fn, './data/test/', 1, (256, 256),
        record_defaults=[[''], ['0'], ['']])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        while True:
            try:
                im, lab = sess.run([image_batch, label_batch])
            except tf.errors.OutOfRangeError:
                print('Out of range!')
                break


if __name__ == '__main__':
    main()
