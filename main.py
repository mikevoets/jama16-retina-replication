import tensorflow as tf
import numpy as np
import scipy as sp
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import utils.images as im
import utils.convnets as cv

# Development
import pdb
#

IMAGE_TRAIN_DIR = './data/train/'
IMAGE_TRAIN_LABELS = './data/trainLabels.csv'

BATCH_SIZE = 50


def main():
    # Filename queue for image data (jpeg) from data directory
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(IMAGE_TRAIN_DIR + '*.jpeg')
    )

    # Filename for labels (csv) from data directory
    labels_csv_path = tf.train.string_input_producer([IMAGE_TRAIN_LABELS])

    # Read the images and labels
    image, label = im.read_images(filename_queue, labels_csv_path)

    # Convolutional Layer 1
    filter_size1 = 5        # Convolution filters are 5 x 5 pixels
    num_filters1 = 16       # There are 16 of these filters

    # Convolutional Layer 2
    filter_size2 = 5        # Convolution filters are 5 x 5 pixels
    num_filters2 = 36       # There are 36 of these filters

    # Fully connected Layer
    fc_size = 128           # Number of neurons in fully connected layer

    # Tuple with height and width of images used to reshape arrays
    im_shape = im.image_size(IMAGE_TRAIN_DIR)

    # List with height and with of images
    im_size = list(im_shape)

    # Number of color channels for the images: rgb (3)
    num_channels = 3

    # Number of classes, one class for each grade scale
    num_classes = 5

    with tf.Session() as sess:
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image_tensor = sess.run([image, label])
        print(image_tensor)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
