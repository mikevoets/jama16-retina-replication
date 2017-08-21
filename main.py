import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import pdb
import PIL.Image
import os
from utils.images import get_image_size

IMAGE_TRAIN_DIR = './data/train/'
IMAGE_TRAIN_LABELS = './data/trainLabels.csv'

BATCH_SIZE = 50


def read_images(filename_queue, labels_csv_path):
    # Reading and decoding labels in csv-format
    csv_reader = tf.TextLineReader()
    record_defaults = [[''], ['0']]
    _, csv_content = csv_reader.read(labels_csv_path)
    im_name, label = tf.decode_csv(
        csv_content, record_defaults=record_defaults)

    # Reading and decoding images in jpeg-format
    im_reader = tf.WholeFileReader()
    im_filename, im_content = im_reader.read(filename_queue)
    image = tf.image.decode_jpeg(im_content)
    image = tf.cast(image, tf.float32) / 255.
    return image, label


def image_size(directory):
    first_im_path = os.listdir(directory)[0]
    return get_image_size(directory + first_im_path)


def main():
    # Configuration of Neural Network

    # Filename queue for image data (jpeg) from data directory
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(IMAGE_TRAIN_DIR + '*.jpeg')
    )

    # Filename for labels (csv) from data directory
    labels_csv_path = tf.train.string_input_producer([IMAGE_TRAIN_LABELS])

    # Read the images and labels
    image, label = read_images(filename_queue, labels_csv_path)

    # Convolutional Layer 1
    filter_size1 = 5        # Convolution filters are 5 x 5 pixels
    num_filters1 = 16       # There are 16 of these filters

    # Convolutional Layer 2
    filter_size2 = 5        # Convolution filters are 5 x 5 pixels
    num_filters2 = 36       # There are 36 of these filters

    # Fully connected Layer
    fc_size = 128           # Number of neurons in fully connected layer

    im_size = image_size(IMAGE_TRAIN_DIR)

    with tf.Session() as sess:
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image_tensor = sess.run([image, label])

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
