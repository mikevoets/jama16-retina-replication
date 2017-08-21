import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy as sp
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
from utils.images import get_image_size

# Development
import pdb
#

IMAGE_TRAIN_DIR = './data/train/'
IMAGE_TRAIN_LABELS = './data/trainLabels.csv'

BATCH_SIZE = 50


def read_images(filename_queue, labels_csv_path):
    # Reading and decoding labels in csv-format
    csv_reader = tf.TextLineReader()
    record_defaults = [[''], ['0']]
    _, csv_content = csv_reader.read(labels_csv_path)
    label = tf.decode_csv(
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


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumarate(axes.flat):
        # Plot image
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis
        ax.set_xlabel(xlabel)

        # Remove tickets from the plot
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the polot is shown correctly with multiple plots
    plt.show()


def main():
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

    # Tuple with height and width of images used to reshape arrays
    im_shape = image_size(IMAGE_TRAIN_DIR)

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
