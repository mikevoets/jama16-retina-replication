import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
from datetime import timedelta

# Functions and classes for loading an using the Inception model.
import inception
from inception import transfer_values_cache

# Use Pretty Tensor to define new classifiers.
import prettytensor as pt

# Use the EyePacs dataset.
import eyepacs
from eyepacs import num_classes

# For debugging purposes.
import pdb


def plot_images(images, cls_true, cls_pred=None, smooth=True):
    """
    Function to plot images in a grid.
    Writing the true and predicted classes below each image.
    """

    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(
                    cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Show the plots.
    plt.show()



def main():
    # Define the location of the EyePacs data set.
    eyepacs.data_path = "data/eyepacs"

    # Extract if necessary.
    eyepacs.maybe_extract()

    # Define the download location of the Inception model.
    inception.data_dir = "data/inception/"

    # Download the Inception model if necessary.
    inception.maybe_download()

    # Load the Inception model.
    model = inception.Inception()

    # Load images from the EyePacs training set.
    images_train, cls_train, labels_train = eyepacs.load_training_data()

    # Load images from the EyePacs test set.
    images_test, cls_test, labels_test = eyepacs.load_training_data()

    # Set the file-paths for the caches of the training and test-set.
    file_path_cache_train = os.path.join(
        eyepacs.data_path, 'inception_eyepacs_train.pkl')
    file_path_cache_test = os.path.join(
        eyepacs.data_path, 'inception_eyepacs_test.pkl')

    print("Processing Inception transfer-values for training-images...")

    # Scale images because Inception needs pixels to be between 0 and 255,
    # while the CIFAR-10 functions return pixels between 0.0 and 1.0
    images_scaled = images_train * 255.0

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while True:
            try:
                images, labels = sess.run([images_scaled, labels_train])
            except tf.errors.OutOfRangeError:
                break

            # If transfer-values have already been calculated then reload them,
            # otherwise calculate them and save them to a cache-file.
            transfer_values_train = transfer_values_cache(
                cache_path=file_path_cache_train, images=images, model=model)

        # Safely queue coordinator and stop threads
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
