import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
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

# Use sklearn for analysis of transfer-values
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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


def plot_transfer_values(i, transfer_values, test=False):
    """
    Function to plot an example image and its corresponding transfer-values.
    """
    print("Input image:")

    if test:
        image_ex, label_ex, _ = eyepacs.load_test_data(i)
    else:
        image_ex, label_ex, _ = eyepacs.load_training_data(i)

    image, label = eyepacs.session_run(image_ex, label_ex)

    # Plot the i'th image from the data-set.
    plt.imshow(image, interpolation='nearest')
    plt.show()

    print("Transfer-values for the image using Inception model:")

    # Transform the transfer-values into an image.
    img = transfer_values[i]
    img = img.reshape((32, 64))

    # Plot the image for the transfer-values.
    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()


def plot_scatter(values, cls):
    """
    Function to plot a color-map with different colors for each class.
    """
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

    # Get the color for each sample.
    colors = cmap[cls]

    # Extract the x- and y-values.
    x = values[:, 0]
    y = values[:, 1]

    # Plot it.
    plt.scatter(x, y, color=colors)
    plt.show()


def plot_transfer_values_analysis(values, cls):
    # New PCA-object with target array-length to 2.
    pca = PCA(n_components=2)

    # Use PCA to reduce the transfer-value arrays from 2048 to 2 elements.
    transfer_values_reduced = pca.fit_transform(values)

    # Plot the transfer-values that have been reduced using PCA.
    plot_scatter(transfer_values_reduced, cls)

    # Do dimensionality reduction using t-SNE.
    # Unfortunately t-SNE is very slow, so we use PCA to reduce the
    # transfer-values from 2048 to 50 elements.
    pca = PCA(n_components=50)
    transfer_values_50d = pca.fit_transform(values)

    # Create a new t-SNE object for final dimensionality reduction.
    tsne = TSNE(n_components=2)

    # Perform the final reduction using t-SNE.
    transfer_values_reduced = tsne.fit_transform(transfer_values_50d)

    # Plot the transfer-values.
    plot_scatter(transfer_values_reduced, cls)


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

    # Load image tensor from the EyePacs training set.
    images_train, cls_train, labels_train = eyepacs.load_training_data()

    print("Processing Inception transfer-values for training-images...")

    file_path_cache_train = os.path.join(
        eyepacs.data_path, 'inception_eyepacs_train.pkl')

    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    transfer_values_train = transfer_values_cache(
        cache_path=file_path_cache_train,
        image_paths=eyepacs.get_training_image_paths(),
        model=model
    )

    print("Processing Inception transfer-values for test-images...")

    file_path_cache_test = os.path.join(
        eyepacs.data_path, 'inception_eyepacs_test.pkl')

    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    transfer_values_test = transfer_values_cache(
        cache_path=file_path_cache_test,
        image_paths=eyepacs.get_test_image_paths(),
        model=model
    )

    # Retrieve the class-labels from the training set.
    training_cls = eyepacs.get_training_cls()

    # Plot analysis of transfer-values using PCA and t-SNE.
    # plot_transfer_values_analysis(
    #    values=transfer_values_train, cls=training_cls)


if __name__ == '__main__':
    main()
