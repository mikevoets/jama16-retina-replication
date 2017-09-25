########################################################################
#
# Functions for downloading the EyePacs data-set from the internet
# and loading it into memory.
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_extract() to extract the data-set
#    if it is not already extracted in the given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################

import numpy as np
import pandas as pd
import os
import download
import glob
import re
import pdb  # TODO: Remove this line
import pickle
from PIL import Image
from fnmatch import fnmatch
from dataset import one_hot_encoded

########################################################################

# Directory where you want to extract and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "data/eyepacs/"

# File name for the training-set.
train_data_filename = "train.tar.gz"

# File name for the labels of the training-set.
train_labels_filename = "trainLabels.csv"

# File name for the test-set.
test_data_filename = "test.tar.gz"

# File name for the labels of the test-set.
test_labels_filename = "testLabels.csv"

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 256

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 5

########################################################################
# Various constants used to allocate arrays of the correct size.

# Batch size.
_train_batch_size = 20

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = len(get_file_paths())

# Total number of batches.
_num_batches_train = _num_images_train / _train_batch_size

########################################################################
# Private functions for downloading, unpacking and loading data-files.


def _convert_images(raw):
    """
    Convert images from the EyePacs format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _get_extract_path(filename):
    """
    Returns the path for the directory the data-set has been extracted to.
    """
    return os.path.join(data_path, filename.split('.')[0])


def _get_extract_file_paths(match):
    """
    Returns a list of file paths that match the match string.
    """
    return [f for f in os.listdir(data_path) if fnmatch(f, match + '.*')]


def _maybe_extract_data(filename_match):
    """
    Extracts a compressed or archived data-file from the EyePacs data-set.
    """
    filenames = _get_extract_file_matches(filename_match)

    extract_path = _get_extract_path(filename_match)

    for f in filenames:
        print("Extracting %s..." % f)

        file_path = os.path.join(data_path, f)

        download.maybe_extract(file_path=file_path, extract_dir=data_path)


def _get_file_paths(test=False):
    """
    Returns a list of sorted file names for a data-set.
    """
    data_dir = "test" if test else "train"

    filename_match = os.path.join(data_dir, '*.jpeg')
    filename_list = glob.glob(filename_match)
    # Sort the filename list.
    filename_list = sorted(
        filename_list, key=lambda fn: int(re.split('[./_]', fn)[-3]))

    return filename_list


def _strip_base_filename(e):
    """
    Helper function for stripping base filename from path.
    """
    return re.split('[./]', e)[-2]


def _base_filename(e):
    """
    Translates the list into values with base filename, without extension.
    """
    if isinstance(e, list):
        base = [_strip_base_filename(x) for x in e]
    else:
        base = _strip_base_filename(x)

    return base


def _get_cls(test=False):
    """
    Returns a numpy array with class labels from the data-set.
    """
    labels_fn = test_labels_filename if test else train_labels_filename
    labels_path = os.path.join(data_path, labels_fn)

    labels = pd.read_csv(labels_path, delimiter=",")

    return labels


def _filter_cls(image_list, cls=None, test=False):
    """
    Returns a filtered list of class labels.
    """
    if cls is None:
        cls = _get_cls(test=test)

    images = _base_filename(image_list)

    filtered = [y for y in cls.loc[cls['image'].isin(images)]['level']]

    return filtered


def _get_images(image_fn_list):
    """
    Returns a numpy array with images from the data-set.
    """
    x = np.array([np.asarray(Image.open(fn)) for fn in image_fn_list])

    return x


def _load_data(batch_num=None, test=False):
    """
    Load a pickled data-file from the EyePacs data-set
    and return the converted images (see above) and the class-number
    for each image.
    """
    image_fn_list = _get_file_path(test=test)
    cls = _get_cls(test=test)

    if batch_num is None:
        # Retrieve the images and convert them.
        raw_images = _get_images(imags_fn_list)
        images = _convert_images(raw_images)

        # Retrieve the class labels.
        labels = np.array(cls)
        return images, labels

    begin = batch_num
    end = min(begin + _train_batch_size, _num_images_train)

    # Retrieve the images for this current batch.
    batch = image_fn_list[i:j]
    raw_images = _get_images(batch)

    # Convert the images.
    images = _convert_images(raw_images)

    # Retrieve the class labels for this current batch.
    labels = np.array(_filter_cls(batch, cls=cls, test=test))

    return images, labels


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


def load_training_data():
    """
    Load all the training-data for the EyePacs data-set.

    The data-set is split into 5 data-files which are merged here.

    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(batch_num=i)

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data():
    """
    Load all the test-data for the EyePacs data-set.

    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images, cls = _load_data(test=True)

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

########################################################################
