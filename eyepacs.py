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
# 3) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 4) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################

import tensorflow as tf
import os
import download
import csv
import pandas as pd
from re import split
from glob import glob
from fnmatch import fnmatch
from dataset import one_hot_encoded
from preprocess import scale_normalize

########################################################################

# Directory where you want to extract and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "data/eyepacs/"

# Directory to training and test-set relative from the data path.
train_subpath = "train/"
test_subpath = "test/"

# Directory to where preprocessed training and test-sets should
# be uploaded to.
train_pre_subpath = "preprocessed/train/"
test_pre_subpath = "preprocessed/test/"

# File name for the training-set.
train_data_filename = "train.7z"

# File name for the labels of the training-set.
train_labels_filename = "trainLabels.csv"

# File name for the extracted labels of the training-set.
train_labels_extracted = "trainLabels.part.csv"

# File name for the test-set.
test_data_filename = "test.7z"

# File name for the labels of the test-set.
test_labels_filename = "testLabels.csv"

# File name for the extracted labels of the training-set.
test_labels_extracted = "testLabels.part.csv"

########################################################################
# Various constants used to allocate arrays of the correct size.

# Batch size.
_batch_size = 20

# Total number of threads.
_num_threads = 2

# Minimal buffer size after dequeueing.
_min_after_dequeue = 10

########################################################################
# Various constants for the size of the images.

# Width and height of each image.
img_size = 256

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Number of classes.
num_classes = 5

########################################################################
# Private functions for downloading, unpacking and loading data-files.


def _get_images_path(test=False):
    """
    Returns the path for the directory the processed data-set.
    """
    if test:
        images_dir = test_pre_subpath
    else:
        images_dir = train_pre_subpath

    return os.path.join(data_path, images_dir)


def _get_extract_path(test=False):
    """
    Returns the path for the directory the data-set has been extracted to.
    """
    if test:
        extract_dir = test_subpath
    else:
        extract_dir = train_subpath

    return os.path.join(data_path, extract_dir)


def _get_extract_file_paths(test=False):
    """
    Returns a list of file paths that match the match string.
    """
    match = test_data_filename if test else train_data_filename

    return [f for f in os.listdir(data_path) if fnmatch(f, match + '.*')]


def _strip_base_filename(e):
    """
    Helper function for stripping base filename from path.
    """
    return split('[./]', e)[-2]


def _base_filename(e):
    """
    Translates the list into values with base filename, without extension.
    """
    if isinstance(e, list):
        base = [_strip_base_filename(x) for x in e]
    else:
        base = _strip_base_filename(x)

    return base


def _get_image_paths(test=False, extension=None):
    if extension is None:
        extension = ".jpeg"

    # Get the directory where the data-set resides.
    images_dir = _get_images_path(test=test)

    # The file paths should match the following regexp.
    filename_match = os.path.join(images_dir, "*" + extension)

    return glob(filename_match)


def _get_base_file_paths(test=False):
    """
    Returns a list of sorted file names for a data-set.
    """
    # Get file paths.
    file_paths = _get_image_paths(test=test, extension=".jpeg")

    # Get base filenames.
    base_filenames = _base_filename(file_paths)

    # Sort the filename list.
    base_filenames = sorted(
        base_filenames, key=lambda fn: int(split('_', fn)[-2]))

    return base_filenames


def _get_data_path(filename="", test=None):
    """
    Returns the file path to a file relative from the data path.

    If filename is blank, the data directory path is returned.

    If test is either false or true, the path is returned relative from
    either the test or the training data directory, respectively.
    """
    if test is not None:
        if test:
            return data_path + test_pre_subpath + filename
        else:
            return data_path + train_pre_subpath + filename
    else:
        return data_path + "/" + filename


def maybe_extract_labels(test=False):
    """
    Helper function for extracting labels.
    """
    print("Extracting labels...")

    image_fns = _get_base_file_paths(test=test)

    if test:
        labels_src_fn = test_labels_filename
        labels_dest_fn = test_labels_extracted
    else:
        labels_src_fn = train_labels_filename
        labels_dest_fn = train_labels_extracted

    # Retrieve the paths labels should be exported to.
    labels_src_path = _get_data_path(labels_src_fn)
    labels_dest_path = _get_data_path(labels_dest_fn)

    if not os.path.exists(labels_dest_path):
        with open(labels_dest_path, 'wt') as w:
            with open(labels_src_path, 'rt') as r:
                reader = csv.reader(r, delimiter=",")
                for i, line in enumerate(reader):
                    if line[0] in image_fns:
                        w.write(",".join(line) + "\n")
        print("Done.")
    else:
        print("Labels already extracted.")


def _maybe_extract_images(test=False):
    """
    Extracts a compressed or archived data-file from the EyePacs data-set.
    """
    filenames = _get_extract_file_paths(test=test)

    for f in filenames:
        print("Extracting %s..." % f)

        file_path = os.path.join(data_path, f)

        download.maybe_extract(file_path=file_path, extract_dir=data_path)


def _convert_images(raw):
    """
    Convert images from the EyePacs format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """
    # Convert the raw images from the data-files to floating-points.
    raw_float = tf.cast(raw, tf.float32) / 255.0

    # Explicitily set size of image.
    images = tf.image.resize_images(raw_float, img_shape)

    return images


def _get_images(file_path):
    """
    Reads an image, converts and returns tensor.
    """
    raw = tf.read_file(file_path)
    raw = tf.image.decode_jpeg(raw, channels=num_channels)

    # Convert image.
    images = _convert_images(raw)

    return images


def _get_labels_path(test=False):
    """
    Helper function for retrieving path of labels.
    """
    if test:
        filename = test_labels_extracted
    else:
        filename = train_labels_extracted

    file_path = _get_data_path(filename)

    return file_path


def _read_labels(test=False):
    """
    Returns a tensor for a label.
    """
    record_defaults = [[''], [0], ['']]
    record_defaults = record_defaults if test else [[''], [0]]
    labels_path = _get_labels_path(test=test)

    # Reading and decoding labels in csv-format.
    csv_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = csv_reader.read(tf.train.string_input_producer([labels_path]))
    row = tf.decode_csv(csv_row, record_defaults=record_defaults)

    return row


def _get_label(i, test=False):
    """
    Returns tensor for i-th label.
    """
    labels_path = _get_labels_path(test=test)

    with open(labels_path, 'rt') as r:
        reader = csv.reader(r, delimiter=",")
        for num, line in enumerate(reader):
            # Skip the header line (hence - 1).
            if i == num - 1:
                return line[0], int(line[1])


def _read_image(i, test=False):
    """
    Returns an image and label tensor representing the i-th image
    in the dataset.
    """
    image_name, cls = _get_label(i, test=test)

    if image_name is None:
        raise ValueError("Out of bound index!")

    label = tf.constant(cls, shape=(1,), dtype=tf.int32)

    images_path = _get_data_path(image_name + '.jpeg', test=test)
    image = _get_images(images_path)

    # Retrieve one-hot encoded class labels
    label_one_hot = tf.one_hot(
        indices=label, depth=num_classes, dtype=tf.float32)

    return image, label, label_one_hot


def _read_images(test=False):
    """
    Returns an image tensor and its corresponding label tensor.
    """
    row = _read_labels()

    # Extract image identifier and class label from file.
    image_names = row[0]
    labels = row[1]

    # Retrieve image
    images_path = _get_data_path(image_names + '.jpeg', test=test)
    images = _get_images(images_path)

    return images, labels


def _load_data(test=False):
    """
    Load a batch from the EyePacs data-set
    and return the converted images (see above) and the class-number
    for each image. Also return a one-hot encoded version of class-labels.
    """
    # Retrieve images and labels.
    images, labels = _read_images(test=test)
    capacity = _min_after_dequeue + (_num_threads + 1)*_batch_size

    # Retrieve batch from Tensorflow batch feeder.
    image_batch, label_batch = tf.train.shuffle_batch(
        [images, labels], batch_size=_batch_size, capacity=capacity,
        min_after_dequeue=_min_after_dequeue, num_threads=_num_threads
    )

    # Retrieve one-hot encoded class labels
    label_one_hot = tf.one_hot(
        indices=label_batch, depth=num_classes, dtype=tf.float32)

    return image_batch, label_batch, label_one_hot


def _get_cls(test=False):
    """
    Returns a numpy array with class labels from the data-set.
    """
    labels_path = _get_labels_path(test=test)

    labels = pd.read_csv(labels_path, delimiter=",",
                         usecols=[1], names=["level"])

    cls = labels["level"]

    return cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def _maybe_preprocess(test=False):
    """
    Helper function for preprocessing of images.
    """
    # Find the path where processed images should be uploaded to.
    save_path = _get_images_path(test=test)

    # Get image path.
    images_dir = _get_extract_path(test=test)

    # Only continue unless the directory does not exist.
    if not os.path.exists(save_path):
        print("Preprocessing images...")

        # Create directory for preprocessed images.
        os.makedirs(save_path)

        # Preprocess images.
        scale_normalize(images_path=images_dir, save_path=save_path)
    else:
        print("Images seem already preprocessed.")


def _session_iterate(*args):
    """
    Helper function for iterating over data-set.
    """
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                yield sess.run(args)

        except tf.errors.OutOfRangeError:
            pass

        finally:
            # Safely queue coordinator and stop threads.
            coord.request_stop()

        coord.join(threads)


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


def get_training_cls():
    """
    Returns a numpy array with class labels from the training-set.
    """

    return _get_cls()


def get_test_cls():
    """
    Returns a numpy array with class labels from the test-set.
    """

    return _get_cls(test=True)


def session_run(*args):
    """
    Wrapper function for running a session once.
    """

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        result = sess.run(args)

    return result


def session_iterate(*args):
    """
    Wrapper function for starting up a session and iterating over the data-set.
    """

    yield _session_iterate(args)


def get_training_image_paths():
    """
    Returns a list with all the image paths for the training-set.
    """

    return _get_image_paths()


def get_test_image_paths():
    """
    Returns a list with all the image paths for the test-set.
    """

    return _get_image_paths(test=True)


def load_training_data(i=None):
    """
    Load all the training-data for the EyePacs data-set.

    The data-set is split into 5 data-files which are merged here.

    Returns the images, class-numbers and one-hot encoded class-labels.

    If i is defined, load the i-th image from the dataset.
    """

    if i is None:
        return _load_data()

    return _read_image(i)


def load_test_data(i=None):
    """
    Load all the test-data for the EyePacs data-set.

    Returns the images, class-numbers and one-hot encoded class-labels.

    If i is defined, load the i-th image from the dataset.
    """

    if i is None:
        return _load_data(test=True)

    return _read_image(i, test=True)


def maybe_preprocess():
    """
    Preprocesses the training and test data-set if it hasn't been
    preprocessed before.

    Scale normalizes each image by finding the circle mask of the fundus
    and scaling it to 299px in diameter.
    """

    _maybe_preprocess()
    _maybe_preprocess(test=True)


def maybe_extract_images():
    """
    Extracts the training and test data-set if it hasn't been
    extracted before.
    """

    _maybe_extract_images()
    _maybe_extract_images(test=True)


def maybe_extract_labels():
    """
    Extracts the training and test data-set labels if they haven't
    been extracted before.
    """

    _maybe_extract_labels()
    _maybe_extract_labels(test=True)


########################################################################
