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
import numpy as np
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
val_pre_subpath = "preprocessed/val/"
test_pre_subpath = "preprocessed/test/"

# File name for the training-set.
train_data_filename = "train.7z"

# File name for the labels of the training-set.
train_labels_filename = "trainLabels.csv"

# File name for the extracted labels of the training-set.
train_labels_extracted = "trainLabels.part.csv"

# File name for the extracted labels of the validation-set.
val_labels_extracted = "valLabels.part.csv"

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
img_size = 299

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


def _get_image_paths(test=False, extension=None, images_dir=None):
    if extension is None:
        extension = ".jpeg"

    # Get the directory where the data-set resides.
    if images_dir is None:
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


def _maybe_extract_labels(test=False):
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
    csv_reader = tf.TextLineReader()
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
            if i == num:
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


def _session_iterate(args, session=None):
    """
    Helper function for iterating over data-set.
    """
    def iterate(args, session):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=session)

        try:
            while not coord.should_stop():
                yield session.run(args)

        except tf.errors.OutOfRangeError:
            pass

        finally:
            # Safely queue coordinator and stop threads.
            coord.request_stop()

        coord.join(threads)

    if session is not None:
        iterate(args=args, session=session)
    else:
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            iterate(args=args, session=sess)


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.

# Class for getting batches of the EyePacs data set.



def get_training_cls(split=None):
    """
    Returns a numpy array with class labels from the training-set.

    If split is defined, split the set into two according to this float value.
    """
    cls = _get_cls()

    # Split the class-numbers into two (training and validation).
    if split is not None:
        num_cls = cls[0].size
        split_at = int(num_cls*(1.0 - split))

        return (cls[0][:split_at], cls[1][:split_at]), \
               (cls[0][split_at:], cls[1][split_at:])
    else:
        return cls


def get_test_cls():
    """
    Returns a numpy array with class labels from the test-set.
    """

    return _get_cls(test=True)


def session_run(args, session=None):
    """
    Wrapper function for running a session once.
    """
    if session is not None:
        result = session.run(args)
    else:
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            result = sess.run(args)

    return result


def session_iterate(args, session=None):
    """
    Wrapper function for starting up a session and iterating over the data-set.
    """

    yield _session_iterate(args=args, session=session)


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

    # Also create the validation preprocess directory.
    val_dir = os.path.join(data_path, val_pre_subpath)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)


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


def split_training_and_validation(split=0.0):
    """
    Rearranges training and validation regarding to the split parameter.
    Assumes the images have been preprocessed and placed in the
    directory of the preprocessed images defined.

    Split = 0.0 means only training data, split = 1.0 means only validation.
    """

    def maybe_move(images, to_directory):
        # Helper function for splitting training and validation images.
        # tbm stands for "to be moved"
        tbm = [x for x in images if x.find(to_directory) == -1]

        if len(tbm) == 0:
            return False

        # Move the images to other set if necessary.
        for x in tbm:
            # Find the new destination of the image.
            image_fn = x.split("/")[-1]
            new_dest = os.path.join(to_directory, image_fn)

            # Move the image to the validation set.
            os.rename(x, new_dest)

        return True

    def reset_labels(images, orig_labels_path, labels_path):
        # Helper function for resetting labels.
        base_fns = _base_filename(images)

        with open(labels_path, 'wt') as w:
            with open(orig_labels_path, 'rt') as r:
                reader = csv.reader(r, delimiter=",")
                for i, line in enumerate(reader):
                    if line[0] in base_fns:
                        w.write(",".join(line) + "\n")

    if not (split >= 0.0 and split <= 1.0):
        # Raise an error if split is not between 0.0 and 1.0.
        raise ValueError("Split should be between 0.0 and 1.0.")

    train_images_path = os.path.join(data_path, train_pre_subpath)
    val_images_path = os.path.join(data_path, val_pre_subpath)

    train_labels_orig = os.path.join(data_path, train_labels_filename)
    train_labels_path = os.path.join(data_path, train_labels_extracted)
    val_labels_path = os.path.join(data_path, val_labels_extracted)

    if not (os.path.exists(train_images_path) or
            os.path.exists(val_images_path)):
        # Raise an error if none of the preprocessed directories exists.
        raise TypeError("Preprocess images first.")

    # Retrieve all files from both previous training-set and validation-set.
    image_paths = []
    image_paths += _get_image_paths(images_dir=train_images_path)
    image_paths += _get_image_paths(images_dir=val_images_path)

    # Total number of images.
    num_images = len(image_paths)

    # Define the split.
    split_at = int(num_images*(1.0 - split))

    # Get all the paths to images belonging to either set.
    train_images = image_paths[:split_at]
    val_images = image_paths[split_at:]

    # Find the images that are in the wrong directory.
    res1 = maybe_move(images=train_images, to_directory=train_images_path)
    res2 = maybe_move(images=val_images, to_directory=val_images_path)

    # Return early if here hasn't been any movements.
    if res1 is False and res2 is False:
        return

    # Reset labels for each set.
    reset_labels(images=train_images,
                 orig_labels_path=train_labels_orig,
                 labels_path=train_labels_path)
    reset_labels(images=val_images,
                 orig_labels_path=train_labels_orig,
                 labels_path=val_labels_path)


# Class for retrieving images from the EyePacs data-set.
class EyepacsBatcher(object):
    # Class constructor.
    def __init__(self, training=True, validation=False, test=False):
        self.index = 0
        self.num_images = images.shape[0]
        self.epoch = 0

    # Mini-batching method.
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        # When all the training data is ran, shuffle it.
        if self.index_in_epoch > self.num_images:
            # Find a new order.
            new_order = np.arange(self.num_images)
            np.random.shuffle(new_order)

            # Apply the new order to images and labels.
            self.images = self.images[new_order]
            self.labels = self.labels[new_order]

            # Start new epoch.
            self.epoch += 1
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_images

        end = self.index_in_epoch
        return self.images[start:end], self.labels[start:end], self.epoch


########################################################################
