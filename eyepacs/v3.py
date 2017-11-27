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

import os
import download
import csv
import numpy as np
import sys
import random
from shutil import copyfile
from re import split
from PIL import Image
from glob import glob
from time import sleep
from fnmatch import fnmatch
from dataset import one_hot_encoded
from preprocess import scale_normalize, resize

########################################################################

FEATURE_DIR = 'data/features'

# set of resampling weights that yields balanced classes
BALANCE_WEIGHTS = np.array([1.3609453700116234,  14.378223495702006,
                            6.637566137566138, 40.235967926689575,
                            49.612994350282484])

no_augmentation_params = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': False,
    'allow_stretch': False,
}

# Directory where you want to extract and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "/data/eyepacs/"

# Directory to training and test-set relative from the data path.
train_subpath = "train/"
test_subpath = "test/"

# Directory to where preprocessed training and test-sets should
# be uploaded to.
train_pre_subpath = "preprocessed/512/train/"
val_pre_subpath = "preprocessed/512/test/"
test_pre_subpath = "preprocessed/512/val/"

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

# Counter for printing status.
_cnt = 0

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
        base = _strip_base_filename(e)

    return base


def _get_image_paths(test=False, extension=None, images_dir=None):
    if extension is None:
        extension = ".jpeg"

    # Get the directory where the data-set resides.
    if images_dir is None:
        images_dir = _get_images_path(test=test)

    # The file paths should match the following regexp.
    filename_match = os.path.join(images_dir, "**", "*" + extension)

    return glob(filename_match, recursive=True)


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
    print_status("Extracting labels...")

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
        print_status("Done.")
    else:
        print_status("Labels already extracted.")


def _maybe_extract_images(test=False):
    """
    Extracts a compressed or archived data-file from the EyePacs data-set.
    """
    filenames = _get_extract_file_paths(test=test)

    for f in filenames:
        print_status("Extracting %s..." % f)

        file_path = os.path.join(data_path, f)

        download.maybe_extract(file_path=file_path, extract_dir=data_path)


def _maybe_preprocess(test=False, convert=True):
    """
    Helper function for preprocessing of images.
    """
    # Find the path where processed images should be uploaded to.
    save_path = os.path.join(_get_images_path(test=test))

    # Get image path.
    images_dir = _get_extract_path(test=test)

    # Get paths of images that are to be preprocessed.
    image_paths = _get_image_paths(images_dir=images_dir)

    # Get paths of images that already are preprocessed.
    preprocessed_paths = _get_image_paths(images_dir=save_path)

    # Get paths of images that yet have to be preprocessed.
    images_to_preprocess = set(_base_filename(image_paths)) \
        - set(_base_filename(preprocessed_paths))

    # Only continue unless the directory does not exist.
    if len(images_to_preprocess) > 0:
        print_status("Preprocessing images...")

        # Get the full paths of images.
        preprocess_fns = [os.path.join(images_dir, im + '.jpeg')
                          for im in images_to_preprocess]

        # Create directory for preprocessed images if necessary.
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Preprocess images.
        scale_normalize(image_paths=preprocess_fns, save_path=save_path,
                        diameter=512)
    else:
        print_status("Images already preprocessed.")


def _maybe_move(images, to_directory, copy=False):
    # Helper function for splitting training and validation images.
    # tbm stands for "to be moved"
    tbm = [x for x in images if x.find(to_directory) == -1]

    if len(tbm) == 0:
        return False

    # Move the images to other set if necessary.
    for x in tbm:
        # Find the new destination of the image.
        partitioned_fn = x.split("/")

        subdir = None
        # Consider if the images are partioned in subfolders.
        try:
            subdir = int(partitioned_fn[-2])

            if not (subdir >= 0 and subdir < num_classes):
                raise ValueError
        except ValueError:
            # Images are not partitioned.
            pass

        image_fn = partitioned_fn[-1]

        if subdir is not None:
            group_path = os.path.join(to_directory, str(subdir))

            # Check if group subdirectory has been made before.
            # If not, make it.
            if not os.path.exists(group_path):
                os.makedirs(group_path)

            new_dest = os.path.join(group_path, image_fn)
        else:
            new_dest = os.path.join(to_directory, image_fn)

        # Move the image to the other directory set.
        if copy:
            copyfile(x, new_dest)
        else:
            os.rename(x, new_dest)

    return True


def print_status(msg):
    global _cnt
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(ERASE_LINE + CURSOR_UP_ONE)
    msg = "\r[Preprocessing {0:>2}] - {1}".format(_cnt, msg)
    sys.stdout.write(msg)
    sys.stdout.flush()
    _cnt += 1

########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.

# Class for getting batches of the EyePacs data set.


def maybe_preprocess():
    """
    Preprocesses the training and test data-set if it hasn't been
    preprocessed before.

    Scale normalizes each image by finding the circle mask of the fundus
    and scaling it to 512px in diameter.
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


def maybe_create_subdirs_group_by_labels():
    """
    Creates subdirectories grouped by labels for each data-set
    if they haven't been created before.
    """
    def _maybe_create(images_dir, labels_path):
        # Helper function for creating subdirectories.
        print_status("Creating subdirectories...")

        # Skipping if there are no labels.
        if not os.path.exists(labels_path):
            print_status("Skipping {}".format(labels_path))
            return

        # Skip if subdirectories already have been created.
        num_directories = len(os.listdir(images_dir))
        if num_directories > 0 and num_directories <= num_classes:
            print_status("Subdirectories already created!")
            return

        # Read labels file.
        with open(labels_path, 'rt') as r:
            reader = csv.reader(r, delimiter=",")
            for i, line in enumerate(reader):
                # Deduct the image filename and label.
                im_name = line[0] + '.jpeg'
                label = line[1]

                # Check if group subdirectory has been made before.
                # If not, make it.
                im_path = os.path.join(images_dir, im_name)
                group_path = os.path.join(images_dir, label)

                if not os.path.exists(group_path):
                    os.makedirs(group_path)

                # Move image to subdirectory.
                new_im_path = os.path.join(group_path, im_name)
                os.rename(im_path, new_im_path)

    # Retrieve all data-sets.
    train_images_path = os.path.join(data_path, train_pre_subpath)
    val_images_path = os.path.join(data_path, val_pre_subpath)
    test_images_path = os.path.join(data_path, test_pre_subpath)

    # Retrieve all labels.
    train_labels_path = os.path.join(data_path, train_labels_extracted)
    val_labels_path = os.path.join(data_path, val_labels_extracted)
    test_labels_path = os.path.join(data_path, test_labels_extracted)

    # Create all subdirectories.
    _maybe_create(train_images_path, train_labels_path)
    _maybe_create(val_images_path, val_labels_path)
    _maybe_create(test_images_path, test_labels_path)


def split_training_and_validation(split=0.0, seed=None):
    """
    Rearranges training and validation regarding to the split parameter.
    Assumes the images have been preprocessed and placed in the
    directory of the preprocessed images defined.

    Split = 0.0 means only training data, split = 1.0 means only validation.
    """

    def _reset_labels(images, orig_labels_path, labels_path):
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

    # Randomize image paths.
    if seed is not None:
        random.seed(seed)

    random.shuffle(image_paths)

    # Total number of images.
    num_images = len(image_paths)

    # Define the split.
    split_at = int(num_images*(1.0 - split))

    # Get all the paths to images belonging to either set.
    train_images = image_paths[:split_at]
    val_images = image_paths[split_at:]

    print_status("Checks if Training/Validation-Set should be rearranged.")

    # Find the images that are in the wrong directory.
    res1 = _maybe_move(images=train_images, to_directory=train_images_path)
    res2 = _maybe_move(images=val_images, to_directory=val_images_path)

    # Return early if here hasn't been any movements.
    if res1 is False and res2 is False:
        print_status("No changes necessary.")
        return

    print_status("New Training/Validation-Set Split: {:>3.2%}".format(1.0 - split))
    print_status("Resetting labels...")

    # Reset labels for each set.
    _reset_labels(images=train_images,
                  orig_labels_path=train_labels_orig,
                  labels_path=train_labels_path)
    _reset_labels(images=val_images,
                  orig_labels_path=train_labels_orig,
                  labels_path=val_labels_path)

    print_status("Resetted all labels!")


def maybe_convert(convert_to=[256, 128]):
    """
    Converts images and places to another directory.
    """
    for pre_subpath in [train_pre_subpath, test_pre_subpath, val_pre_subpath]:
        for size in convert_to:
            path = os.path.join(data_path, pre_subpath)
            new_path = path.replace('512', str(size))

            old_images_paths = _get_image_paths(path)

            # Copy the images from the images dir to the new dir.
            _maybe_move(old_images_paths, new_path, copy=True)

            # Convert the images to size.
            images_paths = _get_image_paths(images_dir=new_path)
            resize(images_paths, size=size)


def wait_until_available():
    """
    Sleeps until data has been found.

    Gives opportunity for end user to upload data to container.
    """
    counter = 0
    ready_file = os.path.join(data_path, '.ready')

    while True:
        if os.path.exists(ready_file):
            break

        print("{0:>4} : Waiting until I find .ready in {1} folder".
              format(counter, data_path))

        sleep(5)
        counter += 1


########################################################################
