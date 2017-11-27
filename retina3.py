import os
import sys
import importlib

from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.api.keras.models import Sequential

# Use the EyePacs dataset.
import eyepacs.v3 as eye

# For debugging purposes.
import pdb

# Ignore Tensorflow logs.
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

########################################################################
# Various constants.

# Define the ratio of training-validation data.
validation_split = 0.1

# Seed for shuffling training-validation data.
seed = 448

########################################################################
# Initializer functions

# Set locations of dataset.
eye.data_path = "data/eyepacs"
eye.train_pre_subpath = "preprocessed/train/"
eye.val_pre_subpath = "preprocessed/test/"
eye.test_pre_subpath = "preprocessed/val/"

# Block and wait until data is available.
eye.wait_until_available()

# Extract if necessary.
eye.maybe_extract_images()

# Preprocess if necessary.
eye.maybe_preprocess()

# Extract labels if necessary.
eye.maybe_extract_labels()

# Create labels-grouped subdirectories if necessary.
eye.maybe_create_subdirs_group_by_labels()

# Split training and validation set.
# eye.split_training_and_validation(split=validation_split, seed=seed)
# eye.maybe_convert()


def load_module(mod):
    return importlib.import_module(mod.replace('/', '.').split('.py')[0])


config = load_module('eyepacs/configs/512x512-5.py').config


def make_model(layers):
    pdb.set_trace()


train_datagen = ImageDataGenerator(
    zoom_range=config.get('augmentation_params')['zoom_range'],
    rotation_range=config.get('augmentation_params')['rotation_range'],
    shear_range=config.get('augmentation_params')['shear_range'],
    width_shift_range=config.get('augmentation_params')['width_shift_range'],
    height_shift_range=config.get('augmentation_params')['height_shift_range'],
    horizontal_flip=config.get('augmentation_params')['horizontal_flip'],
    vertical_flip=config.get('augmentation_params')['vertical_flip']
)

model = make_model(config.layers)
