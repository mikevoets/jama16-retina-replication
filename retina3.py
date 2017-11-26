import os
import sys
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
eye.split_training_and_validation(split=validation_split, seed=seed)

# eye.maybe_convert()
