import os
import sys
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import importlib

from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from PIL import Image

from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint, EarlyStopping

from dataset import one_hot_encoded

# Use the EyePacs dataset.
import eyepacs.v3 as eye

# For debugging purposes.
import pdb

# Ignore Tensorflow logs.
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

########################################################################
# Various constants.

# Fully-connected layer size.
fully_connected_size = 1024

# Define the ratio of training-validation data.
validation_split = 0.1

# Seed for shuffling training-validation data.
seed = 448

########################################################################


def get_num_files():
    """Get number of files by searching directory recursively"""
    return len(eye._get_image_paths(extension=".jpeg"))


def find_num_train_images():
    """Helper function for finding amount of training images."""
    train_images_dir = os.path.join(eye.data_path, eye.train_pre_subpath)

    return len(eye._get_image_paths(images_dir=train_images_dir))


def find_num_val_images():
    """Helper function for finding amount of training images."""
    val_images_dir = os.path.join(eye.data_path, eye.val_pre_subpath)

    return len(eye._get_image_paths(images_dir=val_images_dir))


def load_module(mod):
    return importlib.import_module(mod.replace('/', '.').split('.py')[0])


def transform_target(y):
    t = np.zeros((y.shape[0], 2), dtype=np.float32)
    # Convert one-hot encoded to labels.
    h = y.argmax(axis=1)
    # Transform the m-class y to a 2-label y.
    t[np.argwhere(h > 1).reshape(-1), 0] = 1
    t[np.argwhere(h > 2).reshape(-1), 1] = 1
    return t


def multiclass_flow_from_directory(flow_from_directory_gen, m_class_getter):
    for x, y in flow_from_directory_gen:
        yield x, m_class_getter(y)


# Get config and set directories.
module = load_module('eyepacs/configs/299_iv3.py')
config = module.config

# Set locations of dataset.
eye.data_path = "data/eyepacs/"
eye.train_pre_subpath = config.get('train_dir')
eye.val_pre_subpath = config.get('val_dir')

num_epochs = 10
num_images = find_num_train_images()
num_val_images = find_num_val_images()
batch_size = config.get('batch_size_train')

print("Find images...")

train_datagen = ImageDataGenerator(**config.get('augmentation_params'))
val_datagen = ImageDataGenerator(**config.get('augmentation_params'))

train_generator = train_datagen.flow_from_directory(
        os.path.join(eye.data_path, eye.train_pre_subpath),
        target_size=(config.get('width'), config.get('height')),
        batch_size=batch_size)

validation_generator = val_datagen.flow_from_directory(
        os.path.join(eye.data_path, eye.val_pre_subpath),
        target_size=(config.get('width'), config.get('height')),
        batch_size=batch_size)


def print_ensemble_history():
    models = []

    # Load all saved models.
    for i in range(0, 10):
        model = module.initialize_model()
        model.load_weights('weights/{0:f}-{1}-{2}.hdf5'.format(
                               config.get('compile_params')
                                     .get('optimizer')
                                     .get_config()
                                     .get('lr'),
                               config.get('name'), i))
        models.append(model)

    ensemble = module.ensemble(models)
    ensemble.compile(**config.get('compile_params'))

    loss = ensemble.evaluate_generator(
        multiclass_flow_from_directory(validation_generator,
                                       transform_target),
        steps=find_num_val_images() // config.get('batch_size_train'))

    print(loss)


for i in range(0, 10):
    print("Settings: Num Epochs: {}, Batch Size: {}"
          .format(num_epochs, batch_size))

    print("Setup model {}...".format(i+1))

    model = module.initialize_model()
    model.compile(**config.get('compile_params'))

    print("Start training {}...".format(i+1))

    model.fit_generator(
        multiclass_flow_from_directory(train_generator, transform_target),
        epochs=num_epochs,
        steps_per_epoch=num_images // batch_size,
        validation_data=multiclass_flow_from_directory(validation_generator,
                                                       transform_target),
        validation_steps=num_val_images // batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.1, patience=3),
            ModelCheckpoint('weights/{0:f}-{1}-{2}.hdf5'.format(
                                       config.get('compile_params')
                                             .get('optimizer')
                                             .get_config()
                                             .get('lr'),
                                       config.get('name'), i),
                                   monitor='val_loss',
                                   save_weights_only='val_loss',
                                   save_best_only=True)])

print("Ensemble history:")
print_ensemble_history()
