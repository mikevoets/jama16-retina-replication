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

from tensorflow.contrib.keras.api.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.contrib.keras.api.keras.models import Model, load_model
from tensorflow.contrib.keras.api.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.contrib.keras.api.keras.optimizers import SGD

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
# Initializer functions

# Set locations of dataset.
eye.data_path = "data/eyepacs"
eye.train_pre_subpath = "preprocessed/128/train"
eye.val_pre_subpath = "preprocessed/128/val"

# Block and wait until data is available.
# eye.wait_until_available()

# Extract if necessary.
# eye.maybe_extract_images()

# Preprocess if necessary.
# eye.maybe_preprocess()

# Extract labels if necessary.
# eye.maybe_extract_labels()

# Create labels-grouped subdirectories if necessary.
# eye.maybe_create_subdirs_group_by_labels()

# Split training and validation set.
# eye.split_training_and_validation(split=validation_split, seed=seed)

########################################################################


def get_num_files():
    """Get number of files by searching directory recursively"""
    return len(eye._get_image_paths(extension=".jpeg"))


def setup_to_predict(base_model):
    """
    Set all layers to trainable.

    Args:
    model: keras model
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


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


# Training...

config = load_module('eyepacs/configs/128_5x5.py').config

num_epochs = 10
num_images = find_num_train_images()
num_val_images = find_num_val_images()
batch_size = config.get('batch_size_train')

print("Settings: Num Epochs: {}, Batch Size: {}"
      .format(num_epochs, batch_size))
print("Find images...")

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        config.get('train_dir'),
        target_size=(config.get('width'), config.get('height')),
        batch_size=batch_size)

validation_generator = val_datagen.flow_from_directory(
        config.get('val_dir'),
        target_size=(config.get('width'), config.get('height')),
        batch_size=batch_size)

print("Setup model...")

base_model = InceptionV3(weights='imagenet', include_top=False)
model = setup_to_predict(base_model)

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=SGD(lr=3e-3, momentum=0.9, nesterov=True),
              loss='binary_crossentropy', metrics=['accuracy'])

print("Start training...")

model.fit_generator(
    multiclass_flow_from_directory(train_generator, transform_target),
    epochs=num_epochs,
    steps_per_epoch=num_images // batch_size,
    validation_data=multiclass_flow_from_directory(validation_generator,
                                                   transform_target),
    validation_steps=num_val_images // batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0),
               ModelCheckpoint('retina4-weights-128.hdf5',
                               monitor='val_loss',
                               save_weights_only='val_loss',
                               save_best_only=True)]
)
