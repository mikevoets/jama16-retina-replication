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
import eyepacs.v2 as eye
from eyepacs.v2 import num_classes

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


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet

    Args:
    base_model: keras model excluding top
    nb_classes: # of classes

    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # New fully-connected layer, with random initializers.
    x = Dense(fully_connected_size, activation='relu')(x)

    # New softmax classifier.
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def setup_to_finetune(model, num_layers_freeze):
    """Freeze the bottom num_iv3_layers_freeze and retrain the remaining top layers.

    note: num_iv3_layers_freeze corresponds to the top 2 inception blocks
          in the inception v3 architecture

    Args:
    model: keras model
    """
    for layer in model.layers[:num_layers_freeze]:
        layer.trainable = False
    for layer in model.layers[num_layers_freeze:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=3e-3, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy', metrics=['accuracy'])


def find_num_train_images():
    """Helper function for finding amount of training images."""
    train_images_dir = os.path.join(eye.data_path, eye.train_pre_subpath)

    return len(eye._get_image_paths(images_dir=train_images_dir))


def find_num_val_images():
    """Helper function for finding amount of training images."""
    val_images_dir = os.path.join(eye.data_path, eye.val_pre_subpath)

    return len(eye._get_image_paths(images_dir=val_images_dir))


def train(config, num_epochs, num_layers_freeze, transfer_num_epochs=2):
    """
    Use transfer learning and fine-tuning to train a network on a new dataset
    """
    num_images = find_num_train_images()
    num_val_images = find_num_val_images()
    batch_size = config.get('batch_size_train')

    print()
    print("Settings: Num Epochs: {}, Batch Size: {}, Freeze Layers: {}"
          .format(num_epochs, batch_size, num_layers_freeze))
    print("Find images...")

    train_datagen = ImageDataGenerator(**config.get('augmentation_params'))
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
    model = add_new_last_layer(base_model, num_classes)

    # First train only the top layers.
    # I.e. freeze all convolutional InceptionV3 layers.
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model.
    model.compile(optimizer=SGD(lr=3e-3, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy')

    print("Train the model on the new retina data for a few epochs...")

    class_weight_dict = dict(enumerate(class_weight.compute_class_weight(
        'balanced',
        np.unique(train_generator.classes),
        train_generator.classes
    )))

    model.fit_generator(
        train_generator,
        epochs=transfer_num_epochs,
        class_weight=class_weight_dict,
        steps_per_epoch=num_images // batch_size,
        validation_data=validation_generator,
        validation_steps=num_val_images // batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0)]
    )

    print("Fine-tuning model...")

    setup_to_finetune(model, num_layers_freeze)

    print("Start training again...")

    model.fit_generator(
        train_generator,
        epochs=num_epochs,
        class_weight=class_weight_dict,
        steps_per_epoch=num_images // batch_size,
        validation_data=validation_generator,
        validation_steps=num_val_images // batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0),
                   ModelCheckpoint('retina2-weights-128.hdf5',
                                   monitor='val_loss',
                                   save_weights_only=True,
                                   save_best_only=True)]
    )


def load_module(mod):
    return importlib.import_module(mod.replace('/', '.').split('.py')[0])


if __name__ == "__main__":
    config128 = load_module('eyepacs/configs/128_5x5.py').config
    train(config128, 200, 178, transfer_num_epochs=50)
