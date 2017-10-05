import os
import sys
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.contrib.keras.api.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.api.keras.optimizers import SGD

from dataset import one_hot_encoded

# Use the EyePacs dataset.
import eyepacs.v2
from eyepacs.v2 import num_classes

# For debugging purposes.
import pdb

# Ignore Tensorflow logs.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

########################################################################
# Various constants.

# Shape of a preprocessed image.
image_shape = (299, 299)

# Number of epochs.
num_epochs = 3

# Batch size.
batch_size = 32

# Fully-connected layer size.
fully_connected_size = 1024

# Amount of Inception V3 layers to freeze.
num_iv3_layers_freeze = 172

# Define the ratio of training-validation data.
validation_split = 0.1

########################################################################
# Initializer functions

# Extract if necessary.
eyepacs.v2.maybe_extract_images()

# Preprocess if necessary.
eyepacs.v2.maybe_preprocess()

# Extract labels if necessary.
eyepacs.v2.maybe_extract_labels()

# Create labels-grouped subdirectories if necessary.
eyepacs.v2.maybe_create_subdirs_group_by_labels()

# Split training and validation set.
eyepacs.v2.split_training_and_validation(split=validation_split)

########################################################################


def get_num_files(test=False):
    """Get number of files by searching directory recursively"""
    return len(eyepacs.v2._get_image_paths(test=test, extension=".jpeg"))


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


def setup_to_finetune(model):
    """Freeze the bottom num_iv3_layers_freeze and retrain the remaining top layers.

    note: num_iv3_layers_freeze corresponds to the top 2 inception blocks
          in the inception v3 architecture

    Args:
    model: keras model
    """
    for layer in model.layers[:num_iv3_layers_freeze]:
        layer.trainable = False
    for layer in model.layers[num_iv3_layers_freeze:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])


def find_num_train_images():
    """Helper function for finding amount of training images."""
    train_images_dir = os.path.join(
        eyepacs.v2.data_path, eyepacs.v2.train_pre_subpath)

    return len(eyepacs.v2._get_image_paths(images_dir=train_images_dir))


def train(args):
    """
    Use transfer learning and fine-tuning to train a network on a new dataset
    """
    num_images = find_num_train_images()
    num_epochs = int(args.num_epochs)
    batch_size = int(args.batch_size)

    print()
    print("Find images...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            os.path.join(eyepacs.v2.data_path, eyepacs.v2.train_pre_subpath),
            target_size=image_shape,
            batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(
            os.path.join(eyepacs.v2.data_path, eyepacs.v2.val_pre_subpath),
            target_size=image_shape,
            batch_size=batch_size)

    print("Setup model...")

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = add_new_last_layer(base_model, num_classes)

    # First train only the top layers.
    # I.e. freeze all convolutional InceptionV3 layers.
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model.
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    print("Train the model on the new retina data for a few epochs...")

    model.fit_generator(
            train_generator,
            steps_per_epoch=int(num_images/batch_size),
            epochs=num_epochs,
            validation_data=validation_generator,
            validation_steps=50)

    print("Fine-tuning model...")

    setup_to_finetune(model)

    print("Start training again...")

    model.fit_generator(
            train_generator,
            steps_per_epoch=int(num_images/batch_size),
            epochs=num_epochs,
            validation_data=validation_generator,
            validation_steps=50)

    history_ft = model.save(args.output_model_file)

    if args.plot:
        plot_training(history_ft)


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--num_epochs", default=num_epochs)
    a.add_argument("--batch_size", default=batch_size)
    a.add_argument("--output_model_file", default="inceptionv3-ft.model")
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()

    train(args)
