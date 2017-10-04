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
import eyepacs
from eyepacs import num_classes

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
validation_split = 0.2

########################################################################
# Initializer functions

# Extract if necessary.
eyepacs.maybe_extract_images()

# Preprocess if necessary.
eyepacs.maybe_preprocess()

# Extract labels if necessary.
eyepacs.maybe_extract_labels()

# Split training and validation set.
eyepacs.split_training_and_validation(split=validation_split)

########################################################################


def get_num_files(test=False):
    """Get number of files by searching directory recursively"""
    return len(eyepacs._get_image_paths(test=test, extension=".jpeg"))


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


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

    # New fully-connected layer, with random initializers
    x = Dense(fully_connected_size, activation='relu')(x)

    # New softmax classifier
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks
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


def image_generator(labels_path, images_path):
    """
    Helper method for generating training samples.
    """
    while 1:
        # Open label file.
        with open(labels_path, 'rt') as r:
            reader = csv.reader(r, delimiter=",")

            # Read csv label file.
            for num, line in enumerate(reader):
                # Retrieve image path.
                image_path = os.path.join(images_path, line[0] + '.jpeg')

                # Retrieve image.
                image = Image.open(image_path)
                image = np.array(image.getdata())
                image = 2*(image.reshape((-1, image_shape[0], image_shape[1], 3))
                           .astype(np.float32)/255) - 1

                # Retrieve label.
                label = int(line[1])

                # Convert to one-hot-encoding.
                cls_one_hot = one_hot_encoded(
                    class_numbers=label, num_classes=num_classes)

                cls_one_hot = cls_one_hot.reshape((-1, num_classes))

                yield (image, cls_one_hot)


def train_generator():
    labels_path = os.path.join(
        eyepacs.data_path, eyepacs.train_labels_extracted)
    images_path = os.path.join(eyepacs.data_path, eyepacs.train_pre_subpath)

    return image_generator(labels_path=labels_path, images_path=images_path)


def val_generator():
    labels_path = os.path.join(eyepacs.data_path, eyepacs.val_labels_extracted)
    images_path = os.path.join(eyepacs.data_path, eyepacs.val_pre_subpath)

    return image_generator(labels_path=labels_path, images_path=images_path)


def train(args):
    """
    Use transfer learning and fine-tuning to train a network on a new dataset
    """
    num_training_samples = get_num_files()
    num_epochs = int(args.num_epochs)
    batch_size = int(args.batch_size)

    # setup model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = add_new_last_layer(base_model, num_classes)

    # Transfer learning.
    #setup_to_transfer_learn(model, base_model)

    #history_tl = model.fit_generator(
    #    train_generator(),
    #    epochs=num_epochs,
    #    steps_per_epoch=num_training_samples,
    #    class_weight='auto')

    # Fine-tuning.
    setup_to_finetune(model)

    history_ft = model.fit_generator(
        train_generator(),
        epochs=num_epochs,
        steps_per_epoch=num_training_samples,
        validation_data=val_generator(),
        validation_steps=800,
        class_weight='auto')

    model.save(args.output_model_file)

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
