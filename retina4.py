import os
import sys
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import importlib
from math import ceil

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.utils import class_weight
from PIL import Image

from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

from dataset import one_hot_encoded

# Use the EyePacs dataset.
import eyepacs.v3 as eye

# For debugging purposes.
import pdb

# Ignore Tensorflow logs.
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

########################################################################


class RocAucMetricCallback(Callback):
    def __init__(self, data, steps, val_true):
        super().__init__()
        self.data = data
        self.steps = steps
        self.val_true = val_true

    def on_train_begin(self, logs={}):
        if 'val_roc_auc' not in self.params['metrics']:
            self.params['metrics'].append('val_roc_auc')
        if 'val_sensitivity' not in self.params['metrics']:
            self.params['metrics'].append('val_sensitivity')
        if 'val_specificity' not in self.params['metrics']:
            self.params['metrics'].append('val_specificity')

    def on_epoch_end(self, epoch, logs={}):
        logs['val_roc_auc'] = float('-inf')

        y_score = self.model.predict_generator(self.data, self.steps)
        # Only calculate metrics on RDR (label 0).
        logs['val_roc_auc'] = roc_auc_score(self.val_true[:, 0], y_score[:, 0])

        target_names = ["Moderate+", "Severe+"]

        print("\nIteration {} ________________________________________________"
              "______________________".format(epoch))
        print("\nClassification report")
        print(classification_report(self.val_true, np.rint(y_score),
                                    target_names=target_names))

        for i in range(2):
            y_pred = np.rint(y_score[:, i])
            _confusion_matrix = confusion_matrix(self.val_true[:, i], y_pred)
            print("\nConfusion matrix for {}".format(target_names[i]))
            print(_confusion_matrix)

            tn, fp, fn, tp = _confusion_matrix.ravel()
            print("Sensitivity: {:5f}, Specificity: {:5f}"
                  .format(tp / (tp+fn), tn / (tn+fp)))

        print("\n_____________________________________________________________"
              "______________________\n")


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


def transform_target(y, one_hot=True):
    t = np.zeros((y.shape[0], 2), dtype=np.float32)
    # Convert one-hot encoded to labels.
    if one_hot is True:
        y = y.argmax(axis=1)
    # Transform the m-class y to a 2-label y.
    t[np.argwhere(y > 1).reshape(-1), 0] = 1
    t[np.argwhere(y > 2).reshape(-1), 1] = 1
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
eye.test_pre_subpath = config.get('test_dir')

num_epochs = 200
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
        batch_size=batch_size,
        shuffle=False)


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
        steps=ceil(find_num_val_images() / config.get('batch_size_train')))

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
        steps_per_epoch=ceil(num_images / batch_size),
        validation_data=multiclass_flow_from_directory(validation_generator,
                                                       transform_target),
        validation_steps=ceil(num_val_images / batch_size),
        callbacks=[RocAucMetricCallback(
                       data=multiclass_flow_from_directory(
                                validation_generator, transform_target),
                       steps=ceil(num_val_images / batch_size),
                       val_true=transform_target(validation_generator.classes,
                                                 one_hot=False)),
                   EarlyStopping(
                       monitor='val_roc_auc', mode='max', patience=3),
                   ModelCheckpoint(
                       'weights/{0:f}-{1}-{2}.hdf5'.format(
                           config.get('compile_params')
                                 .get('optimizer')
                                 .get_config()
                                 .get('lr'),
                           config.get('name'), i),
                       monitor='val_roc_auc',
                       mode='max',
                       save_weights_only='val_roc_auc',
                       save_best_only=True)])

print("Ensemble history:")
print_ensemble_history()
