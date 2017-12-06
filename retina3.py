import os
import sys
import importlib
import numpy as np
import h5py

from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Lambda
from tensorflow.contrib.keras.api.keras.optimizers import SGD
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard

from sklearn.utils import class_weight
from collections import OrderedDict
from computations import MEAN, STD

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
eye.data_path = "data/eyepacs/"
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

# Convert the images to 256 and 128 pixels.
# eye.maybe_convert()


def load_module(mod):
    return importlib.import_module(mod.replace('/', '.').split('.py')[0])


config = load_module('eyepacs/configs/128_5x5.py').config


def make_model(layers):
    model = Sequential()

    for func, params in layers:
        if 'lambda' in params and params['lambda']:
            del params['lambda']
            model.add(Lambda(func, arguments=params))
        else:
            model.add(func(**params))

    return model


def find_num_train_images():
    """Helper function for finding amount of training images."""
    train_images_dir = os.path.join(eye.data_path, eye.train_pre_subpath)

    return len(eye._get_image_paths(images_dir=train_images_dir))


def find_num_val_images():
    """Helper function for finding amount of validation images."""
    val_images_dir = os.path.join(eye.data_path, eye.val_pre_subpath)

    return len(eye._get_image_paths(images_dir=val_images_dir))


model = make_model(config.layers)
model.compile(optimizer=SGD(lr=3e-3, momentum=0.9, nesterov=True),
              loss='mean_squared_error', metrics=['accuracy'])

layer_dict = dict([(layer.name, layer) for layer in model.layers])

train_datagen = ImageDataGenerator(**config.get('augmentation_params'))
train_datagen.mean = MEAN
train_datagen.std = STD
train_generator = train_datagen.flow_from_directory(
    config.get('train_dir'),
    target_size=(config.get('width'), config.get('height')),
    batch_size=config.get('batch_size_train'))

val_datagen = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True)
val_datagen.mean = MEAN
val_datagen.std = STD
val_generator = val_datagen.flow_from_directory(
    config.get('val_dir'),
    target_size=(config.get('width'), config.get('height')),
    batch_size=config.get('batch_size_train'))

class_weight = class_weight.compute_class_weight(
    'balanced', np.unique(train_generator.classes), train_generator.classes)

def learning_rate_schedule(epoch):
    s = config.get('learn_rate_schedule')

    lr = s[0]
    for k, v in OrderedDict(sorted(s.items())).items():
        if epoch < k:
            break
        lr = v

    return lr


def load_weights(name, obj):
    for key, val in obj.attrs.items():
        if name in layer_dict:
            try:
                # Load kernel and bias weights.
                weights = [obj[v].value for v in val]
                layer_dict[name].set_weights(weights)
            except ValueError:
                # Do not load due to shape mismatch.
                pass


# f = h5py.File('weights-128.hdf5', 'r')
# f.visititems(load_weights)

callbacks = [
    ModelCheckpoint(
        'weights-128-local.hdf5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True),
    TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        batch_size=find_num_val_images() // config.get('batch_size_train'),
        write_images=True),
    LearningRateScheduler(learning_rate_schedule)
]

model.fit_generator(
    train_generator,
    epochs=200,
    class_weight=dict(enumerate(class_weight)),
    steps_per_epoch=find_num_train_images() // config.get('batch_size_train'),
    validation_data=val_generator,
    validation_steps=find_num_val_images() // config.get('batch_size_train'),
    callbacks=callbacks
)
