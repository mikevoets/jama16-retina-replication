import os
from random import shuffle

import tensorflow as tf

BRIGHTNESS_MAX_DELTA = 0.125
SATURATION_LOWER = 0.5
SATURATION_UPPER = 1.5
HUE_MAX_DELTA = 0.2
CONTRAST_LOWER = 0.5
CONTRAST_UPPER = 1.5


def _tfrecord_dataset_from_folder(folder, ext='.tfrecord'):
    tfrecords = [os.path.join(folder, n)
                 for n in os.listdir(folder) if n.endswith(ext)]
    if len(tfrecords) == 0:
        raise FileNotFoundError(f"No {ext} files found in {folder}.")
    return tf.data.TFRecordDataset(tfrecords)


def _parse_example(proto, num_channels, normalization_fn, augmentation=True):
    features = {"image/encoded": tf.io.FixedLenFeature((), tf.string),
                "image/format": tf.io.FixedLenFeature((), tf.string),
                "image/class/label": tf.io.FixedLenFeature((), tf.int64),
                "image/height": tf.io.FixedLenFeature((), tf.int64),
                "image/width": tf.io.FixedLenFeature((), tf.int64)}
    parsed = tf.io.parse_single_example(proto, features)

    image = tf.io.decode_jpeg(parsed["image/encoded"], num_channels)
    image = normalization_fn(image)

    # Apply data augmentations randomly.
    augmentations = [
        {'fn': tf.image.random_flip_left_right},
        {'fn': tf.image.random_brightness,
         'args': [BRIGHTNESS_MAX_DELTA]},
        {'fn': tf.image.random_saturation,
         'args': [SATURATION_LOWER, SATURATION_UPPER]},
        {'fn': tf.image.random_hue,
         'args': [HUE_MAX_DELTA]},
        {'fn': tf.image.random_contrast,
         'args': [CONTRAST_LOWER, CONTRAST_UPPER]}]

    # The order of augmentations is fixed when the input pipeline is built
    #  (and is seeded with `random.seed` in train.py).
    shuffle(augmentations)

    if augmentation:
        for aug in augmentations:
            if 'args' in aug:
                image = aug['fn'](image, *aug['args'])
            else:
                image = aug['fn'](image)

    label = tf.cast(
                tf.reshape(parsed["image/class/label"], [-1]),
                tf.float32)

    # Images are returned with channels last (NHWC), which works on both CPUs
    #  and GPUs.
    return image, label


def initialize_dataset(image_dir, batch_size, num_epochs=1,
                       num_workers=tf.data.AUTOTUNE, prefetch_buffer_size=None,
                       shuffle_buffer_size=None,
                       normalization_fn=tf.image.per_image_standardization,
                       num_channels=3, augmentation=True):
    # Retrieve data set from folder.
    dataset = _tfrecord_dataset_from_folder(image_dir)

    dataset = dataset.map(
        lambda e: _parse_example(e, num_channels, normalization_fn,
                                 augmentation),
        num_parallel_calls=num_workers)

    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    if prefetch_buffer_size is not None:
        dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset
