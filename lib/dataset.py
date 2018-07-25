import tensorflow as tf
import os


def _tfrecord_dataset_from_folder(folder, ext='.tfrecord'):
    tfrecords = [os.path.join(folder, n)
                 for n in os.listdir(folder) if n.endswith(ext)]
    return tf.data.TFRecordDataset(tfrecords)


def _parse_example(proto, num_channels, image_data_format):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string),
                "image/format": tf.FixedLenFeature((), tf.string),
                "image/class/label": tf.FixedLenFeature((), tf.int64),
                "image/height": tf.FixedLenFeature((), tf.int64),
                "image/width": tf.FixedLenFeature((), tf.int64)}
    parsed = tf.parse_single_example(proto, features)

    # Standardize image.
    image = tf.image.per_image_standardization(
        tf.image.decode_jpeg(parsed["image/encoded"], num_channels))

    if image_data_format == 'channels_first':
        image = tf.transpose(image, [2, 0, 1])

    label = tf.cast(
        tf.reshape(parsed["image/class/label"], [-1]),
        tf.float32)

    return image, label


def initialize_dataset(image_dir, batch_size, num_epochs=1,
                       num_workers=1, prefetch_buffer_size=None,
                       shuffle_buffer_size=None,
                       image_data_format='channels_last',
                       num_channels=3, image_dim=[299, 299]):
    # Retrieve data set from pattern.
    dataset = _tfrecord_dataset_from_folder(image_dir)

    dataset = dataset.map(
        lambda e: _parse_example(e, num_channels, image_data_format),
        num_parallel_calls=num_workers)

    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    if prefetch_buffer_size is not None:
        dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset
