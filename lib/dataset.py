import tensorflow as tf
import os


def _tfrecord_dataset_from_folder(folder, ext='.tfrecord'):
    tfrecords = [os.path.join(folder, n)
                 for n in os.listdir(folder) if n.endswith(ext)]
    return tf.data.TFRecordDataset(tfrecords)


def _parse_example(proto, image_dim):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string),
                "image/format": tf.FixedLenFeature((), tf.string),
                "image/class/label": tf.FixedLenFeature((), tf.int64),
                "image/height": tf.FixedLenFeature((), tf.int64),
                "image/width": tf.FixedLenFeature((), tf.int64)}
    parsed = tf.parse_single_example(proto, features)

    # Rescale to 1./255.
    image = tf.image.convert_image_dtype(
        tf.image.decode_jpeg(parsed["image/encoded"]), tf.float32)

    image = tf.reshape(image, image_dim)
    label = tf.cast(parsed["image/class/label"], tf.int32)

    return image, label


def initialize_dataset(image_dir, batch_size, num_epochs=1,
                       num_workers=1, prefetch_buffer_size=None,
                       shuffle_buffer_size=None,
                       image_data_format='channels_last',
                       num_channels=3):
    # Retrieve data set from pattern.
    dataset = _tfrecord_dataset_from_folder(image_dir)

    # Find metadata file with image dimensions.
    image_dim_fn = os.path.join(image_dir, 'dimensions.txt')

    # Parse dimensions.txt file in data set folder for image dimensions.
    with open(image_dim_fn, 'r') as f:
        image_dims = list(filter(None, f.read().split('\n')))

        if len(image_dims) > 1:
            raise TypeError(
                "can't initialize dataset with multiple image dims")

        image_dim = [int(x) for x in image_dims[0].split('x')]

    # Specify image shape.
    if image_data_format == 'channels_first':
        image_dim = [num_channels, image_dim[0], image_dim[1]]
    elif image_data_format == 'channels_last':
        image_dim = [image_dim[0], image_dim[1], num_channels]
    else:
        raise TypeError('invalid image date format setting')

    dataset = dataset.map(lambda e: _parse_example(e, image_dim),
                          num_parallel_calls=num_workers)

    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    if prefetch_buffer_size is not None:
        dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset
