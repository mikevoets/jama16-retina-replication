"""Convert a directory of images, grouped by class, into TFRecord shards.

Expects `dataset_dir` to contain one subdirectory per class (e.g. `0` and `1`)
with JPEG images. Writes `train-XXXXX-of-YYYYY.tfrecord` shards (and
`validation-...` shards if `--validation_size` > 0), `labels.txt` and
`dimensions.txt` into `dataset_dir`.

This replaces the `create_tfrecords` git submodule that was used by the
TensorFlow 1 version of this code, and writes records in the same format.
"""
import argparse
import math
import os
import random
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import tensorflow as tf  # noqa: E402

LABELS_FILENAME = 'labels.txt'
DIMENSIONS_FILENAME = 'dimensions.txt'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_data),
        'image/format': _bytes_feature(image_format),
        'image/class/label': _int64_feature(class_id),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
    }))


def get_filenames_and_classes(dataset_dir):
    """Returns paths to all images and the sorted list of class names."""
    directories = []
    class_names = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            photo_filenames.append(os.path.join(directory, filename))

    return photo_filenames, sorted(class_names)


def dataset_filename(dataset_dir, split_name, shard_id, num_shards):
    return os.path.join(dataset_dir, '%s-%05d-of-%05d.tfrecord' % (
        split_name, shard_id, num_shards))


def dataset_exists(dataset_dir, num_shards):
    return all(
        os.path.exists(dataset_filename(dataset_dir, split, shard, num_shards))
        for split in ['train', 'validation'] for shard in range(num_shards))


def convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir,
                    num_shards):
    num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))
    dims = set()

    for shard_id in range(num_shards):
        output_filename = dataset_filename(
            dataset_dir, split_name, shard_id, num_shards)

        with tf.io.TFRecordWriter(output_filename) as writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))

            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, len(filenames), shard_id))
                sys.stdout.flush()

                with open(filenames[i], 'rb') as f:
                    image_data = f.read()

                image = tf.io.decode_jpeg(image_data, channels=3)
                height, width = int(image.shape[0]), int(image.shape[1])
                dims.add((width, height))

                class_name = os.path.basename(os.path.dirname(filenames[i]))
                class_id = class_names_to_ids[class_name]

                example = image_to_tfexample(
                    image_data, b'jpg', height, width, class_id)
                writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()

    with open(os.path.join(dataset_dir, DIMENSIONS_FILENAME), 'w') as f:
        for d in dims:
            f.write("{0}x{1}\n".format(*d))


def write_label_file(labels_to_class_names, dataset_dir):
    with open(os.path.join(dataset_dir, LABELS_FILENAME), 'w') as f:
        for label, class_name in labels_to_class_names.items():
            f.write('%d:%s\n' % (label, class_name))


def main():
    parser = argparse.ArgumentParser(
        description="Convert a directory of images grouped by class into "
                    "TFRecord files.")
    parser.add_argument("--dataset_dir", required=True,
                        help="directory with one subdirectory per class")
    parser.add_argument("--validation_size", type=float, default=0.,
                        help="proportion of examples used for validation")
    parser.add_argument("--num_shards", type=int, default=2,
                        help="number of shards to split the TFRecords into")
    parser.add_argument("--random_seed", type=int, default=0,
                        help="random seed for the train/validation split")
    args = parser.parse_args()

    if dataset_exists(args.dataset_dir, args.num_shards):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    photo_filenames, class_names = get_filenames_and_classes(args.dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    num_validation = int(args.validation_size * len(photo_filenames))

    # Divide the data set into a training and a validation set.
    random.seed(args.random_seed)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]

    convert_dataset('train', training_filenames, class_names_to_ids,
                    args.dataset_dir, args.num_shards)

    if num_validation > 0:
        convert_dataset('validation', validation_filenames,
                        class_names_to_ids, args.dataset_dir, args.num_shards)

    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, args.dataset_dir)

    print('\nFinished converting the dataset!')


if __name__ == "__main__":
    main()
