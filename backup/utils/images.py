import struct
import imghdr
import os
import tensorflow as tf
import numpy as np

# Development
import pdb
#


def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception:   # IGNORE:W0703
                return
        else:
            return
        return width, height


def image_size(directory):
    first_im_path = os.listdir(directory)[0]
    return get_image_size(directory + first_im_path)


def read_images(labels_path, image_dir, im_size, record_defaults=None):
    if record_defaults is None:
        record_defaults = [[''], [0]]
    # Reading and decoding labels in csv-format
    csv_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = csv_reader.read(tf.train.string_input_producer([labels_path]))
    row = tf.decode_csv(csv_row, record_defaults=record_defaults)

    im_path = row[0]
    label = row[1]
    # Reading and decoding images in jpeg-format
    image_fn = image_dir + im_path + '.jpeg'
    image = tf.read_file(image_fn)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.

    # Explicitily set size of image
    image = tf.image.resize_images(image, im_size)

    return image, label


def input_pipeline(labels_path,
                   image_dir,
                   batch_size,
                   im_size,
                   num_classes,
                   record_defaults=None,
                   min_after_dequeue=10,
                   num_threads=1):
    # Retrieve example image and label
    example, label = read_images(
        labels_path, image_dir, im_size, record_defaults=record_defaults)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    capacity = min_after_dequeue + num_threads * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue, num_threads=num_threads
    )

    labels_batch_one_hot = tf.one_hot(
        indices=label_batch, depth=num_classes, dtype=tf.float32)
    return example_batch, labels_batch_one_hot


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumarate(axes.flat):
        # Plot image
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis
        ax.set_xlabel(xlabel)

        # Remove tickets from the plot
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the polot is shown correctly with multiple plots
    plt.show()
