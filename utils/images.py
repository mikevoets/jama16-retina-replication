import struct
import imghdr
import os
import matplotlib.pyplot as plt
import tensorflow as tf


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


def read_images(filename_queue, labels_csv_path):
    # Reading and decoding labels in csv-format
    csv_reader = tf.TextLineReader()
    record_defaults = [[''], ['0']]
    _, csv_content = csv_reader.read(labels_csv_path)
    label = tf.decode_csv(
        csv_content, record_defaults=record_defaults)

    # Reading and decoding images in jpeg-format
    im_reader = tf.WholeFileReader()
    im_filename, im_content = im_reader.read(filename_queue)
    image = tf.image.decode_jpeg(im_content)
    image = tf.cast(image, tf.float32) / 255.
    return image, label


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
