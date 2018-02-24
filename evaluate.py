import os
import sys
import argparse
import random
import tensorflow as tf
import numpy as np
import lib.dataset
import lib.evaluation

print(f"Numpy version: {np.__version__}")
print(f"Tensorflow version: {tf.__version__}")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
random.seed(432)

# Default settings.
default_eyepacs_dir = "./data/eyepacs/bin2/test"
default_messidor2_dir = "./data/messidor2/bin2"
default_load_model_path = "./tmp/model"
default_batch_size = 32

parser = argparse.ArgumentParser(
                    description="Evaluate performance of trained graph "
                                "on test data set. "
                                "Specify --data_dir if you use the -o param.")
parser.add_argument("-m", "--messidor2", action="store_true",
                    help="evaluate performance on Messidor-2")
parser.add_argument("-e", "--eyepacs", action="store_true",
                    help="evaluate performance on EyePacs set")
parser.add_argument("-o", "--other", action="store_true",
                    help="evaluate performance on your own dataset")
parser.add_argument("--data_dir", help="directory where data set resides")
parser.add_argument("-lm", "--load_model_path",
                    help="path to where graph model should be loaded from",
                    default=default_load_model_path)
parser.add_argument("-b", "--batch_size",
                    help="batch size", default=default_batch_size)

args = parser.parse_args()

if bool(args.eyepacs) == bool(args.messidor2) == bool(args.other):
    print("Can only evaluate one data set at once!")
    parser.print_help()
    sys.exit(2)

if args.data_dir is not None:
    data_dir = str(args.data_dir)
elif args.eyepacs:
    data_dir = default_eyepacs_dir
elif args.messidor2:
    data_dir = default_messidor2_dir
elif args.other and args.data_dir is None:
    print("Please specify --data_dir.")
    parser.print_help()
    sys.exit(2)

load_model_path = str(args.load_model_path)
batch_size = int(args.batch_size)

# Other setting variables.
num_channels = 3
num_workers = 8
shuffle_buffer_size = 5000
prefetch_buffer_size = 100 * batch_size

# Set image datas format to channels first if GPU is available.
if tf.test.is_gpu_available():
    print("Found GPU! Using channels first as default image data format.")
    image_data_format = 'channels_first'
else:
    image_data_format = 'channels_last'

# Start session.
sess = tf.Session()
tf.keras.backend.set_session(sess)
tf.keras.backend.set_learning_phase(False)
tf.keras.backend.set_image_data_format(image_data_format)

# Load the meta graph and restore variables from training.
saver = tf.train.import_meta_graph("{}.meta".format(load_model_path))
saver.restore(sess, load_model_path)

# Initialize the test set.
test_dataset = lib.dataset.initialize_dataset(
    data_dir, batch_size,
    num_workers=num_workers, prefetch_buffer_size=prefetch_buffer_size,
    shuffle_buffer_size=shuffle_buffer_size,
    image_data_format=image_data_format, num_channels=num_channels)

# Create an initialize iterators.
iterator = tf.data.Iterator.from_structure(
    test_dataset.output_types, test_dataset.output_shapes)

test_images, test_labels = iterator.get_next()

test_init_op = iterator.make_initializer(test_dataset)

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y = graph.get_tensor_by_name("y:0")


def get_feed_dict():
    x_test, y_test = sess.run([test_images, test_labels])
    return {x: x_test, y: y_test}


# Perform the evaluation.
lib.evaluation.perform_test(sess=sess, init_op=test_init_op,
                            feed_dict_fn=get_feed_dict)

sess.close()
sys.exit(0)
