import re
import os
import sys
import argparse
import random
import tensorflow as tf
import numpy as np
import lib.dataset
import lib.evaluation
from glob import glob

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
                    help="path to where graph model should be loaded from "
                         "creates an ensemble if paths are comma separated "
                         "or a regexp",
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

# Check if the model path has comma-separated entries.
if "," in load_model_path:
    load_model_paths = load_model_path.split(",")
# Check if the model path has a regexp character.
elif any(char in load_model_path for char in '*+?'):
    load_model_paths = [".".join(x.split(".")[:-1])
                        for x in glob("{}*".format(load_model_path))]
    load_model_paths = list(set(load_model_paths))
else:
    load_model_paths = [load_model_paths]

print("Found model(s):\n{}".format("\n".join(load_model_paths)))

# Other setting variables.
num_channels = 3
num_workers = 8
prefetch_buffer_size = 2 * batch_size

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

# Initialize the test set.
test_dataset = lib.dataset.initialize_dataset(
    data_dir, batch_size,
    num_workers=num_workers, prefetch_buffer_size=prefetch_buffer_size,
    image_data_format=image_data_format, num_channels=num_channels)

# Create an initialize iterators.
iterator = tf.data.Iterator.from_structure(
    test_dataset.output_types, test_dataset.output_shapes)

test_images, test_labels = iterator.get_next()

test_init_op = iterator.make_initializer(test_dataset)

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y = graph.get_tensor_by_name("y:0")
predictions = graph.get_tensor_by_name("predictions:0")
avg_pred = graph.get_tensor_by_name("avg_pred:0")

def feed_images():
    x_test, y_test = sess.run([test_images, test_labels])
    return {x: x_test, y: y_test}

all_predictions = []

for model_path in load_model_paths:
    # Load the meta graph and restore variables from training.
    saver = tf.train.import_meta_graph("{}.meta".format(model_path))
    saver.restore(sess, model_path)

    # Perform the evaluation.
    test_predictions = lib.evaluation.perform_test(
        sess=sess, init_op=test_init_op, feed_dict_fn=feed_images,
        custom_tensors=predictions)

    all_predictions.append(test_predictions)

# Convert the predictions to a numpy array.
all_predictions = np.array(all_predictions)

# Calculate the linear average of all predictions.
average_predictions = np.mean(all_predictions, axis=0)

# Use these predictions for printing evaluation results.
lib.evaluation.perform_test(
    sess=sess, init_op=test_init_op,
    feed_dict_fn=lambda f: {avg_pred: average_predictions},
    batch_mode=False)

sess.close()
sys.exit(0)
