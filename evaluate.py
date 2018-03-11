import re
import os
import sys
import argparse
import random
import tensorflow as tf
import numpy as np
import lib.dataset
import lib.evaluation
import lib.metrics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from glob import glob

print(f"Numpy version: {np.__version__}")
print(f"Tensorflow version: {tf.__version__}")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
random.seed(432)

# Default settings.
default_eyepacs_dir = "./data/eyepacs/bin2/test"
default_messidor_dir = "./data/messidor/bin2"
default_load_model_path = "./tmp/model"
default_batch_size = 32

parser = argparse.ArgumentParser(
                    description="Evaluate performance of trained graph "
                                "on test data set. "
                                "Specify --data_dir if you use the -o param.")
parser.add_argument("-m", "--messidor", action="store_true",
                    help="evaluate performance on Messidor-Original")
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
parser.add_argument("-sr", "--save_roc_plot_path",
                    help="path to where roc plot should be saved")
parser.add_argument("-b", "--batch_size",
                    help="batch size", default=default_batch_size)
parser.add_argument("-op", "--operating_threshold",
                    help="operating threshold", default=0.5)

args = parser.parse_args()

if bool(args.eyepacs) == bool(args.messidor) == bool(args.other):
    print("Can only evaluate one data set at once!")
    parser.print_help()
    sys.exit(2)

if args.data_dir is not None:
    data_dir = str(args.data_dir)
elif args.eyepacs:
    data_dir = default_eyepacs_dir
elif args.messidor:
    data_dir = default_messidor_dir
elif args.other and args.data_dir is None:
    print("Please specify --data_dir.")
    parser.print_help()
    sys.exit(2)

load_model_path = str(args.load_model_path)
batch_size = int(args.batch_size)
save_roc_plot_path = str(args.save_roc_plot_path)
operating_threshold = float(args.operating_threshold)

# Check if the model path has comma-separated entries.
if "," in load_model_path:
    load_model_paths = load_model_path.split(",")
# Check if the model path has a regexp character.
elif any(char in load_model_path for char in '*+?'):
    load_model_paths = [".".join(x.split(".")[:-1])
                        for x in glob("{}*".format(load_model_path))]
    load_model_paths = list(set(load_model_paths))
else:
    load_model_paths = [load_model_path]

print("""
Evaluating: {},
Saving AUROC plot at: {},
Using operating treshold: {},
""".format(data_dir, save_roc_plot_path, operating_threshold))
print("Trying to load model(s):\n{}".format("\n".join(load_model_paths)))

# Other setting variables.
num_channels = 3
num_workers = 8
prefetch_buffer_size = 2 * batch_size
num_thresholds = 200
kepsilon = 1e-7

# Define thresholds.
thresholds = lib.metrics.generate_thresholds(num_thresholds, kepsilon) \
                + [operating_threshold]

# Set image datas format to channels first if GPU is available.
if tf.test.is_gpu_available():
    print("Found GPU! Using channels first as default image data format.")
    image_data_format = 'channels_first'
else:
    image_data_format = 'channels_last'


def save_roc_plot(specificities, sensitivities, auc):
    fig = plt.figure()
    plt.plot(np.array([(1.0 - n) for n in sensitivities]),
             specificities,
             color="darkorange", lw=2,
             label="ROC curve (area = {:0.2f})".format(auc))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Sensitivity")
    plt.ylabel("Specificity")
    plt.title("Receiver operating curve")
    plt.legend(loc="lower right")
    fig.savefig(save_roc_plot_path)


got_all_y = False
all_y = []


def feed_images(sess, x_tensor, y_tensor, x_batcher, y_batcher):
    _x, _y = sess.run([x_batcher, y_batcher])
    if not got_all_y:
        all_y.append(_y)
    return {x_tensor: _x, y_tensor: _y}


eval_graph = tf.Graph()
with eval_graph.as_default() as g:
    # Variable for average predictions.
    average_predictions = tf.placeholder(tf.float32, shape=[None, 1])
    all_labels = tf.placeholder(tf.float32, shape=[None, 1])

    # Metrics for finding best validation set.
    tp, update_tp, reset_tp = lib.metrics.create_reset_metric(
        tf.metrics.true_positives_at_thresholds, scope='tp',
        labels=all_labels, predictions=average_predictions,
        thresholds=thresholds)

    fp, update_fp, reset_fp = lib.metrics.create_reset_metric(
        tf.metrics.false_positives_at_thresholds, scope='fp',
        labels=all_labels, predictions=average_predictions,
        thresholds=thresholds)

    fn, update_fn, reset_fn = lib.metrics.create_reset_metric(
        tf.metrics.false_negatives_at_thresholds, scope='fn',
        labels=all_labels, predictions=average_predictions,
        thresholds=thresholds)

    tn, update_tn, reset_tn = lib.metrics.create_reset_metric(
        tf.metrics.true_negatives_at_thresholds, scope='tn',
        labels=all_labels, predictions=average_predictions,
        thresholds=thresholds)

    # Last element presents the metrics at operating threshold.
    confusion_matrix = lib.metrics.confusion_matrix(
        tp[-1], fp[-1], fn[-1], tn[-1], scope='confusion_matrix')

    brier, update_brier, reset_brier = lib.metrics.create_reset_metric(
        tf.metrics.mean_squared_error, scope='brier',
        labels=all_labels, predictions=average_predictions)

    auc, update_auc, reset_auc = lib.metrics.create_reset_metric(
        tf.metrics.auc, scope='auc',
        labels=all_labels, predictions=average_predictions)

    specificities = tf.div(tn, tn + fp + kepsilon)
    sensitivities = tf.div(tp, tp + fn + kepsilon)


all_predictions = []

for model_path in load_model_paths:
    # Start session.
    with tf.Session(graph=tf.Graph()) as sess:
        tf.keras.backend.set_session(sess)
        tf.keras.backend.set_learning_phase(False)
        tf.keras.backend.set_image_data_format(image_data_format)

        # Load the meta graph and restore variables from training.
        saver = tf.train.import_meta_graph("{}.meta".format(model_path))
        saver.restore(sess, model_path)

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")

        try:
            predictions = graph.get_tensor_by_name("predictions:0")
        except KeyError:
            predictions = graph.get_tensor_by_name("predictions/Sigmoid:0")

        # Initialize the test set.
        test_dataset = lib.dataset.initialize_dataset(
            data_dir, batch_size,
            num_workers=num_workers, prefetch_buffer_size=prefetch_buffer_size,
            image_data_format=image_data_format, num_channels=num_channels)

        # Create an iterator.
        iterator = tf.data.Iterator.from_structure(
            test_dataset.output_types, test_dataset.output_shapes)

        test_images, test_labels = iterator.get_next()

        test_init_op = iterator.make_initializer(test_dataset)

	    # Perform the evaluation.
        test_predictions = lib.evaluation.perform_test(
            sess=sess, init_op=test_init_op,
            feed_dict_fn=feed_images,
            feed_dict_args={"sess": sess, "x_tensor": x, "y_tensor": y,
                            "x_batcher": test_images, "y_batcher": test_labels},
            custom_tensors=[predictions])

        all_predictions.append(test_predictions[0])

    tf.reset_default_graph()
    got_all_y = True

# Convert the predictions to a numpy array.
all_predictions = np.array(all_predictions)

# Calculate the linear average of all predictions.
avg_pred = np.mean(all_predictions, axis=0)

# Convert all labels to numpy array.
all_y = np.vstack(all_y)

# Use these predictions for printing evaluation results.
with tf.Session(graph=eval_graph) as sess:
    # Reset all streaming variables.
    sess.run([reset_tp, reset_fp, reset_fn, reset_tn, reset_brier, reset_auc])

    # Update all streaming variables with predictions.
    sess.run([update_tp, update_fp, update_fn, update_tn,
              update_brier, update_auc],
              feed_dict={average_predictions: avg_pred, all_labels: all_y})

    # Retrieve confusion matrix and estimated roc auc score.
    test_conf_matrix, test_brier, test_auc, test_specificities, \
        test_sensitivities = \
            sess.run([confusion_matrix, brier, auc, specificities,
                      sensitivities])

    # Plot and save ROC curve figure to a specified path.
    if save_roc_plot_path is not None:
        save_roc_plot(test_specificities, test_sensitivities, test_auc)

    # Print total roc auc score for validation.
    print(f"Brier score: {test_brier:6.4}, AUC: {test_auc:10.8}")

    # Print confusion matrix.
    print(f"Confusion matrix at operating threshold {operating_threshold:0.3f}")
    print(test_conf_matrix[0])

    # Print sentivities and specificities.
    for idx in range(num_thresholds + 1):
        print("Specificity: {0:0.4f}, Sensitivity: {1:0.4f} at " \
              "Operating Threshold {2:0.4f}." \
              .format(test_specificities[idx], test_sensitivities[idx],
                      thresholds[idx]))

sys.exit(0)
