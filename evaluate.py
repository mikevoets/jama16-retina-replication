import argparse
import os
import random
import sys
from glob import glob

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

import lib.dataset  # noqa: E402
import lib.evaluation  # noqa: E402
import lib.metrics  # noqa: E402
import lib.model  # noqa: E402
from lib.preprocess import rescale_min_1_to_1  # noqa: E402

print(f"Numpy version: {np.__version__}")
print(f"Tensorflow version: {tf.__version__}")

random.seed(432)

# Default settings.
default_eyepacs_dir = "./data/eyepacs/bin2/test"
default_messidor_dir = "./data/messidor/bin2"
default_messidor2_dir = "./data/messidor2/bin2"
default_load_model_path = "./tmp/model"
default_save_operating_thresholds_path = "./tmp/test_op_pts.csv"
default_batch_size = 32

parser = argparse.ArgumentParser(
                    description="Evaluate performance of trained graph "
                                "on test data set. "
                                "Specify --data_dir if you use the -o param.")
parser.add_argument("-m", "--messidor", action="store_true",
                    help="evaluate performance on Messidor-Original")
parser.add_argument("-m2", "--messidor2", action="store_true",
                    help="evaluate performance on Messidor-2")
parser.add_argument("-e", "--eyepacs", action="store_true",
                    help="evaluate performance on EyePacs set")
parser.add_argument("-o", "--other", action="store_true",
                    help="evaluate performance on your own dataset")
parser.add_argument("--data_dir", help="directory where data set resides")
parser.add_argument("-lm", "--load_model_path",
                    help="path to where model should be loaded from; "
                         "creates an ensemble if paths are comma separated "
                         "or a glob pattern",
                    default=default_load_model_path)
parser.add_argument("-so", "--save_operating_thresholds_path",
                    help="path to where operating points metrics should be saved",
                    default=default_save_operating_thresholds_path)
parser.add_argument("-b", "--batch_size",
                    help="batch size", default=default_batch_size)
parser.add_argument("-op", "--operating_threshold",
                    help="operating threshold", default=0.5)

args = parser.parse_args()

if sum([bool(args.eyepacs), bool(args.messidor), bool(args.messidor2),
        bool(args.other)]) != 1:
    print("Specify exactly one data set to evaluate!")
    parser.print_help()
    sys.exit(2)

if args.data_dir is not None:
    data_dir = str(args.data_dir)
elif args.eyepacs:
    data_dir = default_eyepacs_dir
elif args.messidor:
    data_dir = default_messidor_dir
elif args.messidor2:
    data_dir = default_messidor2_dir
else:
    print("Please specify --data_dir.")
    parser.print_help()
    sys.exit(2)

load_model_path = str(args.load_model_path)
batch_size = int(args.batch_size)
save_operating_thresholds_path = str(args.save_operating_thresholds_path)
operating_threshold = float(args.operating_threshold)

# Check if the model path has comma-separated entries.
if "," in load_model_path:
    load_model_paths = load_model_path.split(",")
# Check if the model path is a glob pattern.
elif any(char in load_model_path for char in '*?['):
    load_model_paths = sorted(set(
        stem for stem in map(lib.model.model_path_stem,
                             glob("{}*".format(load_model_path)))
        if stem is not None))
else:
    load_model_paths = [load_model_path]

if len(load_model_paths) == 0:
    print(f"No models found matching {load_model_path}.")
    sys.exit(1)

print("""
Evaluating: {},
Saving operating thresholds metrics at: {},
Using operating threshold: {},
""".format(data_dir, save_operating_thresholds_path, operating_threshold))
print("Trying to load model(s):\n{}".format("\n".join(load_model_paths)))

# Other setting variables.
num_channels = 3
num_thresholds = 200

# Define thresholds.
thresholds = lib.metrics.generate_thresholds(num_thresholds)

all_predictions = []
all_labels = None

for model_path in load_model_paths:
    model, is_legacy = lib.model.load_model(model_path)

    if is_legacy:
        # Evaluate the published TensorFlow 1 models exactly like the
        #  original version of this script did: images are standardized per
        #  image, and batch normalization uses the statistics of each batch.
        #  See `lib.model.load_legacy_checkpoint`.
        print(f"{model_path}: legacy TensorFlow 1 checkpoint.")
        normalization_fn = tf.image.per_image_standardization
    else:
        # Use the same normalization as during training.
        normalization_fn = rescale_min_1_to_1

    # Initialize the test set.
    test_dataset = lib.dataset.initialize_dataset(
        data_dir, batch_size, prefetch_buffer_size=tf.data.AUTOTUNE,
        num_channels=num_channels, normalization_fn=normalization_fn,
        augmentation=False)

    # Perform the evaluation.
    test_predictions, test_labels = lib.evaluation.predict(
        model, test_dataset, batch_statistics=is_legacy)

    if all_labels is None:
        all_labels = test_labels

    all_predictions.append(test_predictions)

# Calculate the linear average of all predictions.
avg_pred = np.mean(np.array(all_predictions), axis=0)

# Print Brier score, AUC and confusion matrix.
lib.evaluation.report(all_labels, avg_pred, operating_threshold)

# Print sensitivity and specificity at operating threshold.
op_tp, op_fp, op_fn, op_tn = lib.metrics.confusion_at_thresholds(
    all_labels, avg_pred, [operating_threshold])
op_specificities, op_sensitivities = \
    lib.metrics.specificities_sensitivities(op_tp, op_fp, op_fn, op_tn)

print("Specificity: {0:0.4f}, Sensitivity: {1:0.4f} at "
      "Operating Threshold {2:0.4f}."
      .format(op_specificities[0], op_sensitivities[0], operating_threshold))

# Write sensitivities and specificities to file.
os.makedirs(os.path.dirname(os.path.abspath(save_operating_thresholds_path)),
            exist_ok=True)
lib.evaluation.write_operating_thresholds(
    save_operating_thresholds_path, all_labels, avg_pred, thresholds)

sys.exit(0)
