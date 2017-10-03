import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import time
import os
from datetime import timedelta

# Functions and classes for loading an using the Inception model.
import inception
from inception import transfer_values_cache

# Use Pretty Tensor to define new classifiers.
import prettytensor as pt

# Use the EyePacs dataset.
import eyepacs
from eyepacs import num_classes

# Use sklearn for analysis of transfer-values and confusion matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# For debugging purposes.
import pdb

# Ignore Tensorflow logs.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

########################################################################
# Various constants.

# Batch size for training.
batch_size = 128

# Split training set for validation.
validation_split = 0.2

########################################################################
# Initializer functions

# Define the location of the EyePacs data set.
eyepacs.data_path = "data/eyepacs/"

# Extract if necessary.
eyepacs.maybe_extract_images()

# Preprocess if necessary.
eyepacs.maybe_preprocess()

# Extract labels if necessary.
eyepacs.maybe_extract_labels()

# Define the download location of the Inception model.
inception.data_dir = "data/inception/"

# Download the Inception model if necessary.
inception.maybe_download()

# Load the Inception model.
model = inception.Inception()

print("Processing Inception transfer-values for training-images...")

file_path_cache_train = os.path.join(
    eyepacs.data_path, 'inception_eyepacs_train.pkl')

# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_train, transfer_values_validation = transfer_values_cache(
    cache_path=file_path_cache_train,
    image_paths=eyepacs.get_training_image_paths(),
    model=model,
    split=validation_split
)

print("Processing Inception transfer-values for test-images...")

file_path_cache_test = os.path.join(
    eyepacs.data_path, 'inception_eyepacs_test.pkl')

# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_test = transfer_values_cache(
    cache_path=file_path_cache_test,
    image_paths=eyepacs.get_test_image_paths(),
    model=model
)

# Retrieve the training and validation set labels.
training_labels, validation_labels = eyepacs.get_training_cls(
    split=validation_split)

# Retrieve the class-numbers and one-hot-encoded labels for each set.
cls_training, labels_training = training_labels
cls_validation, labels_validation = validation_labels

# Retrieve the class-labels from the test-set.
cls_test, labels_test = eyepacs.get_test_cls()

# New classifier on top of Inception
transfer_len = model.transfer_len

# Create a placeholder for inputting the transfer-values
# from the Inception model into the new network.
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')

# Create another for inputting the true class-label of each image.
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

# Calculate the true class via argmax.
y_true_cls = tf.argmax(y_true, axis=1)

# Create the neural network for doing the classification
# on top of Inception.
# Wrap the transfer-values are a Pretty Tensor.
x_pretty = pt.wrap(x)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        fully_connected(size=1024, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

# Optimization method
global_step = tf.Variable(
    initial_value=0, name='global_step', trainable=False)

# Use Adam Optimizer with inbuilt well-performing
# stochastic gradient descent.
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

# The output of the network is an array with 5 elements.
# The predicted class number is the index of the largest element
# in the array.
y_pred_cls = tf.argmax(y_pred, axis=1)

# Create an array of booleans whether the predicted class equals
# the true class of each image.
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# Calculate the accuracy by taking the average of correct prediction
# by type-casting to 1 and 0.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create a batcher instance to provide the data in mini-batches.
batcher = eyepacs.TrainBatcher(transfer_values_train, labels_training)

# Start a TensorFlow session.
session = tf.Session()

# Initialize the global variables.
session.run(tf.global_variables_initializer())


def plot_images(images, cls_true, cls_pred=None, smooth=True):
    """
    Function to plot images in a grid.
    Writing the true and predicted classes below each image.
    """

    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = cls_true[i]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = cls_pred[i]

                xlabel = "True: {0}\nPred: {1}".format(
                    cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Show the plots.
    plt.show()


def plot_transfer_values(i, transfer_values, test=False):
    """
    Function to plot an example image and its corresponding transfer-values.
    """
    print("Input image:")

    if test:
        image_ex, label_ex, _ = eyepacs.load_test_data(i)
    else:
        image_ex, label_ex, _ = eyepacs.load_training_data(i)

    image, label = eyepacs.session_run(args=[image_ex, label_ex],
                                       session=session)

    # Plot the i'th image from the data-set.
    plt.imshow(image, interpolation='nearest')
    plt.show()

    print("Transfer-values for the image using Inception model:")

    # Transform the transfer-values into an image.
    img = transfer_values[i]
    img = img.reshape((32, 64))

    # Plot the image for the transfer-values.
    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()


def plot_scatter(values, cls):
    """
    Function to plot a color-map with different colors for each class.
    """
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

    # Get the color for each sample.
    colors = cmap[cls]

    # Extract the x- and y-values.
    x = values[:, 0]
    y = values[:, 1]

    # Plot it.
    plt.scatter(x, y, color=colors)
    plt.show()


def plot_transfer_values_analysis(values, cls):
    """
    Function to plot analysis of transfer values.

    Plots a PCA (reduction to 2), and a t-SNE by first reducing to 50
    with PCA, and then reducing from 50 to 2 to t-SNE.
    """
    # New PCA-object with target array-length to 2.
    pca = PCA(n_components=2)

    # Use PCA to reduce the transfer-value arrays from 2048 to 2 elements.
    transfer_values_reduced = pca.fit_transform(values)

    # Plot the transfer-values that have been reduced using PCA.
    plot_scatter(transfer_values_reduced, cls)

    # Do dimensionality reduction using t-SNE.
    # Unfortunately t-SNE is very slow, so we use PCA to reduce the
    # transfer-values from 2048 to 50 elements.
    pca = PCA(n_components=50)
    transfer_values_50d = pca.fit_transform(values)

    # Create a new t-SNE object for final dimensionality reduction.
    tsne = TSNE(n_components=2)

    # Perform the final reduction using t-SNE.
    transfer_values_reduced = tsne.fit_transform(transfer_values_50d)

    # Plot the transfer-values.
    plot_scatter(transfer_values_reduced, cls)


def optimize(num_iterations):
    """
    Helper function to perform optimization.
    """
    # Start time for printing time-usage.
    start_time = time.time()

    # Keep track of epoch number.
    epoch = 0

    # For each iteration.
    for i in range(num_iterations):
        # Get a batch of training examples.
        x_batch, y_true_batch, current_epoch = batcher.next_batch(batch_size)

        # Validate the current classifier against validation set.
        if epoch < current_epoch:
            assert epoch == current_epoch - 1

            feed_dict_validation = {x: transfer_values_validation,
                                    y_true: labels_validation}

            validation_loss = session.run(loss, feed_dict=feed_dict_validation)

            print("Validation Loss: {0:>6.4}".format(validation_loss))

            epoch = current_epoch

        # Put the batch into a dict for placeholder variables.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer.
        # We want to retrieve the global_step counter.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations.
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the current accuracy on the training-batch.
            batch_acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Print status.
            msg = ("Epoch: {0:>3}, Global Step: {1:>6}, "
                   "Training Batch Accuracy: {2:>6.1%}")
            print(msg.format(epoch, i_global, batch_acc))

    # End.
    end_time = time.time()

    # Print time difference between start and end-times.
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):
    """
    Helper function to plot example errors.
    """
    # Negate the boolean array.
    incorrect_idx = np.where(correct == False)[0]

    # Get up to 9 example images that have been misclassified.
    n = min(9, len(incorrect_idx))

    # Get at least first 9 incorrect.
    examples_incorrect = incorrect_idx[0:n]

    # Get predicted classes for corresponding images.
    cls_pred = cls_pred[examples_incorrect]

    # Get true classes for corresponding images.
    cls_true = cls_test[examples_incorrect]

    # Initialize numpy array for images.
    images = np.zeros(shape=[n, *eyepacs.img_shape, eyepacs.num_channels],
                      dtype=float)

    # Retrieve images.
    for i, idx in enumerate(examples_incorrect):
        image, _, _ = eyepacs.load_test_data(idx)
        images[i] = eyepacs.session_run(args=[image], session=session)[0]

    # Plot n images.
    plot_images(images=images, cls_true=cls_true, cls_pred=cls_pred)


def plot_confusion_matrix(cls_pred):
    """
    Helper function to plot confusion matrix.
    """
    cm = confusion_matrix(y_true=cls_test, y_pred=cls_pred)

    # Print the confusion matrix.
    for i in range(num_classes):
        class_name = "({}) {}".format(i, i)
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))


def predict_cls(transfer_values, labels, cls_true):
    """
    Helper function for calculating classifications.
    """
    # Number of images.
    num_images = len(transfer_values)

    # Preallocate array for predicted classes.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Calculate the predicted classes for the batches.
    i = 0

    # For each image.
    while i < num_images:
        j = min(i + batch_size, num_images)

        # Create a feed dictionary with images and labels.
        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}

        # Calculate the predicted class.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Reset i.
        i = j

    # Create an array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


def classification_accuracy(correct):
    """
    Helper function for calculcating the classification accuracy.
    """
    return correct.mean(), correct.sum()


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    """
    Helper function for printing the classification accuracy on test-set."
    """
    # Calculate the predicted classes for the test-set.
    correct, cls_pred = predict_cls(transfer_values=transfer_values_test,
                                    labels=labels_test, cls_true=cls_test)

    # Classification accuracy.
    acc, num_correct = classification_accuracy(correct)

    # Number of correctly classified images.
    num_images = len(correct)

    # Print accuracy.
    msg = "Accuracy on test-set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples if necessary.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot confusion matrix if necessary.
    if show_confusion_matrix:
        print("Confusion matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


def main():
    # Plot analysis of transfer-values using PCA and t-SNE.
    # plot_transfer_values_analysis(
    #    values=transfer_values_train, cls=cls_training)

    optimize(num_iterations=1000)

    print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=True)


if __name__ == '__main__':
    main()
