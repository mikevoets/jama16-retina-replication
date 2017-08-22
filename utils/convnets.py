import tensorflow as tf


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,               # The previous layer
                   num_input_channels,  # Num. channels in prev. layer
                   filter_size,         # Width and height of each filter
                   num_filters,         # Number of filters
                   use_pooling=True):   # Use 2x2 max-pooling

    # Shape of the filter-weights for the convolution
    # This format is determined by the TensorFlow API
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights (= filters) with the given shape
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution
    layer += biases

    # Down-sample the image resolution if necessary
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # ReLU activation function
    # Calculates max(x, 0) for each input pixel x
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions
    layer = tf.nn.relu(layer)

    # NB: ReLU is normally executed before the pooling
    # but since relu(max_pool(x)) == max_pool(relu(x))
    # we can save 75% of the relu-operations by max-pooling first

    return layer, weights


def flatten_layer(layer):
    # Get the shape of the input layer
    layer_shape = layer.get_shape()

    # NB: The shape of the input layer is assumed to be:
    # layer_shape = [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features]
    # NB: We just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping
    layer_flat = tf.reshape(layer, [-1, num_features])

    # NB: The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    return layer_flat, num_features


def new_fc_layer(input,             # The previous layer
                 num_inputs,        # Num. inputs from prev. layer
                 num_outputs,       # Num. outputs
                 use_relu=True):    # Use ReLU?

    # Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values
    layer = tf.matmul(input, weights) + biases

    # Use ReLU if necessary
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
