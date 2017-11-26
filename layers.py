from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, LeakyReLU, Lambda, Dropout
from tensorflow.contrib.keras.api.keras.optimizers import SGD
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.initializers import Orthogonal, Constant


def conv2d_params(num_filters, filter_size=(3, 3), padding='same',
                  activation=LeakyReLU(alpha=0.1), W=Orthogonal(gain=1.0),
                  b=Constant(value=0.05), untie_biases=True, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': filter_size,
        'activation': activation,
        'W': W,
        'b': b,
        'untie_biases': untie_biases
    }
    args.update(kwargs)
    return args


def pool2d_params(pool_size=3, stride=(2, 2), **kwargs):
    args = {
        'pool_size': pool_size,
        'stride': stride
    }
    args.update(kwargs)
    return args


def dense_params(num_units, activation=LeakyReLU(alpha=0.1),
                 W=Orthogonal(gain=1.0), b=Constant(value=0.05), **kwargs):
    args = {
        'num_units': num_units,
        'activation': activation,
        'W': W,
        'b': b
    }
    args.update(kwargs)
    return args


class RMSPooling2D(MaxPooling2D):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, epsilon=1e-12, **kwargs):
        super(FeaturePooling2D, self).__init__(pool_size, strides, padding,
                                               data_format, **kwargs)
        self.epsilon = epsilon

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        output = K.pool2d(K.square(inputs), pool_size, strides,
                          padding, data_format, pool_mode='avg')
        return K.sqrt(out + epsilon)


class FeaturePooling2D(MaxPooling2D):
    def __init__(self, pool_size=(2, 2), axis=1, strides=None,
                 padding='valid', pool_function=K.max,
                 data_format=None, **kwargs):
        super(FeaturePooling2D, self).__init__(pool_size, strides, padding,
                                               data_format, **kwargs)
        self.axis = axis
        self.pool_function = pool_function

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        input_shape = tuple(inputs.shape)
        num_feature_maps = input_shape[self.axis]
        num_feature_maps_out = num_feature_maps // pool_size

        pool_shape = (input_shape[:self.axis] +
                      (num_feature_maps_out, pool_size) +
                      input_shape[self.axis+1:])

        input_reshaped = inputs.reshape(pool_shape)
        return self.pool_function(input_reshaped, axis=self.axis + 1)
