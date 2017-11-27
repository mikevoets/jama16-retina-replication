from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, LeakyReLU, Dropout
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.initializers import Orthogonal, Constant
from tensorflow.contrib.keras.api.keras.regularizers import l2


def conv2d_params(filters, kernel_size=(3, 3), padding='same',
                  W=Orthogonal(gain=1.0), b=Constant(value=0.05),
                  W_regularizer=l2(5e-4), **kwargs):
    args = {
        'filters': filters,
        'kernel_size': kernel_size,
        'kernel_initializer': W,
        'bias_initializer': b,
        'kernel_regularizer': W_regularizer,
        'padding': padding
    }
    args.update(kwargs)
    return args


def pool2d_params(pool_size=3, strides=(2, 2), **kwargs):
    args = {
        'pool_size': pool_size,
        'strides': strides
    }
    args.update(kwargs)
    return args


def dense_params(units, W=Orthogonal(gain=1.0), b=Constant(value=0.05),
                 W_regularizer=l2(5e-4), **kwargs):
    args = {
        'units': units,
        'kernel_initializer': W,
        'bias_initializer': b,
        'kernel_regularizer': W_regularizer
    }
    args.update(kwargs)
    return args


class RMSPooling2D(MaxPooling2D):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, epsilon=1e-12, **kwargs):
        super(RMSPooling2D, self).__init__(pool_size, strides, padding,
                                           data_format, **kwargs)
        self.epsilon = epsilon

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        output = K.pool2d(K.square(inputs), pool_size, strides,
                          padding, data_format, pool_mode='avg')
        return K.sqrt(output + epsilon)


def Maxout(x, units=None):
    """
    Maxout as in the paper `Maxout Networks <http://arxiv.org/abs/1302.4389>`_.

    Args:
        x (tf.Tensor): a NHWC or NC tensor. Channel has to be known.
        units (int): a int. Must be divisible by C.

    Returns:
        tf.Tensor: of shape NHW(C/units) named ``output``.
    """
    input_shape = x.get_shape().as_list()
    ndim = len(input_shape)
    assert ndim == 4 or ndim == 2

    data_format = K.image_data_format()

    if data_format == 'channels_first':
        ch = input_shape[1]
    else:
        ch = input_shape[-1]

    if units is None:
        units = ch / 2
    assert ch is not None and ch % units == 0

    if ndim == 4:
        if data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 2, 3, 1))
        x = K.reshape(x, (-1, input_shape[2], input_shape[3], ch // units, units))
        x = K.max(x, axis=3)
        if data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 3, 1, 2))
    else:
        x = K.reshape(x, (-1, ch // units, units))
        x = K.max(x, axis=1)

    return x
