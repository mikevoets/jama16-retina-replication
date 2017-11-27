from layers import *
from config import Config
from eyepacs.v3 import BALANCE_WEIGHTS

conf = {
    'name': __name__.split('.')[-1],
    'width': 112,
    'height': 112,
    'train_dir': 'data/eyepacs/preprocessed/128/train',
    'test_dir': 'data/eyepacs/preprocessed/128/test',
    'batch_size_train': 32,
    'batch_size_test': 8,
    'balance_weights': BALANCE_WEIGHTS,
    'balance_ratio': 0.975,
    'final_balance_weights': [1, 2, 2, 2, 2],
    'augmentation_params': {
        'rescale': 1./255,
        'samplewise_center': True,
        'samplewise_std_normalization': True,
        'zoom_range': [1 / 1.15, 1.15],
        'rotation_range': 360.,
        'shear_range': 0.,
        'width_shift_range': 0.4,
        'height_shift_range': 0.4,
        'horizontal_flip': True,
        'vertical_flip': True
    },
    'weight_decay': 5e-4,
    'sigma': 0.5,
    'learn_rate_schedule': {
        0: 3e-3,
        150: 3e-4,
        220: 3e-5,
        250: 'stop'
    }
}

n = 32

layers = [
    (Conv2D, conv2d_params(n, kernel_size=(5, 5), strides=(2, 2),
                           input_shape=(conf['width'], conf['height'], 3))),
    (LeakyReLU, {'alpha': 0.1}),
    (Conv2D, conv2d_params(n)),
    (LeakyReLU, {'alpha': 0.1}),
    (MaxPooling2D, pool2d_params()),
    (Conv2D, conv2d_params(2 * n, kernel_size=(5, 5), strides=(2, 2))),
    (LeakyReLU, {'alpha': 0.1}),
    (Conv2D, conv2d_params(2 * n)),
    (LeakyReLU, {'alpha': 0.1}),
    (Conv2D, conv2d_params(2 * n)),
    (LeakyReLU, {'alpha': 0.1}),
    (MaxPooling2D, pool2d_params()),
    (Conv2D, conv2d_params(4 * n)),
    (LeakyReLU, {'alpha': 0.1}),
    (Conv2D, conv2d_params(4 * n)),
    (LeakyReLU, {'alpha': 0.1}),
    (Conv2D, conv2d_params(4 * n)),
    (LeakyReLU, {'alpha': 0.1}),
    (MaxPooling2D, pool2d_params()),
    (Conv2D, conv2d_params(8 * n)),
    (LeakyReLU, {'alpha': 0.1}),
    (Conv2D, conv2d_params(8 * n)),
    (LeakyReLU, {'alpha': 0.1}),
    (Conv2D, conv2d_params(8 * n)),
    (LeakyReLU, {'alpha': 0.1}),
    (MaxPooling2D, pool2d_params()),
    (Conv2D, conv2d_params(16 * n)),
    (LeakyReLU, {'alpha': 0.1}),
    (Conv2D, conv2d_params(16 * n)),
    (LeakyReLU, {'alpha': 0.1}),
    (RMSPooling2D, pool2d_params(strides=(3, 3))),
    (Dropout, {'rate': 0.5}),
    (Dense, dense_params(1024)),
    (LeakyReLU, {'alpha': 0.1}),
    (Maxout, {'lambda': True, 'units': 512}),
    (Dropout, {'rate': 0.5}),
    (Dense, dense_params(1024)),
    (LeakyReLU, {'alpha': 0.1}),
    (Maxout, {'lambda': True, 'units': 512}),
    (Flatten, {}),
    (Dense, {'units': 5, 'activation': 'softmax'})
]

config = Config(layers=layers, conf=conf)
