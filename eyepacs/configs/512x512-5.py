from layers import *
from config import Config
from eyepacs.v3 import BALANCE_WEIGHTS

config = {
    'name': __name__.split('.')[-1],
    'width': 448,
    'height': 448,
    'train_dir': eyepacs.v3.train_medium_dir,
    'test_dir': eyepacs.v3.test_medium_dir,
    'batch_size_train': 32,
    'batch_size_test': 8,
    'balance_weights': BALANCE_WEIGHTS,
    'balance_ratio': 0.975,
    'final_balance_weights': [1, 2, 2, 2, 2],
    'augmentation_params': {
        'zoom_range': [1 / 1.15, 1.15],
        'rotation_range': 360,
        'shear_range': 0.,
        'width_shift_range': 0.4,
        'height_shift_range': 0.4,
        'horizontal_flip': True,
        'vertical_flip': True
    },
    'weight_decay' = 5e-4,
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
    (Input, {'shape': (None, 3, config['width'], config['height'])}),
    (Conv2D, conv2d_params(n, filter_size=(5, 5), stride=(2, 2))),
    (Conv2D, conv2d_params(n)),
    (MaxPooling2D, pool_params()),
    (Conv2D, conv2d_params(2 * n, filter_size(5, 5), stride(2, 2))),
    (Conv2D, conv2d_params(2 * n)),
    (Conv2D, conv2d_params(2 * n)),
    (MaxPooling2D, pool_params()),
    (Conv2D, conv2d_params(4 * n)),
    (Conv2D, conv2d_params(4 * n)),
    (Conv2D, conv2d_params(4 * n)),
    (MaxPooling2D, pool_params()),
    (Conv2D, conv2d_params(8 * n)),
    (Conv2D, conv2d_params(8 * n)),
    (Conv2D, conv2d_params(8 * n)),
    (MaxPooling2D, pool_params()),
    (Conv2D, conv2d_params(16 * n)),
    (Conv2D, conv2d_params(16 * n)),
    (RMSPooling2D, pool_params(stride=(3, 3))),
    (Dropout, {'rate': 0.5}),
    (Dense, dense_params(1024)),
    (FeaturePooling2D, {'pool_size': 2}),
    (Dropout, {'rate': 0.5}),
    (Dense, dense_params(1024)),
    (FeaturePooling2D, {'pool_size': 2}),
    (Dense, {'num_units': 1})
]

config = Config(layers=layers, conf=conf)
