from datetime import datetime
import pprint
import os

import numpy as np

from eyepacs.v3 import FEATURE_DIR


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


mkdir(FEATURE_DIR)


class Config(object):
    def __init__(self, layers, conf=None):
        self.layers = layers
        self.conf = conf
        pprint.pprint(conf)

    def get(self, k, default=None):
        return self.conf.get(k, default)

    def weights_epoch(self):
        path = "weights/{}/epochs".format(self.conf['name'])
        mkdir(path)
        return os.path.join(path, '{epoch}_{timestamp}_{loss}.pkl')

    def weights_best(self):
        path = "weights/{}/best".format(self.conf['name'])
        mkdir(path)
        return os.path.join(path, '{epoch}_{timestamp}_{loss}.pkl')

    def weights_file(self):
        path = "weights/{}".format(self.conf['name'])
        mkdir(path)
        return os.path.join(path, 'weights.pkl')

    def retrain_weights_file(self):
        path = "weights/{}/retrain".format(self.conf['name'])
        mkdir(path)
        return os.path.join(path, 'weights.pkl')

    def final_weights_file(self):
        path = "weights/{}".format(self.conf['name'])
        mkdir(path)
        return os.path.join(path, 'weights_final.pkl')

    def get_features_fname(self, n_iter, skip=0, test=False):
        fname = '{}_{}_mean_iter_{}_skip_{}.npy'.format(
            self.conf['name'], ('test' if test else 'train'),  n_iter, skip)
        return os.path.join(FEATURE_DIR, fname)

    def get_std_fname(self, n_iter, skip=0, test=False):
        fname = '{}_{}_std_iter_{}_skip_{}.npy'.format(
            self.conf['name'], ('test' if test else 'train'), n_iter, skip)
        return os.path.join(FEATURE_DIR, fname)

    def save_features(self, X, n_iter, skip=0, test=False):
        np.save(open(self.get_features_fname(n_iter, skip=skip,
                                             test=test), 'wb'), X)

    def save_std(self, X, n_iter, skip=0, test=False):
        np.save(open(self.get_std_fname(n_iter, skip=skip,
                                        test=test), 'wb'), X)

    def load_features(self, test=False):
        return np.load(open(self.get_features_fname(test=test)))
