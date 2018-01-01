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
    def __init__(self, conf=None):
        self.conf = conf
        pprint.pprint(conf)

    def get(self, k, default=None):
        return self.conf.get(k, default)
