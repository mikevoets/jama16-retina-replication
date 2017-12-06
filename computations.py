import os
import eyepacs.v3 as eye
import numpy as np
from PIL import Image

# Channel stds.
STD = np.array([74.69171454, 53.94257818, 43.4777137], dtype=np.float32)

# Channel means.
MEAN = np.array([93.57430267, 65.09854889, 46.50986099], dtype=np.float32)


def train_images_files():
    """Helper function for finding paths to training images."""
    train_images_dir = os.path.join(eye.data_path, eye.train_pre_subpath)

    return eye._get_image_paths(images_dir=train_images_dir)


image_files = train_images_files()


def load_image(fname):
    if isinstance(fname, str):
        return np.array(Image.open(fname), dtype=np.float32).transpose(2, 1, 0)
    else:
        return np.array([load_image(f) for f in fname])


def compute_mean(files, batch_size=128):
    m = np.zeros(3)
    batches = 0
    for i in range(0, len(files), batch_size):
        images = load_image(files[i:i + batch_size])
        m += images.mean(axis=(0, 2, 3))
        batches += 1
    return (m / batches).astype(np.float32)


def compute_std(files, batch_size=128):
    s = np.zeros(3)
    s2 = np.zeros(3)
    shape = None
    for i in range(0, len(files), batch_size):
        images = np.array(load_image(files[i:i + batch_size]),
                          dtype=np.float64)
        shape = images.shape
        s += images.sum(axis=(0, 2, 3))
        s2 += np.power(images, 2).sum(axis=(0, 2, 3))
    n = len(files) * shape[2] * shape[3]
    var = (s2 - s**2.0 / n) / (n - 1)
    return np.sqrt(var)
