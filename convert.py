import os
import sys
import importlib
import csv
from preprocess import scale_normalize, resize
from distutils.dir_util import copy_tree

# TODO: Flags input/output directory




for pre_subpath in [eye.test_pre_subpath]:
    eye.print_status("Converting images in {}.".format(pre_subpath))

    for size in [299, 256, 128]:
        eye.print_status("Converting images to {} pixels.".format(size))

        path = os.path.join(eye.data_path, pre_subpath)
        new_path = path.replace('512', str(size))

        if os.path.exists(new_path):
            eye.print_status("Already converted.")
            next

        # Copy files to new path.
        copy_tree(path, new_path)

        # Convert the images to size.
        old_images_paths = eye._get_image_paths(path)
        images_paths = eye._get_image_paths(images_dir=new_path)
        resize(images_paths, size=size)


sys.exit(0)


def _maybe_create(images_dir, labels_path):
    # Helper function for creating subdirectories.
    eye.print_status("Creating subdirectories...")

    # Skipping if there are no labels.
    if not os.path.exists(labels_path):
        eye.print_status("Skipping {}".format(labels_path))
        return

    # Skip if subdirectories already have been created.
    num_directories = len(os.listdir(images_dir))
    if num_directories > 0 and num_directories <= eye.num_classes:
        eye.print_status("Subdirectories already created!")
        return

    # Read labels file.
    with open(labels_path, 'rt') as r:
        reader = csv.reader(r, delimiter=",")
        for i, line in enumerate(reader):
            # Deduct the image filename and label.
            im_name = line[0] + '.jpeg'
            label = line[1]

            # Check if group subdirectory has been made before.
            # If not, make it.
            im_path = os.path.join(images_dir, im_name)
            group_path = os.path.join(images_dir, label)

            if not os.path.exists(group_path):
                os.makedirs(group_path)

            # Move image to subdirectory.
            new_im_path = os.path.join(group_path, im_name)
            try:
                os.rename(im_path, new_im_path)
            except FileNotFoundError:
                eye.print_status("Skipping {}".format(im_name))


# Retrieve all data-sets.
test_images_path = os.path.join(eye.data_path, eye.test_pre_subpath)

# Retrieve all labels.
test_labels_path = os.path.join(eye.data_path, eye.test_labels_filename)

# Create all subdirectories.
_maybe_create(test_images_path, test_labels_path)


sys.exit(0)

# Find the path where processed images should be uploaded to.
save_path = os.path.join(eye.data_path, eye.test_pre_subpath)

# Get image path.
images_dir = os.path.join(eye.data_path, eye.test_subpath)

# Get paths of images that are to be preprocessed.
image_paths = eye._get_image_paths(images_dir=images_dir)

# Get paths of images that already are preprocessed.
preprocessed_paths = eye._get_image_paths(images_dir=save_path)

# Get paths of images that yet have to be preprocessed.
images_to_preprocess = set(eye._base_filename(image_paths)) \
    - set(eye._base_filename(preprocessed_paths))

# Only continue unless the directory does not exist.
if len(images_to_preprocess) > 0:
    eye.print_status("Preprocessing images...")

    # Get the full paths of images.
    preprocess_fns = [os.path.join(images_dir, im + '.jpeg')
                      for im in images_to_preprocess]

    # Create directory for preprocessed images if necessary.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Preprocess images.
    scale_normalize(image_paths=preprocess_fns, save_path=save_path,
                    diameter=512)
else:
    eye.print_status("Images already preprocessed.")
