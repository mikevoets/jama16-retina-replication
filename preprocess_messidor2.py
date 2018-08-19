import argparse
import csv
import sys
from shutil import rmtree
from PIL import Image
from glob import glob
from os import makedirs, rename
from os.path import join, splitext, basename, exists
from lib.preprocess import resize_and_center_fundus

parser = argparse.ArgumentParser(description='Preprocess Messidor-2 data set.')
parser.add_argument("--data_dir", help="Directory where Messidor-2 resides.",
                    default="data/messidor2")

args = parser.parse_args()
data_dir = str(args.data_dir)

labels = join(data_dir, 'labels.csv')

# Create directories for grades.
[makedirs(join(data_dir, str(i))) for i in [0, 1]
        if not exists(join(data_dir, str(i)))]

# Create a tmp directory for saving temporary preprocessing files.
tmp_path = join(data_dir, 'tmp')
if exists(tmp_path):
    rmtree(tmp_path)
makedirs(tmp_path)

failed_images = []

with open(labels, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)

    for i, row in enumerate(reader):
        basename, grade = row

        im_paths = glob(join(data_dir, "Messidor-2/{}*".format(basename)))

        # Find contour of eye fundus in image, and scale
        #  diameter of fundus to 299 pixels and crop the edges.
        res = resize_and_center_fundus(save_path=tmp_path,
                                       image_paths=im_paths,
                                       diameter=299, verbosity=0)

        # Status message.
        msg = "\r- Preprocessing pair of image: {0:>7}".format(i+1)
        sys.stdout.write(msg)
        sys.stdout.flush()

        if res != 2:
            failed_images.append(basename)
            continue

        # Move the files from the tmp folder to the right grade folder.
        for j in range(2):
            new_filename = "{0}.00{1}.jpg".format(basename, j)

            rename(join(tmp_path, new_filename),
                   join(data_dir, str(int(grade)), new_filename))

# Clean tmp folder.
rmtree(tmp_path)

print("Could not preprocess {} images.".format(len(failed_images)))
print(", ".join(failed_images))
