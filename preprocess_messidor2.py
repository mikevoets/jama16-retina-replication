import xlrd
import zipfile
import argparse
import sys
from shutil import rmtree
from PIL import Image
from glob import glob
from os import makedirs, rename
from os.path import join, splitext, basename, exists
from lib.preprocess import scale_normalize

parser = argparse.ArgumentParser(description='Preprocess Messidor-2 data set.')
parser.add_argument("--data_dir", help="Directory where Messidor-2 resides.",
                    default="data/messidor2")

args = parser.parse_args()
data_dir = str(args.data_dir)

# Create directories for grades.
[makedirs(join(data_dir, str(i))) for i in [0, 1, 2, 3]
        if not exists(join(data_dir, str(i)))]

# Create a tmp directory for saving temporary preprocessing files.
tmp_path = join(data_dir, 'tmp')
if exists(tmp_path):
    rmtree(tmp_path)
makedirs(tmp_path)

# Find shard zip files.
shards_paths = glob(join(data_dir, "*.zip"))

for shard in shards_paths:
    shard_name = splitext(basename(shard))[0]
    shard_unpack_dir = join(data_dir, shard_name)

    # Unzip shard.
    print(f"Unzipping {shard_name}...")
    if exists(shard_unpack_dir):
        rmtree(shard_unpack_dir)

    zip_ref = zipfile.ZipFile(shard, 'r')
    zip_ref.extractall(shard_unpack_dir)
    zip_ref.close()

    # Open annotations file for shard.
    annotations_path = join(
            data_dir, f"Annotation_{shard_name}.xls")
    workbook = xlrd.open_workbook(annotations_path)
    worksheet = workbook.sheet_by_index(0)

    # Parse annotations file.
    for num, row in enumerate(range(1, worksheet.nrows)):
        filename = worksheet.cell(row, 0).value
        grade = worksheet.cell(row, 2).value

        im_path = glob(join(shard_unpack_dir, "**/{}".format(filename)),
                       recursive=True)[0]

        # Find contour of eye fundus in image, and scale
        #  diameter of fundus to 299 pixels and crop the edges.
        res = scale_normalize(save_path=tmp_path, image_path=im_path,
                              diameter=299, verbosity=0)

        # Status-message.
        msg = "\r- Preprocessing image: {0:>6} / {1}".format(
                num+1, worksheet.nrows-1)

        # Print the status message.
        sys.stdout.write(msg)
        sys.stdout.flush()

        if res != 1:
            continue

        new_filename = "{0}.jpg".format(splitext(basename(im_path))[0])

        # Move the file from the tmp folder to the right grade folder.
        rename(join(tmp_path, new_filename),
               join(data_dir, str(int(grade)), new_filename))

    print()
    rmtree(shard_unpack_dir)

# Clean tmp folder.
rmtree(tmp_path)
