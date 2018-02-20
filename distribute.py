import os
import argparse
import sys
from glob import glob
from shutil import copyfile
from lib.common import print_status

parser = argparse.ArgumentParser(description='Redistribute data set.')
parser.add_argument("source", help="Pool to take from.")
parser.add_argument("dest", help="Destination of new data set.")
parser.add_argument("n0", help="Amount to take from class 0.", type=int)
parser.add_argument("n1", help="Amount to take from class 1.", type=int)
parser.add_argument("n2", help="Amount to take from class 2.", type=int)
parser.add_argument("n3", help="Amount to take from class 3.", type=int)
parser.add_argument("n4", help="Amount to take from class 4.", type=int)

args = parser.parse_args()
pool = args.source

requested = []
class_images = []
total = 0

for i in range(5):
    requested.append(args.__getattribute__("n{}".format(i)))

    filename_match = os.path.join(pool, str(i), "*.jpeg")
    paths = glob(filename_match)
    class_images.append(paths)

    if requested[i] > len(class_images[i]):
        raise TypeError("Max images in class {} is {}!"
                        .format(i, len(class_images[i])))

    total += len(paths)

print("Found {} images!".format(total))

for i in range(5):
    print(" {} -> {} images".format(i, len(class_images[i])))

train_path = os.path.join(args.dest, 'train')
test_path = os.path.join(args.dest, 'test')

if os.path.exists(args.dest):
    print("Destination does already exist!")
    sys.exit(-1)
else:
    os.makedirs(args.dest)
    os.makedirs(train_path)
    os.makedirs(test_path)

    for i in range(5):
        os.makedirs(os.path.join(train_path, str(i)))
        os.makedirs(os.path.join(test_path, str(i)))

test_train_split = []

for i in range(5):
    train_split = class_images[i][:requested[i]]
    test_split = class_images[i][requested[i]:]

    for j in range(len(train_split)):
        print_status("Copying class {} (train): {} / {}"
                     .format(i, j, len(train_split)))
        new_path = train_split[j].replace(pool, train_path)
        copyfile(train_split[j], new_path)

    for j in range(len(test_split)):
        print_status("Copying class {} (test): {} / {}"
                     .format(i, j, len(test_split)))
        new_path = test_split[j].replace(pool, test_path)
        copyfile(test_split[j], new_path)
