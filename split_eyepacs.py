"""Distribute the preprocessed EyePACS pool into a training and a test set.

Images with DR grade 0 or 1 go to class 0 (no referable DR), and images with
grade 2, 3 or 4 go to class 1. The split only depends on the image names and
the seed: the names are sorted before they are shuffled with a seeded
random.Random, so the result does not depend on the file system, the OpenSSL
version or the shell tools. The split is written to <output_dir>/split.csv.
"""
import argparse
import csv
import os
import random
import shutil
import sys

# The study's distribution numbers (all images / gradable images only).
COUNTS = {
    False: {"class0": 48784, "class0_train": 40688, "class1_train": 16458},
    True: {"class0": 39202, "class0_train": 31106, "class1_train": 12582},
}


def list_images(pool_dir, grades):
    names = []
    for grade in grades:
        directory = os.path.join(pool_dir, str(grade))
        names += [os.path.join(str(grade), f) for f in os.listdir(directory)
                  if f.lower().endswith(".jpg")]
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only_gradable", action="store_true")
    args = parser.parse_args()

    counts = COUNTS[args.only_gradable]
    rng = random.Random(args.seed)

    class0 = list_images(args.pool_dir, [0, 1])
    class1 = list_images(args.pool_dir, [2, 3, 4])

    if len(class0) < counts["class0"] or len(class1) <= counts["class1_train"]:
        sys.exit("Not enough images in {}: found {} with grade 0-1 and {} "
                 "with grade 2-4.".format(args.pool_dir, len(class0),
                                          len(class1)))

    rng.shuffle(class0)
    rng.shuffle(class1)
    class0 = class0[:counts["class0"]]

    split = (
        [("train", "0", f) for f in class0[:counts["class0_train"]]] +
        [("test", "0", f) for f in class0[counts["class0_train"]:]] +
        [("train", "1", f) for f in class1[:counts["class1_train"]]] +
        [("test", "1", f) for f in class1[counts["class1_train"]:]]
    )

    for subset, label, path in split:
        target = os.path.join(args.output_dir, subset, label)
        os.makedirs(target, exist_ok=True)
        shutil.copy(os.path.join(args.pool_dir, path), target)

    with open(os.path.join(args.output_dir, "split.csv"), "w",
              newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "grade", "set", "class"])
        for subset, label, path in sorted(split, key=lambda r: r[2]):
            grade, filename = path.split(os.sep)
            writer.writerow([os.path.splitext(filename)[0], grade, subset,
                             label])

    for subset in ["train", "test"]:
        for label in ["0", "1"]:
            n = sum(1 for s, l, _ in split if s == subset and l == label)
            print("{} set, class {}: {} images".format(subset, label, n))


if __name__ == "__main__":
    main()
