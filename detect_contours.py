import numpy as np
import argparse
import cv2
import os
import math
from glob import glob
from pylab import array, plot, show, axis, arange, figure, uint8

# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image-dir", required=True, help="Directory path to the images")
ap.add_argument("-n", "--take-n", required=True, help="Analyse max n images")
args = vars(ap.parse_args())


def get_image_paths(directory):
    # The file paths should match the following regexp.
    filename_match = os.path.join(directory, "*.jpeg")

    return glob(filename_match)


# Retrieve the image paths from the directory.
image_paths = np.array(get_image_paths(args["image_dir"]))

# Number of images in the set.
num_images = len(image_paths)

# Create a random index.
idx = np.random.choice(num_images, size=int(args["take_n"]), replace=False)

for image_path in image_paths[idx]:
    # Load the image and clone it for output.
    image = cv2.imread(image_path)
    image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
    orig = image.copy()
    output = image.copy()

    # Increase contrast on image.
    maxIntensity = 255.0
    x = arange(maxIntensity)

    # Parameters for manipulating image data.
    phi = 1.2
    theta = 1
    y = (maxIntensity/phi)*(x/(maxIntensity/theta))**0.5

    # Decrease intensity such that dark pixels become much darker,
    # and bright pixels become slightly dark.
    image = (maxIntensity/phi)*(image/(maxIntensity/theta))**2
    image = array(image, dtype=uint8)

    # Get the gray-scale of the image.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Detecting contours for {0}...".format(image_path))
    # Detect circle(s) in the image.
    cnts = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # At least ensure that some circles were found.
    if len(cnts) > 0:
        print("Contours detected...")
        # Find the largest contour in the mask.
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Only draw the countour if its larger than a certain size.
        if radius > 10:
            # Draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle.
            cv2.circle(output, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(output, center, 5, (0, 128, 255), -1)

        # Show the output image.
        cv2.imshow(image_path, np.hstack([orig, output]))
        cv2.waitKey(0)
    else:
        print("No circles detected...")
