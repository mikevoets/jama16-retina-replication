########################################################################
#
# Functions for preprocessing data-set.
#
# Implemented in Python 3.5
#
########################################################################

import os
import sys
import cv2
from pylab import array, arange, uint8

########################################################################


def _increase_contrast(image):
    """
    Helper function for increasing contrast of image.
    """
    # Create a local copy of the image.
    copy = image.copy()

    maxIntensity = 255.0
    x = arange(maxIntensity)

    # Parameters for manipulating image data.
    phi = 1.3
    theta = 1.5
    y = (maxIntensity/phi)*(x/(maxIntensity/theta))**0.5

    # Decrease intensity such that dark pixels become much darker,
    # and bright pixels become slightly dark.
    copy = (maxIntensity/phi)*(copy/(maxIntensity/theta))**2
    copy = array(copy, dtype=uint8)

    return copy


def _find_contours(image):
    """
    Helper function for finding contours of image.

    Returns coordinates of contours.
    """
    # Increase constrast in image to increase changes of finding
    # contours.
    processed = _increase_contrast(image)

    # Get the gray-scale of the image.
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # Detect contour(s) in the image.
    cnts = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # At least ensure that some contours were found.
    if len(cnts) > 0:
        # Find the largest contour in the mask.
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Assume the radius is of a certain size.
        if radius > 100:
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            return (center, radius)


def _get_filename(file_path):
    """
    Helper function to get filename including extension.
    """
    return file_path.split("/")[-1]


def _scale_normalize(image, diameter):
    """
    Helper function for scale normalizing image.
    """
    copy = image.copy()

    # Find largest contour in image.
    contours = _find_contours(image)

    # Return unless we have gotten some result contours.
    if contours is None:
        return None

    center, radius = contours

    # Calculate the min and max-boundaries for cropping the image.
    x_min = max(0, int(center[0] - radius))
    y_min = max(0, int(center[1] - radius))
    z = int(radius*2)
    x_max = x_min + z
    y_max = y_min + z

    # Crop the image.
    copy = copy[y_min:y_max, x_min:x_max]

    # Scale the image.
    fx = fy = (diameter / 2) / radius
    copy = cv2.resize(copy, (0, 0), fx=fx, fy=fy)

    # Add padding to image.
    shape = copy.shape

    # Get the border shape size.
    top = bottom = int((diameter - shape[0])/2)
    left = right = int((diameter - shape[1])/2)

    # Add 1 pixel if necessary.
    if shape[0] + top + bottom == diameter - 1:
        top += 1

    if shape[1] + left + right == diameter - 1:
        left += 1

    # Define border of the image.
    border = [top, bottom, left, right]

    # Add border.
    copy = cv2.copyMakeBorder(copy, *border,
                              borderType=cv2.BORDER_CONSTANT,
                              value=[0, 0, 0])
    # Return the image.
    return copy


def _get_image_paths(images_path):
    """
    Helper function for getting file paths to images.
    """
    return [os.path.join(images_path, fn) for fn in os.listdir(images_path)]


def _scale_normalize_all(image_paths, save_path, diameter):
    # Get the total amount of images.
    num_images = len(image_paths)

    # For each image in the specified directory.
    for i, image_path in enumerate(image_paths):
        # Status-message.
        msg = "\r- Preprocessing image: {0:>6} / {1}".format(i + 1, num_images)

        # Print the status message.
        sys.stdout.write(msg)
        sys.stdout.flush()

        try:
            # Load the image and clone it for output.
            image = cv2.imread(image_path)

            # Scale normalize the image.
            processed = _scale_normalize(image, diameter=diameter)

            if processed is None:
                print("Could not preprocess {}...".format(image_path))
            else:
                # Get the save path for the processed image.
                image_filename = _get_filename(image_path)
                output_path = os.path.join(save_path, image_filename)

                # Save the image.
                cv2.imwrite(output_path, processed,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        except AttributeError:
            print("Could not preprocess {}...".format(image_path))


########################################################################


def scale_normalize(save_path=None, images_path=None, image_paths=None,
                    diameter=512):
    """
    Function for normalizing scale of images.

    :param save_path:
        Required. Saves preprocessed image to the given path.

    :param images_path:
        Optional. Path to directory in where images reside.

    :param image:
        Optional. Numpy array/OpenCV image with 8-bit image.

    :param diameter:
        Optional. Result diameter of fundus. Defaults to 512.

    :return:
        Nothing.
    """
    if save_path is None:
        raise ValueError("Save path not specified!")

    if image_paths is not None:
        _scale_normalize_all(image_paths=image_paths, save_path=save_path,
                             diameter=diameter)

    elif images_path is not None:
        # Get the paths to all images.
        image_paths = _get_image_paths(images_path)
        # Scale all images.
        _scale_normalize_all(image_paths=image_paths, save_path=save_path,
                             diameter=diameter)


def resize(images_paths, size=512):
    """
    Function for resizing images.

    :param images_paths:
        Required. Paths to images.

    :param size:
        Optional. Size to which resize to. Defaults to 512.

    :return:
        Nothing.
    """
    for image_path in images_paths:
        image = cv2.imread(image_path)

        # Resize the image.
        cv2.resize(image, (size, size))

        # Save the image.
        cv2.imwrite(image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
