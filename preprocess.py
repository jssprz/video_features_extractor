#!/usr/bin/env python
"""Defines the several functions to pre-process a video
"""

import skimage
import numpy as np
import cv2

__author__ = "jssprz"
__version__ = "0.0.1"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"


def resize_frame(image, target_height=224, target_width=224):
    """

    :param image:
    :param target_height:
    :param target_width:
    :return:
    """
    if len(image.shape) == 2:
        # Copy a single-channel gray-scale image three times into a three-channel image
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * target_height / height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height, int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length: resized_image.shape[0] - cropping_length]
    return cv2.resize(resized_image, (target_height, target_width))


def preprocess_frame(image, target_height=224, target_width=224):
    """

    :param image:
    :param target_height:
    :param target_width:
    :return:
    """
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_float(image).astype(np.float32)
    # Whitening based on the mean (RGB format) of the image on the ILSVRC data set
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image -= np.array(mean)
    image /= np.array(std)
    return image
