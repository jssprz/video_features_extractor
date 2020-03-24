#!/usr/bin/env python
"""Defines the several functions to pre-process a video
"""

from PIL import Image, ImageOps
import skimage
import numpy as np
import cv2
import torchvision
import torchvision.transforms as transforms

__author__ = "jssprz"
__version__ = "0.0.1"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

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

def scale_frame(img, h, w):
    resized = cv2.resize(img, (h, w), interpolation = cv2.INTER_LINEAR)
    return resized

def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx, :]

def preprocess_frame(image, scale_size=256, crop_size=224, mean=[.485, .456, .406], std=[.229, .224, .225], normalize_input=False):
    image = np.asarray(image)
    if normalize_input:
        image *= (255.0/image.max().as_array(np.float32))
    image = scale_frame(image, scale_size, scale_size)
    image = crop_center(image, crop_size, crop_size).astype(np.float32)
    image -= np.array(mean).astype(np.float32)
    image /= np.array(std).astype(np.float32)
    return image
    
