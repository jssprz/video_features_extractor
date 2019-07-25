#!/usr/bin/env python
"""Defines the MotionEncoder class
MotionEncoder extracts the motion features from a sequence of frames
"""

import torch
import torch.nn as nn

from .c3d import C3D

__author__ = "jssprz"
__version__ = "0.0.1"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"


class MotionEncoder(nn.Module):
    """
    """
    def __init__(self, extractor_name, extractor_path):
        super(MotionEncoder, self).__init__()
        if extractor_name == 'c3d':
            self.extractor = C3D()

            pretrained_dict = torch.load(extractor_path)
            model_dict = self.extractor.state_dict()
            model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
            self.extractor.load_state_dict(model_dict)

        self.feature_size = self.extractor.fc7.in_features

    def forward(self, x):
        return self.extractor(x)
