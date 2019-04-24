#!/usr/bin/env python
"""Defines the MotionEncoder class
MotionEncoder extracts the motion features from a
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "jssprz"
__version__ = "0.0.1"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"


class C3D(nn.Module):
    """C3D model (https://github.com/DavideA/c3d-pytorch/blob/master/C3D_model.py)
    """

    def __init__(self):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        h = self.pool1(F.relu(self.conv1(x)))
        h = self.pool2(F.relu(self.conv2(h)))

        h = F.relu(self.conv3a(h))
        h = self.pool3(F.relu(self.conv3b(h)))

        h = F.relu(self.conv4a(h))
        h = self.pool4(F.relu(self.conv4b(h)))

        h = F.relu(self.conv5a(h))
        h = self.pool5(F.relu(self.conv5b(h)))

        h = h.view(-1, 8192)
        h = self.dropout(F.relu(self.fc6(h)))
        h = self.dropout(F.relu(self.fc7(h)))

        return h


class MotionEncoder(nn.Module):
    """
    """
    def __init__(self, extractor_name, extractor_path, use_pretrained=True):
        super(MotionEncoder, self).__init__()
        if extractor_name == 'c3d':
            self.extractor = C3D()
            pretrained_dict = torch.load(extractor_path)
            model_dict = self.extractor.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.extractor.load_state_dict(model_dict)

    @property
    def feature_size(self):
        return self.extractor.fc7.in_features

    def forward(self, x):
        return self.extractor(x)
