#!/usr/bin/env python
"""
"""

import torch
import torch.nn as nn
import torchvision.models as models

__author__ = "jssprz"
__version__ = "0.0.1"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"


class AppearanceEncoder(nn.Module):
    def __init__(self, extractor_name, extractor_model_path, use_pretrained=False):
        super(AppearanceEncoder, self).__init__()

        # initialize the visual feature extractor
        if extractor_name == 'resnet18':
            self.extractor = models.resnet18(pretrained=use_pretrained)
        elif extractor_name == 'resnet34':
            self.extractor = models.resnet34(pretrained=use_pretrained)
        elif extractor_name == 'resnet50':
            self.extractor = models.resnet50(pretrained=use_pretrained)
        elif extractor_name == 'resnet101':
            self.extractor = models.resnet101(pretrained=use_pretrained)
        elif extractor_name == 'resnet152':
            self.extractor = models.resnet152(pretrained=use_pretrained)

        self.extractor.load_state_dict(torch.load(extractor_model_path))

        # remove the last fully connected layer
        del self.extractor.fc

    @property
    def feature_size(self):
        return self.extractor.fc.in_features

    def forward(self, x):
        x = self.extractor.conv1(x)
        x = self.extractor.bn1(x)
        x = self.extractor.relu(x)
        x = self.extractor.maxpool(x)

        x = self.extractor.layer1(x)
        x = self.extractor.layer2(x)
        x = self.extractor.layer3(x)
        x = self.extractor.layer4(x)

        x = self.extractor.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
