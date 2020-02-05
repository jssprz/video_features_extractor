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


class CNN(nn.Module):
    def __init__(self, extractor_name, extractor_model_path, use_my_resnet=False, use_pretrained=False):
        super(CNN, self).__init__()

        # initialize the visual feature extractor
        if extractor_name == 'resnet18':
            self.resnet = models.resnet18(pretrained=use_pretrained)
        elif extractor_name == 'resnet34':
            self.resnet = models.resnet34(pretrained=use_pretrained)
        elif extractor_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=use_pretrained)
        elif extractor_name == 'resnet101':
            self.resnet = models.resnet101(pretrained=use_pretrained)
        elif extractor_name == 'resnet152':
            self.resnet = models.resnet152(pretrained=use_pretrained)

        self.use_my_resnet = use_my_resnet
        if use_my_resnet:
            self.avg_pool = nn.AdaptiveAvgPool2d((14, 14))

        # remove the last fully connected layer
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    @property
    def feature_size(self):
        return self.resnet.fc.in_features

    def original_forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def my_forward(self, x, att_size=14):
        x = x.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(2).mean(2).squeeze()
        att = self.avg_pool(x).squeeze().permute(1, 2, 0)

        return fc, att

    def forward(self, x):
        return self.my_forward(x)[0] if self.use_my_resnet else self.original_forward(x)