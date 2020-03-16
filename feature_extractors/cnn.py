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
    def __init__(self, extractor_name, input_size=224, use_my_resnet=False, use_pretrained=False):
        super(CNN, self).__init__()

        self.__input_size = input_size
        self.__input_mean = [0.485, 0.456, 0.406]
        self.__input_std = [0.229, 0.224, 0.225]

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
        elif extractor_name == 'resnext50':
            self.extractor = models.resnext50_32x4d(retrained=use_pretrained)
        elif extractor_name == 'resnext101':
            self.extractor = models.resnext101_32x8d(pretrained=use_pretrained)
        elif extractor_name == 'resnext101-8d-wsl':
            self.extractor = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        elif extractor_name == 'resnext101-16d-wsl':
            self.extractor = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
        elif extractor_name == 'resnext101-32d-wsl':
            self.extractor = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x32d_wsl')
        elif extractor_name == 'resnext101-48d-wsl':
            self.extractor = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
        else:
            raise ValueError('{} is not a correct extractor name'.format(extractor_name))

        self.__feature_size = self.extractor.fc.in_features

        self.use_my_resnet = use_my_resnet
        if use_my_resnet:
            self.avg_pool = nn.AdaptiveAvgPool2d((14, 14))
        else:
            modules=list(self.extractor.children())[:-1]
            self.extractor = nn.Sequential(*modules)

        # remove the last fully connected layer
        # self.extractor = nn.Sequential(*list(self.extractor.children())[:-1])

    def load_pretrained(self, model_weights_path):
        # self.extractor.load_state_dict(torch.load(model_weights_path))
        pretrained_dict = torch.load(model_weights_path)
        model_dict = self.extractor.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.extractor.load_state_dict(model_dict)
        
    @property
    def crop_size(self):
        return self.__input_size
    
    @property
    def scale_size(self):
        return self.__input_size * 256 // 224
        
    @property
    def feature_size(self):
        return self.__feature_size

    def original_forward(self, x):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        return x

    def my_forward(self, x, att_size=14):
        x = x.unsqueeze(0)

        x = self.extractor.conv1(x)
        x = self.extractor.bn1(x)
        x = self.extractor.relu(x)
        x = self.extractor.maxpool(x)

        x = self.extractor.layer1(x)
        x = self.extractor.layer2(x)
        x = self.extractor.layer3(x)
        x = self.extractor.layer4(x)

        fc = x.mean(2).mean(2).squeeze()
        att = self.avg_pool(x).squeeze().permute(1, 2, 0)

        return fc, att

    def forward(self, x):
        return self.my_forward(x)[0] if self.use_my_resnet else self.original_forward(x)