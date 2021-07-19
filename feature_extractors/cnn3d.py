#!/usr/bin/env python
"""
"""

import torch
import torch.nn as nn
import torchvision.models as models
# from feature_extractors.video_resnet import r2plus1d_18

__author__ = "jssprz"
__version__ = "0.0.1"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"


class CNN3D(nn.Module):
    def __init__(self, extractor_name, input_size=224, use_pretrained=False, get_probs=False):
        super(CNN, self).__init__()

        self.__input_size = input_size
        self.__input_mean = [0.43216, 0.394666, 0.37645]
        self.__input_std = [0.22803, 0.22145, 0.216989]

        # self.__input_mean = [0]
        # self.__input_std = [1]

        # initialize the visual feature extractor
        if extractor_name == 'r3d_18':
            self.extractor = models.video.r3d_18(pretrained=use_pretrained, progress=True)
        elif extractor_name == 'mc3_18':
            self.extractor = models.video.mc3_18(pretrained=use_pretrained, progress=True)
        elif extractor_name == 'r2plus1d_18':
            self.extractor = models.video.r2plus1d_18(pretrained=use_pretrained, progress=True)
        else:
            raise ValueError('{} is not a correct extractor name'.format(extractor_name))

        self.__feature_size = self.extractor.fc.in_features
            
        self.get_probs = get_probs
        if not get_probs:
            modules=list(self.extractor.children())[:-1]
            self.extractor = nn.Sequential(*modules)
        else:
            self.__feature_size = self.extractor.fc.out_features

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
    
    @property
    def input_mean(self):
        return self.__input_mean
    
    @property
    def input_std(self):
        return self.__input_std

    def original_forward(self, x):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        if self.get_probs:
            x = torch.softmax(x, dim=1)
        return x

    def forward(self, x):
        return self.original_forward(x)