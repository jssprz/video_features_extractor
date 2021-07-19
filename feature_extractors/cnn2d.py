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


class CNN2D(nn.Module):
    def __init__(self, extractor_name, input_size=224, use_my_resnet=False, use_pretrained=False, get_probs=False):
        super(CNN2D, self).__init__()

        if use_pretrained:
            self.__input_size = input_size
            self.__input_mean = [0.485, 0.456, 0.406]
            self.__input_std = [0.229, 0.224, 0.225]
        else:
            self.__input_size = input_size
            self.__input_mean = [0]
            self.__input_std = [1]

        # initialize the visual feature extractor
        self.extractor = None
        if extractor_name == 'alexnet':
            self.extractor = models.alexnet(pretrained=use_pretrained)
        elif 'vgg' in extractor_name:
            if extractor_name == 'vgg11':
                self.extractor = models.vgg11(pretrained=use_pretrained) 
            elif extractor_name == 'vgg11_bn':
                self.extractor = models.vgg11_bn(pretrained=use_pretrained) 
            elif extractor_name == 'vgg13':
                self.extractor = models.vgg13(pretrained=use_pretrained) 
            elif extractor_name == 'vgg13_bn':
                self.extractor = models.vgg13_bn(pretrained=use_pretrained) 
            elif extractor_name == 'vgg16':
                self.extractor = models.vgg16(pretrained=use_pretrained) 
            elif extractor_name == 'vgg16_bn':
                self.extractor = models.vgg16_bn(pretrained=use_pretrained) 
            elif extractor_name == 'vgg19':
                self.extractor = models.vgg19(pretrained=use_pretrained) 
            elif extractor_name == 'vgg19_bn':
                self.extractor = models.vgg19_bn(pretrained=use_pretrained) 
        elif 'resnet' in extractor_name:
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
        elif 'squeezenet' in extractor_name:
            if extractor_name == 'squeezenet1_0':
                self.extractor = models.squeezenet1_0(pretrained=use_pretrained) 
            elif extractor_name == 'squeezenet1_1':
                self.extractor = models.squeezenet1_1(pretrained=use_pretrained) 
        elif 'densenet' in extractor_name:
            if extractor_name == 'densenet121':
                self.extractor = models.densenet121(pretrained=use_pretrained) 
            elif extractor_name == 'densenet169':
                self.extractor = models.densenet169(pretrained=use_pretrained) 
            elif extractor_name == 'densenet161':
                self.extractor = models.densenet161(pretrained=use_pretrained) 
            elif extractor_name == 'densenet201':
                self.extractor = models.densenet201(pretrained=use_pretrained)
        elif  extractor_name == 'inception_v3':
            self.extractor = models.inception_v3(pretrained=use_pretrained)
        if extractor_name == 'googlenet':
            self.extractor = models.googlenet(pretrained=use_pretrained)
        elif 'shufflenet' in extractor_name:
            if extractor_name == 'shufflenet_v2_x0_5':
                self.extractor = models.shufflenet_v2_x0_5(pretrained=use_pretrained)
            elif extractor_name == 'shufflenet_v2_x1_0':
                self.extractor = models.shufflenet_v2_x1_0(pretrained=use_pretrained)
            elif extractor_name == 'shufflenet_v2_x1_5':
                self.extractor = models.shufflenet_v2_x1_5(pretrained=use_pretrained)
            elif extractor_name == 'shufflenet_v2_x2_0':
                self.extractor = models.shufflenet_v2_x2_0(pretrained=use_pretrained)
        elif 'mobilenet' in extractor_name:
            if extractor_name == 'mobilenet_v2':
                self.extractor = models.mobilenet_v2(pretrained=use_pretrained)
            elif extractor_name == 'mobilenet_v3_large':
                self.extractor = models.mobilenet_v3_large(pretrained=use_pretrained)
            elif extractor_name == 'mobilenet_v3_small':
                self.extractor = models.mobilenet_v3_small(pretrained=use_pretrained)
            elif extractor_name == 'shufflenet_v2_x2_0':
                self.extractor = models.shufflenet_v2_x2_0(pretrained=use_pretrained)
        elif 'resnext' in extractor_name:
            if extractor_name == 'resnext50':
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
        elif 'wide_resnet' in extractor_name:
            if extractor_name == 'wide_resnet50_2':
                self.extractor = models.wide_resnet50_2(retrained=use_pretrained)
            elif extractor_name == 'wide_resnet101_2':
                self.extractor = models.wide_resnet101_2(pretrained=use_pretrained)
        elif 'mnasnet' in extractor_name:
            if extractor_name == 'mnasnet0_5':
                self.extractor = models.mnasnet0_5(retrained=use_pretrained)
            elif extractor_name == 'mnasnet0_75':
                self.extractor = models.mnasnet0_75(pretrained=use_pretrained)
            elif extractor_name == 'mnasnet1_0':
                self.extractor = models.mnasnet1_0(pretrained=use_pretrained)
            elif extractor_name == 'mnasnet1_3':
                self.extractor = models.mnasnet1_3(pretrained=use_pretrained)
        
        if self.extractor == None:
            raise ValueError('{} is not a correct extractor name'.format(extractor_name))

        self.__feature_size = self.extractor.fc.in_features
            
        self.use_my_resnet = use_my_resnet
        self.get_probs = get_probs
        if use_my_resnet:
            self.att_size = 14
            self.avg_pool = nn.AdaptiveAvgPool2d((self.att_size, self.att_size))
        elif not get_probs:
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
        # x = self.extractor.conv1(x)
        # x = self.extractor.bn1(x)
        # x = self.extractor.relu(x)
        # x = self.extractor.maxpool(x)

        # x = self.extractor.layer1(x)
        # x = self.extractor.layer2(x)
        # x = self.extractor.layer3(x)
        # x = self.extractor.layer4(x)

        # x = self.extractor.avgpool(x)
        # x = torch.flatten(x, 1)
        return x

    def my_forward(self, x):
        # x = x.unsqueeze(0)

        x = self.extractor.conv1(x)
        x = self.extractor.bn1(x)
        x = self.extractor.relu(x)
        x = self.extractor.maxpool(x)

        x = self.extractor.layer1(x)
        x = self.extractor.layer2(x)
        x = self.extractor.layer3(x)
        x = self.extractor.layer4(x)
        print(x.size())

        fc = x.mean(2).mean(2)
        return fc
        # att = self.avg_pool(x).view(-1, self.att_size, self.att_size).permute(1, 2, 0)
        
        # print(fc.size(), att.size())

        return fc, att

    def forward(self, x):
        return self.my_forward(x)[0] if self.use_my_resnet else self.original_forward(x)