import torch
import torch.nn as nn
import torchvision.models as models


class RCNN(nn.Module):
    def __init__(self, extractor_name, input_size=224, use_pretrained=False, get_probs=False):
        super(RCNN, self).__init__()

        self.get_probs = get_probs

        if extractor_name == 'fasterrcnn_resnet50_fpn':
            self.extractor = models.detection.fasterrcnn_resnet50_fpn(pretrained=use_pretrained)
        elif extractor_name == 'fasterrcnn_mobilenet_v3_large_320_fpn':
            self.extractor = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=use_pretrained)
        elif extractor_name == 'retinanet_resnet50_fpn':
            self.extractor = models.detection.retinanet_resnet50_fpn(pretrained=use_pretrained)
        elif extractor_name == 'maskrcnn_resnet50_fpn':
            self.extractor = models.detection.maskrcnn_resnet50_fpn(pretrained=use_pretrained)
        elif extractor_name == 'keypointrcnn_resnet50_fpn':
            self.extractor = models.detection.keypointrcnn_resnet50_fpn(pretrained=use_pretrained)

        self.__feature_size = self.extractor.fc.in_features

        self.get_probs = get_probs
        if not get_probs:
            modules=list(self.extractor.children())[:-1]
            self.extractor = nn.Sequential(*modules)
        else:
            self.__feature_size = self.extractor.fc.out_features

    def forward(self, x_list):
        x_list = self.extractor(x_list)
        
        for x_dict in x_list:
            # TODO: process the output dictionary for each image in the list
            x = torch.flatten(x, 1)
            if self.get_probs:
                x = torch.softmax(x, dim=1)
            return x