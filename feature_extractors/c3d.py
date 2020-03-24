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


class C3D_tf(nn.Module):
    def __init__(self, input_size=112, num_segments=16, pretrained_parts='finetune'):
        super(C3D_tf, self).__init__()
        
        self.__input_size = input_size
        self.__input_mean = [104, 117, 128]
        self.__input_std = [1]
        
        import tf_model_zoo
        self.extractor = getattr(tf_model_zoo, 'C3DRes18')(num_segments=num_segments, pretrained_parts=pretrained_parts)
        self.extractor.last_layer_name = 'fc8'
        
        self.__feature_size = self.extractor.fc8.in_features
        
    @property
    def feature_size(self):
        return self.__feature_size
    
    @property
    def crop_size(self):
        return self.__input_size
    
    @property
    def scale_size(self):
        return self.__input_size * 256 // 224
    
    @property
    def input_mean(self):
        return self.__input_mean
    
    @property
    def input_std(self):
        return self.__input_std
        
    def forward(self, x):
        return self.extractor(x)


class C3D_Original(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(C3D_Original, self).__init__()

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
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
        self.__feature_size = self.fc7.in_features
        
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

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        probs = self.softmax(logits)

        return logits
    
    
class C3D(nn.Module):
    """C3D model (https://github.com/DavideA/c3d-pytorch/blob/master/C3D_model.py)
    """

    def __init__(self, input_size=112):
        super(C3D, self).__init__()
        
        self.__input_size = input_size
        self.__input_mean = [104, 117, 128]
        self.__input_std = [1]
        
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
        
        self.__feature_size = self.fc7.in_features

        self.dropout = nn.Dropout(p=0.5)
        
    def load_pretrained(self, model_weights_path):
        pretrained_dict = torch.load(model_weights_path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        
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
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))

        return h