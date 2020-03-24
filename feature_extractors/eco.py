# import argparse
import os
import sys
import time
import shutil

import torch
import torch.nn as nn
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.model_zoo as model_zoo
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.utils import clip_grad_norm_

# from dataset import TSNDataSet
# from models import TSN
from feature_extractors.ops.transforms import *
# from opts import parser


from feature_extractors.ops.basic_ops import ConsensusModule, Identity

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, pretrained_parts, modality, base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True, dropout=0.8, crop_num=1, partial_bn=True, get_global_pool=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.pretrained_parts = pretrained_parts
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.base_model_name = base_model
        self.get_global_pool = get_global_pool
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)

        # zc comments
        feature_dim = self._prepare_tsn(num_class)
        # modules = list(self.modules())
        # print(modules)
        # zc comments end

        '''
        # zc: print "NN variable name"
        zc_params = self.base_model.state_dict()
        for zc_k in zc_params.items():
            print(zc_k)
        # zc: print "Specified layer's weight and bias"
        print(zc_params['conv1_7x7_s2.weight'])
        print(zc_params['conv1_7x7_s2.bias'])
        '''

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            xavier_uniform_(getattr(self.base_model, self.base_model.last_layer_name).weight)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            xavier_uniform_(self.new_fc.weight)
            constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'C3DRes18':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)(num_segments=self.num_segments, pretrained_parts=self.pretrained_parts)
            self.base_model.last_layer_name = 'fc8'
            self.input_size = 112
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif base_model == 'ECO':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)(num_segments=self.num_segments, pretrained_parts=self.pretrained_parts)
            self.base_model.last_layer_name = 'fc_final'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif base_model == 'ECOfull' :
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)(num_segments=self.num_segments, pretrained_parts=self.pretrained_parts)
            self.base_model.last_layer_name = 'fc_final'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)


        elif base_model == 'BN2to1D':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)(num_segments=self.num_segments)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        else:
            print("No BN layer Freezing.")

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_3d_conv_weight = []
        first_3d_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_2d_cnt = 0
        conv_3d_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            # (conv1d or conv2d) 1st layer's params will be append to list: first_conv_weight & first_conv_bias, total num 1 respectively(1 conv2d)
            # (conv1d or conv2d or Linear) from 2nd layers' params will be append to list: normal_weight & normal_bias, total num 69 respectively(68 Conv2d + 1 Linear)
            if isinstance(m, torch.nn.Conv2d):
                ps = list(m.parameters())
                conv_2d_cnt += 1
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_3d_cnt += 1
                if conv_3d_cnt == 1:
                    first_3d_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_3d_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            # (BatchNorm1d or BatchNorm2d) params will be append to list: bn, total num 2 (enabled pbn, so only: 1st BN layer's weight + 1st BN layer's bias)
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # 4
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': first_3d_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_3d_conv_weight"},
            {'params': first_3d_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_3d_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def get_optim_policies_BN2to1D(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        last_conv_weight = []
        last_conv_bias = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            # (conv1d or conv2d) 1st layer's params will be append to list: first_conv_weight & first_conv_bias, total num 1 respectively(1 conv2d)
            # (conv1d or conv2d or Linear) from 2nd layers' params will be append to list: normal_weight & normal_bias, total num 69 respectively(68 Conv2d + 1 Linear)
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                last_conv_weight.append(ps[0])
                if len(ps) == 2:
                    last_conv_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            # (BatchNorm1d or BatchNorm2d) params will be append to list: bn, total num 2 (enabled pbn, so only: 1st BN layer's weight + 1st BN layer's bias)
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # 4
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
             {'params': last_conv_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "last_conv_weight"},
            {'params': last_conv_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "last_conv_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
#         print(input.size(), input.mean())
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        # input.size(): [32, 9, 224, 224]
        # after view() func: [96, 3, 224, 224]
        # print(input.view((-1, sample_len) + input.size()[-2:]).size())
        if self.base_model_name == "C3DRes18":
            before_permute = input.view((-1, sample_len) + input.size()[-2:])
            input_var = torch.transpose(before_permute.view((-1, self.num_segments) + before_permute.size()[1:]), 1, 2)
        else:
            input_var = input.view((-1, sample_len) + input.size()[-2:])
            
#         print(input_var.size(), input.mean())
        base_out, global_pool = self.base_model(input_var)
#         print(base_out.size(), base_out.mean())
        
        if self.get_global_pool:
            base_out = global_pool

        # zc comments
        if self.dropout > 0:
            base_out = self.new_fc(base_out)
            
#         print(base_out.size(), base_out.mean())

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        # zc comments end
        
#         print(base_out.size(), base_out.mean())
        
        if self.reshape:
            if self.base_model_name == 'C3DRes18':
                output = base_out
                output = self.consensus(base_out)
#                 print(output.size(), output.mean())
                return output
            elif self.base_model_name == 'ECO':
                output = base_out
                output = self.consensus(base_out)
#                 print(output.size(), output.mean())                
                return output
            elif self.base_model_name == 'ECOfull':
                output = base_out
                output = self.consensus(base_out)
#                 print(output.size(), output.mean())                
                return output
            else:
                # base_out.size(): [32, 3, 101], [batch_size, num_segments, num_class] respectively
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
                # output.size(): [32, 1, 101]
                output = self.consensus(base_out)
                # output after squeeze(1): [32, 101], forward() returns size: [batch_size, num_class]
#                 print(output.size(), output.mean())
                return output.squeeze(1)


    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224
    
    @property
    def feature_size(self):
        return self.base_model.fc.in_features

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        
        
def init_ECO(model_dict, pretrained_parts, net_model2D=None, net_model3D=None, net_modelECO=None):

    weight_url_2d='https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'

    if pretrained_parts == "scratch":
        new_state_dict = {}
    elif pretrained_parts == "2D":
        if net_model2D is not None:
            pretrained_dict_2d = torch.load(net_model2D)
            print(("=> loading model - 2D net:  '{}'".format(net_model2D)))
        else:
            weight_url_2d='https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'
            pretrained_dict_2d = torch.utils.model_zoo.load_url(weight_url_2d)
            print(("=> loading model - 2D net-url:  '{}'".format(weight_url_2d)))

        #print(pretrained_dict_2d)
        for k, v in pretrained_dict_2d['state_dict'].items():
            if "module.base_model."+k in model_dict:
                print("k is in model dict", k)
            else:
                print("Problem!")
                print("k: {}, size: {}".format(k,v.shape))
       
        new_state_dict = {"module.base_model."+k: v for k, v in pretrained_dict_2d['state_dict'].items() if "module.base_model."+k in model_dict}
    elif pretrained_parts == "3D":
        new_state_dict = {}
        if net_model3D is not None:
            pretrained_dict_3d = torch.load(net_model3D)
            print(("=> loading model - 3D net:  '{}'".format(net_model3D)))
        else:
            pretrained_dict_3d = torch.load("models/Kinetic-400-eco/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
            print(("=> loading model - 3D net-url:  '{}'".format("models/Kinetic-400-eco/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")))

        for k, v in pretrained_dict_3d['state_dict'].items():
            if (k in model_dict) and (v.size() == model_dict[k].size()):
                new_state_dict[k] = v

        res3a_2_weight_chunk = torch.chunk(pretrained_dict_3d["state_dict"]["module.base_model.res3a_2.weight"], 4, 1)
        new_state_dict["module.base_model.res3a_2.weight"] = torch.cat((res3a_2_weight_chunk[0], res3a_2_weight_chunk[1], res3a_2_weight_chunk[2]), 1)
    elif pretrained_parts == "finetune":
        print(net_modelECO)
        print("88"*40)
        if net_modelECO is not None:
            pretrained_dict = torch.load(net_modelECO)
            print(("=> loading model-finetune: '{}'".format(net_modelECO)))
        else:
            pretrained_dict = torch.load("models/Kinetic-400-eco/eco_lite_rgb_16F_kinetics_v3.pth.tar")
            print(("=> loading model-finetune-url: '{}'".format("models/Kinetic-400-eco/eco_lite_rgb_16F_kinetics_v3.pth.tar")))

        new_state_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if (k in model_dict) and (v.size() == model_dict[k].size())}
        print("*"*50)
        print("Start finetuning ..")
    elif pretrained_parts == "both":
        # Load the 2D net pretrained model
        if net_model2D is not None:
            pretrained_dict_2d = torch.load(net_model2D)
            print(("=> loading model - 2D net:  '{}'".format(args.net_model2D)))
        else:
            weight_url_2d='https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'
            pretrained_dict_2d = torch.utils.model_zoo.load_url(weight_url_2d)
            print(("=> loading model - 2D net-url:  '{}'".format(weight_url_2d)))
        # Load the 3D net pretrained model
        if net_model3D is not None:
            pretrained_dict_3d = torch.load(net_model3D)
            print(("=> loading model - 3D net:  '{}'".format(net_model3D)))
        else:
            pretrained_dict_3d = torch.load("models/Kinetic-400-eco/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
            print(("=> loading model - 3D net-url:  '{}'".format("models/Kinetic-400-eco/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")))

        new_state_dict = {"module.base_model."+k: v for k, v in pretrained_dict_2d['state_dict'].items() if "module.base_model."+k in model_dict}

        for k, v in pretrained_dict_3d['state_dict'].items():
            if (k in model_dict) and (v.size() == model_dict[k].size()):
                new_state_dict[k] = v

        res3a_2_weight_chunk = torch.chunk(pretrained_dict_3d["state_dict"]["module.base_model.res3a_2.weight"], 4, 1)
        new_state_dict["module.base_model.res3a_2.weight"] = torch.cat((res3a_2_weight_chunk[0], res3a_2_weight_chunk[1], res3a_2_weight_chunk[2]), 1)
    return new_state_dict


def init_ECOfull(model_dict, pretrained_parts, net_model2D=None, net_model3D=None, net_modelECO=None):
    weight_url_2d='https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'

    if pretrained_parts == "scratch":     
        new_state_dict = {}
    elif pretrained_parts == "2D":
        pretrained_dict_2d = torch.utils.model_zoo.load_url(weight_url_2d)
        new_state_dict = {"module.base_model."+k: v for k, v in pretrained_dict_2d['state_dict'].items() if "module.base_model."+k in model_dict}
    elif pretrained_parts == "3D":
        new_state_dict = {}
        pretrained_dict_3d = torch.load("models/Kinetic-400-eco/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
        for k, v in pretrained_dict_3d['state_dict'].items():
            if (k in model_dict) and (v.size() == model_dict[k].size()):
                new_state_dict[k] = v

        res3a_2_weight_chunk = torch.chunk(pretrained_dict_3d["state_dict"]["module.base_model.res3a_2.weight"], 4, 1)
        new_state_dict["module.base_model.res3a_2.weight"] = torch.cat((res3a_2_weight_chunk[0], res3a_2_weight_chunk[1], res3a_2_weight_chunk[2]), 1)
    elif pretrained_parts == "finetune":
        print(net_modelECO)
        print("88"*40)
        if net_modelECO is not None:
            pretrained_dict = torch.load(net_modelECO)
            print(("=> loading model-finetune: '{}'".format(args.net_modelECO)))
        else:
            pretrained_dict = torch.load("models/Kinetic-400-eco/ECO_Full_rgb_model_Kinetics.pth.tar")
            print(("=> loading model-finetune-url: '{}'".format("models/Kinetic-400-eco/ECO_Full_rgb_model_Kinetics.pth.tar")))

        new_state_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if (k in model_dict) and (v.size() == model_dict[k].size())}
        print("*"*50)
        print("Start finetuning ..")
    elif pretrained_parts == "both":
        # Load the 2D net pretrained model
        if net_model2D is not None:
            pretrained_dict_2d = torch.load(net_model2D)
            print(("=> loading model - 2D net:  '{}'".format(net_model2D)))
        else:
            weight_url_2d='https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'
            pretrained_dict_2d = torch.utils.model_zoo.load_url(weight_url_2d)
            print(("=> loading model - 2D net-url:  '{}'".format(weight_url_2d)))

        new_state_dict = {"module.base_model."+k: v for k, v in pretrained_dict_2d['state_dict'].items() if "module.base_model."+k in model_dict}

        # Load the 3D net pretrained model
        if net_model3D is not None:
            pretrained_dict_3d = torch.load(net_model3D)
            print(("=> loading model - 3D net:  '{}'".format(net_model3D)))
        else:
            pretrained_dict_3d = torch.load("models/Kinetic-400-eco/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
            print(("=> loading model - 3D net-url:  '{}'".format("models/Kinetic-400-eco/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")))

        for k, v in pretrained_dict_3d['state_dict'].items():
            if (k in model_dict) and (v.size() == model_dict[k].size()):
                new_state_dict[k] = v

        #res3a_2_weight_chunk = torch.chunk(pretrained_dict_3d["state_dict"]["module.base_model.res3a_2.weight"], 4, 1)
        #new_state_dict["module.base_model.res3a_2.weight"] = torch.cat((res3a_2_weight_chunk[0], res3a_2_weight_chunk[1], res3a_2_weight_chunk[2]), 1)

    return new_state_dict

def init_C3DRes18(model_dict, pretrained_parts):

    if pretrained_parts == "scratch":
        new_state_dict = {}
    elif pretrained_parts == "3D":
        pretrained_dict = torch.load("models/Kinetic-400-eco/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
        new_state_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if (k in model_dict) and (v.size() == model_dict[k].size())}
    else:
        raise ValueError('For C3DRes18, "--pretrained_parts" can only be chosen from [scratch, 3D]')

    return new_state_dict


def init_model(num_class, num_segments, pretrained_parts, modality, arch, consensus_type, dropout, no_partialbn, resume_chpt, get_global_pool, gpus):
#     global args, best_prec1
#     args = parser.parse_args()

    print("------------------------------------")
    print("Environment Versions:")
    print("- Python: {}".format(sys.version))
    print("- PyTorch: {}".format(torch.__version__))
    print("- TorchVison: {}".format(torchvision.__version__))

#     args_dict = args.__dict__
#     print("------------------------------------")
#     print(args.arch+" Configurations:")
#     for key in args_dict.keys():
#         print("- {}: {}".format(key, args_dict[key]))
#     print("------------------------------------")

#     if args.dataset == 'ucf101':
#         num_class = 101
#         rgb_read_format = "{:05d}.jpg"
#     elif args.dataset == 'hmdb51':
#         num_class = 51
#         rgb_read_format = "{:05d}.jpg"
#     elif args.dataset == 'kinetics':
#         num_class = 400
#         rgb_read_format = "{:04d}.jpg"
#     elif args.dataset == 'something':
#         num_class = 174
#         rgb_read_format = "{:04d}.jpg"
#     else:
#         raise ValueError('Unknown dataset '+args.dataset)

    model = TSN(num_class, num_segments, pretrained_parts, modality, base_model=arch,
                consensus_type=consensus_type, dropout=dropout, partial_bn=not no_partialbn, get_global_pool=get_global_pool)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std

    # Optimizer s also support specifying per-parameter options. 
    # To do this, pass in an iterable of dict s. 
    # Each of them will define a separate parameter group, 
    # and should contain a params key, containing a list of parameters belonging to it. 
    # Other keys should match the keyword arguments accepted by the optimizers, 
    # and will be used as optimization options for this group.
    policies = model.get_optim_policies()

    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    model_dict = model.state_dict()

    print("pretrained_parts: ", pretrained_parts)

    if resume_chpt:
        if os.path.isfile(resume_chpt):
            print(("=> loading checkpoint '{}'".format(resume_chpt)))
            checkpoint = torch.load(resume_chpt)
            # if not checkpoint['lr']:
            if "lr" not in checkpoint.keys():
                lr = input("No 'lr' attribute found in resume model, please input the 'lr' manually: ")
                lr = float(lr)
            else:
                lr = checkpoint['lr']
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            
            base_dict = checkpoint['state_dict']
            replace_dict = {'module.base_model.fc_final.weight': 'module.new_fc.weight',
                            'module.base_model.fc_final.bias': 'module.new_fc.bias'}
            for k, v in replace_dict.items():
                if k in base_dict:
                    base_dict[v] = base_dict.pop(k)
            
            model.load_state_dict(base_dict)
            print(("=> loaded checkpoint '{}' (epoch: {}, lr: {})"
                  .format(resume_chpt, checkpoint['epoch'], lr)))
        else:
            print(("=> no checkpoint found at '{}'".format(resume_chpt)))
    else:
        if arch == "ECO":
            new_state_dict = init_ECO(model_dict, pretrained_parts)
        elif arch == "ECOfull":
            new_state_dict = init_ECOfull(model_dict, pretrained_parts)
        elif arch == "C3DRes18":
            new_state_dict = init_C3DRes18(model_dict, pretrained_parts)

        un_init_dict_keys = [k for k in model_dict.keys() if k not in new_state_dict]
        print("un_init_dict_keys: ", un_init_dict_keys)
        print("\n------------------------------------")

        for k in un_init_dict_keys:
            new_state_dict[k] = torch.DoubleTensor(model_dict[k].size()).zero_()
            if 'weight' in k:
                if 'bn' in k:
                    print("{} init as: 1".format(k))
                    constant_(new_state_dict[k], 1)
                else:
                    print("{} init as: xavier".format(k))
                    xavier_uniform_(new_state_dict[k])
            elif 'bias' in k:
                print("{} init as: 0".format(k))
                constant_(new_state_dict[k], 0)

        print("------------------------------------")
        model.load_state_dict(new_state_dict)
            
    return model, crop_size, scale_size, input_mean, input_std



    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        #input_mean = [0,0,0] #for debugging
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=args.rgb_prefix+rgb_read_format if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+rgb_read_format,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=args.rgb_prefix+rgb_read_format if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+rgb_read_format,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       #Stack(roll=(args.arch == 'C3DRes18') or (args.arch == 'ECO') or (args.arch == 'ECOfull') or (args.arch == 'ECO_2FC')),
                       #ToTorchFormatTensor(div=(args.arch != 'C3DRes18') and (args.arch != 'ECO') and (args.arch != 'ECOfull') and (args.arch != 'ECO_2FC')),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,nesterov=args.nesterov)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    saturate_cnt = 0
    exp_num = 0

    for epoch in range(args.start_epoch, args.epochs):
        if saturate_cnt == args.num_saturate:
            exp_num = exp_num + 1
            saturate_cnt = 0
            print("- Learning rate decreases by a factor of '{}'".format(10**(exp_num)))
        adjust_learning_rate(optimizer, epoch, args.lr_steps, exp_num)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            if is_best:
                saturate_cnt = 0
            else:
                saturate_cnt = saturate_cnt + 1

            print("- Validation Prec@1 saturates for {} epochs.".format(saturate_cnt))
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'lr': optimizer.param_groups[-1]['lr'],
            }, is_best)