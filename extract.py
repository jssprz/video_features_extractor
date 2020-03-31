#!/usr/bin/env python
"""

"""

import os
import argparse
import h5py
import time
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from utils import get_freer_gpu
from preprocess import resize_frame, preprocess_frame, ToTensorWithoutScaling, ToFloatTensorInZeroOne
from sample_frames import sample_frames, sample_frames2
from feature_extractors.cnn import CNN
from feature_extractors.c3d import C3D
from feature_extractors.i3dpt import I3D
from feature_extractors.eco import init_model as ECO
from feature_extractors.tsm import init_model as TSM
from configuration_file import ConfigurationFile

__author__ = "jssprz"
__version__ = "0.0.1"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None

def extract_features(file_name, extractor_name, extractor, dataset_name, device, config, feature_size, transformer):
    """

    :type c3d_extractor:
    :param device:
    :param config:
    :param frame_shape:
    :param dataset_name:
    :param cnn_extractor:
    :param c3d_extractor:
    :param params:
    :return:
    """

    if not os.path.exists(config.features_dir):
        os.makedirs(config.features_dir)

    # Read the video list and let the videos sort by id in ascending order
    if dataset_name == 'MSVD':
        videos = [os.path.join(config.data_dir, v) for v in sorted(os.listdir(config.data_dir))]
        map_file = open(config.mapping_path, 'w')
    elif dataset_name == 'MSR-VTT':
        videos = [os.path.join(config.data_dir, v) for v in sorted(os.listdir(config.data_dir), key=lambda x: int(x[5:-4]))]
    elif dataset_name == 'M-VAD':
        videos = [os.path.join(config.data_dir, v) for v in sorted(os.listdir(config.data_dir), key=lambda x: int(x[5:-4]))]
    else:
        with open(os.path.join(config.data_dir, 'list.txt')) as f:
            videos = [os.path.join(config.data_dir, path.strip()) for path in f.readlines()]

    # Create an hdf5 file that saves video features
    feature_h5_path = os.path.join(config.features_dir, file_name)
    if os.path.exists(feature_h5_path):
        # If the hdf5 file already exists, it has been processed before,
        # perhaps it has not been completely processed.
        # Read using r+ (read and write) mode to avoid overwriting previously saved data
        h5 = h5py.File(feature_h5_path, 'r+')
    else:
        h5 = h5py.File(feature_h5_path, 'w')

    if dataset_name in list(h5.keys()):
        dataset = h5[dataset_name]
        if extractor_name in list(dataset.keys()):
            dataset_model = dataset[extractor_name]
        elif extractor_name.split('_')[-1] == 'features':
            dataset_model = dataset.create_dataset(extractor_name, (config.num_videos, config.max_frames,
                                                                    feature_size), dtype='float32')
        elif extractor_name.split('_')[-1] == 'globals': 
            dataset_model = dataset.create_dataset(extractor_name, (config.num_videos, 1, feature_size), dtype='float32')
        dataset_counts = dataset['count_features']
    else:
        dataset = h5.create_group(dataset_name)
        if extractor_name.split('_')[-1] == 'features':
            dataset_model = dataset.create_dataset(extractor_name, (config.num_videos, config.max_frames,
                                                                    feature_size), dtype='float32')
        elif extractor_name.split('_')[-1] == 'globals':
            dataset_model = dataset.create_dataset(extractor_name, (config.num_videos, 1, feature_size), dtype='float32')
        dataset_counts = dataset.create_dataset('count_features', (config.num_videos,), dtype='int')
                                   
    extractor.to(device)
    extractor.eval()

    for i, video_path in enumerate(videos):
        if dataset_name == 'MSVD':
            map_name_id = '{}\tvideo{}\n'.format(video_path.split('/')[-1][:-4], i)
            map_file.write(map_name_id)

        # Extract video frames and video tiles
        frame_list, clip_list, frame_count = sample_frames(video_path, config.max_frames, config.frame_sample_rate,
                                                           config.frame_sample_overlap)
#         sampled_frames, frame_count = sample_frames2(video_path, num_segments=16, segment_length=1)

        if i % 50 == 0 or frame_count == 0:
            print('%d\t%s\t%d' % (i, video_path.split('/')[-1], frame_count))

        # If the number of frames is less than max_frames, then the remaining part is complemented by 0
#         dataset_model[i] = np.zeros((config.max_frames, extractor.feature_size), dtype='float32')

        if extractor_name == 'cnn_features': 
            # Preprocess frames and then convert it into (batch, channel, height, width) format
#             frame_list = np.array([preprocess_frame(x, scale_size=scale_size, crop_size=crop_size,
#                                                     mean=input_mean, std=input_std, normalize_input=True) 
#                                    for x in frame_list])
#             frame_list = torch.from_numpy(frame_list.transpose((0, 3, 1, 2))).to(device)
#             frame_list = torch.cat([preprocess_frame(x, scale_size=scale_size, crop_size=crop_size,
#                                                     mean=input_mean, std=input_std, normalize_input=True).unsqueeze(0) 
#                                    for x in frame_list], dim=0).to(device)
            frame_list = torch.cat([transformer(x).unsqueeze(0) for x in frame_list], dim=0).to(device)

            # Extracting cnn features of sampled frames first
            features = extractor(frame_list)
            print(features.size(), features.mean())
        elif extractor_name in ['cnn_globals', 'cnn_sem_globals']:
            # frame_list = torch.cat([torch.from_numpy(np.array(x)).unsqueeze(0) for x in frame_list], dim=0).to(device)
            frame_list = torch.cat([transformer(x).unsqueeze(0) for x in frame_list], dim=0).to(device)
            # features = extractor(transformer(frame_list))
            features = extractor(frame_list.transpose(0,1).unsqueeze(0))
            print(features.size(), features.min(), features.max(), features.mean())
        elif extractor_name in ['c3d_features', 'i3d_features']:
            # Preprocess frames of the video fragments to extract motion features
#             clip_list = np.array([[preprocess_frame(x, scale_size=extractor.scale_size, crop_size=extractor.crop_size,
#                                                     mean=extractor.input_mean, std=extractor.input_std) for x in clip] for clip in clip_list])
#             clip_list = clip_list.transpose((0, 4, 1, 2, 3)).astype(np.float32)
#             clip_list = torch.from_numpy(clip_list).to(device)
            clips_tensors = []
            for clip in clip_list:
                t = torch.cat([transformer(x).unsqueeze(0) for x in clip], dim=0).unsqueeze(0)
                clips_tensors.append(t)
            clip_list = torch.cat(clips_tensors, dim=0).transpose(1,2).to(device)

            # Extracting c3d features
            features = extractor(clip_list)
            print(features.size(), features.mean())
        elif extractor_name in ['c3d_globals', 'i3d_globals']:
            # Preprocess frames of the video fragments to extract motion features
            frames = np.array([preprocess_frame(x, scale_size=scale_size, crop_size=crop_size,
                                                mean=input_mean, std=input_std) for x in frame_list])
            frames = torch.from_numpy(frames.transpose((0, 3, 1, 2)).astype(np.float32)).unsqueeze(2).to(device)

            # Extracting i3d features of sampled frames first
            features = extractor(frames)[1]
            print(features.size(), features.mean())
        elif extractor_name in ['eco_features', 'tsm_features', 'eco_sem_features', 'tsm_sem_features']:
            features = []
            for clip in clip_list:
                clip_frames = torch.cat([torch.from_numpy(preprocess_frame(x, scale_size=scale_size, crop_size=crop_size,
                                                                          mean=input_mean, std=input_std))
                                         for x in clip], dim=2).transpose(0,2).unsqueeze(0).to(device)
                # Extracting eco features from current clip
                features.append(extractor(clip_frames))
            features = torch.cat(features, dim=0)
            if extractor_name in ['eco_sem_features', 'tsm_sem_features']:
                probs = torch.softmax(features, dim=1)
                print(probs.size(), probs.max(), probs.min())
        elif extractor_name in ['eco_globals', 'tsm_globals', 'eco_sem_globals', 'tsm_sem_globals']:
            features = []
            frames = torch.cat([torch.from_numpy(preprocess_frame(x, scale_size=scale_size, crop_size=crop_size,
                                                                  mean=input_mean, std=input_std))
                                 for x in frame_list], dim=2).transpose(0,2).unsqueeze(0).to(device)
            # Extracting eco-semantic features from sampled frames
            features = extractor(frames)
            if extractor_name in ['eco_sem_globals', 'tsm_sem_globals']:
                probs = torch.softmax(features, dim=1)
                print(probs.size(), probs.max(), probs.min())

        if features.size(0) > 1:
            dataset_model[i] = np.zeros((config.max_frames, feature_size), dtype='float32')
            dataset_model[i, :features.size(0), :] = features.data.cpu().numpy()
            dataset_counts[i] = features.size(0)
        else:
            dataset_model[i] = features.data.cpu().numpy()

    h5.close()

    if dataset_name == 'MSVD':
        map_file.close()


def main(args, config):   
    if torch.cuda.is_available():
        freer_gpu_ids = get_freer_gpu()[0]
        device = torch.device('cuda:{}'.format(freer_gpu_ids))
        print('Running on freer device: cuda:{}'.format(freer_gpu_ids))
    else:
        device = torch.device('cpu')
        print('Running on cpu device')
    
    file_name = 'features_linspace{}_{}-{}.h5'.format(config.frame_sample_rate, config.max_frames, '-'.join(args.features))

    for feats_name in args.features:
        if feats_name == 'cnn_features':
            print('Extracting CNN for {} dataset'.format(args.dataset_name))
            cnn_use_torch_weights = (config.cnn_pretrained_path == '')
            model = CNN(config.cnn_model, input_size=224, use_pretrained=cnn_use_torch_weights, use_my_resnet=False)
            if not cnn_use_torch_weights:
                model.load_pretrained(config.cnn_pretrained_path)
            transformer = transforms.Compose([transforms.Scale(model.scale_size),
                                            transforms.CenterCrop(model.crop_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=model.input_mean, std=model.input_std)])
            with torch.no_grad():
                extract_features(file_name, feats_name, model, args.dataset_name, device, config, model.feature_size, transformer)
        if feats_name in ['cnn_globals', 'cnn_sem_globals']:
            print('Extracting ResNet (2+1)D for {} dataset'.format(args.dataset_name))
            model = CNN('r2plus1d_18', input_size=112, use_pretrained=True, use_my_resnet=False, get_probs=feats_name=='cnn_sem_globals')
            transformer = transforms.Compose([transforms.Resize((128, 171)),
                                            transforms.CenterCrop((112, 112)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                                                std=[0.22803, 0.22145, 0.216989]),
                                            ])
            with torch.no_grad():
                extract_features(file_name, feats_name, model, args.dataset_name, device, config, model.feature_size, transformer)
        if feats_name in ['c3d_features', 'c3d_globals']:
            print('Extracting C3D for {} dataset'.format(args.dataset_name))
            model = C3D()
            model.load_pretrained(config.c3d_pretrained_path)
            transformer = transforms.Compose([transforms.Scale((200, 112)),
                                            transforms.CenterCrop(112),
                                            ToTensorWithoutScaling(),
                                            transforms.Normalize(mean=model.input_mean, std=model.input_std)])
            with torch.no_grad():
                extract_features(file_name, feats_name, model, args.dataset_name, device, config, model.feature_size, transformer)
        if feats_name in ['c3d_globals', 'i3d_globals']:
            print('Extracting I3D for {} dataset'.format(args.dataset_name))
            model = I3D(modality='rgb')
            model.load_state_dict(torch.load(config.i3d_pretrained_path))
            with torch.no_grad():
                extract_features(file_name, feats_name, model, args.dataset_name, device, config, feature_size=model.feature_size,
                              crop_size=model.crop_size, scale_size=model.scale_size, input_mean=model.input_mean, 
                              input_std=model.input_std)
        if feats_name in ['eco_features', 'eco_globals']:
            print('Extracting ECOfull for {} dataset'.format(args.dataset_name))
            model, crop_size, scale_size, input_mean, input_std = ECO(num_class=400, num_segments=config.frame_sample_rate, 
                                                                    pretrained_parts='finetune', #'2D', '3D',
                                                                    modality='RGB', 
                                                                    arch='ECOfull',  # 'ECOfull' 'ECO' 'C3DRes18'
                                                                    consensus_type='identity', #'avg',
                                                                    dropout=0, 
                                                                    no_partialbn=True, #False, 
                                                                    resume_chpt='',
                                                                    get_global_pool=True,
                                                                    gpus=[device])
            with torch.no_grad():
                extract_features(file_name, feats_name, model, args.dataset_name, device, config, 1536, crop_size, scale_size, input_mean,
                                 input_std)
        if feats_name in ['eco_sem_features', 'eco_sem_globals']:
            print('Extracting ECO-Smantic for {} dataset'.format(args.dataset_name))
            model, crop_size, scale_size, input_mean, input_std = ECO(num_class=400, num_segments=config.frame_sample_rate, 
                                                                    pretrained_parts='finetune', #'2D', '3D',
                                                                    modality='RGB', 
                                                                    arch='ECOfull',  # 'ECOfull' 'ECO' 'C3DRes18'
                                                                    consensus_type='identity', #'avg',
                                                                    dropout=.8, 
                                                                    no_partialbn=True, #False, 
                                                                    resume_chpt='',
                                                                    get_global_pool=False,
                                                                    gpus=[device])
            with torch.no_grad():
                extract_features(file_name, 'eco_sem_features', model, args.dataset_name, device, config, 400, crop_size, scale_size,
                                 input_mean, input_std)
        if feats_name in ['tsm_features', 'tsm_globals']:
            print('Extracting {} for {} dataset'.format(feats_name, args.dataset_name))
            model, crop_size, scale_size, input_mean, input_std = TSM(num_class=174, num_segments=config.frame_sample_rate, 
                                                                    modality='RGB', 
                                                                    arch='resnet50',  # 'resnet101'
                                                                    consensus_type='avg', # 'avg' 'identity'
                                                                    dropout=0, 
                                                                    img_feature_dim=256,
                                                                    no_partialbn=True, #False,
                                                                    pretrain='imagenet',
                                                                    is_shift=True, 
                                                                    shift_div=8, 
                                                                    shift_place='blockers', 
                                                                    non_local=False,
                                                                    temporal_pool=False,
                                                                    resume_chkpt='./models/Smth-Smth-v2-tsm/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment16_e45.pth',
                                                                    get_global_pool=True,
                                                                    gpus=[device])
        if feats_name in ['tsm_sem_features', 'tsm_sem_globals']:
            print('Extracting {} for {} dataset'.format(feats_name, args.dataset_name))
            model, crop_size, scale_size, input_mean, input_std = TSM(num_class=174, num_segments=config.frame_sample_rate, 
                                                                    modality='RGB', 
                                                                    arch='resnet50',  # 'resnet101'
                                                                    consensus_type='avg', # 'avg' 'identity'
                                                                    dropout=.8, 
                                                                    img_feature_dim=256,
                                                                    no_partialbn=True, #False,
                                                                    pretrain='imagenet',
                                                                    is_shift=True, 
                                                                    shift_div=8, 
                                                                    shift_place='blockers', 
                                                                    non_local=False,
                                                                    temporal_pool=False,
                                                                    resume_chkpt='./models/Smth-Smth-v2-tsm/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment16_e45.pth',
                                                                    get_global_pool=False,
                                                                    gpus=[device])
            with torch.no_grad():
                extract_features(file_name, feats_name, model, args.dataset_name, device, config, 174, crop_size, scale_size, input_mean, input_std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a captioning model for a specific dataset.')
    parser.add_argument('-ds', '--dataset_name', type=str, default='MSVD',
                        help='Set The name of the dataset (default is MSVD).')
    parser.add_argument('-f','--features', nargs='+', 
                        help='<Required> Set the names of features to be extracted', required=True)
    parser.add_argument('-config', '--config_file', type=str, required=True,
                        help='<Required> Set the path to the config file with other configuration params')

    args = parser.parse_args()

    assert args.dataset_name in ['MSVD', 'M-VAD', 'MSR-VTT', 'TRECVID']
    for f_name in args.features:
      assert f_name in ['cnn_features', 'cnn_globals', 'cnn_sem_globals', 'c3d_features', 'c3d_globals', 'i3d_features', 'i3d_globals', 'eco_features', 'eco_globals', 'eco_sem_features', 'eco_sem_globals', 'tsm_sem_features', 'tsm_sem_globals', 'tsm_features', 'tsm_globals']
    
    config = ConfigurationFile(args.config_file, args.dataset_name)

    while True:
        try:
            main(args, config)
            print('Extraction of all features finished!!')
        except OSError:
            time.sleep(10)
            print('\ntrying again...')