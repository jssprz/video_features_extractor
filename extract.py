#!/usr/bin/env python
"""

"""

import os
import argparse
import h5py
import numpy as np
import torch
from torch.autograd import Variable

from utils import get_freer_gpu
from preprocess import resize_frame, preprocess_frame
from sample_frames import sample_frames, sample_frames2
from feature_extractors.cnn import CNN
from feature_extractors.c3d import C3D
from feature_extractors.i3dpt import I3D
from feature_extractors.eco import init_model
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

def extract_features(extractor_name, extractor, dataset_name, device, config, feature_size, crop_size, scale_size, input_mean, input_std):
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
    feature_h5_path = os.path.join(config.features_dir, 'temp_features_{}space_{}.h5'.format('lin' if config.frame_sample_rate == -1 else config.frame_sample_rate, config.max_frames))
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
        else:
            dataset_model = dataset.create_dataset(extractor_name, (config.num_videos, config.max_frames,
                                                                    feature_size), dtype='float32')
        dataset_counts = dataset['count_features']
    else:
        dataset = h5.create_group(dataset_name)
        dataset_model = dataset.create_dataset(extractor_name, (config.num_videos, config.max_frames,
                                                                feature_size), dtype='float32')
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
        feats_count = len(frame_list)
#         sampled_frames, frame_count = sample_frames2(video_path, num_segments=16, segment_length=1)

        if i % 50 == 0 or frame_count == 0 or feats_count == 0:
            print('%d\t%s\t%d' % (i, video_path.split('/')[-1], frame_count))

        # If the number of frames is less than max_frames, then the remaining part is complemented by 0
#         dataset_model[i] = np.zeros((config.max_frames, extractor.feature_size), dtype='float32')
        dataset_model[i] = np.zeros((config.max_frames, feature_size), dtype='float32')

        if extractor_name == 'cnn_features': 
            # Preprocess frames and then convert it into (batch, channel, height, width) format
            frame_list = np.array([preprocess_frame(x, scale_size=scale_size, crop_size=crop_size,
                                                    mean=input_mean, std=input_std) 
                                   for x in frame_list])
            frame_list = torch.from_numpy(frame_list.transpose((0, 3, 1, 2))).to(device)

            # Extracting cnn features of sampled frames first
            features = extractor(frame_list)
            print(features.size(), features.mean())
        elif extractor_name == 'c3d_features':
            # Preprocess frames of the video fragments to extract motion features
            clip_list = np.array([[resize_frame(x, 112, 112) for x in clip] for clip in clip_list])
            clip_list = clip_list.transpose((0, 4, 1, 2, 3)).astype(np.float32)
            clip_list = Variable(torch.from_numpy(clip_list), volatile=True).to(device)

            # Extracting c3d features
            features = extractor(clip_list)
        elif extractor_name == 'i3d_features':
            # Preprocess frames of the video fragments to extract motion features
            clip_list = np.array([[resize_frame(x, 196, 196) for x in clip] for clip in clip_list])
            clip_list = clip_list.transpose((0, 4, 1, 2, 3)).astype(np.float32)
            clip_list = Variable(torch.from_numpy(clip_list), volatile=True).to(device)

            # Extracting i3d features of sampled frames first
            features = extractor(clip_list)
        elif extractor_name == 'eco_features':
            # Preprocess frames of the video fragments to extract motion features
# #           clip_list = np.array([[resize_frame(x, 112, 112) for x in clip] for clip in clip_list])
#             clip_list = np.array([[resize_frame(x, 224, 224) for x in clip] for clip in clip_list])
#             clip_list = clip_list.transpose((0, 4, 1, 2, 3)).astype(np.float32)
#             clip_list = Variable(torch.from_numpy(clip_list)).to(device)
            
#             # Extracting i3d features of sampled frames first
#             features = extractor(clip_list)
#             print(features.size(), features.mean())
#             print(features)
    
#             overflow = len(all_frames)%16
#             if overflow:
#                 all_frames = all_frames[:-overflow]
#             all_frames = np.array([resize_frame(x, 112, 112) for x in all_frames]).astype(np.float32).transpose((0,3,1,2))
            features = []
            for clip in clip_list:
                clip_frames = torch.cat([torch.from_numpy(preprocess_frame(x, scale_size=scale_size, crop_size=crop_size,
                                                                          mean=input_mean, std=input_std))
                                         for x in clip], dim=2).transpose(0,2).unsqueeze(0).to(device)
                features.append(extractor(clip_frames))
            features = torch.cat(features, dim=0)
#             features = l2norm(features)            
            print(features.size(), features.mean())

        feats_count = features.size(0)
#         print(feats_count)
        dataset_model[i, :feats_count, :] = features.data.cpu().numpy()
        dataset_counts[i] = feats_count

    h5.close()

    if dataset_name == 'MSVD':
        map_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a captioning model for a specific dataset.')
    parser.add_argument('-ds', '--dataset_name', type=str, default='MSR-VTT',
                        help='the name of the dataset (default is MSVD).')
    parser.add_argument('-config', '--config_file', type=str, required=True,
                        help='the path to the config file with all params')

    args = parser.parse_args()

    assert args.dataset_name in ['MSVD', 'M-VAD', 'MSR-VTT', 'TRECVID']
    
    config = ConfigurationFile(args.config_file, args.dataset_name)
    
    if torch.cuda.is_available():
        freer_gpu_ids = get_freer_gpu()[0]
        device = torch.device('cuda:{}'.format(freer_gpu_ids))
        print('Running on freer device: cuda:{}'.format(freer_gpu_ids))
    else:
        device = torch.device('cpu')
        print('Running on cpu device')
    
#     frame_shape = (config.frame_shape_channels, config.frame_shape_height, config.frame_shape_width)
    
    print('\nExtracting CNN for {} dataset'.format(args.dataset_name))

    cnn_use_torch_weights = (config.cnn_pretrained_path == '')
    model = CNN(config.cnn_model, input_size=224, use_pretrained=cnn_use_torch_weights)
    if not cnn_use_torch_weights:
        model.load_pretrained(config.cnn_pretrained_path)
    with torch.no_grad():
        extract_features('cnn_features', model, args.dataset_name, device, config, feature_size=model.feature_size,
                         crop_size=model.crop_size, scale_size=model.scale_size, input_mean=model.input_mean, input_std=model.input_std)

#     print('\nExtracting C3D for {} dataset'.format(args.dataset_name))
    
#     model = C3D()
#     model.load_pretrained(config.c3d_pretrained_path)
#     with torch.no_grad():
#         extract_features('c3d_features', model, args.dataset_name, device, frame_shape, config)

#     print('\nExtracting I3D for {} dataset'.format(args.dataset_name))
        
#     model = I3D(modality='rgb')
#     model.load_state_dict(torch.load(config.i3d_pretrained_path))
#     with torch.no_grad():
#         extract_features('i3d_features', model, args.dataset_name, device, frame_shape, config)
        
    print('\nExtracting TSM for {} dataset'.format(args.dataset_name))
    
    model, crop_size, scale_size, input_mean, input_std = init_model(num_class=400, num_segments=16, 
                                                                     pretrained_parts='finetune', #'2D', '3D',
                                                                     modality='RGB', 
                                                                     arch='ECOfull', #'ECO', 'C3DRes18',
                                                                     consensus_type='identity', #'avg',
                                                                     dropout=0, 
                                                                     no_partialbn=True, #False, 
                                                                     resume_chpt='',
                                                                     gpus=[device])
    with torch.no_grad():
        extract_features('eco_features', model, args.dataset_name, device, config, 1536, crop_size, scale_size, input_mean, input_std)