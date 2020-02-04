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
from sample_frames import sample_frames
from feature_extractors.cnn import CNN
from feature_extractors.c3d import C3D
from feature_extractors.i3dpt import I3D
from feature_extractors.tsm import TSN
from configuration_file import ConfigurationFile

__author__ = "jssprz"
__version__ = "0.0.1"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"


def extract_features(extractor_name, extractor, dataset_name, device, frame_shape, config):
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
    feature_h5_path = os.path.join(config.features_dir, 'features_{}space_{}.h5'.format('lin' if config.frame_sample_rate == -1 else config.frame_sample_rate, config.max_frames))
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
                                                              extractor.feature_size), dtype='float32')
        dataset_counts = dataset['count_features']
    else:
        dataset = h5.create_group(dataset_name)
        dataset_model = dataset.create_dataset(extractor_name, (config.num_videos, config.max_frames,
                                                              extractor.feature_size), dtype='float32')
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

        if i % 50 == 0 or frame_count == 0 or feats_count == 0:
            print('%d\t%s\t%d\t%d' % (i, video_path.split('/')[-1], frame_count, feats_count))

        # If the number of frames is less than max_frames, then the remaining part is complemented by 0
        dataset_model[i] = np.zeros((config.max_frames, extractor.feature_size), dtype='float32')

        if extractor_name == 'cnn_features': 
            # Preprocess frames and then convert it into (batch, channel, height, width) format
            frame_list = np.array([preprocess_frame(x, frame_shape[1], frame_shape[2]) for x in frame_list])
            frame_list = frame_list.transpose((0, 3, 1, 2))
            frame_list = Variable(torch.from_numpy(frame_list), volatile=True).to(device)

            # Extracting cnn features of sampled frames first
            features = extractor(frame_list)
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
        elif extractor_name == 'tsm_features':
            # Preprocess frames of the video fragments to extract motion features
            clip_list = 
            clip_list = Variable(torch.from_numpy(clip_list), volatile=True).to(device)
            
            # Extracting i3d features of sampled frames first
            features = extractor(clip_list)

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
        freer_gpu_ids = get_freer_gpu()
        device = torch.device('cuda:{}'.format(freer_gpu_ids[0]))
        print('Running on cuda: {} devices sort'.format(freer_gpu_ids))
    else:
        device = torch.device('cpu')
        print('Running on cpu device')
    
    frame_shape = (config.frame_shape_channels, config.frame_shape_height, config.frame_shape_width)
    
    print('\nExtracting CNN for {} dataset'.format(args.dataset_name))

    cnn_use_torch_weights = (config.cnn_pretrained_path == '')
    model = CNN(config.cnn_model, cnn_use_torch_weights)
    if not cnn_use_torch_weights:
        model.load_pretrained(config.cnn_pretrained_path)
    with torch.no_grad():
        extract_features('cnn_features', model, args.dataset_name, device, frame_shape, config)

    print('\nExtracting C3D for {} dataset'.format(args.dataset_name))
    
    model = C3D()
    model.load_pretrained(config.c3d_pretrained_path)
    with torch.no_grad():
        extract_features('c3d_features', model, args.dataset_name, device, frame_shape, config)

    print('\nExtracting I3D for {} dataset'.format(args.dataset_name))
        
    model = I3D(modality='rgb')
    model.load_state_dict(torch.load(config.i3d_pretrained_path))
    with torch.no_grad():
        extract_features('i3d_features', model, args.dataset_name, device, frame_shape, config)
        
    print('\nExtracting TSM for {} dataset'.format(args.dataset_name))
    
    model = TSN()
    model.load_state_dict(torch.load(config.tsm_pretrained_path))
    with torch.no_grad():
        extract_features('tsm_features', model, args.dataset_name, device, frame_shape, config)