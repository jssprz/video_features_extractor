#!/usr/bin/env python
"""

"""

import os
import argparse
import h5py
import numpy as np
import torch

from utils import get_freer_gpu
from preprocess import resize_frame, preprocess_frame
from sample_frames import sample_frames
from feature_extractors.cnn import AppearanceEncoder
from feature_extractors.feats_extractor import MotionEncoder
from feature_extractors.i3dpt import I3D
from configuration_file import ConfigurationFile

__author__ = "jssprz"
__version__ = "0.0.1"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"


def extract_features(cnn_extractor, c3d_extractor, i3d_extractor, dataset_name, frame_shape, config, device):
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

    # Read the video list and let the videos sort by id in ascending order
    if dataset_name == 'MSVD':
        videos = [os.path.join(config.data_dir, v) for v in sorted(os.listdir(config.data_dir))]
        map_file = open(config.mapping_path, 'w')
    elif dataset_name == 'MSR-VTT':
        videos = [os.path.join(config.data_dir, v) for v in sorted(os.listdir(config.data_dir), key=lambda x: int(x[5:-4]))]
    elif dataset_name == 'M-VAD':
        videos = [os.path.join(config.data_dir, v) for v in sorted(os.listdir(config.data_dir), key=lambda x: int(x[3:-4]))]
    else:
        with open(os.path.join(config.data_dir, 'list.txt')) as f:
            videos = [os.path.join(config.data_dir, path) for path in f.readlines()]

    features_dir = os.path.join(config.data_dir, 'features')
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # Create an hdf5 file that saves video features
    feature_h5_path = os.path.join(features_dir, 'features.h5')
    if os.path.exists(feature_h5_path):
        # If the hdf5 file already exists, it has been processed before,
        # perhaps it has not been completely processed.
        # Read using r+ (read and write) mode to avoid overwriting previously saved data
        h5 = h5py.File(feature_h5_path, 'r+')
    else:
        h5 = h5py.File(feature_h5_path, 'w')

    if dataset_name in list(h5.keys()):
        dataset = h5[dataset_name]
        dataset_cnn = dataset['cnn_feats']
        dataset_c3d = dataset['c3d_features']
        dataset_i3d = dataset['i3d_features']
        dataset_counts = dataset['i3d_feats']
    else:
        dataset = h5.create_group(dataset_name)
        dataset_cnn = dataset.create_dataset('cnn_feats', (config.num_videos, config.max_frames,
                                                           cnn_extractor.feature_size()), dtype='float32')
        dataset_c3d = dataset.create_dataset('c3d_features', (config.num_videos, config.max_frames,
                                                              c3d_extractor.feature_size()), dtype='float32')
        dataset_i3d = dataset.create_dataset('i3d_feats', (config.num_videos, config.max_frames,
                                                           i3d_extractor.feature_size()), dtype='float32')
        dataset_counts = dataset.create_dataset('i3d_feats', (config.num_videos,), dtype='int')

    for i, video_path in enumerate(videos):
        if dataset_name == 'MSVD':
            map_name_id = '{}\tvideo{}\n'.format(video_path.split('/')[-1][:-4], i)
            map_file.write(map_name_id)

        # Extract video frames and video tiles
        frame_list, clip_list, frame_count = sample_frames(video_path, config.max_frames, config.frame_sample_rate,
                                                           config.frame_sample_overlap)
        feats_count = len(frame_list)

        if i % 100 == 0 or frame_count == 0 or feats_count == 0:
            print('%d\t%s\t%d\t%d' % (i, video_path.split('/')[-1], frame_count, feats_count))

        # Preprocess frames and then convert it into (batch, channel, height, width) format
        frame_list = np.array([preprocess_frame(x, frame_shape[1], frame_shape[2]) for x in frame_list])
        frame_list = frame_list.transpose((0, 3, 1, 2))
        frame_list = torch.from_numpy(frame_list).to(device)

        # If the number of frames is less than max_frames, then the remaining part is complemented by 0
        cnn_features = np.zeros((config.max_frames, cnn_extractor.feature_size()), dtype='float32')
        c3d_features = np.zeros((config.max_frames, c3d_extractor.feature_size()), dtype='float32')
        i3d_features = np.zeros((config.max_frames, i3d_extractor.feature_size()), dtype='float32')

        # Extracting cnn features of sampled frames first
        cnn = cnn_extractor(frame_list)

        # Extracting i3d features of sampled frames first
        i3d = i3d_extractor(frame_list)[1]

        # Preprocess frames of the video fragments to extract motion features
        clip_list = np.array([[resize_frame(x, 112, 112) for x in clip] for clip in clip_list])
        clip_list = clip_list.transpose((0, 4, 1, 2, 3)).astype(np.float32)
        clip_list = torch.from_numpy(clip_list).to(device)

        # Extracting c3d features
        c3d = c3d_extractor(clip_list)

        cnn_features[:len(frame_list), :] = cnn.data.cpu().numpy()
        c3d_features[:len(frame_list), :] = c3d.data.cpu().numpy()
        i3d_features[:len(frame_list), :] = i3d.data.cpu().numpy()

        dataset_cnn[i] = cnn_features
        dataset_c3d[i] = c3d_features
        dataset_i3d[i] = i3d_features
        dataset_counts[i] = feats_count

    h5.close()

    if dataset_name == 'MSVD':
        map_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a captioning model for a specific dataset.')
    parser.add_argument('-ds', '--dataset_name', type=str, default='MSVD',
                        help='the name of the dataset (default is MSVD).')
    parser.add_argument('-config', '--config_file', type=str, required=True,
                        help='the path to the config file with all params')

    args = parser.parse_args()

    assert args.dataset_name in ['MSVD', 'M-VAD', 'MSR-VTT', 'TRECVID']

    config = ConfigurationFile(args.config_file, args.dataset_name)

    if torch.cuda.is_available():
        freer_gpu_id = get_freer_gpu()
        device = torch.device('cuda:{}'.format(freer_gpu_id))
        torch.cuda.empty_cache()
        print('Running on cuda:{} device'.format(freer_gpu_id))
    else:
        device = torch.device('cpu')
        print('Running on cpu device')

    cnn_extractor = AppearanceEncoder(config.cnn_model, config.cnn_pretrained_path)
    c3d_extractor = MotionEncoder('c3d', config.c3d_pretrained_path)
    i3d_rgb_extractor = I3D(modality='rgb')

    cnn_extractor.eval()
    c3d_extractor.eval()
    i3d_rgb_extractor.eval()

    i3d_rgb_extractor.load_state_dict(torch.load(config.i3d_pretrained_path))

    cnn_extractor.to(device)
    c3d_extractor.to(device)
    i3d_rgb_extractor.to(device)

    frame_shape = (config.frame_shape_channels, config.frame_shape_height, config.frame_shape_width)

    extract_features(cnn_extractor, c3d_extractor, i3d_rgb_extractor, args.ds, frame_shape, config, device)
