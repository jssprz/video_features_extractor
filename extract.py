#!/usr/bin/env python
"""

"""

import os
import sys
import argparse
import h5py
from configparser import ConfigParser
import numpy as np
import torch
from torch.autograd import Variable

from .utils import get_freer_gpu
from .preprocess import resize_frame, preprocess_frame
from .sample_frames import sample_frames

sys.path.append('video-features-extractor')
from appearance_features import AppearanceEncoder
from motion_features import MotionEncoder

__author__ = "jssprz"
__version__ = "0.0.1"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"


def extract_features(aencoder, mencoder, dataset_name, params):
    """

    :param aencoder:
    :param mencoder:
    :param params:
    :return:
    """

    assert os.listdir(params['data_dir']) == params['num_videos'], \
        'All videos are not stored in {}'.format(params['data_dir'])

    # Read the video list and let the videos sort by id in ascending order
    if dataset_name == 'MSVD':
        videos = sorted(os.listdir(params['data_dir']))
        map_file = open(params['mapping_path'], 'w')
    elif dataset_name == 'MSR-VTT':
        videos = sorted(os.listdir(params['data_dir']), key=lambda x: int(x[5:-4]))
    elif dataset_name == 'M-VAD':
        videos = sorted(os.listdir(params['data_dir']), key=lambda x: int(x[3:-4]))

    feature_h5_path = os.path.join(params['data_dir'], 'features/features.h5')

    # Create an hdf5 file that saves video features
    if os.path.exists(feature_h5_path):
        # If the hdf5 file already exists, it has been processed before,
        # perhaps it has not been completely processed.
        # Read using r+ (read and write) mode to avoid overwriting previously saved data
        h5 = h5py.File(feature_h5_path, 'r+')
    else:
        h5 = h5py.File(feature_h5_path, 'w')

    if dataset_name in list(h5.keys()):
        dataset = h5[dataset_name]
        dataset_afeats = dataset['a_feats']
        dataset_mfeats = dataset['m_feats']
        #     dataset_cfeats = dataset['c_feats']
        dataset_counts = dataset['n_feats']
    else:
        dataset = h5.create_group(dataset_name)
        dataset_afeats = dataset.create_dataset('a_feats', (params['num_videos'], params['max_frames'],
                                                            aencoder.feature_size()), dtype='float32')
        dataset_mfeats = dataset.create_dataset('m_feats', (params['num_videos'], params['max_frames'],
                                                            mencoder.feature_size()), dtype='float32')
        # dataset_cfeats = dataset.create_dataset('c_feats', (params['num_videos'], params['max_frames'],
        #                                                     aencoder.feature_size() + mencoder.feature_size()),
        #                                         dtype='float32')
        dataset_counts = dataset.create_dataset('n_feats', (params['num_videos'],), dtype='int')

    for i, video in enumerate(videos):
        if dataset_name == 'MSVD':
            map_nameid = '%s\tvideo%d\n' % (video[:-4], i)
            map_file.write(map_nameid)

        video_path = os.path.join(params['data_dir'], video)

        # Extract video frames and video tiles
        frame_list, clip_list, frame_count = sample_frames(video_path, params['max_frames'],
                                                           params['frame_sample_rate'], params['frame_sample_overlap'])
        feats_count = len(frame_list)

        if i % 100 == 0 or frame_count == 0 or feats_count == 0:
            print('%d\t%s\t%d\t%d' % (i, video, frame_count, feats_count))

        # Do the image and then convert it into (batch, channel, height, width) format
        frame_list = np.array([preprocess_frame(x, frame_shape[1], frame_shape[2]) for x in frame_list])
        frame_list = frame_list.transpose((0, 3, 1, 2))
        frame_list = Variable(torch.from_numpy(frame_list), volatile=True).to(device)

        # If the number of frames is less than max_frames, then the remaining part is complemented by 0
        afeats = np.zeros((params['max_frames'], aencoder.feature_size()), dtype='float32')
        mfeats = np.zeros((params['max_frames'], mencoder.feature_size()), dtype='float32')
        #     cfeats = np.zeros((max_frames, c_feature_size), dtype='float32')

        # Extracting apparience features of sampled frames first
        af = aencoder(frame_list)

        # Preprocess frames of the video fragments to extract motion features
        clip_list = np.array([[resize_frame(x, 112, 112)
                               for x in clip] for clip in clip_list])
        clip_list = clip_list.transpose(0, 4, 1, 2, 3).astype(np.float32)
        clip_list = Variable(torch.from_numpy(clip_list), volatile=True).to(device)

        # Extracting motion features
        mf = mencoder(clip_list)

        afeats[:len(frame_list), :] = af.data.cpu().numpy()
        mfeats[:len(frame_list), :] = mf.data.cpu().numpy()
        #     cfeats[:len(frame_list), :] = torch.cat([af, mf], dim=1).data.cpu().numpy()

        dataset_afeats[i] = afeats
        dataset_mfeats[i] = mfeats
        #     dataset_cfeats[i] = cfeats
        dataset_counts[i] = feats_count

    h5.close()

    if dataset_name == 'MSVD':
        map_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a captioning model for a specific dataset.')
    parser.add_argument('-dataset_name', '--ds', type=str, default='MSVD',
                        help='the name of the dataset (default is MSVD).')
    parser.add_argument('-config_file', '--config', type=str, required=True,
                        help='the path to the config file with all params')

    args = parser.parse_args()

    assert args.ds in ['MSVD', 'M-VAD', 'MSR-VTT']

    config = ConfigParser()
    config.read(args.config)
    ds_params = config[args.ds]

    if torch.cuda.is_available():
        freer_gpu_id = get_freer_gpu()
        device = torch.device('cuda:{}'.format(freer_gpu_id))
        torch.cuda.empty_cache()
        print('Running on cuda:{} device'.format(freer_gpu_id))
    else:
        device = torch.device('cpu')
        print('Running on cpu device')

    aencoder = AppearanceEncoder(ds_params['appearance_model'], ds_params['appearance_pretrained_path'])
    aencoder.eval()
    aencoder.to(device)

    mencoder = MotionEncoder(ds_params['motion_model'], ds_params['motion_pretrained_path'])
    mencoder.eval()
    mencoder.to(device)

    extract_features(aencoder, mencoder, args.ds, ds_params)
