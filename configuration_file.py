#!/usr/bin/env python
from configparser import ConfigParser

__author__ = 'jssprz'
__version__ = '0.0.1'
__email__ = 'jperezmartin90@gmail.com'
__maintainer__ = 'jssprz'
__status__ = 'Development'


class ConfigurationFile:
    def __init__(self, config_path, section_name):
        self.__config = ConfigParser()
        self.__config.read(config_path)

        try:
            section = self.__config[section_name]
        except Exception:
            raise ValueError(" {} is not a valid section".format(section_name))

        self.__dataset_name = section['dataset_name']
        self.__data_dir = section['data_dir']
        self.__features_dir = section['features_dir']
        self.__num_videos = int(section['num_videos']) if 'num_videos' in section else None 

        self.__mapping_path = section['mapping_path'] if 'mapping_path' in section else None
        self.__datainfo_path = section['datainfo_path'] if 'datainfo_path' in section else None
        self.__test_vid_list_json = section['test_vid_list_json'] if 'test_vid_list_json' in section else None

        self.__cnn_model = section['cnn_model']
        self.__cnn_pretrained_path = section['cnn_pretrained_path']
        self.__c3d_pretrained_path = section['c3d_pretrained_path']
        self.__i3d_pretrained_path = section['i3d_pretrained_path']
        self.__tsm_pretrained_path = section['tsm_pretrained_path']
        self.__frame_sample_rate = int(section['frame_sample_rate'])
        self.__clip_size = int(section['clip_size'])
        self.__max_frames = int(section['max_frames'])
        
    @property
    def dataset_name(self):
        return self.__dataset_name

    @property
    def data_dir(self):
        return self.__data_dir

    @property
    def features_dir(self):
        return self.__features_dir

    @property
    def num_videos(self):
        return self.__num_videos

    @property
    def mapping_path(self):
        return self.__mapping_path
    
    @property
    def datainfo_path(self):
        return self.__datainfo_path
    
    @property
    def test_vid_list_json(self):
        return self.__test_vid_list_json
    
    @property
    def cnn_model(self):
        return self.__cnn_model
    
    @property
    def cnn_pretrained_path(self):
        return self.__cnn_pretrained_path
    
    @property
    def c3d_pretrained_path(self):
        return self.__c3d_pretrained_path
    
    @property
    def i3d_pretrained_path(self):
        return self.__i3d_pretrained_path
    
    @property
    def tsm_pretrained_path(self):
        return self.__tsm_pretrained_path

    @property
    def frame_sample_rate(self):
        return self.__frame_sample_rate

    @property
    def clip_size(self):
        return self.__clip_size

    @property
    def max_frames(self):
        return self.__max_frames
