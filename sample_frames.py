#!/usr/bin/env python
"""Defines the sample_frames function
This function select sequence of frames from a video
"""

import os
# import cv2
import imageio
import numpy as np
from PIL import Image


def sample_frames(sample_type, video_path, max_frames, frame_sample_rate, chunk_size, segment_secs=None, all_fragments=None):
    """Samples video frames reduces computational effort. Taking max_frames frames at equal intervals
    """

    assert os.path.exists(video_path), 'video path {} doesn\'t exist'.format(video_path)

    try:
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
        # cap = cv2.VideoCapture(video_path)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        
        if segment_secs is not None:
            start, end = tuple(segment_secs)
            start_index = start * fps  # int(max(start - .5, 0) * fps)
            stop_index = end * fps  # int(max(end - .5, 0) * fps)
            
            # cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
            reader.set_image_index(start_index)
        else:
            start_index = 0
            # stop_index = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stop_index = int(reader.get_meta_data()['duration'] * fps)
    except Exception as e:
        print('Cannot process {} because {}'.format(video_path, e))
        pass
    else:
        frames, frames_ts, fragments_mask = [], [], []
        index = start_index
        while index < stop_index:
            # ret, frame = cap.read()
            frame = reader.get_next_data()
            # if ret is not True:
            #     break
    
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if all_fragments is not None:
                mark = 0
                for f in all_fragments:
                    start, end = tuple(f)
                    if timestamp >= start and timestamp <= end:
                        mark = 1
                fragments_mask.append(mark)
                
            # Convert the BGR image into an RGB image, because the later model uses the RGB format.
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame = Image.fromarray(frame)
            frames.append(frame)
            frames_ts.append(timestamp)
            index += 1

        if not len(frames):
            print('video-clip without frames in segment: {}, fragments: {}'.format(segment_secs, all_fragments))
            return [], [], [], 0, []
        
        #assert len(frames), 'video-clip without frames in segment: {}, fragments: {}'.format(segment_secs, all_fragments)
        # frames = np.array(frames)

        # print(fragments_mask, all_fragments)
        
        if sample_type == 'dynamic':
            frames_per_side = frame_sample_rate // 2
            if frame_sample_rate * max_frames < len(frames):
                indices = np.linspace(frames_per_side, len(frames) - frames_per_side + 1, max_frames, endpoint=False, dtype=int)
            elif frame_sample_rate < len(frames):
                indices = list(range(frames_per_side, len(frames) - frames_per_side, frame_sample_rate))
            else:
                indices = [len(frames)//2]
        elif sample_type == 'fixed':
            # sample max_frames frames always
            indices = np.linspace(0, len(frames), max_frames, endpoint=False, dtype=int)
        
        # chunk_list = [frames[max(i - chunk_size//2, 0): i + chunk_size//2, len(frames))] for i in indices]
        
        chunk_list = []
        for i in indices:
            ss = max(i - chunk_size//2, 0)
            to = min(i + chunk_size//2, len(frames))
            
            if ss == 0:
                to = min(chunk_size, len(frames))
            elif to == len(frames):
                ss = max(len(frames) - chunk_size, 0)                 
            
            chunk_list.append(frames[ss:to])

        frame_list = [frames[i] for i in indices]
        frame_ts_list = [frames_ts[i] for i in indices]
        if all_fragments is not None:
            fragments_mask = [fragments_mask[i] for i in indices]
        
        # if frame_sample_rate == -1:
        #     indices = np.linspace(frames_per_side, len(frames) - frames_per_side + 1, max_frames, endpoint=False, dtype=int)
        #     chunk_list = [frames[i - frames_per_side: i + frames_per_side] for i in indices]
        # else:
        #     frames_per_side = frame_sample_rate // 2
        #     sample_step = frame_sample_rate - frame_sample_overlap
        #     max_index = len(frames) - frames_per_side
            
        #     indices = []
        #     i = frames_per_side
        #     while i <= max_index and len(indices) < max_frames:
        #         indices.append(i)
        #         i += sample_step
                
        #     chunk_list = [frames[i - frames_per_side: i + frames_per_side] for i in indices]
        
        # chunk_list = np.array(chunk_list)
        # frame_list = frames[indices]
        # frame_list = [frames[i] for i in indices]
        return frame_list, frame_ts_list, chunk_list, len(frames), fragments_mask


# def sample_frames(sample_type, video_path, max_frames, frame_sample_rate, chunk_size, segment_secs=None, all_fragments=None):
#     """Samples video frames reduces computational effort. Taking max_frames frames at equal intervals
#     """

#     assert os.path.exists(video_path), 'video path {} doesn\'t exist'.format(video_path)

#     try:
#         cap = cv2.VideoCapture(video_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
        
#         if segment_secs is not None:
#             start, end = tuple(segment_secs)
#             start_index = start * fps  # int(max(start - .5, 0) * fps)
#             stop_index = end * fps  # int(max(end - .5, 0) * fps)
#             cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
#         else:
#             start_index = 0
#             stop_index = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_count = stop_index - start_index
#     except Exception as e:
#         print('Cannot process {} because {}'.format(video_path, e))
#         pass
#     else:
#         frames, frames_ts, fragments_mask = [], [], []
#         index, calc_timestamp = start_index, 0
#         while index < stop_index:
#             ret, frame = cap.read()
#             if ret is not True:
#                 break
    
#             timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
#             if all_fragments is not None:
#                 mark = 0
#                 for f in all_fragments:
#                     start, end = tuple(f)
#                     if timestamp >= start and timestamp <= end:
#                         mark = 1
#                 fragments_mask.append(mark)
                
#             # Convert the BGR image into an RGB image, because the later model uses the RGB format.
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             frame = Image.fromarray(frame)
#             frames.append(frame)
#             frames_ts.append(timestamp)
#             index += 1

#         if not len(frames):
#             print('video-clip without frames in segment: {}, fragments: {}'.format(segment_secs, all_fragments))
#             return [], [], [], 0, []
        
#         #assert len(frames), 'video-clip without frames in segment: {}, fragments: {}'.format(segment_secs, all_fragments)
#         # frames = np.array(frames)

#         # print(fragments_mask, all_fragments)
        
#         if sample_type == 'dynamic':
#             frames_per_side = frame_sample_rate // 2
#             if frame_sample_rate * max_frames < len(frames):
#                 indices = np.linspace(frames_per_side, len(frames) - frames_per_side + 1, max_frames, endpoint=False, dtype=int)
#             elif frame_sample_rate < len(frames):
#                 indices = list(range(frames_per_side, len(frames) - frames_per_side, frame_sample_rate))
#             else:
#                 indices = [len(frames)//2]
#         elif sample_type == 'fixed':
#             # sample max_frames frames always
#             indices = np.linspace(0, len(frames), max_frames, endpoint=False, dtype=int)
        
#         # chunk_list = [frames[max(i - chunk_size//2, 0): i + chunk_size//2, len(frames))] for i in indices]
        
#         chunk_list = []
#         for i in indices:
#             ss = max(i - chunk_size//2, 0)
#             to = min(i + chunk_size//2, len(frames))
            
#             if ss == 0:
#                 to = min(chunk_size, len(frames))
#             elif to == len(frames):
#                 ss = max(len(frames) - chunk_size, 0)                 
            
#             chunk_list.append(frames[ss:to])

#         frame_list = [frames[i] for i in indices]
#         frame_ts_list = [frames_ts[i] for i in indices]
#         if all_fragments is not None:
#             fragments_mask = [fragments_mask[i] for i in indices]
        
#         # if frame_sample_rate == -1:
#         #     indices = np.linspace(frames_per_side, len(frames) - frames_per_side + 1, max_frames, endpoint=False, dtype=int)
#         #     chunk_list = [frames[i - frames_per_side: i + frames_per_side] for i in indices]
#         # else:
#         #     frames_per_side = frame_sample_rate // 2
#         #     sample_step = frame_sample_rate - frame_sample_overlap
#         #     max_index = len(frames) - frames_per_side
            
#         #     indices = []
#         #     i = frames_per_side
#         #     while i <= max_index and len(indices) < max_frames:
#         #         indices.append(i)
#         #         i += sample_step
                
#         #     chunk_list = [frames[i - frames_per_side: i + frames_per_side] for i in indices]
        
#         # chunk_list = np.array(chunk_list)
#         # frame_list = frames[indices]
#         # frame_list = [frames[i] for i in indices]
#         return frame_list, frame_ts_list, chunk_list, len(frames), fragments_mask


def sample_frames2(video_path, num_segments, segment_length):
    """Samples video frames reduces computational effort. Taking max_frames frames at equal intervals
    """

    assert os.path.exists(video_path), 'video path {} doesn\'t exist'.format(video_path)

    try:
        cap = cv2.VideoCapture(video_path)
    except Exception as e:
        print('Can not open {} because {}'.format(video_path, e))
        pass
    else:
        frames = []
        i = 0
        while True:
            ret, frame = cap.read()
            if ret is not True:
                break
            # Convert the BGR image into an RGB image, because the later model uses the RGB format.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        frames = np.array(frames)
        
        num_frames = len(frames)
        if num_frames > num_segments + segment_length - 1:
            tick = (num_frames - segment_length + 1) / float(num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        else:
            offsets = np.zeros((num_segments,))
        offsets+=1
        
        images = []
        for seg_ind in offsets:
            p = int(seg_ind)
            for i in range(segment_length):
                images.append(frames[p])
                if p < num_frames:
                    p += 1
                    
        return images, num_frames