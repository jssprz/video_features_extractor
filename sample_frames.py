#!/usr/bin/env python
"""Defines the sample_frames function
This function select sequence of frames from a video
"""

import os
import cv2
import numpy as np


def sample_frames(video_path, max_frames, frame_sample_rate, frame_sample_overlap):
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
#             if i % 2 == 0:
            frames.append(frame)
            i += 1

        frames = np.array(frames)
        
        frames_per_side = frame_sample_rate // 2
        if frame_sample_rate == -1:
            indices = np.linspace(frames_per_side, len(frames) - frames_per_side + 1, max_frames, endpoint=False, dtype=int)
            clip_list = [frames[i - frames_per_side: i + frames_per_side] for i in indices]
        else:
            frames_per_side = frame_sample_rate // 2
            sample_step = frame_sample_rate - frame_sample_overlap
            max_index = len(frames) - frames_per_side
            
            indices = []
            i = frames_per_side
            while i <= max_index and len(indices) < max_frames:
                indices.append(i)
                i += sample_step
                
            clip_list = [frames[i - frames_per_side: i + frames_per_side] for i in indices]
        
        clip_list = np.array(clip_list)
        frame_list = frames[indices]
        return frame_list, clip_list, len(frames)
    
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