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

    assert os.path.exists(video_path), 'video path doesn\'t exist'

    try:
        cap = cv2.VideoCapture(video_path)
    except Exception as e:
        print('Can not open {} because {}'.format(video_path, e))
        pass
    else:
        frames = []
        while True:
            ret, frame = cap.read()
            # Convert the BGR image into an RGB image, because the later model uses the RGB format.
            if ret is not True:
                break
            frame = frame[:, :, ::-1]
            frames.append(frame)

        frames_per_side = frame_sample_rate // 2
        sample_step = frame_sample_rate - frame_sample_overlap
        max_index = len(frames) - frames_per_side

        # while max_frames*frame_sample_rate >= frame_count-frame_sample_rate:
        #   max_frames -= 1
        # indices = np.linspace(frame_sample_rate, frame_count - frames_per_side, max_frames, endpoint=False, dtype=int)

        indices = []
        i = frames_per_side
        while i <= max_index and len(indices) < max_frames:
            indices.append(i)
            i += sample_step

        frames = np.array(frames)
        frame_list = frames[indices]
        clip_list = [frames[i - frames_per_side: i + frames_per_side] for i in indices]
        clip_list = np.array(clip_list)
        return frame_list, clip_list, len(frames)
