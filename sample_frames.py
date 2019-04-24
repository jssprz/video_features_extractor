#!/usr/bin/env python
"""Defines the sample_frames function
This function select sequence of frames from a video
"""

import cv2
import numpy as np


def sample_frames(video_path, max_frames, frame_sample_rate, frame_sample_overlap, train=True):
    """Samples video frames reduces computational effort. Taking max_frames frames at equal intervals
    """

    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        # Convert the BGR image into an RGB image, because the later model uses the RGB format.
        if ret is False:
            break
        frame = frame[:, :, ::-1]
        frames.append(frame)
        frame_count += 1

    frames_per_side = frame_sample_rate // 2
    sample_step, max_index = frame_sample_rate - frame_sample_overlap, frame_count - frames_per_side

    indices = []
    i = frames_per_side
    while i <= max_index and len(indices) < max_frames:
        indices.append(i)
        i += sample_step

    # while max_frames*frame_sample_rate >= frame_count-frame_sample_rate:
    #   max_frames -= 1
    # indices = np.linspace(frame_sample_rate, frame_count - frames_per_side, max_frames, endpoint=False, dtype=int)

    frames = np.array(frames)
    frame_list = frames[indices]
    clip_list = [frames[i - frames_per_side: i + frames_per_side] for i in indices]
    clip_list = np.array(clip_list)
    return frame_list, clip_list, frame_count
