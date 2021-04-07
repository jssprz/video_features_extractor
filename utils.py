#!/usr/bin/env python
"""Defines some useful functions
"""

import os
import numpy as np
# import nltk

__author__ = "jssprz"
__version__ = "0.0.1"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argsort(memory_available)[::-1]


# def get_tags_feat(captions, tags_vocab):
#     result = np.zeros((len(tags_vocab),), dtype='int')
#     for caption in captions:
#         tags = [x for x in nltk.tokenize.word_tokenize(caption.lower()) if x in key_words]
#         for tag in tags:
#             result[tags_vocab(tag)] = 1
#     return result
