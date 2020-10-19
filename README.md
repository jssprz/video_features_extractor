# Video Features Extractor

Extracting several kinds of visual representations from videos.


## Supported visual features

The following frame-level (`*_features`) and video-level (`*_gloabal`) visual representations are supported:
1. 2D-CNN (`cnn_features`, `cnn_globals`, `cnn_sem_globals`)
2. 3D-CNN (`c3d_features`, `c3d_globals`, `i3d_features`, `i3d_globals`)
3. ECO (`eco_features`, `eco_globals`, `eco_sem_features`, `eco_sem_globals`)
4. TSM (`tsm_features`, `tsm_globals`, `tsm_sem_features`, `tsm_sem_globals`)

Note: `*_sem_*` representations are based on the classification level (probability distribution) of respective models.


## Supported Datasets

This package has been tested for extracting visual representations from videos of the following video-caption datasets:
1. MSVD 
2. M-VAD
3. MSR-VTT
4. TRECVID-2020
5. TRECVID-2020-Test
6. TGIF
7. VATEX
8. ActivityNet
9. ActivityNet-Test
10. ActivityNet-Fragments
