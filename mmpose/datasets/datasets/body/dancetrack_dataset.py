# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
from .posetrack21_dataset import PoseTrack21Dataset


@DATASETS.register_module()
class DanceTrackPoseDataset(PoseTrack21Dataset):
    """DanceTrack with PoseTrack21-style pose annotations (19-keypoint union).

    Expects the merged JSON written by
    ``tools/dataset_converters/dancetrack_pseudo_pose.py``.  Image / track
    fields match PoseTrack21 (``seq_name``, ``frame_id``, ``good_frame``,
    globally unique ``track_id``).  Keypoints are COCO-17 followed by
    synthesized ``head_bottom`` and ``head_top``::

        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle',
        17: 'head_bottom',
        18: 'head_top'
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/dancetrack.py')
