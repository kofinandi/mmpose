# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import defaultdict
from typing import List, Optional, Tuple

from mmengine.fileio import exists, get_local_path
from xtcocotools.coco import COCO

from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class PoseTrack21Dataset(BaseCocoStyleDataset):
    """PoseTrack21 dataset for multi-person 2D pose estimation and tracking.

    "PoseTrack21: A Dataset for Person Search, Multi-Object Tracking and
    Multi-Person Pose Tracking", CVPR`2022.
    More details: https://github.com/anDoer/PoseTrack21

    Expects the merged annotation file written by
    ``tools/dataset_converters/preprocess_posetrack21.py``, which adds the
    ``width``/``height``, ``area``, ``frame_id``, ``seq_name``,
    ``good_frame`` and globally unique ``track_id`` fields that the raw
    per-video release omits.

    Keypoints use the PoseTrack-17 ordering, which is identical to
    PoseTrack18 -- hence the shared ``METAINFO`` file::

        0: 'nose',
        1: 'head_bottom',
        2: 'head_top',
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
        16: 'right_ankle'

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If set,
            detected bboxes will be used instead of ground-truth bboxes.
            Default: ``None``.
        data_mode (str): ``'topdown'`` or ``'bottomup'``.
            Default: ``'topdown'``
        good_frame_mask (bool): If ``True``, load only frames whose
            ``good_frame`` annotation field is ``True``, i.e. the ~44% of
            PoseTrack frames that actually carry pose annotations (every 4th
            frame, plus a dense consecutive block per video). If ``False``,
            every frame with an image on disk is loaded -- the unlabeled ones
            carrying no GT instances -- so that consumers needing a
            temporally continuous sequence (e.g. the post-processing tracker)
            can see every frame. Default: ``True``.
        max_frames_per_sequence (int, optional): If set, keep at most this
            many frames per ``seq_name``, chosen by ascending ``frame_id``.
            Useful for lightweight benchmark splits.
        metainfo (dict, optional): Meta information for dataset.
            Default: ``None``.
        data_root (str, optional): Root directory for data. Default: ``None``.
        data_prefix (dict, optional): Prefix for data. Default:
            ``dict(img='')``.
        filter_cfg (dict, optional): Config for filtering. Default: ``None``.
        indices (int or Sequence[int], optional): Use only the first N samples.
            Default: ``None``.
        serialize_data (bool, optional): Whether to serialize data.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: ``[]``.
        test_mode (bool, optional): Whether in test phase. Default: ``False``.
        lazy_init (bool, optional): Whether to defer loading annotations.
            Default: ``False``.
        max_refetch (int, optional): Max cycles to fetch a valid sample.
            Default: ``1000``.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/posetrack18.py')

    def __init__(self,
                 good_frame_mask: bool = True,
                 max_frames_per_sequence: Optional[int] = None,
                 **kwargs) -> None:
        if (max_frames_per_sequence is not None
                and max_frames_per_sequence <= 0):
            raise ValueError(
                'PoseTrack21Dataset: max_frames_per_sequence must be a '
                f'positive integer, got {max_frames_per_sequence}.')
        self.good_frame_mask = good_frame_mask
        self.max_frames_per_sequence = max_frames_per_sequence
        super().__init__(**kwargs)

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load COCO annotations with an optional labeled-frame filter."""
        assert exists(self.ann_file), (
            f'Annotation file `{self.ann_file}`does not exist')

        with get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)
        if 'categories' in self.coco.dataset:
            self._metainfo['CLASSES'] = self.coco.loadCats(
                self.coco.getCatIds())

        candidate_images: List[dict] = []

        for img_id in self.coco.getImgIds():
            if img_id % self.sample_interval != 0:
                continue
            img = self.coco.loadImgs(img_id)[0]

            if self.good_frame_mask and not img.get('good_frame', True):
                continue

            img.update({
                'img_id': img_id,
                'img_path': osp.join(self.data_prefix['img'],
                                     img['file_name']),
            })
            candidate_images.append(img)

        if self.max_frames_per_sequence is not None:
            candidate_images = self._cap_frames_per_sequence(candidate_images)

        instance_list = []
        image_list = []
        for img in candidate_images:
            img_id = img['img_id']
            image_list.append(img)

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            for ann in self.coco.loadAnns(ann_ids):
                instance_info = self.parse_data_info(
                    dict(raw_ann_info=ann, raw_img_info=img))
                if not instance_info:
                    continue
                instance_list.append(instance_info)

        return instance_list, image_list

    def _cap_frames_per_sequence(self, images: List[dict]) -> List[dict]:
        """Keep the first ``max_frames_per_sequence`` frames per sequence."""
        by_sequence: dict = defaultdict(list)
        for img in images:
            by_sequence[img.get('seq_name', '')].append(img)

        capped: List[dict] = []
        for seq_name in sorted(by_sequence):
            seq_frames = sorted(
                by_sequence[seq_name], key=lambda img: img.get('frame_id', 0))
            capped.extend(seq_frames[:self.max_frames_per_sequence])
        return capped
