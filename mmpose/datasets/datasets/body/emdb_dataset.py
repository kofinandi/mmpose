# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import defaultdict
from typing import List, Optional, Tuple

from mmengine.fileio import exists, get_local_path
from xtcocotools.coco import COCO

from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class EmdbDataset(BaseCocoStyleDataset):
    """EMDB dataset for 2D human pose estimation.

    EMDB: The Electromagnetic Database of Global 3D Human Pose and Shape
    in the Wild (ICCV 2023).
    More details: https://ait.ethz.ch/emdb

    EMDB keypoints use COCO-17 ordering (converted from SMPL-24 in the
    preprocessor). See ``tools/dataset_converters/preprocess_emdb.py``.

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path.
        data_mode (str): ``'topdown'`` or ``'bottomup'``. Default: ``'topdown'``
        emdb1 (bool): If ``True``, load only EMDB 1 benchmark sequences.
            Default: ``False``.
        emdb2 (bool): If ``True``, load only EMDB 2 benchmark sequences.
            Default: ``False``. Cannot be ``True`` together with ``emdb1``.
        good_frame_mask (bool): If ``True``, load only frames marked valid
            in ``good_frames_mask``. Default: ``True``.
        max_frames_per_sequence (int, optional): If set, keep at most this
            many frames per ``seq_name``, chosen by ascending ``frame_id``.
            Useful for lightweight benchmark splits (e.g. ``emdb-mini``).
        metainfo (dict, optional): Meta information for dataset.
        data_root (str, optional): Root directory for data.
        data_prefix (dict, optional): Prefix for data.
        filter_cfg (dict, optional): Config for filtering.
        indices (int or Sequence[int], optional): Use only first N samples.
        serialize_data (bool, optional): Whether to serialize data.
        pipeline (list, optional): Processing pipeline.
        test_mode (bool, optional): Whether in test phase.
        lazy_init (bool, optional): Whether to defer loading annotations.
        max_refetch (int, optional): Max cycles to get a valid image.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/emdb.py')

    def __init__(self,
                 emdb1: bool = False,
                 emdb2: bool = False,
                 good_frame_mask: bool = True,
                 max_frames_per_sequence: Optional[int] = None,
                 **kwargs) -> None:
        if emdb1 and emdb2:
            raise ValueError(
                'EmdbDataset: emdb1 and emdb2 cannot both be True. '
                'Select at most one benchmark split.')
        if (max_frames_per_sequence is not None
                and max_frames_per_sequence <= 0):
            raise ValueError(
                'EmdbDataset: max_frames_per_sequence must be a positive '
                f'integer, got {max_frames_per_sequence}.')
        self.emdb1 = emdb1
        self.emdb2 = emdb2
        self.good_frame_mask = good_frame_mask
        self.max_frames_per_sequence = max_frames_per_sequence
        super().__init__(**kwargs)

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load COCO annotations with EMDB split / frame filters."""
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

            if self.emdb1 and not img.get('emdb1', False):
                continue
            if self.emdb2 and not img.get('emdb2', False):
                continue
            if self.good_frame_mask and not img.get('good_frame', True):
                continue

            img.update({
                'img_id': img_id,
                'img_path': osp.join(self.data_prefix['img'], img['file_name']),
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
