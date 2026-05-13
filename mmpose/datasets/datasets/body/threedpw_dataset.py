# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class ThreeDPWDataset(BaseCocoStyleDataset):
    """3DPW dataset for 2D human pose estimation.

    "Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving
    Camera", ECCV`2018.
    More details can be found in the `paper
    <https://arxiv.org/abs/1811.09751>`__ .

    3DPW keypoints (COCO-17 ordering)::

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
        16: 'right_ankle'

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If set,
            detected bboxes will be used instead of ground-truth bboxes.
            Default: ``None``.
        data_mode (str): ``'topdown'`` or ``'bottomup'``. Default: ``'topdown'``
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

    METAINFO: dict = dict(from_file='configs/_base_/datasets/threedpw.py')


@DATASETS.register_module()
class ThreeDPWVideoDataset(BaseCocoStyleDataset):
    """3DPW dataset with video-sequence support for temporal pose models.

    Extends :class:`ThreeDPWDataset` with the ability to sample neighboring
    frames around each center frame, following the pattern of
    :class:`PoseTrack18VideoDataset`.

    The annotation JSON must include ``nframes`` and ``frame_id`` fields per
    image entry (these are written by
    ``tools/dataset_converters/preprocess_threedpw.py``).

    3DPW image filenames follow the pattern ``image_XXXXX.jpg`` (5-digit
    zero-padded index with the ``image_`` prefix).

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. Default: ``None``.
        data_mode (str): ``'topdown'`` or ``'bottomup'``. Default: ``'topdown'``
        frame_weights (list[float]): Per-frame aggregation weights.  The
            first weight corresponds to the center frame; remaining weights
            correspond to sampled supporting frames in ascending index order.
            Must sum to 1.  Default: ``[0.0, 1.0]``.
        frame_sampler_mode (str): ``'fixed'`` -- use explicit ``frame_indices``;
            ``'random'`` -- sample from ``frame_range``.  Default: ``'random'``.
        frame_range (int | list[int], optional): Half-range (int) or
            ``[low, high]`` (inclusive) from which supporting frame offsets
            are drawn.  Required when ``frame_sampler_mode='random'``.
        num_sampled_frame (int, optional): Number of supporting frames to
            sample.  Required when ``frame_sampler_mode='random'``.
        frame_indices (list[int], optional): Fixed frame offsets (including
            ``0`` for center).  Required when ``frame_sampler_mode='fixed'``.
        metainfo (dict, optional): Dataset meta information. Default: ``None``.
        data_root (str, optional): Root directory for data. Default: ``None``.
        data_prefix (dict, optional): Prefix for data. Default:
            ``dict(img='')``.
        filter_cfg (dict, optional): Config for filtering. Default: ``None``.
        indices (int or Sequence[int], optional): Use only first N samples.
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

    METAINFO: dict = dict(from_file='configs/_base_/datasets/threedpw.py')

    def __init__(self,
                 ann_file: str = '',
                 bbox_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 frame_weights: List[Union[int, float]] = [0.0, 1.0],
                 frame_sampler_mode: str = 'random',
                 frame_range: Optional[Union[int, List[int]]] = None,
                 num_sampled_frame: Optional[int] = None,
                 frame_indices: Optional[Sequence[int]] = None,
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        assert abs(sum(frame_weights) - 1.0) < 1e-6, (
            f'`frame_weights` must sum to 1.0, got {frame_weights}.')
        for w in frame_weights:
            assert w >= 0, '`frame_weights` must be non-negative.'
        self.frame_weights = np.array(frame_weights)

        if frame_sampler_mode not in {'fixed', 'random'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid '
                f'frame_sampler_mode: {frame_sampler_mode!r}. '
                "Must be 'fixed' or 'random'.")
        self.frame_sampler_mode = frame_sampler_mode

        if frame_sampler_mode == 'random':
            assert frame_range is not None, (
                "`frame_sampler_mode='random'` requires `frame_range`.")
            if isinstance(frame_range, int):
                assert frame_range >= 0
                self.frame_range = [-frame_range, frame_range]
            else:
                assert (len(frame_range) == 2
                        and frame_range[0] <= 0 <= frame_range[1]
                        and frame_range[1] > frame_range[0]), (
                            f'Invalid `frame_range`: {frame_range}')
                self.frame_range = list(frame_range)

            assert num_sampled_frame is not None, (
                "`frame_sampler_mode='random'` requires `num_sampled_frame`.")
            assert len(frame_weights) == num_sampled_frame + 1, (
                f'len(frame_weights)={len(frame_weights)} must equal '
                f'num_sampled_frame+1={num_sampled_frame + 1}.')
            self.num_sampled_frame = num_sampled_frame
            self.frame_indices = None
            self.frame_range = self.frame_range

        else:  # fixed
            assert frame_indices is not None, (
                "`frame_sampler_mode='fixed'` requires `frame_indices`.")
            assert len(frame_weights) == len(frame_indices), (
                f'len(frame_weights) must equal len(frame_indices).')
            self.frame_indices = sorted(frame_indices)
            self.num_sampled_frame = None
            self.frame_range = None

        super().__init__(
            ann_file=ann_file,
            bbox_file=bbox_file,
            data_mode=data_mode,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw annotation and build multi-frame image path list.

        The center frame path is first; supporting frames are appended in
        ascending offset order.  At test time the center frame is omitted
        from the supporting list (offset 0 is skipped, matching PoseTrack18
        behavior).

        Args:
            raw_data_info (dict): Contains ``'raw_ann_info'`` and
                ``'raw_img_info'`` from the COCO loader.

        Returns:
            dict | None: Parsed sample with ``img_path`` as a list of paths,
            or ``None`` if the annotation is invalid.
        """
        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']

        if 'bbox' not in ann or 'keypoints' not in ann:
            return None
        if max(ann['keypoints']) == 0:
            return None

        img_w, img_h = img['width'], img['height']
        x, y, w, h = ann['bbox']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        _kpts = np.array(ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
        keypoints = _kpts[..., :2]
        keypoints_visible = np.minimum(1, _kpts[..., 2])

        nframes = int(img['nframes'])
        frame_id = int(img['frame_id'])

        # Build center frame path
        center_img_path = osp.join(
            self.data_prefix.get('img', ''), img['file_name'])
        img_paths = [center_img_path]

        # Determine supporting frame offsets
        if self.frame_sampler_mode == 'fixed':
            offsets = self.frame_indices
        else:
            low, high = self.frame_range
            offsets = np.random.randint(
                low, high + 1, self.num_sampled_frame).tolist()

        seq_dir = osp.dirname(center_img_path)
        for offset in offsets:
            if self.test_mode and offset == 0:
                continue
            sup_frame_id = int(np.clip(frame_id + offset, 0, nframes - 1))
            sup_path = osp.join(
                seq_dir, f'image_{sup_frame_id:05d}.jpg')
            img_paths.append(sup_path)

        data_info = {
            'img_id': int(img.get('frame_id', ann['image_id'])),
            'img_path': img_paths,
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': ann['num_keypoints'],
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'frame_weights': self.frame_weights,
            'id': ann['id'],
        }
        return data_info
