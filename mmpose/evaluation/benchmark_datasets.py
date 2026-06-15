# Copyright (c) OpenMMLab. All rights reserved.
"""Registry and helpers for cross-dataset benchmark evaluation.

Maps standard COCO-trained model configs onto alternate test datasets
(CrowdPose, MPII, AIC, OCHuman, EMDB) without duplicating per-model config
files.  Swap ``test_dataloader.dataset``, insert ``KeypointConverter`` before
``PackPoseInputs``, and use ``CocoMetric(gt_from_samples=True)`` unless
``preserve_evaluator`` is set on the dataset spec (e.g. EMDB temporal metrics).

``dst='coco'`` assumes 17-keypoint COCO-trained checkpoints. Whole-body or
Halpe models need a different destination convention and are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mmengine.config import Config

# meta_keys required for CocoMetric(gt_from_samples=True) on non-COCO sets
EXTENDED_PACK_META_KEYS = (
    'id', 'img_id', 'img_path', 'category_id', 'crowd_index',
    'ori_shape', 'img_shape', 'input_size', 'input_center',
    'input_scale', 'flip', 'flip_direction', 'flip_indices',
    'raw_ann_info', 'dataset_name', 'area',
)

BENCHMARK_TEST_DATASET_NAMES = (
    'coco', 'crowdpose', 'mpii', 'aic', 'ochuman', 'emdb')


@dataclass(frozen=True)
class BenchmarkTestDataset:
    """Metadata for a benchmark test split."""

    dataset_type: str
    data_root: str
    ann_file: str
    data_prefix: dict
    keypoint_src: str
    extended_meta_keys: bool = True
    dataset_kwargs: Optional[dict] = None
    preserve_evaluator: bool = False
    # Extra metric configs appended to the evaluator by build_evaluator in
    # benchmark_e2e.py.  These metrics receive dataset_meta but NOT
    # gt_from_samples (they manage their own state via process()).
    extra_metrics: Optional[List[dict]] = None


BENCHMARK_TEST_DATASETS: Dict[str, BenchmarkTestDataset] = {
    'coco': BenchmarkTestDataset(
        dataset_type='CocoDataset',
        data_root='data/',
        ann_file='coco/annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='coco/val2017/'),
        keypoint_src='coco',
    ),
    'crowdpose': BenchmarkTestDataset(
        dataset_type='CrowdPoseDataset',
        data_root='data/',
        ann_file='crowdpose/annotations/mmpose_crowdpose_trainval.json',
        data_prefix=dict(img='crowdpose/images/'),
        keypoint_src='crowdpose',
    ),
    'mpii': BenchmarkTestDataset(
        dataset_type='MpiiDataset',
        data_root='data/',
        ann_file='mpii/annotations/mpii_val.json',
        data_prefix=dict(img='mpii/images/'),
        keypoint_src='mpii',
    ),
    'aic': BenchmarkTestDataset(
        dataset_type='AicDataset',
        data_root='data/',
        ann_file='aic/annotations/aic_val.json',
        data_prefix=dict(
            img='aic/ai_challenger_keypoint'
            '_validation_20170911/keypoint_validation_images_20170911/'),
        keypoint_src='aic',
    ),
    'ochuman': BenchmarkTestDataset(
        dataset_type='OCHumanDataset',
        data_root='data/',
        ann_file='ochuman/annotations/'
        'ochuman_coco_format_val_range_0.00_1.00.json',
        data_prefix=dict(img='ochuman/images/'),
        keypoint_src='ochuman',
    ),
    'emdb': BenchmarkTestDataset(
        dataset_type='EmdbDataset',
        data_root='data/emdb/',
        ann_file='annotations/emdb_all.json',
        data_prefix=dict(img=''),
        keypoint_src='emdb',
        dataset_kwargs=dict(
            emdb1=True,
            emdb2=False,
            good_frame_mask=True,
        ),
        extra_metrics=[
            dict(type='MPJVE', prefix='emdb'),
            dict(type='MPJAE', prefix='emdb'),
            dict(type='MPJVE', norm_item=['bbox', 'torso'], prefix='emdb'),
            dict(type='MPJAE', norm_item=['bbox', 'torso'], prefix='emdb'),
        ],
    ),
}
