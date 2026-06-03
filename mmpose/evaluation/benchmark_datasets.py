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

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


BENCHMARK_TEST_DATASETS: Dict[str, Optional[BenchmarkTestDataset]] = {
    'coco': None,
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
    ),
}


def _is_keypoint_converter(step) -> bool:
    return isinstance(step, dict) and step.get('type') == 'KeypointConverter'


def _find_pack_pose_inputs_index(pipeline: list) -> Optional[int]:
    for i in range(len(pipeline) - 1, -1, -1):
        step = pipeline[i]
        if isinstance(step, dict) and step.get('type') == 'PackPoseInputs':
            return i
    return None


def _build_evaluator_with_gt_from_samples(cfg: Config) -> dict:
    """Return CocoMetric config with gt_from_samples, preserving other kwargs."""
    ev = cfg.test_evaluator
    if isinstance(ev, list):
        if len(ev) != 1:
            raise ValueError(
                'apply_benchmark_test_dataset only supports a single '
                f'test_evaluator entry, got {len(ev)}')
        base = dict(ev[0])
    elif isinstance(ev, dict):
        base = dict(ev)
    else:
        base = {}

    base['type'] = 'CocoMetric'
    base['gt_from_samples'] = True
    base.pop('ann_file', None)
    return base


def _inject_keypoint_converter(
    pipeline: list,
    keypoint_src: str,
    dst: str,
    extended_meta_keys: bool,
) -> list:
    """Insert KeypointConverter immediately before PackPoseInputs."""
    pipeline = [
        dict(step) if isinstance(step, dict) else step
        for step in pipeline
        if not _is_keypoint_converter(step)
    ]

    pack_idx = _find_pack_pose_inputs_index(pipeline)
    if pack_idx is None:
        raise ValueError(
            'test_dataloader.dataset.pipeline must contain PackPoseInputs '
            'to apply a benchmark test dataset override')

    if extended_meta_keys:
        pack = dict(pipeline[pack_idx])
        pack['meta_keys'] = EXTENDED_PACK_META_KEYS
        pipeline[pack_idx] = pack

    converter = dict(type='KeypointConverter', src=keypoint_src, dst=dst)
    return pipeline[:pack_idx] + [converter] + pipeline[pack_idx:]


def apply_benchmark_test_dataset(
    cfg: Config,
    dataset_name: str,
    dst: str = 'coco',
) -> None:
    """Override ``test_dataloader`` / ``test_evaluator`` for cross-dataset eval.

    When ``dataset_name`` is ``'coco'``, the config is left unchanged so
    explicit conversion configs (e.g. ``*_crowdpose_conversion.py``) work
    as-is when no override is requested.

    Args:
        cfg: Pose config loaded from file (mutated in place).
        dataset_name: One of :data:`BENCHMARK_TEST_DATASET_NAMES`.
        dst: Keypoint convention for model outputs (default ``'coco'``).
    """
    if dataset_name not in BENCHMARK_TEST_DATASETS:
        raise ValueError(
            f'Unknown test dataset {dataset_name!r}. '
            f'Choose from: {", ".join(BENCHMARK_TEST_DATASET_NAMES)}')

    spec = BENCHMARK_TEST_DATASETS[dataset_name]
    if spec is None:
        return

    ds = cfg.test_dataloader.dataset
    data_mode = ds.get('data_mode', 'topdown')
    pipeline = _inject_keypoint_converter(
        list(ds.pipeline),
        spec.keypoint_src,
        dst,
        spec.extended_meta_keys,
    )

    dataset_cfg = dict(
        type=spec.dataset_type,
        data_root=spec.data_root,
        data_mode=data_mode,
        ann_file=spec.ann_file,
        data_prefix=spec.data_prefix,
        pipeline=pipeline,
        test_mode=True,
    )
    if spec.dataset_kwargs:
        dataset_cfg.update(spec.dataset_kwargs)
    cfg.test_dataloader.dataset = dataset_cfg

    if not spec.preserve_evaluator:
        cfg.test_evaluator = _build_evaluator_with_gt_from_samples(cfg)
