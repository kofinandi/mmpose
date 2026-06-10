# Copyright (c) OpenMMLab. All rights reserved.
"""Unified data loading for benchmarking bottomup and topdown models.

Loads GT annotations unfiltered (preserving ``area``, ``iscrowd``, etc.) and
prefetches images, providing a single :class:`UnifiedSample` per image for
both bottomup and topdown inference pipelines.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mmcv
import numpy as np
from mmengine.registry import DATASETS, TRANSFORMS, init_default_scope

from mmpose.evaluation.benchmark_datasets import BENCHMARK_TEST_DATASETS
from mmpose.structures.bbox import bbox_xyxy2xywh


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GTInstance:
    """Per-instance ground-truth annotation in COCO-17 keypoint format."""

    keypoints: np.ndarray           # (K, 2) float32
    keypoints_visible: np.ndarray   # (K,)   float32, binary 0/1
    bbox: np.ndarray                # (4,)   float32, xyxy
    area: float
    id: int
    category_id: int = 1
    iscrowd: int = 0
    num_keypoints: int = 0          # visible kps in COCO-17 space
    # Raw COCO visibility (0=unlabeled, 1=occluded, 2=visible), when
    # available from raw_ann_info. None for non-COCO / converted datasets.
    keypoints_visible_coco: Optional[np.ndarray] = None


@dataclass
class UnifiedSample:
    """One image with its GT annotations and prefetched pixel data."""

    img_id: int
    img_path: str
    image: np.ndarray                   # BGR HWC uint8
    ori_shape: Tuple[int, int]          # (H, W)
    gt_instances: List[GTInstance] = field(default_factory=list)
    crowd_index: Optional[float] = None  # CrowdPose crowdIndex


# ---------------------------------------------------------------------------
# Validity helper (mirrors BaseCocoStyleDataset._is_valid_instance)
# ---------------------------------------------------------------------------

def is_valid_instance(gt: GTInstance) -> bool:
    """Return True when ``gt`` is a valid detection target.

    Used by the mock detector to select GT bboxes that represent real,
    labelled persons.  Replicates
    :meth:`BaseCocoStyleDataset._is_valid_instance`.
    """
    if gt.iscrowd:
        return False
    if gt.num_keypoints == 0:
        return False
    w = gt.bbox[2] - gt.bbox[0]
    h = gt.bbox[3] - gt.bbox[1]
    if w <= 0 or h <= 0:
        return False
    if np.max(gt.keypoints) <= 0:
        return False
    return True


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_unified_samples(
    dataset_name: str,
    num_frames: Optional[int] = None,
) -> List[UnifiedSample]:
    """Load GT annotations and prefetch images for all supported datasets.

    Returns one :class:`UnifiedSample` per unique image (including images
    with zero annotations so the denominator for recall is correct).
    Keypoints are always converted to COCO-17 format.

    The raw annotation data is obtained via
    :meth:`BaseCocoStyleDataset._load_annotations` **without** any
    ``_is_valid_instance`` or ``_get_topdown_data_infos`` filtering, so
    every annotation (including ``iscrowd=1``) is preserved.

    Args:
        dataset_name: One of ``'coco', 'crowdpose', 'mpii', 'aic',
            'ochuman', 'emdb'``.
        num_frames: If set, cap the number of unique images loaded.

    Returns:
        List of :class:`UnifiedSample` in dataset order.
    """
    init_default_scope('mmpose')

    spec = BENCHMARK_TEST_DATASETS[dataset_name]

    # ── Resolve dataset build config ───────────────────────────────────────
    ds_cfg = dict(
        type=spec.dataset_type,
        data_root=spec.data_root,
        ann_file=spec.ann_file,
        data_prefix=spec.data_prefix,
        data_mode='topdown',
        pipeline=[],
        test_mode=True,
        lazy_init=True,
    )
    if spec.dataset_kwargs:
        ds_cfg.update(spec.dataset_kwargs)
    keypoint_src = spec.keypoint_src

    # Build dataset with lazy_init=True so full_init (which calls
    # _get_topdown_data_infos filtering) is never triggered.
    dataset = DATASETS.build(ds_cfg)
    instance_list, image_list = dataset._load_annotations()

    # ── Image metadata lookup ──────────────────────────────────────────────
    img_info: Dict[int, dict] = {}
    for img in image_list:
        iid = int(img['img_id'])
        if iid not in img_info:
            img_info[iid] = {
                'img_path': img['img_path'],
                # CrowdPose stores crowdIndex in the raw COCO image dict
                'crowd_index': img.get('crowdIndex') or img.get('crowd_index'),
            }

    # ── Ordered unique img_ids with optional num_frames cap ─────────────
    seen_img_ids: List[int] = []
    seen_set: set = set()
    for img in image_list:
        iid = int(img['img_id'])
        if iid not in seen_set:
            seen_set.add(iid)
            seen_img_ids.append(iid)
            if num_frames is not None and len(seen_img_ids) >= num_frames:
                break

    # ── Group instances by img_id ─────────────────────────────────────────
    img_to_instances: Dict[int, list] = {iid: [] for iid in seen_img_ids}
    for inst in instance_list:
        iid = int(inst['img_id'])
        if iid in img_to_instances:
            img_to_instances[iid].append(inst)

    # ── Keypoint converter ────────────────────────────────────────────────
    kp_converter = None
    if keypoint_src != 'coco':
        kp_converter = TRANSFORMS.build(
            dict(type='KeypointConverter', src=keypoint_src, dst='coco'))

    def _parse_instance(inst: dict) -> GTInstance:
        """Convert a raw dataset instance dict into a :class:`GTInstance`."""
        kpts = np.asarray(inst['keypoints'], dtype=np.float32)

        kv_raw = inst.get('keypoints_visible')
        if kv_raw is None:
            # Fall back to ones if visibility not stored
            kv = np.ones(kpts.shape[:-1], dtype=np.float32)
        else:
            kv = np.asarray(kv_raw, dtype=np.float32)

        # Normalise shapes for KeypointConverter: (N, K, 2) and (N, K)
        if kpts.ndim == 2:
            kpts = kpts[None]       # (1, K, 2)
        if kv.ndim == 1:
            kv = kv[None]           # (1, K)
        elif kv.ndim == 3:
            kv = kv[:, :, 0]       # (N, K, 2) → (N, K): take visibility

        if kp_converter is not None:
            tmp = kp_converter({'keypoints': kpts, 'keypoints_visible': kv})
            kpts_out = tmp['keypoints']          # (1, K_dst, 2)
            kv_out = tmp['keypoints_visible']    # (1, K_dst, 2): [vis, weight]
            kpts_final: np.ndarray = kpts_out[0]        # (K_dst, 2)
            kv_final: np.ndarray = kv_out[0, :, 0]     # (K_dst,)
        else:
            kpts_final = kpts[0]                        # (K, 2)
            kv_final = kv[0] if kv.ndim >= 2 else kv   # (K,)

        num_kpts = int(np.sum(kv_final > 0))

        bbox = np.asarray(inst['bbox'], dtype=np.float32).reshape(4)

        area_raw = inst.get('area')
        if area_raw is not None:
            area = float(np.asarray(area_raw).flat[0])
        else:
            area = 0.0
        if area <= 0:
            area = float(
                max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) * 0.53, 1.0))

        iscrowd = int(inst.get('iscrowd', 0))

        cat_id = inst.get('category_id', 1)
        try:
            cat_id = int(np.asarray(cat_id).flat[0])
        except (TypeError, ValueError):
            cat_id = 1

        inst_id = inst.get('id', 0)
        try:
            inst_id = int(np.asarray(inst_id).flat[0])
        except (TypeError, ValueError):
            inst_id = 0

        # Recover raw COCO visibility (0/1/2) from raw_ann_info when no
        # keypoint converter is applied.  _load_annotations clips v to [0,1]
        # via np.minimum(1, v), so we re-read from the original flat list.
        kv_coco: Optional[np.ndarray] = None
        if kp_converter is None:
            raw_kp_list = inst.get('raw_ann_info', {}).get('keypoints', [])
            if raw_kp_list:
                raw_arr = np.asarray(raw_kp_list, dtype=np.float32).reshape(-1, 3)
                if len(raw_arr) == len(kpts_final):
                    kv_coco = raw_arr[:, 2]  # raw v column: 0/1/2

        return GTInstance(
            keypoints=kpts_final,
            keypoints_visible=kv_final,
            bbox=bbox,
            area=area,
            id=inst_id,
            category_id=cat_id,
            iscrowd=iscrowd,
            num_keypoints=num_kpts,
            keypoints_visible_coco=kv_coco,
        )

    # ── Prefetch images ───────────────────────────────────────────────────
    print(f'Prefetching {len(seen_img_ids)} images ...')
    samples: List[UnifiedSample] = []
    for img_id in seen_img_ids:
        info = img_info.get(img_id)
        if info is None:
            insts = img_to_instances.get(img_id, [])
            if insts:
                img_path = insts[0]['img_path']
                crowd_index = None
            else:
                continue  # cannot determine image path
        else:
            img_path = info['img_path']
            crowd_index = info.get('crowd_index')

        image = mmcv.imread(img_path)
        if image is None:
            print(f'Warning: could not read {img_path}, skipping.')
            continue
        h, w = image.shape[:2]

        gt_instances = [
            _parse_instance(inst) for inst in img_to_instances[img_id]
        ]

        samples.append(UnifiedSample(
            img_id=img_id,
            img_path=img_path,
            image=image,
            ori_shape=(h, w),
            gt_instances=gt_instances,
            crowd_index=crowd_index,
        ))

    print(f'Prefetch complete. Loaded {len(samples)} images.')
    return samples


# ---------------------------------------------------------------------------
# Detection annotation builder (for --det-metrics)
# ---------------------------------------------------------------------------

def build_det_ann_from_samples(
    samples: List[UnifiedSample],
    cache_dir: Optional[str] = None,
) -> str:
    """Build a COCO bbox-detection GT JSON from :class:`UnifiedSample` list.

    Replaces ``convert_pose_gt_to_coco_det_ann`` / ``resolve_det_ann_file``
    for the ``--det-metrics`` path.  Preserves the real ``iscrowd`` value
    from each annotation (previously hardcoded to 0).

    Args:
        samples: List of loaded unified samples.
        cache_dir: Directory for caching the JSON output.

    Returns:
        Path to the COCO-format annotation JSON file.
    """
    if cache_dir is None:
        cache_dir = osp.join('work_dirs', '.cache', 'coco_det_ann')
    os.makedirs(cache_dir, exist_ok=True)

    key = ','.join(f'{s.img_id}:{len(s.gt_instances)}' for s in samples)
    digest = hashlib.md5(key.encode()).hexdigest()
    out_path = osp.join(cache_dir, f'unified_{digest}.json')
    if osp.isfile(out_path):
        return out_path

    images = []
    annotations = []

    for sample in samples:
        h, w = sample.ori_shape
        images.append({
            'id': int(sample.img_id),
            'file_name': osp.basename(sample.img_path),
            'width': w,
            'height': h,
        })
        for inst in sample.gt_instances:
            bbox_xywh = bbox_xyxy2xywh(
                inst.bbox.reshape(1, 4).astype(np.float32))[0]
            annotations.append({
                'id': int(inst.id),
                'image_id': int(sample.img_id),
                'category_id': int(inst.category_id),
                'bbox': [float(x) for x in bbox_xywh],
                'area': float(inst.area),
                'iscrowd': int(inst.iscrowd),
            })

    coco_json = {
        'info': {'description': 'COCO det GT built by mmpose benchmark loader'},
        'images': images,
        'annotations': annotations,
        'categories': [
            {'supercategory': 'person', 'id': 1, 'name': 'person'}
        ],
    }
    with open(out_path, 'w') as f:
        json.dump(coco_json, f)
    return out_path
