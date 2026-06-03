# Copyright (c) OpenMMLab. All rights reserved.
"""Convert pose-dataset GT to COCO bbox-detection annotation format.

Used to evaluate detector AP/AR on datasets
whose native annotation files are not compatible with mmdet CocoMetric
(e.g. MPII list format, CrowdPose/AIC missing ``area``/``iscrowd``).
"""

import hashlib
import json
import os
import os.path as osp
from typing import List, Optional, Tuple

import mmcv
import numpy as np
from mmengine.config import Config
from mmengine.registry import DATASETS, TRANSFORMS, init_default_scope
from xtcocotools.coco import COCO

from mmpose.structures.bbox import bbox_xyxy2xywh

_DEFAULT_CACHE_DIR = osp.join('work_dirs', '.cache', 'coco_det_ann')


def _init_scope(cfg: Config) -> None:
    scope = cfg.get('default_scope', 'mmpose')
    if scope:
        init_default_scope(scope)


def is_coco_det_ann_file(ann_file: str) -> bool:
    """Return True when *ann_file* is a COCO JSON usable for bbox AP/AR.

    Requires the file to load via ``xtcocotools.COCO`` and every annotation
    to contain both ``area`` and ``iscrowd`` (needed by pycocotools eval).
    """
    if not osp.isfile(ann_file):
        return False
    try:
        coco = COCO(ann_file)
    except (AssertionError, ValueError, KeyError, TypeError):
        return False

    ann_ids = coco.getAnnIds()
    if not ann_ids:
        return True
    for ann in coco.loadAnns(ann_ids):
        if 'area' not in ann or 'iscrowd' not in ann:
            return False
    return True


def _take_instance_field(item: dict, key: str, idx: Optional[int]):
    """Return one instance field, indexing stacked bottom-up arrays."""
    val = item.get(key)
    if val is None:
        return None
    if idx is None:
        return val
    arr = np.asarray(val)
    if arr.ndim == 0 or arr.shape[0] == 1:
        return val
    return arr[idx]


def _orig_area_from_item(item: dict, idx: Optional[int]) -> Optional[float]:
    raw_ann = item.get('raw_ann_info')
    if isinstance(raw_ann, dict):
        area = raw_ann.get('area')
        if area is not None:
            return float(area)
    elif isinstance(raw_ann, list) and raw_ann:
        ann_idx = idx if idx is not None and idx < len(raw_ann) else 0
        ann = raw_ann[ann_idx]
        if isinstance(ann, dict) and ann.get('area') is not None:
            return float(ann['area'])

    area = _take_instance_field(item, 'area', idx)
    if area is not None:
        arr = np.asarray(area).reshape(-1)
        if arr.size == 0:
            return None
        return float(arr[0])
    return None


def _category_id_from_item(item: dict, idx: Optional[int]) -> int:
    cat_id = _take_instance_field(item, 'category_id', idx)
    if cat_id is None:
        return 1
    arr = np.asarray(cat_id).reshape(-1)
    if arr.size == 0:
        return 1
    return int(arr[0])


def _inst_id_from_item(item: dict, idx: Optional[int]) -> Optional[int]:
    inst_id = _take_instance_field(item, 'id', idx)
    if inst_id is None:
        return None
    arr = np.asarray(inst_id).reshape(-1)
    if arr.size == 0:
        return None
    return int(arr[0])


def _build_gt_instance_dict(item: dict, idx: Optional[int] = None) -> dict:
    raw_ann = item.get('raw_ann_info')
    if isinstance(raw_ann, list) and idx is not None and idx < len(raw_ann):
        raw_ann = raw_ann[idx]
    return {
        'keypoints': _take_instance_field(item, 'keypoints', idx),
        'keypoints_visible': _take_instance_field(item, 'keypoints_visible', idx),
        'bbox_scale': _take_instance_field(item, 'bbox_scale', idx),
        'bbox': _take_instance_field(item, 'bbox', idx),
        'orig_area': _orig_area_from_item(item, idx),
        'category_id': _category_id_from_item(item, idx),
        'id': _inst_id_from_item(item, idx),
        'raw_ann_info': raw_ann,
    }


def _item_to_gt_instances(item: dict) -> List[dict]:
    """Expand one dataset item into per-instance GT dicts."""
    kpts = item.get('keypoints')
    if kpts is not None:
        kpts_arr = np.asarray(kpts)
        if kpts_arr.size == 0:
            return []
        if kpts_arr.ndim == 3 and kpts_arr.shape[0] > 1:
            return [
                _build_gt_instance_dict(item, i)
                for i in range(kpts_arr.shape[0])
            ]
        return [_build_gt_instance_dict(item)]

    bbox = item.get('bbox')
    if bbox is not None:
        bbox_arr = np.asarray(bbox)
        if bbox_arr.size == 0:
            return []
        if bbox_arr.ndim == 2 and bbox_arr.shape[0] > 1:
            return [
                _build_gt_instance_dict(item, i)
                for i in range(bbox_arr.shape[0])
            ]
        return [_build_gt_instance_dict(item)]

    return []


def load_pose_gt_per_image(
    pose_cfg: Config,
    num_frames: Optional[int] = None,
) -> List[Tuple[int, str, list]]:
    """Load GT instances grouped by image from the pose test dataset.

    Returns a list of ``(img_id, img_path, gt_instances)`` tuples where each
    ``gt_instance`` dict has keys ``bbox`` (xyxy), ``orig_area``,
    ``category_id``, ``id``, and optionally keypoint fields.
    """
    _init_scope(pose_cfg)
    ds_cfg = pose_cfg.test_dataloader.dataset.to_dict()
    ds_cfg['pipeline'] = []
    dataset = DATASETS.build(ds_cfg)

    kp_converter = None
    for step in pose_cfg.test_dataloader.dataset.pipeline:
        if isinstance(step, dict) and step.get('type') == 'KeypointConverter':
            kp_converter = TRANSFORMS.build(step)
            break

    img_paths: dict = {}
    img_gt_instances: dict = {}

    for idx in range(len(dataset)):
        item = dataset[idx]
        img_id = item['img_id']
        if img_id not in img_paths:
            img_paths[img_id] = item['img_path']
            img_gt_instances[img_id] = []
            if num_frames is not None and len(img_paths) > num_frames:
                del img_paths[img_id]
                del img_gt_instances[img_id]
                break
        if img_id in img_gt_instances:
            if kp_converter is not None:
                item = kp_converter(item)
            img_gt_instances[img_id].extend(_item_to_gt_instances(item))

    return [
        (img_id, img_paths[img_id], img_gt_instances[img_id])
        for img_id in img_paths
    ]


def _cache_path(pose_cfg: Config, num_frames: Optional[int],
                cache_dir: str) -> str:
    ds_cfg = pose_cfg.test_dataloader.dataset
    data_root = ds_cfg.get('data_root', '')
    ann_file = ds_cfg.ann_file
    raw_path = osp.join(data_root, ann_file)
    mtime = osp.getmtime(raw_path) if osp.isfile(raw_path) else 0
    dataset_type = ds_cfg.get('type', '')
    key = f'{raw_path}:{mtime}:{num_frames}:{dataset_type}'
    digest = hashlib.md5(key.encode()).hexdigest()
    os.makedirs(cache_dir, exist_ok=True)
    return osp.join(cache_dir, f'{digest}.json')


def convert_pose_gt_to_coco_det_ann(
    pose_cfg: Config,
    num_frames: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Build a COCO bbox-detection GT JSON from pose-dataset instances.

    Args:
        pose_cfg: Pose model config (uses ``test_dataloader.dataset``).
        num_frames: Cap the number of unique images (``None`` = full set).
        cache_dir: Directory for cached JSON files.

    Returns:
        Path to the COCO-format annotation JSON file.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    out_path = _cache_path(pose_cfg, num_frames, cache_dir)
    if osp.isfile(out_path):
        return out_path

    gt_per_image = load_pose_gt_per_image(pose_cfg, num_frames)

    images = []
    annotations = []
    ann_id = 1

    for img_id, img_path, gt_insts in gt_per_image:
        img = mmcv.imread(img_path)
        h, w = img.shape[:2]
        images.append({
            'id': int(img_id),
            'file_name': osp.basename(img_path),
            'width': w,
            'height': h,
        })

        for inst in gt_insts:
            bbox_xyxy = np.asarray(inst['bbox']).reshape(1, 4)
            bbox_xywh = bbox_xyxy2xywh(bbox_xyxy.astype(np.float32))[0]
            bw, bh = float(bbox_xywh[2]), float(bbox_xywh[3])
            area = inst.get('orig_area')
            if area is None:
                area = bw * bh
            else:
                area = float(area)

            cat_id = inst.get('category_id', 1)
            inst_ann_id = inst.get('id')
            if inst_ann_id is None:
                inst_ann_id = ann_id

            annotations.append({
                'id': int(inst_ann_id),
                'image_id': int(img_id),
                'category_id': int(cat_id),
                'bbox': [float(x) for x in bbox_xywh],
                'area': area,
                'iscrowd': 0,
            })
            ann_id += 1

    coco_json = {
        'info': {
            'description': 'COCO det GT converted from pose dataset by mmpose',
        },
        'images': images,
        'annotations': annotations,
        'categories': [{
            'supercategory': 'person',
            'id': 1,
            'name': 'person',
        }],
    }

    with open(out_path, 'w') as f:
        json.dump(coco_json, f)

    return out_path


def resolve_det_ann_file(
    pose_cfg: Config,
    num_frames: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Return a COCO bbox-det GT path for mmdet CocoMetric.

    Uses the config's ``ann_file`` when it already satisfies
    :func:`is_coco_det_ann_file`, otherwise converts pose GT on the fly.
    """
    ds_cfg = pose_cfg.test_dataloader.dataset
    ann_file = osp.join(ds_cfg.get('data_root', ''), ds_cfg.ann_file)

    if is_coco_det_ann_file(ann_file):
        return ann_file

    out_path = convert_pose_gt_to_coco_det_ann(
        pose_cfg, num_frames=num_frames, cache_dir=cache_dir)
    print(f'Converted pose GT to COCO det annotations: {out_path}')
    return out_path
