# Copyright (c) OpenMMLab. All rights reserved.
"""Per-frame OKS matching and prediction export helpers for benchmarks."""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.structures import InstanceData

from mmpose.evaluation.functional.nms import oks_iou
from mmpose.structures import split_instances


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')


def serialize_pred_instances(instances: Optional[InstanceData]) -> List[dict]:
    """Serialize predicted instances to JSON-friendly dicts."""
    if instances is None:
        return []
    return split_instances(instances)


def serialize_gt_instances(
    gt_source: Union[InstanceData, Sequence[dict], None],
) -> List[dict]:
    """Serialize GT instances from InstanceData or dataset GT dicts."""
    if gt_source is None:
        return []

    if isinstance(gt_source, InstanceData):
        return _serialize_gt_instance_data(gt_source)

    results = []
    for g in gt_source:
        inst: dict = {}
        kpts = g.get('keypoints')
        if kpts is not None:
            kpts = np.asarray(kpts)
            if kpts.ndim == 3 and kpts.shape[0] == 1:
                kpts = kpts[0]
            inst['keypoints'] = kpts.reshape(-1, 2).tolist()

        kv = g.get('keypoints_visible')
        if kv is not None:
            kv = np.asarray(kv)
            if kv.ndim == 3:
                kv = kv[0, :, 0] if kv.shape[-1] == 1 else kv[0]
            elif kv.ndim == 2 and kv.shape[0] == 1:
                kv = kv[0]
            inst['keypoints_visible'] = np.atleast_1d(kv).tolist()

        bbox = g.get('bbox')
        if bbox is not None:
            inst['bbox'] = np.asarray(bbox).reshape(-1)[:4].tolist()

        if g.get('orig_area') is not None:
            inst['orig_area'] = float(g['orig_area'])

        results.append(inst)
    return results


def _serialize_gt_instance_data(gt_inst: InstanceData) -> List[dict]:
    if not hasattr(gt_inst, 'keypoints') or len(gt_inst.keypoints) == 0:
        return []

    results = []
    n = len(gt_inst.keypoints)
    for i in range(n):
        inst = {'keypoints': np.asarray(gt_inst.keypoints[i]).tolist()}
        if hasattr(gt_inst, 'keypoints_visible'):
            vis = gt_inst.keypoints_visible[i]
            if np.asarray(vis).ndim > 1:
                vis = np.asarray(vis).reshape(-1)
            inst['keypoints_visible'] = np.asarray(vis).tolist()
        if hasattr(gt_inst, 'bboxes') and len(gt_inst.bboxes) > i:
            inst['bbox'] = np.asarray(gt_inst.bboxes[i]).reshape(-1)[:4].tolist()
        if hasattr(gt_inst, 'orig_areas') and len(gt_inst.orig_areas) > i:
            inst['orig_area'] = float(gt_inst.orig_areas[i])
        results.append(inst)
    return results


def _to_oks_vector(keypoints: np.ndarray,
                    visibility: Optional[np.ndarray] = None,
                    scores: Optional[np.ndarray] = None) -> np.ndarray:
    """Pack keypoints into (K*3,) vector for :func:`oks_iou`."""
    kpts = np.asarray(keypoints).reshape(-1, 2)
    num_kpts = kpts.shape[0]
    if visibility is not None:
        vis = np.asarray(visibility).reshape(num_kpts)
        if vis.ndim > 1:
            vis = vis[:, 0]
    elif scores is not None:
        vis = np.asarray(scores).reshape(num_kpts)
    else:
        vis = np.ones(num_kpts, dtype=np.float32)

    flat = np.zeros(num_kpts * 3, dtype=np.float32)
    flat[0::3] = kpts[:, 0]
    flat[1::3] = kpts[:, 1]
    flat[2::3] = vis.astype(np.float32)
    return flat


def _instance_area(inst: dict) -> float:
    if inst.get('orig_area') is not None:
        return float(inst['orig_area'])
    bbox = inst.get('bbox')
    if bbox is not None:
        b = np.asarray(bbox).reshape(-1)[:4]
        return float(max((b[2] - b[0]) * (b[3] - b[1]), 1.0))
    kpts = inst.get('keypoints')
    if kpts is not None:
        k = np.asarray(kpts).reshape(-1, 2)
        if len(k) > 0:
            x_min, y_min = k.min(axis=0)
            x_max, y_max = k.max(axis=0)
            return float(max((x_max - x_min) * (y_max - y_min), 1.0))
    return 1.0


def match_instances_oks(
    gt_list: List[dict],
    pred_list: List[dict],
    sigmas: np.ndarray,
    match_thr: float = 0.5,
) -> Tuple[List[dict], dict]:
    """Greedy OKS matching between GT and prediction instance lists."""
    num_gt = len(gt_list)
    num_pred = len(pred_list)
    base_metrics = {
        'mean_oks': 0.0,
        'num_pred': num_pred,
        'num_gt': num_gt,
        'num_matched': 0,
        'gt_recall': 0.0,
    }
    if num_gt == 0 or num_pred == 0:
        return [], base_metrics

    sigmas = np.asarray(sigmas, dtype=np.float32)
    pred_vecs = []
    pred_areas = []
    for pred in pred_list:
        kpts = np.asarray(pred['keypoints'])
        scores = pred.get('keypoint_scores')
        pred_vecs.append(
            _to_oks_vector(kpts, scores=np.asarray(scores)
                           if scores is not None else None))
        pred_areas.append(_instance_area(pred))
    pred_vecs_arr = np.stack(pred_vecs, axis=0)
    pred_areas_arr = np.asarray(pred_areas, dtype=np.float32)

    pairs: List[Tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_list):
        g_vec = _to_oks_vector(
            np.asarray(gt['keypoints']),
            visibility=np.asarray(gt['keypoints_visible'])
            if gt.get('keypoints_visible') is not None else None,
        )
        g_area = _instance_area(gt)
        oks_row = oks_iou(g_vec, pred_vecs_arr, g_area, pred_areas_arr,
                          sigmas)
        for pi, oks_val in enumerate(oks_row):
            pairs.append((float(oks_val), gi, pi))

    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_gt: set = set()
    matched_pred: set = set()
    matches: List[dict] = []
    for oks_val, gi, pi in pairs:
        if oks_val < match_thr:
            break
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append({
            'gt_idx': gi,
            'pred_idx': pi,
            'oks': oks_val,
        })

    num_matched = len(matches)
    mean_oks = float(np.mean([m['oks'] for m in matches])) if matches else 0.0
    metrics = {
        'mean_oks': mean_oks,
        'num_pred': num_pred,
        'num_gt': num_gt,
        'num_matched': num_matched,
        'gt_recall': num_matched / num_gt if num_gt > 0 else 0.0,
    }
    return matches, metrics


def relative_img_path(img_path: str, data_root: str) -> str:
    """Return path relative to dataset data_root when possible."""
    if not img_path:
        return img_path
    data_root = osp.abspath(data_root)
    abs_path = osp.abspath(img_path)
    if abs_path.startswith(data_root + osp.sep) or abs_path == data_root:
        return osp.relpath(abs_path, data_root)
    return img_path


def build_frame_record(
    *,
    img_id: int,
    frame_id: int,
    img_path: str,
    data_root: str,
    ori_shape: Tuple[int, int],
    pred_instances: List[dict],
    gt_instances: List[dict],
    dataset_meta: dict,
    det_bboxes: Optional[np.ndarray] = None,
    det_scores: Optional[np.ndarray] = None,
    match_thr: float = 0.5,
) -> dict:
    """Build one frame dict for ``frames.json``."""
    sigmas = np.asarray(dataset_meta.get('sigmas', []), dtype=np.float32)
    matches, metrics = match_instances_oks(
        gt_instances, pred_instances, sigmas, match_thr=match_thr)

    record = {
        'img_id': int(img_id),
        'frame_id': int(frame_id),
        'img_path': relative_img_path(img_path, data_root),
        'ori_shape': [int(ori_shape[0]), int(ori_shape[1])],
        'predictions': {
            'instances': pred_instances,
        },
        'ground_truth': {
            'instances': gt_instances,
        },
        'metrics': metrics,
    }
    if matches:
        record['metrics']['matches'] = matches

    if det_bboxes is not None and len(det_bboxes) > 0:
        record['predictions']['det_bboxes'] = np.asarray(
            det_bboxes, dtype=np.float32).tolist()
    if det_scores is not None and len(det_scores) > 0:
        record['predictions']['det_scores'] = np.asarray(
            det_scores, dtype=np.float32).tolist()

    return record


def _sanitize_dataset_meta(dataset_meta: dict) -> dict:
    """Convert dataset_meta values to JSON-serializable types."""
    out = {}
    for key, val in dataset_meta.items():
        if isinstance(val, np.ndarray):
            out[key] = val.tolist()
        elif isinstance(val, (list, tuple)):
            converted = []
            for item in val:
                if isinstance(item, np.ndarray):
                    converted.append(item.tolist())
                elif isinstance(item, (list, tuple)):
                    converted.append([
                        x.tolist() if isinstance(x, np.ndarray) else x
                        for x in item
                    ])
                else:
                    converted.append(item)
            out[key] = converted
        else:
            out[key] = val
    return out


def save_prediction_bundle(manifest: dict, frames: List[dict],
                           out_dir: str) -> str:
    """Write ``manifest.json`` and ``frames.json`` to *out_dir*."""
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = osp.join(out_dir, 'manifest.json')
    frames_path = osp.join(out_dir, 'frames.json')

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=_json_default)

    with open(frames_path, 'w', encoding='utf-8') as f:
        json.dump(frames, f, indent=2, default=_json_default)

    return out_dir
