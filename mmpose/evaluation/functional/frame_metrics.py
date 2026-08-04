# Copyright (c) OpenMMLab. All rights reserved.
"""Per-frame OKS matching and prediction export helpers for benchmarks."""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.structures import InstanceData

from mmpose.structures import PoseDataSample, split_instances


def normalize_keypoint_visibility(
    visibility: Union[np.ndarray, Sequence, None],
    num_keypoints: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Return per-joint visibility scores with shape ``(K,)``.

    Supports dataset layouts ``(K,)``, ``(1, K)``, ``(1, K, 1)``, and
    ``KeypointConverter`` outputs ``(K, 2)`` / ``(1, K, 2)`` where the second
    channel is a loss weight, not visibility.
    """
    if visibility is None:
        return None

    vis = np.asarray(visibility, dtype=np.float32)
    if vis.ndim == 3:
        vis = vis[0]
    if vis.ndim == 2:
        if vis.shape[0] == 1:
            vis = vis[0]
        elif vis.shape[1] in (1, 2):
            vis = vis[:, 0]

    vis = vis.reshape(-1)
    k = num_keypoints if num_keypoints is not None else vis.size
    if vis.size == k * 2:
        vis = vis.reshape(k, 2)[:, 0]
    elif vis.size > k:
        vis = vis[:k]
    elif vis.size < k:
        padded = np.zeros(k, dtype=np.float32)
        padded[:vis.size] = vis
        vis = padded
    return vis


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
            n_kpts = None
            if inst.get('keypoints') is not None:
                n_kpts = len(inst['keypoints'])
            kv = normalize_keypoint_visibility(kv, n_kpts)
            inst['keypoints_visible'] = kv.tolist()

        bbox = g.get('bbox')
        if bbox is not None:
            inst['bbox'] = np.asarray(bbox).reshape(-1)[:4].tolist()

        if g.get('orig_area') is not None:
            inst['orig_area'] = float(g['orig_area'])

        if g.get('iscrowd') is not None:
            inst['iscrowd'] = int(g['iscrowd'])

        kv_coco = g.get('keypoints_visible_coco')
        if kv_coco is not None:
            inst['keypoints_visible_coco'] = np.asarray(
                kv_coco, dtype=np.float32).reshape(-1).tolist()

        results.append(inst)
    return results


def _serialize_gt_instance_data(gt_inst: InstanceData) -> List[dict]:
    has_keypoints = hasattr(gt_inst, 'keypoints') and len(gt_inst.keypoints) > 0
    has_bboxes = hasattr(gt_inst, 'bboxes') and len(gt_inst.bboxes) > 0
    if not has_keypoints and not has_bboxes:
        return []

    n = len(gt_inst.keypoints) if has_keypoints else len(gt_inst.bboxes)
    results = []
    for i in range(n):
        inst: dict = {}
        if has_keypoints:
            inst['keypoints'] = np.asarray(gt_inst.keypoints[i]).tolist()
        if hasattr(gt_inst, 'keypoints_visible') and len(
                gt_inst.keypoints_visible) > i:
            n_kpts = None
            if has_keypoints:
                n_kpts = int(np.asarray(gt_inst.keypoints[i]).reshape(-1, 2).shape[0])
            vis = normalize_keypoint_visibility(
                gt_inst.keypoints_visible[i], n_kpts)
            inst['keypoints_visible'] = vis.tolist()
        if has_bboxes and len(gt_inst.bboxes) > i:
            inst['bbox'] = np.asarray(gt_inst.bboxes[i]).reshape(-1)[:4].tolist()
        if hasattr(gt_inst, 'orig_areas') and len(gt_inst.orig_areas) > i:
            inst['orig_area'] = float(gt_inst.orig_areas[i])
        if hasattr(gt_inst, 'iscrowd') and len(gt_inst.iscrowd) > i:
            inst['iscrowd'] = int(gt_inst.iscrowd[i])
        results.append(inst)
    return results


def _pad_or_trim_kpts(kpts: np.ndarray, num_kpts: int) -> np.ndarray:
    """Reshape and pad/trim a keypoint array to (num_kpts, 2)."""
    k = kpts.reshape(-1, 2)
    if k.shape[0] == num_kpts:
        return k.astype(np.float32)
    if k.shape[0] > num_kpts:
        return k[:num_kpts].astype(np.float32)
    padded = np.zeros((num_kpts, 2), dtype=np.float32)
    padded[:k.shape[0]] = k
    return padded


def _compute_oks_coco(
    gt_kpts: np.ndarray,   # (K, 2)
    gt_vis: np.ndarray,    # (K,) visibility flags; >0 = labeled
    gt_area: float,
    pred_kpts: np.ndarray, # (N, K, 2)
    oks_vars: np.ndarray,  # (K,) = (2 * sigma_i)^2
) -> np.ndarray:           # (N,) OKS values in [0, 1]
    """Compute OKS between one GT instance and N predictions.

    Implements the COCO keypoint evaluation formula exactly::

        OKS = (Σ_i exp(-d_i² / (2·s²·k_i²)) · δ(v_i > 0)) / (Σ_i δ(v_i > 0))

    where s² = GT area, k_i = sigma_i (so (2·k_i)² = oks_vars_i).
    Only GT keypoints with v_i > 0 (labeled, whether occluded or visible)
    contribute to numerator and denominator. Non-labeled keypoints (v_i == 0)
    are excluded entirely, matching COCOeval behaviour.
    """
    labeled = gt_vis > 0          # (K,) boolean mask
    k1 = int(labeled.sum())
    if k1 == 0:
        return np.zeros(len(pred_kpts), dtype=np.float32)

    dx = pred_kpts[:, :, 0] - gt_kpts[None, :, 0]  # (N, K)
    dy = pred_kpts[:, :, 1] - gt_kpts[None, :, 1]  # (N, K)
    d2 = dx ** 2 + dy ** 2                          # (N, K)

    # COCO exponent: e_i = d_i² / (2 · area · sigma_i²)
    #   With oks_vars_i = (2·sigma_i)² = 4·sigma_i²:
    #   e_i = d_i² / (vars_i · 2 · area) = d_i²/vars_i/area * 0.5
    area = float(gt_area) + np.spacing(1)
    e = d2 / (oks_vars[None, :] * (2.0 * area))  # (N, K)

    # Mask to labeled keypoints only
    e_labeled = e[:, labeled]                         # (N, k1)
    oks = np.sum(np.exp(-e_labeled), axis=1) / k1    # (N,)
    return oks.astype(np.float32)


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


def compute_oks_pairs(
    gt_list: List[dict],
    pred_list: List[dict],
    sigmas: np.ndarray,
) -> Tuple[List[Tuple[float, int, int]], List[int], List[int]]:
    """Compute all-pairs OKS scores between valid GT and predicted instances.

    Shared building block for :func:`match_instances_oks` and any custom
    assignment strategy (e.g. identity-aware "sticky" matching in tracking
    metrics) that needs the full pairwise OKS score matrix rather than just
    the greedily-selected matches.

    "Valid" GT excludes crowd regions and GT with no labeled keypoints;
    "valid" pred excludes instances with no keypoints at all — see
    :func:`match_instances_oks` for the exact semantics.  The two are
    independent: a frame with zero valid GT (e.g. all crowd, or none at
    all) still reports every keypoint-bearing prediction in
    ``valid_pred_idx``, so a caller that counts unmatched-but-present
    predictions as false positives (as :class:`~mmpose.evaluation.metrics.
    MOTA` does) attributes them correctly instead of silently dropping
    them.

    Returns:
        Tuple of:

        - ``pairs``: ``(oks, gt_idx, pred_idx)`` tuples for every valid GT
          x valid pred combination, sorted by OKS descending.
        - ``valid_gt_idx``: indices into ``gt_list`` considered for matching.
        - ``valid_pred_idx``: indices into ``pred_list`` considered for
          matching.
    """
    sigmas = np.asarray(sigmas, dtype=np.float32)
    num_kpts = len(sigmas)
    oks_vars = (sigmas * 2.0) ** 2  # (K,)

    # Separate valid GT from crowd / keypoint-free GT
    valid_gt_idx: List[int] = []
    for gi, gt in enumerate(gt_list):
        if gt.get('iscrowd', 0):
            continue
        if gt.get('keypoints') is None:
            continue
        # Prefer raw COCO vis; fall back to binary vis
        _coco = gt.get('keypoints_visible_coco')
        vis_raw = _coco if _coco is not None else gt.get('keypoints_visible')
        if vis_raw is not None:
            vis = normalize_keypoint_visibility(vis_raw, num_kpts)
            if int(np.sum(vis > 0)) == 0:
                continue  # no labeled keypoints → can't match
        valid_gt_idx.append(gi)

    # Pre-build pred keypoint array (N, K, 2) and track original indices.
    # Computed unconditionally -- *not* gated on ``valid_gt_idx`` -- so a
    # frame with no valid GT still reports which predictions exist; see
    # the docstring note above.
    valid_pred_idx: List[int] = []
    pred_kpts_list: List[np.ndarray] = []
    if num_kpts > 0:
        for pi, pred in enumerate(pred_list):
            kpts = pred.get('keypoints')
            if kpts is None:
                continue
            pred_kpts_list.append(
                _pad_or_trim_kpts(np.asarray(kpts), num_kpts))
            valid_pred_idx.append(pi)

    if not valid_gt_idx or not pred_kpts_list:
        return [], valid_gt_idx, valid_pred_idx

    pred_kpts_arr = np.stack(pred_kpts_list, axis=0)  # (N, K, 2)

    # Score each valid GT against all preds
    pairs: List[Tuple[float, int, int]] = []
    for gi in valid_gt_idx:
        gt = gt_list[gi]
        gt_kpts = _pad_or_trim_kpts(np.asarray(gt['keypoints']), num_kpts)

        _coco = gt.get('keypoints_visible_coco')
        vis_raw = _coco if _coco is not None else gt.get('keypoints_visible')
        if vis_raw is not None:
            gt_vis = normalize_keypoint_visibility(vis_raw, num_kpts)
        else:
            gt_vis = np.ones(num_kpts, dtype=np.float32)

        gt_area = _instance_area(gt)
        oks_vals = _compute_oks_coco(
            gt_kpts, gt_vis, gt_area, pred_kpts_arr, oks_vars)

        for row_pi, oks_val in enumerate(oks_vals):
            pairs.append((float(oks_val), gi, valid_pred_idx[row_pi]))

    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs, valid_gt_idx, valid_pred_idx


def match_instances_oks(
    gt_list: List[dict],
    pred_list: List[dict],
    sigmas: np.ndarray,
    match_thr: float = 0.5,
) -> Tuple[List[dict], dict]:
    """Greedy OKS matching aligned with COCOeval semantics.

    Differences from a naive implementation:

    * **Non-labeled keypoints excluded** – only GT keypoints with visibility
      > 0 contribute to the OKS numerator and denominator (matches
      ``COCOeval`` which applies ``e = e[vg > 0]``).
    * **GT area only** – OKS is normalised by the GT object area, not the
      average of GT and predicted areas.
    * **iscrowd GT skipped** – crowd regions are not counted in ``num_gt``
      and do not generate false negatives (matching ``COCOeval`` which
      ignores crowd GT for recall).
    * **k1 == 0 guard** – if a GT instance has no labeled keypoints, OKS is
      0 (no match).

    The GT visibility flag is taken from ``keypoints_visible_coco`` (raw COCO
    0/1/2) when present, otherwise from ``keypoints_visible`` (binary 0/1).
    Both representations mark labeled keypoints as v > 0.

    See :func:`compute_oks_pairs` for the underlying pairwise-score
    computation shared with other matching strategies.
    """
    num_pred = len(pred_list)
    pairs, valid_gt_idx, _ = compute_oks_pairs(gt_list, pred_list, sigmas)
    num_valid_gt = len(valid_gt_idx)

    base_metrics = {
        'mean_oks': 0.0,
        'num_pred': num_pred,
        'num_gt': num_valid_gt,
        'num_matched': 0,
        'gt_recall': 0.0,
    }
    if not pairs:
        return [], base_metrics

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
        matches.append({'gt_idx': gi, 'pred_idx': pi, 'oks': oks_val})

    num_matched = len(matches)
    mean_oks = float(np.mean([m['oks'] for m in matches])) if matches else 0.0
    metrics = {
        'mean_oks': mean_oks,
        'num_pred': num_pred,
        'num_gt': num_valid_gt,
        'num_matched': num_matched,
        'gt_recall': num_matched / num_valid_gt if num_valid_gt > 0 else 0.0,
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
    pred_ds: PoseDataSample,
    gt_instances: List[dict],
    dataset_meta: dict,
    match_thr: float = 0.5,
) -> dict:
    """Build one frame dict for ``frames.json``.

    Prediction instances, shape, and optional raw detector bboxes are read
    directly from ``pred_ds`` so callers don't need to extract them separately.
    Raw detector output (before keypoint refinement) is stored in
    ``pred_ds.metainfo`` under ``'det_bboxes'`` / ``'det_scores'`` by
    :func:`run_topdown`.
    """
    pred_instances = serialize_pred_instances(pred_ds.pred_instances)
    ori_shape = pred_ds.metainfo.get('ori_shape', (0, 0))
    det_bboxes = pred_ds.metainfo.get('det_bboxes')
    det_scores = pred_ds.metainfo.get('det_scores')

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


def sanitize_dataset_meta(dataset_meta: dict) -> dict:
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
