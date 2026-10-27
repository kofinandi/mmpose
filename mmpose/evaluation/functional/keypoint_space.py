# Copyright (c) OpenMMLab. All rights reserved.
"""Keypoint layout for benchmark evaluation.

``--keypoint-space coco`` (the default) is the fair baseline: ground truth
and predictions are expressed as COCO-17, and a joint that exists on only
one side of the mapping is dropped. ``--keypoint-space dataset`` evaluates
in the target dataset's own convention, and remaps predictions only when
the model emits a different name list. Nothing is interpolated or invented.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mmpose.datasets.transforms.keypoint_registry import (
    get_keypoints, get_mapping, get_metainfo)
from mmpose.evaluation.benchmark_datasets import BENCHMARK_TEST_DATASETS

# COCO-17 torso diagonals: left_shoulder-right_hip, right_shoulder-left_hip.
# Used when the metainfo has no names to resolve.
COCO_TORSO_PAIRS = ((5, 12), (6, 11))

_TORSO_JOINTS = (
    ('left_shoulder', 'right_hip'),
    ('right_shoulder', 'left_hip'),
)

KEYPOINT_SPACES = ('coco', 'dataset')


def eval_convention(keypoint_src: str, keypoint_space: str) -> str:
    """Convention name ground truth is stored in for one benchmark dataset.

    Args:
        keypoint_src: Dataset convention (``BenchmarkTestDataset.keypoint_src``).
        keypoint_space: ``'coco'`` or ``'dataset'``.

    Returns:
        str: ``'coco'`` or ``keypoint_src``.
    """
    if keypoint_space not in KEYPOINT_SPACES:
        raise ValueError(
            f'keypoint_space must be one of {KEYPOINT_SPACES}, '
            f'got {keypoint_space!r}.')
    if keypoint_space == 'coco':
        return 'coco'
    return keypoint_src


def evaluation_metainfo(dataset_name: str, keypoint_space: str) -> dict:
    """Sigmas and names of the layout metrics should score.

    This follows the evaluation convention, not whatever ``init_model``
    attached to the checkpoint. MOTIP configs have no train dataloader, so
    that fallback is COCO even when the network emits another layout.
    """
    spec = BENCHMARK_TEST_DATASETS[dataset_name]
    convention = eval_convention(spec.keypoint_src, keypoint_space)
    return get_metainfo(convention)


def names_from_metainfo(meta: Optional[dict]) -> List[str]:
    """Ordered keypoint names from a parsed metainfo dict."""
    if not meta:
        return []
    id2name = meta.get('keypoint_id2name') or {}
    n = int(meta.get('num_keypoints', len(id2name)))
    return [id2name[i] for i in range(n)]


def model_output_keypoint_names(model) -> List[str]:
    """Names of the joints a model actually writes into ``pred_instances``.

    ``output_keypoint_names`` wins when the estimator sets it (MOTIP's
    COCO-17 reprojection versus its native 15). Otherwise the model's
    ``dataset_meta`` names are used, and COCO-17 is the last resort.
    """
    names = getattr(model, 'output_keypoint_names', None)
    if names:
        return list(names)
    meta = getattr(model, 'dataset_meta', None) or {}
    parsed = names_from_metainfo(meta)
    if parsed:
        return parsed
    return list(get_keypoints('coco'))


def make_gt_converter(keypoint_src: str, keypoint_space: str):
    """``KeypointConverter`` from the dataset layout into the eval layout.

    Returns ``None`` when the two layouts are the same convention, so the
    loader leaves keypoints untouched. The converter matches names only;
    joints without a counterpart stay at zero with visibility 0.
    """
    dst = eval_convention(keypoint_src, keypoint_space)
    if keypoint_src == dst:
        return None
    from mmpose.datasets.transforms.converting import KeypointConverter
    return KeypointConverter(src=keypoint_src, dst=dst)


def convert_keypoint_arrays(
    coords: np.ndarray,
    values: Optional[np.ndarray],
    src_names: Sequence[str],
    dst_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Copy joints that share a name. Unmatched destination joints stay 0.

    Args:
        coords: ``(N, K_src, C)`` keypoint coordinates.
        values: ``(N, K_src)`` scores or visibilities, or ``None``.
        src_names: Source joint names, length ``K_src``.
        dst_names: Destination joint names, length ``K_dst``.

    Returns:
        Tuple of ``(N, K_dst, C)`` coordinates and ``(N, K_dst)`` values.
    """
    coords = np.asarray(coords)
    n = int(coords.shape[0])
    channels = int(coords.shape[-1]) if coords.ndim == 3 else 2
    dst_names = list(dst_names)
    out = np.zeros((n, len(dst_names), channels), dtype=np.float32)
    out_values = np.zeros((n, len(dst_names)), dtype=np.float32)
    if n == 0:
        return out, out_values

    mapping = get_mapping(list(src_names), dst_names)
    if not mapping:
        return out, out_values
    src_idx = [src for src, _ in mapping]
    dst_idx = [dst for _, dst in mapping]
    out[:, dst_idx] = np.asarray(coords[:, src_idx], dtype=np.float32)
    if values is not None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(n, -1)
        out_values[:, dst_idx] = values[:, src_idx]
    return out, out_values


def align_prediction_keypoints(
    pred_by_img_id: Dict[int, object],
    src_names: Sequence[str],
    dst_names: Sequence[str],
) -> None:
    """Rewrite every prediction into ``dst_names`` in place.

    Empty frames (no instances) are resized to the destination joint count
    even when the name lists match, because a few code paths allocate a
    COCO-17 placeholder for a frame the model never ran on. A non-empty
    frame whose joint count disagrees with ``src_names`` is an error: the
    model's declared layout and its tensors have diverged.
    """
    src_names = list(src_names)
    dst_names = list(dst_names)
    same = src_names == dst_names
    dst_k = len(dst_names)
    for ds in pred_by_img_id.values():
        pred = getattr(ds, 'pred_instances', None)
        if pred is None or 'keypoints' not in pred:
            continue
        coords = np.asarray(pred.keypoints)
        if coords.ndim == 2:
            coords = coords[None]
        n = int(coords.shape[0])
        if n == 0:
            if coords.shape[1] != dst_k:
                pred.keypoints = np.zeros((0, dst_k, 2), dtype=np.float32)
                pred.keypoint_scores = np.zeros((0, dst_k), dtype=np.float32)
                if 'keypoints_visible' in pred:
                    pred.keypoints_visible = np.zeros(
                        (0, dst_k), dtype=np.float32)
            continue
        if coords.shape[1] != len(src_names):
            raise ValueError(
                f'A prediction has {coords.shape[1]} joints but the model '
                f'layout lists {len(src_names)} '
                f'({src_names[:4]}...).')
        if same:
            continue
        scores = None
        if 'keypoint_scores' in pred:
            scores = np.asarray(pred.keypoint_scores)
            if scores.ndim == 3:
                scores = scores[..., 0]
        new_coords, new_scores = convert_keypoint_arrays(
            coords[..., :2], scores, src_names, dst_names)
        pred.keypoints = new_coords
        pred.keypoint_scores = new_scores
        if 'keypoints_visible' in pred:
            visible = np.asarray(pred.keypoints_visible)
            if visible.ndim == 3:
                visible = visible[..., 0]
            _, new_visible = convert_keypoint_arrays(
                coords[..., :2], visible, src_names, dst_names)
            pred.keypoints_visible = new_visible


def torso_index_pairs(
        dataset_meta: Optional[dict]) -> Tuple[Tuple[int, int], ...]:
    """Shoulder-hip index pairs for torso-normalised temporal metrics.

    Resolved from keypoint names so a non-COCO layout (CrowdPose shoulders
    are 0 and 1) uses its own indices. Missing names fall back to the
    COCO-17 pairs ``(5, 12)`` and ``(6, 11)``. When the names *are* the
    COCO set, the lookup returns those same indices.
    """
    if not dataset_meta:
        return COCO_TORSO_PAIRS
    name2id = dataset_meta.get('keypoint_name2id')
    if not name2id:
        id2name = dataset_meta.get('keypoint_id2name')
        if id2name:
            name2id = {name: idx for idx, name in id2name.items()}
    if not name2id:
        return COCO_TORSO_PAIRS
    pairs = []
    for left, right in _TORSO_JOINTS:
        if left not in name2id or right not in name2id:
            return COCO_TORSO_PAIRS
        pairs.append((int(name2id[left]), int(name2id[right])))
    return tuple(pairs)
