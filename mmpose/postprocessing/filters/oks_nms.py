# Copyright (c) OpenMMLab. All rights reserved.
"""Confidence thresholding + OKS-based non-maximum suppression filter."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from mmpose.structures import PoseDataSample

from ..base import BaseFilter
from ..registry import POST_PROCESS_FILTERS

# Default COCO-17 sigmas (same as mmpose/evaluation/functional/nms.py)
_COCO17_SIGMAS = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
    1.07, 1.07, .87, .87, .89, .89,
], dtype=np.float32) / 10.0


def _bbox_area(bbox: np.ndarray) -> float:
    """Return area of a single xyxy bbox."""
    b = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
    return float(max((b[2] - b[0]) * (b[3] - b[1]), 1.0))


def _oks_matrix(kpts: np.ndarray, areas: np.ndarray,
                 sigmas: np.ndarray) -> np.ndarray:
    """Pairwise OKS between all instances within a single set.

    Args:
        kpts: Keypoint coordinates ``(N, K, 2)``.
        areas: Per-instance area ``(N,)``, used as the OKS normaliser for
            the "row" instance in each pair.
        sigmas: Per-keypoint sigma values ``(K,)``.

    Returns:
        ``(N, N)`` OKS matrix. Not symmetric in general, since OKS(i, j)
        normalises by ``areas[i]`` while OKS(j, i) normalises by
        ``areas[j]``.
    """
    vars_ = (sigmas * 2.0)**2  # (K,)
    N = kpts.shape[0]

    oks = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        area = float(areas[i]) + np.spacing(1)
        dx = kpts[:, :, 0] - kpts[i:i + 1, :, 0]  # (N, K)
        dy = kpts[:, :, 1] - kpts[i:i + 1, :, 1]  # (N, K)
        d2 = dx**2 + dy**2  # (N, K)
        e = d2 / (vars_[None, :] * (2.0 * area))  # (N, K)
        oks[i] = np.mean(np.exp(-e), axis=1)  # (N,)
    return oks


@POST_PROCESS_FILTERS.register_module()
class OKSNMS(BaseFilter):
    """Confidence thresholding + greedy OKS-based non-maximum suppression.

    A per-frame filter with two stages:

    1. **Confidence thresholding**: instances whose confidence score is
       below ``score_thr`` are discarded outright.
    2. **OKS-based NMS**: among the remaining instances, sorts by score
       (descending) and greedily keeps each instance in turn, suppressing
       any not-yet-suppressed instance whose OKS with it is ``>= oks_thr``.
       This removes duplicate detections of the same person while keeping
       the highest-confidence instance among each duplicate cluster.

    Confidence scores are read from ``pred_instances.bbox_scores`` and/or
    ``pred_instances.keypoint_scores`` (mean over joints), depending on
    ``score_mode``.

    Args:
        score_thr (float): Minimum confidence score required to keep an
            instance. Default: ``0.3``.
        oks_thr (float): OKS threshold above which the lower-scoring of a
            pair of instances is suppressed. Set to ``1.0`` (or higher) to
            disable the NMS stage and only apply confidence thresholding.
            Default: ``0.9``.
        score_mode (str): One of ``'bbox'``, ``'keypoint'``, ``'auto'``.
            ``'bbox'`` uses ``pred_instances.bbox_scores``; ``'keypoint'``
            uses the mean of ``pred_instances.keypoint_scores`` per
            instance; ``'auto'`` prefers ``bbox_scores`` and falls back to
            mean keypoint scores if unavailable. Default: ``'auto'``.
        sigmas (list[float] | None): Per-keypoint sigma values. Defaults to
            COCO-17 sigmas when ``None``.
    """

    online = True

    def __init__(
        self,
        score_thr: float = 0.3,
        oks_thr: float = 0.9,
        score_mode: str = 'auto',
        sigmas: Optional[List[float]] = None,
    ) -> None:
        if score_mode not in ('bbox', 'keypoint', 'auto'):
            raise ValueError(
                "score_mode must be one of 'bbox', 'keypoint', 'auto', "
                f'got {score_mode!r}')
        self.score_thr = float(score_thr)
        self.oks_thr = float(oks_thr)
        self.score_mode = score_mode
        self.sigmas = (
            np.asarray(sigmas, dtype=np.float32)
            if sigmas is not None else _COCO17_SIGMAS)

    def process_frame(
        self,
        ds: PoseDataSample,
        seq_key: str,
    ) -> PoseDataSample:
        instances = ds.pred_instances
        if instances is None or len(instances) == 0:
            return ds

        N = len(instances)
        scores = self._get_scores(instances, N)

        keep_idx = np.where(scores >= self.score_thr)[0]
        if len(keep_idx) > 1 and self.oks_thr < 1.0:
            keep_idx = self._oks_nms(instances, scores, keep_idx)

        keep_idx = np.sort(keep_idx).astype(np.int64)
        return self._select(ds, keep_idx)

    def _get_scores(self, instances, n: int) -> np.ndarray:
        """Return a per-instance confidence score array of shape ``(n,)``."""
        has_bbox = (getattr(instances, 'bbox_scores', None) is not None)
        has_kpt = (getattr(instances, 'keypoint_scores', None) is not None)

        def bbox_scores():
            return np.asarray(
                instances.bbox_scores, dtype=np.float32).reshape(n)

        def keypoint_scores():
            return np.asarray(
                instances.keypoint_scores, dtype=np.float32).mean(axis=1)

        if self.score_mode == 'bbox':
            if not has_bbox:
                raise RuntimeError(
                    "score_mode='bbox' requires pred_instances.bbox_scores.")
            return bbox_scores()
        if self.score_mode == 'keypoint':
            if not has_kpt:
                raise RuntimeError(
                    "score_mode='keypoint' requires "
                    'pred_instances.keypoint_scores.')
            return keypoint_scores()

        # 'auto'
        if has_bbox:
            return bbox_scores()
        if has_kpt:
            return keypoint_scores()
        raise RuntimeError(
            'OKSNMS requires pred_instances.bbox_scores or '
            'pred_instances.keypoint_scores.')

    def _get_areas(self, instances, idx: np.ndarray) -> np.ndarray:
        """Return per-instance areas ``(len(idx),)`` for the given indices."""
        if getattr(instances, 'bboxes', None) is not None:
            bboxes = np.asarray(instances.bboxes, dtype=np.float32)
            if bboxes.ndim == 1:
                bboxes = bboxes[None]
            return np.array(
                [_bbox_area(bboxes[i]) for i in idx], dtype=np.float32)

        # Fallback: bounding box around keypoints
        kpts = np.asarray(instances.keypoints, dtype=np.float32)[idx]
        return np.array([
            float(
                max((kpts[i, :, 0].max() - kpts[i, :, 0].min()) *
                    (kpts[i, :, 1].max() - kpts[i, :, 1].min()), 1.0))
            for i in range(len(idx))
        ], dtype=np.float32)

    def _oks_nms(
        self,
        instances,
        scores: np.ndarray,
        idx: np.ndarray,
    ) -> np.ndarray:
        """Greedily suppress overlapping instances among ``idx``.

        Returns:
            The subset of ``idx`` that survives NMS (unordered).
        """
        kpts = np.asarray(instances.keypoints, dtype=np.float32)[idx]
        areas = self._get_areas(instances, idx)

        K = kpts.shape[1]
        sigmas = self.sigmas
        if sigmas.shape[0] != K:
            sigmas = np.full(K, 0.05, dtype=np.float32)

        oks = _oks_matrix(kpts, areas, sigmas)
        sub_scores = scores[idx]
        order = np.argsort(-sub_scores)

        n = len(idx)
        suppressed = np.zeros(n, dtype=bool)
        keep_local: List[int] = []
        for i in order:
            if suppressed[i]:
                continue
            keep_local.append(int(i))
            overlap = oks[i] >= self.oks_thr
            overlap[i] = False
            suppressed |= overlap

        return idx[np.asarray(keep_local, dtype=np.int64)]

    @staticmethod
    def _select(
        ds: PoseDataSample,
        keep_idx: np.ndarray,
    ) -> PoseDataSample:
        """Return a copy of *ds* with ``pred_instances`` subset to keep_idx."""
        new_ds = ds.new()
        new_ds.set_metainfo(ds.metainfo)
        if hasattr(ds, 'gt_instances'):
            new_ds.gt_instances = ds.gt_instances
        if ds.pred_instances is not None:
            new_ds.pred_instances = ds.pred_instances[keep_idx]
        return new_ds
