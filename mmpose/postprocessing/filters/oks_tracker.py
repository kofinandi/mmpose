# Copyright (c) OpenMMLab. All rights reserved.
"""OKS-based greedy instance tracker."""

from __future__ import annotations

import copy
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
from mmengine.structures import InstanceData

from mmpose.structures import PoseDataSample

from ..base import BaseFilter
from ..registry import POST_PROCESS_FILTERS

# Default COCO-17 sigmas (same as mmpose/evaluation/functional/nms.py)
_COCO17_SIGMAS = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
    1.07, 1.07, .87, .87, .89, .89,
], dtype=np.float32) / 10.0


def _oks_matrix(
    prev_kpts: np.ndarray,    # (M, K, 2)
    curr_kpts: np.ndarray,    # (N, K, 2)
    prev_areas: np.ndarray,   # (M,)
    sigmas: np.ndarray,       # (K,)
) -> np.ndarray:              # (M, N)
    """Compute pairwise OKS between M previous and N current instances.

    Uses only position (no visibility masking) since we are matching
    predictions to predictions, not predictions to GT.  Every keypoint
    contributes equally.

    The OKS formula is:
        OKS(m, n) = mean_k( exp(-d_mk^2 / (2 * area_m * (2*sigma_k)^2)) )
    """
    K = sigmas.shape[0]
    vars_ = (sigmas * 2.0) ** 2  # (K,)

    M = prev_kpts.shape[0]
    N = curr_kpts.shape[0]

    oks = np.zeros((M, N), dtype=np.float32)
    for m in range(M):
        area = float(prev_areas[m]) + np.spacing(1)
        dx = curr_kpts[:, :, 0] - prev_kpts[m:m+1, :, 0]  # (N, K)
        dy = curr_kpts[:, :, 1] - prev_kpts[m:m+1, :, 1]  # (N, K)
        d2 = dx ** 2 + dy ** 2                              # (N, K)
        e = d2 / (vars_[None, :] * (2.0 * area))           # (N, K)
        oks[m] = np.mean(np.exp(-e), axis=1)               # (N,)
    return oks


def _bbox_area(bbox: np.ndarray) -> float:
    """Return area of a single xyxy bbox."""
    b = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
    return float(max((b[2] - b[0]) * (b[3] - b[1]), 1.0))


@POST_PROCESS_FILTERS.register_module()
class OKSTracker(BaseFilter):
    """Greedy OKS-based instance tracker.

    Assigns a ``track_id`` to each predicted instance by matching it to the
    best-matching instance in the previous frame using OKS.  Unmatched
    instances receive a new unique id.  State is reset at sequence boundaries.

    The assigned ``track_ids`` array is stored on
    ``pred_instances.track_ids (N,)`` as ``np.int32``.

    Args:
        match_thr (float): Minimum OKS to accept a match.  Default: ``0.5``.
        sigmas (list[float] | None): Per-keypoint sigma values.  Defaults to
            COCO-17 sigmas when ``None``.
    """

    online = True

    def __init__(
        self,
        match_thr: float = 0.5,
        sigmas: Optional[List[float]] = None,
    ) -> None:
        self.match_thr = float(match_thr)
        self.sigmas = (
            np.asarray(sigmas, dtype=np.float32)
            if sigmas is not None
            else _COCO17_SIGMAS
        )
        self._prev_kpts: Optional[np.ndarray] = None   # (M, K, 2)
        self._prev_areas: Optional[np.ndarray] = None  # (M,)
        self._prev_track_ids: Optional[np.ndarray] = None  # (M,)
        self._next_id: int = 0

    def reset(self) -> None:
        self._prev_kpts = None
        self._prev_areas = None
        self._prev_track_ids = None
        self._next_id = 0

    def process_frame(
        self,
        ds: PoseDataSample,
        seq_key: str,
    ) -> PoseDataSample:
        instances = ds.pred_instances
        if instances is None or len(instances) == 0:
            ds = self._set_track_ids(ds, np.empty(0, dtype=np.int32))
            self._prev_kpts = None
            self._prev_areas = None
            self._prev_track_ids = None
            return ds

        curr_kpts = np.asarray(instances.keypoints, dtype=np.float32)  # (N, K, 2)
        N = curr_kpts.shape[0]
        K = curr_kpts.shape[1]

        # Compute areas from bboxes when available
        if hasattr(instances, 'bboxes') and instances.bboxes is not None:
            bboxes = np.asarray(instances.bboxes, dtype=np.float32)
            if bboxes.ndim == 1:
                bboxes = bboxes[None]
            curr_areas = np.array(
                [_bbox_area(bboxes[i]) for i in range(N)], dtype=np.float32)
        else:
            # Fallback: bounding box around keypoints
            curr_areas = np.array([
                float(max(
                    (curr_kpts[i, :, 0].max() - curr_kpts[i, :, 0].min()) *
                    (curr_kpts[i, :, 1].max() - curr_kpts[i, :, 1].min()), 1.0
                )) for i in range(N)
            ], dtype=np.float32)

        # Adapt sigmas to actual K
        sigmas = self.sigmas
        if sigmas.shape[0] != K:
            sigmas = np.full(K, 0.05, dtype=np.float32)

        track_ids = np.full(N, -1, dtype=np.int32)

        if self._prev_kpts is not None and len(self._prev_kpts) > 0:
            oks = _oks_matrix(
                self._prev_kpts, curr_kpts,
                self._prev_areas, sigmas,
            )  # (M, N)
            # Greedy matching in descending OKS order
            used_prev: set = set()
            used_curr: set = set()
            pairs: List[Tuple[float, int, int]] = []
            M = oks.shape[0]
            for mi in range(M):
                for ni in range(N):
                    pairs.append((float(oks[mi, ni]), mi, ni))
            pairs.sort(key=lambda x: x[0], reverse=True)

            for oks_val, mi, ni in pairs:
                if oks_val < self.match_thr:
                    break
                if mi in used_prev or ni in used_curr:
                    continue
                used_prev.add(mi)
                used_curr.add(ni)
                track_ids[ni] = int(self._prev_track_ids[mi])

        # Assign new IDs to unmatched instances
        for ni in range(N):
            if track_ids[ni] == -1:
                track_ids[ni] = self._next_id
                self._next_id += 1

        # Store state for next frame
        self._prev_kpts = curr_kpts.copy()
        self._prev_areas = curr_areas.copy()
        self._prev_track_ids = track_ids.copy()

        return self._set_track_ids(ds, track_ids)

    @staticmethod
    def _set_track_ids(
        ds: PoseDataSample,
        track_ids: np.ndarray,
    ) -> PoseDataSample:
        """Return a copy of *ds* with track_ids set on pred_instances."""
        new_ds = ds.new()
        new_ds.set_metainfo(ds.metainfo)
        if hasattr(ds, 'gt_instances'):
            new_ds.gt_instances = ds.gt_instances
        if ds.pred_instances is not None:
            # deepcopy ensures _data_fields registry is independent so the
            # track_ids field doesn't leak back to the original ds.
            new_inst = deepcopy(ds.pred_instances)
            new_inst.track_ids = track_ids
            new_ds.pred_instances = new_inst
        return new_ds
