# Copyright (c) OpenMMLab. All rights reserved.
"""OKS-based greedy instance tracker."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

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


def _greedy_match(
    prev_kpts: np.ndarray,
    curr_kpts: np.ndarray,
    prev_areas: np.ndarray,
    sigmas: np.ndarray,
    match_thr: float,
    used_curr: Optional[Set[int]] = None,
) -> Tuple[List[Tuple[int, int]], Set[int]]:
    """Greedy OKS matching from *prev* to *curr* instances.

    Returns:
        pairs: list of ``(prev_idx, curr_idx)`` assignments.
        used_curr: set of matched current-instance indices.
    """
    M = prev_kpts.shape[0]
    N = curr_kpts.shape[0]
    if M == 0 or N == 0:
        return [], set(used_curr or ())

    oks = _oks_matrix(prev_kpts, curr_kpts, prev_areas, sigmas)
    used_prev: Set[int] = set()
    used_curr_set: Set[int] = set(used_curr or ())
    pairs: List[Tuple[float, int, int]] = []
    for mi in range(M):
        for ni in range(N):
            if ni in used_curr_set:
                continue
            pairs.append((float(oks[mi, ni]), mi, ni))
    pairs.sort(key=lambda x: x[0], reverse=True)

    matches: List[Tuple[int, int]] = []
    for oks_val, mi, ni in pairs:
        if oks_val < match_thr:
            break
        if mi in used_prev or ni in used_curr_set:
            continue
        used_prev.add(mi)
        used_curr_set.add(ni)
        matches.append((mi, ni))
    return matches, used_curr_set


@dataclass
class _LostTrack:
    """A track that disappeared but is kept for re-identification."""

    kpts: np.ndarray       # (K, 2)
    area: float
    track_id: int
    age: int               # frames since last seen (1 = lost last frame)


def _bbox_area(bbox: np.ndarray) -> float:
    """Return area of a single xyxy bbox."""
    b = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
    return float(max((b[2] - b[0]) * (b[3] - b[1]), 1.0))


@POST_PROCESS_FILTERS.register_module()
class OKSTracker(BaseFilter):
    """Greedy OKS-based instance tracker.

    Assigns a ``track_id`` to each predicted instance by matching it to the
    best-matching instance in the previous frame using OKS.  Tracks that
    disappear can be re-identified within ``remember_frames`` using their
    last known pose.  Unmatched instances receive a new unique id.  State is
    reset at sequence boundaries.

    The assigned ``track_ids`` array is stored on
    ``pred_instances.track_ids (N,)`` as ``np.int32``.

    Args:
        match_thr (float): Minimum OKS to accept a match.  Default: ``0.5``.
        remember_frames (int): Number of frames to keep lost tracks in memory
            for re-identification.  ``0`` disables memory (previous-frame
            matching only).  Default: ``0``.
        sigmas (list[float] | None): Per-keypoint sigma values.  Defaults to
            COCO-17 sigmas when ``None``.
    """

    online = True

    def __init__(
        self,
        match_thr: float = 0.5,
        remember_frames: int = 0,
        sigmas: Optional[List[float]] = None,
    ) -> None:
        self.match_thr = float(match_thr)
        self.remember_frames = int(remember_frames)
        self.sigmas = (
            np.asarray(sigmas, dtype=np.float32)
            if sigmas is not None
            else _COCO17_SIGMAS
        )
        self._prev_kpts: Optional[np.ndarray] = None   # (M, K, 2)
        self._prev_areas: Optional[np.ndarray] = None  # (M,)
        self._prev_track_ids: Optional[np.ndarray] = None  # (M,)
        self._lost_tracks: List[_LostTrack] = []
        self._next_id: int = 0

    def reset(self) -> None:
        self._prev_kpts = None
        self._prev_areas = None
        self._prev_track_ids = None
        self._lost_tracks = []
        self._next_id = 0

    def _age_lost_tracks(self) -> None:
        """Increment age and drop tracks that exceeded the memory window."""
        if self.remember_frames <= 0:
            self._lost_tracks = []
            return
        aged: List[_LostTrack] = []
        for track in self._lost_tracks:
            track.age += 1
            if track.age <= self.remember_frames:
                aged.append(track)
        self._lost_tracks = aged

    def _remove_lost_track_id(self, track_id: int) -> None:
        """Drop a track id from the lost-track buffer."""
        self._lost_tracks = [
            track for track in self._lost_tracks
            if track.track_id != track_id
        ]

    def _register_lost_tracks(
        self,
        kpts: np.ndarray,
        areas: np.ndarray,
        track_ids: np.ndarray,
        matched_prev: Set[int],
    ) -> None:
        """Move unmatched previous-frame tracks into the lost-track buffer."""
        if self.remember_frames <= 0:
            return
        for mi in range(len(track_ids)):
            if mi in matched_prev:
                continue
            track_id = int(track_ids[mi])
            self._remove_lost_track_id(track_id)
            self._lost_tracks.append(_LostTrack(
                kpts=kpts[mi].copy(),
                area=float(areas[mi]),
                track_id=track_id,
                age=1,
            ))

    def _match_lost_tracks(
        self,
        curr_kpts: np.ndarray,
        sigmas: np.ndarray,
        track_ids: np.ndarray,
        used_curr: Set[int],
    ) -> Set[int]:
        """Try to re-assign IDs from recently lost tracks."""
        if self.remember_frames <= 0 or not self._lost_tracks:
            return used_curr

        lost_kpts = np.stack(
            [track.kpts for track in self._lost_tracks], axis=0)
        lost_areas = np.asarray(
            [track.area for track in self._lost_tracks], dtype=np.float32)

        matches, used_curr = _greedy_match(
            lost_kpts, curr_kpts, lost_areas, sigmas,
            self.match_thr, used_curr=used_curr,
        )

        matched_lost: Set[int] = set()
        for lost_idx, curr_idx in matches:
            track_ids[curr_idx] = self._lost_tracks[lost_idx].track_id
            matched_lost.add(lost_idx)

        if matched_lost:
            self._lost_tracks = [
                track for idx, track in enumerate(self._lost_tracks)
                if idx not in matched_lost
            ]
        return used_curr

    def process_frame(
        self,
        ds: PoseDataSample,
        seq_key: str,
    ) -> PoseDataSample:
        instances = ds.pred_instances
        if instances is None or len(instances) == 0:
            self._age_lost_tracks()
            if (self._prev_kpts is not None and self._prev_track_ids is not None
                    and len(self._prev_kpts) > 0):
                self._register_lost_tracks(
                    self._prev_kpts,
                    self._prev_areas,
                    self._prev_track_ids,
                    matched_prev=set(),
                )
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
        used_curr: Set[int] = set()
        matched_prev: Set[int] = set()

        self._age_lost_tracks()

        if self._prev_kpts is not None and len(self._prev_kpts) > 0:
            matches, used_curr = _greedy_match(
                self._prev_kpts, curr_kpts,
                self._prev_areas, sigmas, self.match_thr,
            )
            for mi, ni in matches:
                track_id = int(self._prev_track_ids[mi])
                track_ids[ni] = track_id
                matched_prev.add(mi)
                self._remove_lost_track_id(track_id)

            self._register_lost_tracks(
                self._prev_kpts,
                self._prev_areas,
                self._prev_track_ids,
                matched_prev,
            )

        used_curr = self._match_lost_tracks(
            curr_kpts, sigmas, track_ids, used_curr)

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
