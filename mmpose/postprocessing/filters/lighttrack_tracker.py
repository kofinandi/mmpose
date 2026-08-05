# Copyright (c) OpenMMLab. All rights reserved.
"""LightTrack's two-stage online pose-tracking association.

    Ning et al., "LightTrack: A Generic Framework for Online Top-Down Human
    Pose Tracking", CVPRW 2020.  https://github.com/Guanghan/lighttrack

Scope of this integration
-------------------------
LightTrack is a full top-down framework: a detector runs on *keyframes*, a
single-person pose estimator runs on every box, and on non-keyframes boxes
are propagated from the previous frame's *keypoints* instead of being
re-detected.  Only its data-association stage is a post-processor, and that
is what this filter implements.  Everything upstream of the association -
keyframe scheduling, pose-guided box propagation, the ``is_target_lost``
check that triggers re-detection - needs to re-run a detector or pose
estimator mid-sequence, which a post-processing filter cannot do and which
is out of scope here: detections and poses already exist for every frame.

With a detection on every frame, this is upstream's ``keyframe_interval=1``
limiting case, where every frame is a keyframe and the association stage
sees exactly what it sees here.

Ported from ``external/lighttrack/demo_video_mobile.py``
(``get_track_id_SpatialConsistency``, ``get_track_id_SGCN``,
``get_pose_matching_score``, ``enlarge_bbox``, ``bbox_invalid``, ``iou``,
plus the ``light_track`` main loop's bookkeeping).  Those functions live in
a module whose imports pull in TensorFlow, a YOLOv3 detector and compiled
NMS kernels, so they are ported rather than imported; only the SGCN network
itself is wrapped (see
:class:`~mmpose.postprocessing.matchers.SGCNPoseMatcher`).

Upstream quirks that are deliberately preserved: the ``+1``-inflated IoU,
integer truncation of the enlarged box margins, and the ``[0, 0, 2, 2]``
sentinel for degenerate boxes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Set

import numpy as np

from mmpose.structures import PoseDataSample

from ..base import BaseFilter
from ..matchers import build_pose_matcher
from ..registry import POST_PROCESS_FILTERS


def enlarge_bbox(bbox: np.ndarray, scale: float) -> np.ndarray:
    """Grow an ``xyxy`` box by ``scale`` around its centre.

    Port of ``enlarge_bbox`` from
    ``external/lighttrack/demo_video_mobile.py``: each side moves out by
    half of ``scale`` times the box's extent (with the margin truncated to
    an integer, as upstream), and a box that ends up degenerate or larger
    than 2000 px collapses to the sentinel ``[0, 0, 2, 2]`` that
    :func:`bbox_invalid` rejects.

    Args:
        bbox: ``xyxy`` box, shape ``(4,)``.
        scale: Fraction of the box's width/height to add on each axis.

    Returns:
        Enlarged ``xyxy`` box, shape ``(4,)``.
    """
    min_x, min_y, max_x, max_y = (float(v) for v in bbox[:4])
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_x < 0:
        margin_x = 2
    if margin_y < 0:
        margin_y = 2

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    width = max_x - min_x
    height = max_y - min_y
    if (max_y < 0 or max_x < 0 or width <= 0 or height <= 0
            or width > 2000 or height > 2000):
        return np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32)

    return np.array([min_x, min_y, max_x, max_y], dtype=np.float32)


def bbox_invalid(bbox: np.ndarray, max_size: float = 2000.0) -> bool:
    """Whether a box is the sentinel or has a degenerate/oversized extent.

    Port of ``bbox_invalid``; upstream works in ``xywh`` and tests the
    width/height entries, which here are derived from the ``xyxy`` box.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    if [x1, y1, x2, y2] == [0.0, 0.0, 2.0, 2.0]:
        return True
    w = x2 - x1
    h = y2 - y1
    return w <= 0 or h <= 0 or w > max_size or h > max_size


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two ``xyxy`` boxes.  Port of ``iou`` from upstream.

    Keeps upstream's ``+1`` inflation of every extent.
    """
    xa = max(float(box_a[0]), float(box_b[0]))
    ya = max(float(box_a[1]), float(box_b[1]))
    xb = min(float(box_a[2]), float(box_b[2]))
    yb = min(float(box_a[3]), float(box_b[3]))

    inter = max(0.0, xb - xa + 1) * max(0.0, yb - ya + 1)
    area_a = (float(box_a[2]) - float(box_a[0]) + 1) * (
        float(box_a[3]) - float(box_a[1]) + 1)
    area_b = (float(box_b[2]) - float(box_b[0]) + 1) * (
        float(box_b[3]) - float(box_b[1]) + 1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


@POST_PROCESS_FILTERS.register_module()
class LightTrackTracker(BaseFilter):
    """Associate detections to tracks by spatial consistency, then pose.

    Two passes over the current frame's detections, in detection order:

    1. **Spatial consistency** - greedily take the previous-frame candidate
       with the highest IoU, if that IoU clears
       ``spatial_consistency_thr``.
    2. **Pose matching** - for detections still unassigned, take the
       remaining candidate with the smallest pose distance, if it is within
       ``pose_matching_threshold``.

    A previous-frame candidate consumed by either pass is removed from the
    pool, so it cannot be claimed twice.  Anything still unmatched starts a
    new track.  Matching only ever looks one frame back: there is no
    lost-track buffer, which is the published behaviour.

    Args:
        pose_matcher: Config for the
            :class:`~mmpose.postprocessing.matchers.BasePoseMatcher` used by
            pass 2.  LightTrack's own matcher is
            :class:`~mmpose.postprocessing.matchers.SGCNPoseMatcher`.
        pose_matching_threshold: Maximum pose distance for a pass-2 match,
            upstream ``0.5`` for the SGCN's embedding distance.  Must be
            re-tuned for any other matcher, whose distances live on a
            different scale.
        spatial_consistency_thr: Minimum IoU for a pass-1 match; upstream
            hard-codes ``0.3``.
        enlarge_scale: Box enlargement before matching, upstream ``0.2``.
        score_thr: Drop detections scoring below this before association.
            Upstream's detector threshold is ``0.4``.
        score_mode: How to read detection confidence, as in
            :class:`~mmpose.postprocessing.filters.OKSNMS`.
        max_bbox_size: Boxes wider or taller than this are invalid,
            upstream ``2000``.
        drop_invalid: Drop detections whose enlarged box is invalid, as the
            upstream loop does.  When ``False`` they are kept and given
            fresh track ids.
    """

    online = True
    requires_images = False

    def __init__(
        self,
        pose_matcher: dict,
        pose_matching_threshold: float = 0.5,
        spatial_consistency_thr: float = 0.3,
        enlarge_scale: float = 0.2,
        score_thr: float = 0.4,
        score_mode: str = 'auto',
        max_bbox_size: float = 2000.0,
        drop_invalid: bool = True,
    ) -> None:
        if score_mode not in ('bbox', 'keypoint', 'auto'):
            raise ValueError(
                "score_mode must be one of 'bbox', 'keypoint', 'auto', "
                f'got {score_mode!r}')

        self.pose_matcher = build_pose_matcher(pose_matcher)
        self.pose_matching_threshold = float(pose_matching_threshold)
        self.spatial_consistency_thr = float(spatial_consistency_thr)
        self.enlarge_scale = float(enlarge_scale)
        self.score_thr = float(score_thr)
        self.score_mode = score_mode
        self.max_bbox_size = float(max_bbox_size)
        self.drop_invalid = bool(drop_invalid)
        self.reset()

    def reset(self) -> None:
        self._prev_bboxes: Optional[np.ndarray] = None
        self._prev_kpts: Optional[np.ndarray] = None
        self._prev_scores: Optional[np.ndarray] = None
        self._prev_track_ids: Optional[np.ndarray] = None
        self._next_id = 0
        self.pose_matcher.reset()

    def _det_scores(self, instances, n: int) -> np.ndarray:
        """Per-instance detection confidence, shape ``(n,)``."""
        has_bbox = getattr(instances, 'bbox_scores', None) is not None
        has_kpt = getattr(instances, 'keypoint_scores', None) is not None

        if self.score_mode == 'bbox' or (self.score_mode == 'auto'
                                         and has_bbox):
            if not has_bbox:
                raise RuntimeError(
                    "score_mode='bbox' requires pred_instances.bbox_scores.")
            return np.asarray(
                instances.bbox_scores, dtype=np.float32).reshape(n)
        if not has_kpt:
            raise RuntimeError(
                'LightTrackTracker requires pred_instances.bbox_scores or '
                'pred_instances.keypoint_scores.')
        return np.asarray(
            instances.keypoint_scores, dtype=np.float32).mean(axis=1)

    # ------------------------------------------------------------------
    # Association passes
    # ------------------------------------------------------------------

    def _spatial_consistency_pass(
        self,
        cur_bboxes: np.ndarray,
        track_ids: np.ndarray,
        pool: List[int],
    ) -> None:
        """First pass: greedy per-detection IoU against the previous frame.

        Port of ``get_track_id_SpatialConsistency``, called in detection
        order by the ``light_track`` main loop.  Mutates ``track_ids`` and
        removes each consumed candidate from ``pool``.

        Args:
            cur_bboxes: Enlarged current-frame boxes, ``(N, 4)``.
            track_ids: ``(N,)`` ids to fill; ``-1`` means unmatched.
            pool: Indices into the previous frame's candidates that are
                still available.
        """
        for i in range(len(cur_bboxes)):
            best_iou = 0.0
            best_slot = -1
            for slot, prev_idx in enumerate(pool):
                score = iou(cur_bboxes[i], self._prev_bboxes[prev_idx])
                if score > best_iou:
                    best_iou = score
                    best_slot = slot
            if best_iou > self.spatial_consistency_thr and best_slot >= 0:
                prev_idx = pool.pop(best_slot)
                track_ids[i] = int(self._prev_track_ids[prev_idx])

    def _pose_matching_pass(
        self,
        cur_bboxes: np.ndarray,
        cur_kpts: np.ndarray,
        cur_scores: Optional[np.ndarray],
        track_ids: np.ndarray,
        pool: List[int],
    ) -> None:
        """Second pass: pose similarity over what the first pass missed.

        Port of ``get_track_id_SGCN``: for each still-unmatched detection,
        take the remaining candidate with the lowest pose distance, subject
        to ``distance <= pose_matching_threshold``.
        """
        unmatched = [i for i in range(len(cur_bboxes)) if track_ids[i] == -1]
        if not unmatched or not pool:
            return

        for i in unmatched:
            if not pool:
                break
            prev_idx = np.asarray(pool, dtype=np.int64)
            dists = self.pose_matcher.distance_matrix(
                cur_kpts[i][None], cur_bboxes[i][None],
                self._prev_kpts[prev_idx], self._prev_bboxes[prev_idx],
                None if cur_scores is None else cur_scores[i][None],
                None if self._prev_scores is None
                else self._prev_scores[prev_idx],
            )[0]
            best_slot = int(np.argmin(dists))
            if dists[best_slot] <= self.pose_matching_threshold:
                consumed = pool.pop(best_slot)
                track_ids[i] = int(self._prev_track_ids[consumed])

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        ds: PoseDataSample,
        seq_key: str,
    ) -> PoseDataSample:
        instances = ds.pred_instances
        n = 0 if instances is None else len(instances)

        if n == 0:
            self._clear_prev()
            return _select(ds, np.zeros(0, dtype=np.int64),
                           np.zeros(0, dtype=np.int32))

        raw_bboxes = np.asarray(
            instances.bboxes, dtype=np.float32).reshape(-1, 4)
        kpts = np.asarray(instances.keypoints, dtype=np.float32)
        kpt_scores = getattr(instances, 'keypoint_scores', None)
        if kpt_scores is not None:
            kpt_scores = np.asarray(kpt_scores, dtype=np.float32)
        det_scores = self._det_scores(instances, n)

        enlarged = np.stack(
            [enlarge_bbox(b, self.enlarge_scale) for b in raw_bboxes])

        keep = [
            i for i in range(n)
            if det_scores[i] >= self.score_thr
            and not (self.drop_invalid
                     and bbox_invalid(enlarged[i], self.max_bbox_size))
        ]
        keep_idx = np.asarray(keep, dtype=np.int64)

        if len(keep_idx) == 0:
            self._clear_prev()
            return _select(ds, keep_idx, np.zeros(0, dtype=np.int32))

        cur_bboxes = enlarged[keep_idx]
        cur_kpts = kpts[keep_idx]
        cur_scores = None if kpt_scores is None else kpt_scores[keep_idx]

        track_ids = np.full(len(keep_idx), -1, dtype=np.int32)
        if self._prev_bboxes is not None and len(self._prev_bboxes) > 0:
            pool = list(range(len(self._prev_bboxes)))
            self._spatial_consistency_pass(cur_bboxes, track_ids, pool)
            self._pose_matching_pass(
                cur_bboxes, cur_kpts, cur_scores, track_ids, pool)

        for i in range(len(track_ids)):
            if track_ids[i] == -1:
                track_ids[i] = self._next_id
                self._next_id += 1

        self._prev_bboxes = cur_bboxes.copy()
        self._prev_kpts = cur_kpts.copy()
        self._prev_scores = None if cur_scores is None else cur_scores.copy()
        self._prev_track_ids = track_ids.copy()

        return _select(ds, keep_idx, track_ids)

    def _clear_prev(self) -> None:
        self._prev_bboxes = None
        self._prev_kpts = None
        self._prev_scores = None
        self._prev_track_ids = None


def _select(
    ds: PoseDataSample,
    keep_idx: np.ndarray,
    track_ids: np.ndarray,
) -> PoseDataSample:
    """Copy of ``ds`` subset to ``keep_idx``, with ``track_ids`` set."""
    new_ds = ds.new()
    new_ds.set_metainfo(ds.metainfo)
    if hasattr(ds, 'gt_instances'):
        new_ds.gt_instances = ds.gt_instances
    if ds.pred_instances is not None:
        new_inst = deepcopy(ds.pred_instances[keep_idx])
        new_inst.track_ids = track_ids
        new_ds.pred_instances = new_inst
    return new_ds
