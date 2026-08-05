# Copyright (c) OpenMMLab. All rights reserved.
"""Bipartite-matching pose linker from *Detect-and-Track*.

    Girdhar et al., "Detect-and-Track: Efficient Pose Estimation in Videos",
    CVPR 2018.  https://github.com/facebookresearch/DetectAndTrack

Scope of this integration
-------------------------
Detect-and-Track is a two-stage framework: a 3D Mask R-CNN produces
per-frame person boxes and poses, and a *separate, purely post-hoc linking
stage* stitches those per-frame detections into tracks.  Only the second
stage is a post-processor, and that is what this filter implements - the
detector/pose stage is whatever produced the prediction bundle being
post-processed.  That division matches the upstream design, where tracking
runs offline over a saved ``detections.pkl``.

The linking stage is reproduced from
``external/DetectAndTrack/lib/core/tracking_engine.py``
(``_prune_bad_detections``, ``_compute_distance_matrix``,
``bipartite_matching_greedy``, ``_compute_matches``,
``_compute_tracks_video``) and ``lib/utils/keypoints.py``
(``compute_head_size``, ``pck_distance``).  The upstream module cannot be
imported - it is Python 2 (``import cPickle``), reads a Detectron ``cfg``
singleton at module scope, and pulls in a compiled Cython ``bbox_overlaps``
- so the matching logic is ported, with the submodule kept for diffing.

Fidelity notes
--------------
* All three published cost types are available.  ``cnn-cosdist`` needs the
  frame images (see :class:`TorchvisionCNNEmbedder`); the filter then
  reports ``requires_images``, so its config must declare
  ``needs_images=True``.
* The **LSTM matcher** (``_compute_tracks_video_lstm``) is not implemented:
  its weights were never released, ``TRACKING.LSTM_TEST.LSTM_WEIGHTS``
  defaults to empty, and the ``lstm`` package it imports is absent from the
  repository.
* ``pose-pck`` normalises joint distances by head size, which upstream
  measures between the PoseTrack ``head_top`` and ``head_bottom`` joints.
  COCO-17 has neither, so ``head_keypoint_ids`` selects a stand-in pair
  (the ears by default) and ``min_head_size_ratio`` floors the result at a
  fraction of the box diagonal - without that floor the ear pair collapses
  to ~0 px whenever the head is turned away, which would make every joint
  count as a PCK match.  This is a substitution: the PCK cost is not
  normalised the way the paper's is.
* Upstream applies **no cost gate** - every pair the assignment produces
  becomes a link, however bad.  That is reproduced; ``max_cost`` is an
  optional extension, off by default.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mmpose.structures import PoseDataSample

from ..base import BaseFilter
from ..matchers import build_appearance_embedder
from ..registry import POST_PROCESS_FILTERS

#: Upstream ``MAX_TRACK_IDS`` / ``FIRST_TRACK_ID``.
_MAX_TRACK_IDS = 999
_FIRST_TRACK_ID = 0

#: Cost types implemented here.  Upstream's ``DISTANCE_METRICS`` default is
#: ``('bbox-overlap', 'cnn-cosdist', 'pose-pck')`` with weights
#: ``(1.0, 0.0, 0.0)``, i.e. IoU-only in the released configuration.
_COST_TYPES = ('bbox-overlap', 'pose-pck', 'cnn-cosdist')


def bbox_overlaps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``xyxy`` boxes.

    Port of ``utils.boxes.bbox_overlaps`` as used by
    ``_compute_pairwise_iou`` in the upstream tracking engine (a compiled
    Cython routine there).

    Args:
        a: Boxes ``(M, 4)`` in ``xyxy`` format.
        b: Boxes ``(N, 4)`` in ``xyxy`` format.

    Returns:
        ``(M, N)`` IoU matrix.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    # Cython bbox_overlaps measures extents inclusively (+1 on each side).
    area_a = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

    iw = (np.minimum(a[:, None, 2], b[None, :, 2])
          - np.maximum(a[:, None, 0], b[None, :, 0]) + 1).clip(min=0)
    ih = (np.minimum(a[:, None, 3], b[None, :, 3])
          - np.maximum(a[:, None, 1], b[None, :, 1]) + 1).clip(min=0)

    inter = iw * ih
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def compute_head_size(
    kpts: np.ndarray,
    head_keypoint_ids: Sequence[int],
    bbox: Optional[np.ndarray] = None,
    min_head_size_ratio: float = 0.0,
) -> float:
    """Head size used to normalise the PCK distance.

    Port of ``compute_head_size`` from
    ``external/DetectAndTrack/lib/utils/keypoints.py``, which measures the
    distance between the PoseTrack ``head_top`` and ``head_bottom`` joints
    (``+ 1`` to avoid zeros).

    COCO-17 has no head_top/head_bottom, so ``head_keypoint_ids`` selects
    whichever pair of joints stands in for the head extent (for COCO the
    ears, ``(3, 4)``).  Because that pair collapses to ~0 px whenever the
    head is turned away or mispredicted - which would make *every* joint
    count as a PCK match - ``min_head_size_ratio`` optionally floors the
    head size at a fraction of the box diagonal.  ``0.0`` reproduces
    upstream's unfloored behaviour.

    Args:
        kpts: Pose, shape ``(K, 2)``.
        head_keypoint_ids: The two joint indices spanning the head.
        bbox: ``xyxy`` box of this pose, used for the floor.
        min_head_size_ratio: Floor as a fraction of the box diagonal.

    Returns:
        Head size in pixels (always positive).
    """
    i, j = head_keypoint_ids
    size = float(np.linalg.norm(kpts[i, :2] - kpts[j, :2])) + 1.0

    if min_head_size_ratio > 0 and bbox is not None:
        w = float(bbox[2]) - float(bbox[0])
        h = float(bbox[3]) - float(bbox[1])
        floor = min_head_size_ratio * float(np.hypot(w, h))
        size = max(size, floor)
    return size


def pck_distance(
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    head_size: float,
    dist_thresh: float = 0.5,
) -> float:
    """``1 - PCKh`` between two poses.

    Port of ``pck_distance`` from
    ``external/DetectAndTrack/lib/utils/keypoints.py``: the fraction of
    joints whose head-normalised distance is below ``dist_thresh``,
    subtracted from one so that it behaves as a cost.  Upstream normalises
    by the head size of ``kpts_a`` (the "reference" pose), which here is
    always the previous frame's track.

    Args:
        kpts_a: Reference pose ``(K, 2)``.
        kpts_b: Candidate pose ``(K, 2)``.
        head_size: Head size of ``kpts_a``, see :func:`compute_head_size`.
        dist_thresh: PCK threshold in head-size units.

    Returns:
        ``1 - PCKh`` in ``[0, 1]``.
    """
    normed = np.linalg.norm(kpts_a[:, :2] - kpts_b[:, :2], axis=-1) / head_size
    return float(1.0 - np.sum(normed < dist_thresh) / normed.size)


def bipartite_matching_greedy(cost: np.ndarray) -> Tuple[List[int], List[int]]:
    """Greedy bipartite matching on a cost matrix.

    Port of ``bipartite_matching_greedy`` from
    ``external/DetectAndTrack/lib/core/tracking_engine.py``: repeatedly take
    the globally cheapest remaining cell and strike out its row and column.

    Args:
        cost: ``(M, N)`` cost matrix.

    Returns:
        ``(row_ids, col_ids)`` of the matched pairs, in the order chosen.
    """
    cost = np.asarray(cost).copy()
    row_ids = np.arange(cost.shape[0])
    col_ids = np.arange(cost.shape[1])
    prev_ids: List[int] = []
    cur_ids: List[int] = []

    while cost.size > 0:
        i, j = np.unravel_index(cost.argmin(), cost.shape)
        prev_ids.append(int(row_ids[i]))
        cur_ids.append(int(col_ids[j]))
        cost = np.delete(np.delete(cost, i, 0), j, 1)
        row_ids = np.delete(row_ids, i, 0)
        col_ids = np.delete(col_ids, j, 0)

    return prev_ids, cur_ids


@POST_PROCESS_FILTERS.register_module()
class DetectAndTrackLinker(BaseFilter):
    """Link per-frame detections into tracks by bipartite matching.

    Each frame's detections are matched against the previous frame's
    surviving detections under a weighted sum of pairwise costs, solved
    either greedily or with the Hungarian algorithm.  Detections that match
    nothing start a new track.  Matching only ever looks one frame back:
    there is no lost-track buffer, so a person missed for a single frame
    gets a new id when they reappear.  That is the published behaviour.

    Args:
        cost_types: Any of ``'bbox-overlap'`` (``1 - IoU``), ``'pose-pck'``
            (``1 - PCKh``), ``'cnn-cosdist'`` (cosine distance between crop
            features - needs images).  Upstream lists all three and zeroes
            the weights it doesn't use.
        cost_weights: One weight per entry of ``cost_types``.  A zero weight
            skips that cost entirely (upstream behaviour), so a
            zero-weighted ``'cnn-cosdist'`` costs nothing and loads no
            images.
        bipart_match_algo: ``'hungarian'`` (upstream default,
            ``scipy.optimize.linear_sum_assignment``) or ``'greedy'``.
        conf_filter_initial_dets: Minimum detection score to consider,
            upstream ``TRACKING.CONF_FILTER_INITIAL_DETS`` (``0.9``
            default, ``0.95`` in the released best config).
        min_box_area: Minimum area, after clipping to the image, for a box
            to be kept.  Upstream ``_get_big_size_boxes`` hard-codes ``50``.
        pck_dist_thresh: PCK threshold in head-size units.
        head_keypoint_ids: Joint pair standing in for the head extent; the
            COCO ears by default.  See :func:`compute_head_size`.
        min_head_size_ratio: Floor on head size as a fraction of the box
            diagonal.  ``0.0`` reproduces upstream exactly.
        max_cost: Optional gate - matches costing more than this are
            rejected.  ``None`` (default) reproduces upstream, which
            accepts every assigned pair.
        max_track_ids: Track ids wrap at this value, upstream ``999``.
        score_mode: How to read detection confidence, as in
            :class:`~mmpose.postprocessing.filters.OKSNMS`.
        appearance_embedder: Config for the
            :class:`~mmpose.postprocessing.matchers.BaseAppearanceEmbedder`
            backing ``'cnn-cosdist'``.  Required when that cost has a
            non-zero weight.
    """

    online = True

    def __init__(
        self,
        cost_types: Sequence[str] = ('bbox-overlap', ),
        cost_weights: Sequence[float] = (1.0, ),
        bipart_match_algo: str = 'hungarian',
        conf_filter_initial_dets: float = 0.95,
        min_box_area: float = 50.0,
        pck_dist_thresh: float = 0.5,
        head_keypoint_ids: Sequence[int] = (3, 4),
        min_head_size_ratio: float = 0.1,
        max_cost: Optional[float] = None,
        max_track_ids: Optional[int] = _MAX_TRACK_IDS,
        score_mode: str = 'auto',
        appearance_embedder: Optional[dict] = None,
    ) -> None:
        if len(cost_types) != len(cost_weights):
            raise ValueError(
                f'cost_types ({len(cost_types)}) and cost_weights '
                f'({len(cost_weights)}) must have the same length.')
        unknown = set(cost_types) - set(_COST_TYPES)
        if unknown:
            raise ValueError(
                f'Unknown cost type(s) {sorted(unknown)}; supported: '
                f'{list(_COST_TYPES)}.')
        if bipart_match_algo not in ('greedy', 'hungarian'):
            raise ValueError(
                "bipart_match_algo must be 'greedy' or 'hungarian', got "
                f'{bipart_match_algo!r}')
        if score_mode not in ('bbox', 'keypoint', 'auto'):
            raise ValueError(
                "score_mode must be one of 'bbox', 'keypoint', 'auto', "
                f'got {score_mode!r}')

        self.cost_types = tuple(cost_types)
        self.cost_weights = tuple(float(w) for w in cost_weights)
        self.bipart_match_algo = bipart_match_algo
        self.conf_filter_initial_dets = float(conf_filter_initial_dets)
        self.min_box_area = float(min_box_area)
        self.pck_dist_thresh = float(pck_dist_thresh)
        self.head_keypoint_ids = tuple(int(i) for i in head_keypoint_ids)
        self.min_head_size_ratio = float(min_head_size_ratio)
        self.max_cost = None if max_cost is None else float(max_cost)
        self.max_track_ids = max_track_ids
        self.score_mode = score_mode

        # A zero weight skips the cost entirely, so only a non-zero
        # cnn-cosdist actually needs pixels.
        active = {
            t for t, w in zip(self.cost_types, self.cost_weights) if w != 0
        }
        self._uses_cnn = 'cnn-cosdist' in active
        self.requires_images = self._uses_cnn

        if self._uses_cnn:
            if appearance_embedder is None:
                raise ValueError(
                    "cost type 'cnn-cosdist' has a non-zero weight but no "
                    'appearance_embedder was configured.')
            self.embedder = build_appearance_embedder(appearance_embedder)
        else:
            self.embedder = None

        self._warned_missing_image = False
        self.reset()

    def reset(self) -> None:
        self._prev_bboxes: Optional[np.ndarray] = None
        self._prev_kpts: Optional[np.ndarray] = None
        self._prev_track_ids: Optional[np.ndarray] = None
        self._prev_feats: Optional[np.ndarray] = None
        self._next_track_id = _FIRST_TRACK_ID
        if self.embedder is not None:
            self.embedder.reset()

    # ------------------------------------------------------------------
    # Detection preparation
    # ------------------------------------------------------------------

    def _scores(self, instances, n: int) -> np.ndarray:
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
                'DetectAndTrackLinker requires pred_instances.bbox_scores '
                'or pred_instances.keypoint_scores.')
        return np.asarray(
            instances.keypoint_scores, dtype=np.float32).mean(axis=1)

    def _prune(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        ori_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Port of ``_prune_bad_detections``.

        Keeps only high-confidence boxes that are big enough once clipped
        to the image.  Upstream clips the boxes *in place* and keeps the
        clipped coordinates, so the clipped boxes are returned here too.

        Args:
            bboxes: ``(N, 4)`` boxes in ``xyxy`` format.
            scores: ``(N,)`` detection confidences.
            ori_shape: ``(height, width)`` of the source image.

        Returns:
            ``(keep_idx, clipped_bboxes)`` where ``clipped_bboxes`` covers
            all ``N`` input boxes and ``keep_idx`` indexes the survivors.
        """
        clipped = np.asarray(bboxes, dtype=np.float32).copy()
        h, w = (float(ori_shape[0]), float(ori_shape[1])) if ori_shape else (
            0.0, 0.0)
        if h > 0 and w > 0:
            clipped[:, 0] = np.maximum(clipped[:, 0], 0.0)
            clipped[:, 1] = np.maximum(clipped[:, 1], 0.0)
            clipped[:, 2] = np.minimum(clipped[:, 2], w)
            clipped[:, 3] = np.minimum(clipped[:, 3], h)

        area = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
        keep = np.where((scores >= self.conf_filter_initial_dets)
                        & (area >= self.min_box_area))[0]
        return keep.astype(np.int64), clipped

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _cost_matrix(
        self,
        prev_bboxes: np.ndarray,
        prev_kpts: np.ndarray,
        cur_bboxes: np.ndarray,
        cur_kpts: np.ndarray,
        prev_feats: Optional[np.ndarray],
        cur_feats: Optional[np.ndarray],
    ) -> np.ndarray:
        """Port of ``_compute_distance_matrix``: weighted sum of costs."""
        parts: List[np.ndarray] = []
        for cost_type, weight in zip(self.cost_types, self.cost_weights):
            if weight == 0:
                continue

            if cost_type == 'bbox-overlap':
                cost = 1.0 - bbox_overlaps(prev_bboxes, cur_bboxes)
            elif cost_type == 'pose-pck':
                cost = np.zeros((len(prev_kpts), len(cur_kpts)),
                                dtype=np.float32)
                for i in range(len(prev_kpts)):
                    head_size = compute_head_size(
                        prev_kpts[i], self.head_keypoint_ids,
                        prev_bboxes[i], self.min_head_size_ratio)
                    for j in range(len(cur_kpts)):
                        cost[i, j] = pck_distance(
                            prev_kpts[i], cur_kpts[j], head_size,
                            self.pck_dist_thresh)
            else:  # 'cnn-cosdist'
                if prev_feats is None or cur_feats is None:
                    # No pixels this frame; upstream has no such case, so
                    # fall back to a constant (i.e. uninformative) term
                    # rather than dropping the frame.
                    cost = np.zeros((len(prev_bboxes), len(cur_bboxes)),
                                    dtype=np.float32)
                else:
                    cost = self.embedder.distance_matrix(prev_feats, cur_feats)

            parts.append(cost.astype(np.float32) * weight)

        if not parts:
            return np.zeros((len(prev_bboxes), len(cur_bboxes)),
                            dtype=np.float32)
        return np.sum(np.stack(parts, axis=0), axis=0)

    def _match(self, cost: np.ndarray) -> np.ndarray:
        """Port of ``_compute_matches``.

        Args:
            cost: ``(M, N)`` cost matrix, previous frame by current frame.

        Returns:
            ``(N,)`` array where entry ``j`` is the previous-frame index
            matched to current detection ``j``, or ``-1`` for no match.
        """
        matches = -np.ones((cost.shape[1], ), dtype=np.int32)
        if cost.size == 0:
            return matches

        if self.bipart_match_algo == 'hungarian':
            from scipy.optimize import linear_sum_assignment
            prev_inds, cur_inds = linear_sum_assignment(cost)
        else:
            prev_inds, cur_inds = bipartite_matching_greedy(cost)

        for pi, ci in zip(prev_inds, cur_inds):
            # Upstream applies no gate at all; max_cost is opt-in.
            if self.max_cost is not None and cost[pi, ci] > self.max_cost:
                continue
            matches[ci] = pi
        return matches

    def _new_track_id(self) -> int:
        """Mint a fresh track id, wrapping at ``max_track_ids``."""
        track_id = self._next_track_id
        self._next_track_id += 1
        if (self.max_track_ids is not None
                and self._next_track_id >= self.max_track_ids):
            self._next_track_id %= self.max_track_ids
        return track_id

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
            self._prev_bboxes = None
            self._prev_kpts = None
            self._prev_track_ids = None
            self._prev_feats = None
            return _select(ds, np.zeros(0, dtype=np.int64),
                           np.zeros(0, dtype=np.int32), None)

        bboxes = np.asarray(instances.bboxes, dtype=np.float32).reshape(-1, 4)
        kpts = np.asarray(instances.keypoints, dtype=np.float32)
        scores = self._scores(instances, n)
        ori_shape = ds.metainfo.get('ori_shape', (0, 0))

        keep_idx, clipped = self._prune(bboxes, scores, ori_shape)
        cur_bboxes = clipped[keep_idx]
        cur_kpts = kpts[keep_idx]

        cur_feats: Optional[np.ndarray] = None
        if self._uses_cnn and len(keep_idx) > 0:
            image = ds.get('img', None)
            if image is None:
                if not self._warned_missing_image:
                    print('Warning: DetectAndTrackLinker has cnn-cosdist '
                          'enabled but a frame arrived without pixels; the '
                          'appearance term is inactive for such frames.')
                    self._warned_missing_image = True
            else:
                cur_feats = self.embedder.embed(image, cur_bboxes)

        track_ids = np.full(len(keep_idx), -1, dtype=np.int32)
        if (self._prev_bboxes is not None and len(self._prev_bboxes) > 0
                and len(keep_idx) > 0):
            cost = self._cost_matrix(
                self._prev_bboxes, self._prev_kpts, cur_bboxes, cur_kpts,
                self._prev_feats, cur_feats)
            matches = self._match(cost)
            for j, prev_idx in enumerate(matches):
                if prev_idx >= 0:
                    track_ids[j] = int(self._prev_track_ids[prev_idx])

        for j in range(len(track_ids)):
            if track_ids[j] == -1:
                track_ids[j] = self._new_track_id()

        self._prev_bboxes = cur_bboxes.copy()
        self._prev_kpts = cur_kpts.copy()
        self._prev_track_ids = track_ids.copy()
        self._prev_feats = cur_feats

        return _select(ds, keep_idx, track_ids, clipped)


def _select(
    ds: PoseDataSample,
    keep_idx: np.ndarray,
    track_ids: np.ndarray,
    clipped_bboxes: Optional[np.ndarray],
) -> PoseDataSample:
    """Return a copy of ``ds`` subset to ``keep_idx`` with ``track_ids`` set.

    ``clipped_bboxes`` (all ``N`` input boxes, clipped to the image) is
    written back for the kept instances, mirroring upstream's in-place
    clipping in ``_prune_bad_detections``.
    """
    new_ds = ds.new()
    new_ds.set_metainfo(ds.metainfo)
    if hasattr(ds, 'gt_instances'):
        new_ds.gt_instances = ds.gt_instances

    if ds.pred_instances is not None:
        new_inst = deepcopy(ds.pred_instances[keep_idx])
        if clipped_bboxes is not None and len(keep_idx) > 0:
            new_inst.bboxes = clipped_bboxes[keep_idx]
        new_inst.track_ids = track_ids
        new_ds.pred_instances = new_inst

    return new_ds
