# Copyright (c) OpenMMLab. All rights reserved.
"""SORT-style online tracker + smoother built around a swappable next-step
predictor (motion model).

Architecture (see ``configs/post_processing/gp_kalman_sort.py``):

1. **Predict** - the owned predictor produces a per-keypoint
   ``(mean, variance, age)`` for every currently tracked instance.
2. **Data association** - detections are matched to track predictions with
   the Hungarian algorithm on an OKS cost matrix (computed only over each
   track's currently "alive" keypoints, see below), gated by ``match_thr``.
3. **Output computation** - matched keypoints are fused (Bayesian update of
   the prediction with the detection); unmatched-but-alive keypoints output
   the raw prediction; dead keypoints output score ``0``; new tracks output
   the raw detection.
4. **Track management** - matched tracks are updated (observed keypoints
   only), unmatched tracks are aged, and new tracks are registered with the
   predictor.

Resolution-agnostic predictor coordinates
------------------------------------------
The owned predictor (e.g. :class:`GPKalmanPredictor`) never sees raw pixel
coordinates. Every keypoint handed to :meth:`BasePredictor.add_track` /
:meth:`BasePredictor.update` is first normalized to the ``[0, 1]`` range by
dividing by the current frame's image width/height (from
``ds.metainfo['ori_shape']``), and every ``Prediction.mean`` read back from
:meth:`BasePredictor.predict` is scaled back to pixel coordinates before use.
This makes any predictor implementation tolerant of the input image size
without needing to know about it itself.

Variances are handled the same way, but only for predictors that declare
``predictor.var_is_normalized = True`` (the :class:`BasePredictor` default):
for those, the ``variances`` passed into :meth:`add_track`/:meth:`update`
and the ``Prediction.var`` read back from :meth:`predict` are assumed to
scale like coordinate\ :sup:`2`, and are divided/multiplied by
``scale_x * scale_y`` alongside ``mean``. Predictors that set
``var_is_normalized = False``, whose variance is intrinsically decoupled
from the coordinate values are left untouched, so this stays a no-op for
them and keeps their output numerically identical regardless of image size.
Ages, OKS, and everything else downstream of the predictor continue to be
expressed in the same (pixel) units as before.

Partial-occlusion handling
---------------------------
Every keypoint has its own age (frames since it was last *observed*, i.e.
had a detection score >= ``keypoint_score_thr``).  A keypoint becomes
"dead" - excluded from association and shown with score ``0`` - once its
age exceeds ``keypoint_max_age``.  This lets a track survive an instance being
partially occluded (e.g. legs behind an object) while only the occluded
keypoints go stale; a dead keypoint revives automatically the moment it is
observed again.  The whole instance is only dropped once *every* keypoint's
age exceeds ``instance_max_age``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment

from mmpose.structures import PoseDataSample

from ..base import BaseFilter
from ..measurement import build_measurement_model
from ..predictors import Prediction, build_predictor
from ..registry import POST_PROCESS_FILTERS

# Default COCO-17 sigmas (same as mmpose/evaluation/functional/nms.py)
_COCO17_SIGMAS = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
    1.07, 1.07, .87, .87, .89, .89,
], dtype=np.float32) / 10.0


def _bbox_area(bbox: np.ndarray) -> float:
    b = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
    return float(max((b[2] - b[0]) * (b[3] - b[1]), 1.0))


def _bbox_from_keypoints(kpts: np.ndarray) -> np.ndarray:
    """Fallback bbox (xyxy) enclosing a single instance's keypoints."""
    x0, y0 = kpts[:, 0].min(), kpts[:, 1].min()
    x1, y1 = kpts[:, 0].max(), kpts[:, 1].max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _to_normalized(
    kpts: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Normalize keypoint coordinates ``(K, 2)`` to ``[0, 1]`` by dividing
    by the image width/height, for consumption by the predictor."""
    out = np.asarray(kpts, dtype=np.float64).copy()
    out[:, 0] /= scale_x
    out[:, 1] /= scale_y
    return out


def _to_pixels(
    mean: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Inverse of :func:`_to_normalized`: scale a predictor's normalized
    ``Prediction.mean`` back to pixel coordinates."""
    out = np.asarray(mean, dtype=np.float32).copy()
    out[:, 0] *= scale_x
    out[:, 1] *= scale_y
    return out


@dataclass
class _TrackMeta:
    """Post-processor-side per-track bookkeeping (not owned by the predictor)."""

    last_bbox: np.ndarray
    last_bbox_score: float
    last_area: float
    last_scores: np.ndarray          # (K,) last observed detection score
    prev_innovation: np.ndarray      # (K, 2) previous fusion innovation
    hits: int = 1                    # number of frames matched to a detection


@POST_PROCESS_FILTERS.register_module()
class PredictiveTracker(BaseFilter):
    """SORT-style tracker/smoother driven by a swappable next-step predictor.

    Args:
        predictor (dict): Config for the predictor submodule (registered in
            ``POST_PROCESS_PREDICTORS``), e.g.
            ``dict(type='GPKalmanPredictor', num_keypoints=17, ...)``.
        measurement_model (dict): Config for the measurement-noise model
            submodule (registered in ``POST_PROCESS_MEASUREMENT_MODELS``)
            that converts a detection's keypoint confidence score (and, at
            fusion time, its innovation against the prediction) into a
            measurement variance ``R``, e.g.
            ``dict(type='PowerScoreMeasurementModel', pixel_scale=1.0,
            min_r=3e-4, score_exp=8.0, inflation_factor=8.0,
            osc_inflate=2.0)``. This is inherently heuristic and how a
            score maps to trustworthiness varies a lot between detector
            architectures, so it is fully swappable/tunable per model
            rather than hard-coded - see
            :class:`~mmpose.postprocessing.measurement.BaseMeasurementModel`.
        match_thr (float): Minimum OKS to accept an association.
        sigmas (list[float] | None): Per-keypoint OKS sigmas. Defaults to
            COCO-17 sigmas when ``None``.
        keypoint_score_thr (float): Minimum detection confidence for a
            keypoint to count as "observed" this frame.
        keypoint_max_age (float): A keypoint becomes dead (no longer
            predicted/matched/output with a real score) once its age
            (frames since last observed) exceeds this value.
        instance_max_age (float): The whole track is discarded once every
            keypoint's age exceeds this value.
        min_hits_to_remember (int): Number of frames a track must have been
            matched to a detection before it is trusted enough to be
            "remembered" (predicted/extrapolated) while lost. A track that
            goes unmatched before reaching this many hits is dropped
            immediately instead of lingering in the lost state - this
            filters out one-off phantom detections that would otherwise be
            predicted for up to ``instance_max_age`` frames after a single
            spurious detection. Default: ``1`` (any track, including a
            single-frame detection, gets remembered).
    """

    online = True

    def __init__(
        self,
        predictor: dict,
        measurement_model: dict,
        match_thr: float = 0.5,
        sigmas: Optional[List[float]] = None,
        keypoint_score_thr: float = 0.3,
        keypoint_max_age: float = 15,
        instance_max_age: float = 30,
        min_hits_to_remember: int = 1,
    ) -> None:
        self.predictor = build_predictor(predictor)
        self.measurement_model = build_measurement_model(measurement_model)
        self.match_thr = float(match_thr)
        self.sigmas = (
            np.asarray(sigmas, dtype=np.float32)
            if sigmas is not None else _COCO17_SIGMAS)

        self.keypoint_score_thr = float(keypoint_score_thr)
        self.keypoint_max_age = float(keypoint_max_age)
        self.instance_max_age = float(instance_max_age)
        self.min_hits_to_remember = int(min_hits_to_remember)

        self._tracks: Dict[int, _TrackMeta] = {}
        self._next_id: int = 0

    def reset(self) -> None:
        self.predictor.reset()
        self._tracks = {}
        self._next_id = 0

    # ------------------------------------------------------------------
    # Resolution-agnostic predictor coordinates
    # ------------------------------------------------------------------

    @staticmethod
    def _image_scale(ds: PoseDataSample) -> tuple:
        """Return ``(scale_x, scale_y)`` = image ``(width, height)`` used to
        normalize/denormalize keypoint coordinates around calls into the
        predictor (see module docstring).

        Falls back to ``(1.0, 1.0)`` (i.e. no normalization, matching the
        previous pixel-coordinate behaviour) when ``ori_shape`` is missing
        or degenerate.
        """
        ori_shape = ds.metainfo.get('ori_shape', None)
        if ori_shape is not None and len(ori_shape) >= 2:
            h, w = float(ori_shape[0]), float(ori_shape[1])
            if h > 0 and w > 0:
                return w, h
        return 1.0, 1.0

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------

    def _oks_row(
        self,
        track_mean: np.ndarray,   # (K, 2)
        track_area: float,
        alive: np.ndarray,        # (K,) bool
        det_kpts: np.ndarray,     # (N, K, 2)
        sigmas: np.ndarray,       # (K,)
    ) -> np.ndarray:
        """OKS between one track prediction and every detection, computed
        only over the track's currently alive keypoints."""
        n = det_kpts.shape[0]
        if not alive.any() or n == 0:
            return np.zeros(n, dtype=np.float32)

        area = float(track_area) + np.spacing(1)
        vars_ = (sigmas[alive] * 2.0) ** 2                    # (K',)
        dx = det_kpts[:, alive, 0] - track_mean[None, alive, 0]  # (N, K')
        dy = det_kpts[:, alive, 1] - track_mean[None, alive, 1]  # (N, K')
        d2 = dx ** 2 + dy ** 2
        e = d2 / (vars_[None, :] * (2.0 * area))
        return np.mean(np.exp(-e), axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_frame(
        self,
        ds: PoseDataSample,
        seq_key: str,
    ) -> PoseDataSample:
        K = self.predictor.num_keypoints
        instances = ds.pred_instances

        # ── Gather detections ───────────────────────────────────────────
        if instances is not None and len(instances) > 0:
            det_kpts = np.asarray(instances.keypoints, dtype=np.float32)
            det_scores = np.asarray(instances.keypoint_scores, dtype=np.float32)
            n_det = det_kpts.shape[0]
            if det_kpts.shape[1] != K:
                raise RuntimeError(
                    f'PredictiveTracker configured for {K} keypoints but '
                    f'detections have {det_kpts.shape[1]}.')

            if getattr(instances, 'bboxes', None) is not None:
                det_bboxes = np.asarray(
                    instances.bboxes, dtype=np.float32).reshape(n_det, 4)
            else:
                det_bboxes = np.stack(
                    [_bbox_from_keypoints(det_kpts[i])
                     for i in range(n_det)])

            if getattr(instances, 'bbox_scores', None) is not None:
                det_bbox_scores = np.asarray(
                    instances.bbox_scores, dtype=np.float32).reshape(n_det)
            else:
                det_bbox_scores = np.ones(n_det, dtype=np.float32)

            det_areas = np.array(
                [_bbox_area(det_bboxes[i]) for i in range(n_det)],
                dtype=np.float32)
        else:
            n_det = 0
            det_kpts = np.zeros((0, K, 2), dtype=np.float32)
            det_scores = np.zeros((0, K), dtype=np.float32)
            det_bboxes = np.zeros((0, 4), dtype=np.float32)
            det_bbox_scores = np.zeros((0,), dtype=np.float32)
            det_areas = np.zeros((0,), dtype=np.float32)

        sigmas = self.sigmas
        if sigmas.shape[0] != K:
            sigmas = np.full(K, 0.05, dtype=np.float32)

        scale_x, scale_y = self._image_scale(ds)
        # `var` scales like coordinate^2, but only for predictors that
        # actually express it in the same normalized units as `mean` (see
        # module docstring and `BasePredictor.var_is_normalized`).
        var_scale = (
            scale_x * scale_y if self.predictor.var_is_normalized else 1.0)

        # ── Step 1: Predict ─────────────────────────────────────────────
        active_ids = list(self._tracks.keys())
        raw_predictions: Dict[int, Prediction] = (
            self.predictor.predict(active_ids) if active_ids else {})
        # The predictor works in normalized [0, 1] coordinates; scale its
        # prediction means (and, if applicable, variances) back to pixels
        # so everything below operates in pixel space exactly as before.
        predictions: Dict[int, Prediction] = {
            tid: Prediction(
                mean=_to_pixels(pred.mean, scale_x, scale_y),
                var=pred.var * var_scale,
                age=pred.age)
            for tid, pred in raw_predictions.items()
        }

        alive_masks: Dict[int, np.ndarray] = {
            tid: (pred.age <= self.keypoint_max_age)
            for tid, pred in predictions.items()
        }

        # ── Step 2: Data association (Hungarian on OKS) ─────────────────
        matchable_ids = [
            tid for tid in active_ids if alive_masks[tid].any()
        ]
        oks = np.zeros((len(matchable_ids), n_det), dtype=np.float32)
        for mi, tid in enumerate(matchable_ids):
            oks[mi] = self._oks_row(
                predictions[tid].mean, self._tracks[tid].last_area,
                alive_masks[tid], det_kpts, sigmas)

        matched_pairs: List[tuple] = []
        matched_track_ids: set = set()
        matched_det_idx: set = set()
        if oks.shape[0] > 0 and oks.shape[1] > 0:
            row_ind, col_ind = linear_sum_assignment(-oks)
            for r, c in zip(row_ind, col_ind):
                if oks[r, c] >= self.match_thr:
                    tid = matchable_ids[r]
                    matched_pairs.append((tid, int(c)))
                    matched_track_ids.add(tid)
                    matched_det_idx.add(int(c))

        unmatched_track_ids = [
            tid for tid in active_ids if tid not in matched_track_ids
        ]
        unmatched_det_idx = [
            i for i in range(n_det) if i not in matched_det_idx
        ]

        out_records: List[dict] = []

        # ── Step 3 & 4: Matched tracks - fuse, update, manage ───────────
        for tid, ni in matched_pairs:
            record = self._process_matched(
                tid, ni, predictions[tid], alive_masks[tid],
                det_kpts, det_scores, det_bboxes, det_bbox_scores, det_areas,
                scale_x, scale_y, var_scale)
            if record is not None:
                out_records.append(record)

        # ── Lost tracks - extrapolate, age, manage ──────────────────────
        for tid in unmatched_track_ids:
            record = self._process_lost(
                tid, predictions.get(tid), alive_masks.get(tid))
            if record is not None:
                out_records.append(record)

        # ── New tracks - register, output raw detection ─────────────────
        for ni in unmatched_det_idx:
            out_records.append(self._process_new(
                ni, det_kpts, det_scores, det_bboxes, det_bbox_scores,
                det_areas, K, scale_x, scale_y, var_scale))

        return self._build_output(ds, out_records, K)

    # ------------------------------------------------------------------
    # Per-branch handlers
    # ------------------------------------------------------------------

    def _process_matched(
        self,
        tid: int,
        ni: int,
        pred: Prediction,
        alive: np.ndarray,
        det_kpts: np.ndarray,
        det_scores: np.ndarray,
        det_bboxes: np.ndarray,
        det_bbox_scores: np.ndarray,
        det_areas: np.ndarray,
        scale_x: float,
        scale_y: float,
        var_scale: float,
    ) -> Optional[dict]:
        track = self._tracks[tid]

        z = det_kpts[ni].astype(np.float64)     # (K, 2)
        s = det_scores[ni]                      # (K,)
        observed = s >= self.keypoint_score_thr  # (K,)

        mu_p = pred.mean.astype(np.float64)      # (K, 2)
        var_p = pred.var.astype(np.float64)      # (K,)
        innov = z - mu_p                         # (K, 2)

        base_var = self.measurement_model.variance(s.astype(np.float64))
        r_var = self.measurement_model.inflate(
            base_var, innov, track.prev_innovation, observed)  # (K,)

        kg = var_p / (var_p + r_var)             # (K,)
        mu_post = mu_p + kg[:, None] * innov     # (K, 2)

        out_kpts = np.where(observed[:, None], mu_post, mu_p).astype(
            np.float32)
        out_scores = np.where(
            observed, s,
            np.where(alive, track.last_scores, 0.0)).astype(np.float32)
        upd_kpts = np.where(observed[:, None], z, 0.0)
        upd_vars = np.where(observed, r_var, 0.0)
        new_age = np.where(observed, 0.0, pred.age + 1.0).astype(np.float32)

        track.prev_innovation = np.where(observed[:, None], innov,
                                          track.prev_innovation)
        track.last_scores = np.where(observed, s,
                                      track.last_scores).astype(np.float32)

        self.predictor.update(
            tid, _to_normalized(upd_kpts, scale_x, scale_y),
            upd_vars / var_scale, valid_mask=observed)

        track.last_bbox = det_bboxes[ni]
        track.last_bbox_score = float(det_bbox_scores[ni])
        track.last_area = float(det_areas[ni])
        track.hits += 1

        if np.all(new_age > self.instance_max_age):
            self.predictor.remove_track(tid)
            del self._tracks[tid]
            return None

        return dict(
            track_id=tid, kpts=out_kpts, scores=out_scores,
            bbox=track.last_bbox, bbox_score=track.last_bbox_score)

    def _process_lost(
        self,
        tid: int,
        pred: Optional[Prediction],
        alive: Optional[np.ndarray],
    ) -> Optional[dict]:
        track = self._tracks[tid]
        if pred is None:
            # Defensive: predictor state missing for a tracked id.
            self.predictor.remove_track(tid)
            del self._tracks[tid]
            return None

        if track.hits < self.min_hits_to_remember:
            # Not established enough to be "remembered" through occlusion
            # - drop immediately rather than extrapolating a likely
            # phantom detection for up to instance_max_age frames.
            self.predictor.remove_track(tid)
            del self._tracks[tid]
            return None

        self.predictor.update(tid, None, None)

        new_age = pred.age + 1.0
        if np.all(new_age > self.instance_max_age):
            self.predictor.remove_track(tid)
            del self._tracks[tid]
            return None

        out_kpts = pred.mean.astype(np.float32).copy()
        out_scores = np.where(alive, track.last_scores, 0.0).astype(np.float32)

        return dict(
            track_id=tid, kpts=out_kpts, scores=out_scores,
            bbox=track.last_bbox, bbox_score=track.last_bbox_score)

    def _process_new(
        self,
        ni: int,
        det_kpts: np.ndarray,
        det_scores: np.ndarray,
        det_bboxes: np.ndarray,
        det_bbox_scores: np.ndarray,
        det_areas: np.ndarray,
        K: int,
        scale_x: float,
        scale_y: float,
        var_scale: float,
    ) -> dict:
        tid = self._next_id
        self._next_id += 1

        kpts = det_kpts[ni]
        scores = det_scores[ni].copy()
        variances = self.measurement_model.variance(scores)

        self.predictor.add_track(
            tid, _to_normalized(kpts, scale_x, scale_y), variances / var_scale)
        self._tracks[tid] = _TrackMeta(
            last_bbox=det_bboxes[ni],
            last_bbox_score=float(det_bbox_scores[ni]),
            last_area=float(det_areas[ni]),
            last_scores=scores,
            prev_innovation=np.zeros((K, 2), dtype=np.float64),
        )

        return dict(
            track_id=tid, kpts=kpts.astype(np.float32), scores=scores,
            bbox=det_bboxes[ni], bbox_score=float(det_bbox_scores[ni]))

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _build_output(
        ds: PoseDataSample,
        records: List[dict],
        num_keypoints: int,
    ) -> PoseDataSample:
        new_ds = ds.new()
        new_ds.set_metainfo(ds.metainfo)
        if hasattr(ds, 'gt_instances'):
            new_ds.gt_instances = ds.gt_instances

        records = sorted(records, key=lambda r: r['track_id'])
        n = len(records)

        pred = InstanceData()
        if n == 0:
            pred.keypoints = np.zeros((0, num_keypoints, 2), dtype=np.float32)
            pred.keypoint_scores = np.zeros((0, num_keypoints), dtype=np.float32)
            pred.bboxes = np.zeros((0, 4), dtype=np.float32)
            pred.bbox_scores = np.zeros(0, dtype=np.float32)
            pred.track_ids = np.zeros(0, dtype=np.int32)
        else:
            pred.keypoints = np.stack(
                [r['kpts'] for r in records]).astype(np.float32)
            pred.keypoint_scores = np.stack(
                [r['scores'] for r in records]).astype(np.float32)
            pred.bboxes = np.stack(
                [r['bbox'] for r in records]).astype(np.float32)
            pred.bbox_scores = np.array(
                [r['bbox_score'] for r in records], dtype=np.float32)
            pred.track_ids = np.array(
                [r['track_id'] for r in records], dtype=np.int32)

        new_ds.pred_instances = pred
        return new_ds
