# Copyright (c) OpenMMLab. All rights reserved.
"""PGPT's pose-guided tracking-by-detection association cascade.

    Zhang et al., "Pose-Guided Tracking-by-Detection: Robust Multi-Person
    Pose Tracking", IEEE TMM 2020.  https://github.com/JDAI-CV/PGPT

Scope of this integration
-------------------------
PGPT uses pose information in two places: video human *detection*, via a
pose-guided single-object tracker (SiamFC) that invents candidate boxes for
people the detector missed; and *data association*, via a two-stage cascade
that falls back from box IoU to a learned appearance embedding.  The
association cascade is a post-processor and is reproduced here from
``external/PGPT/inference/track_and_detect_new.py``
(``Track_And_Detect.match_detection_tracking_oks_iou_embedding``,
``oks_filter``, ``create_id``) and ``inference/match.py``
(``Matcher.oks_iou``, ``oks_nms``, ``iou``, ``distance``,
``associate_detections_to_trackers_iou`` /
``..._embedding``).

Those modules are ported rather than imported: ``track_and_detect_new``
pulls in a compiled ``model.nms.nms_wrapper`` extension built for
``torch.utils.ffi``, ``match.py`` imports ``linear_assignment`` from
``sklearn.utils.linear_assignment_`` (removed in scikit-learn 0.23), and
both rely on ``inference/`` being the working directory.  Only the PoseGCN
network is wrapped - see
:class:`~mmpose.postprocessing.matchers.PGPTPoseGCNEmbedder`.

Fidelity notes
--------------
* **ABSENT - SiamFC track propagation.**  Upstream merges each live track's
  single-object-tracker box into the candidate list (score decayed by
  0.35) before association, so a person the detector misses is still
  carried.  Tracker-generated boxes are out of scope here, so this filter
  associates real detections only.  Combined with upstream's immediate
  deletion of unmatched tracks, a person occluded for even one frame gets
  a new id - that is what this configuration measures.
* **SUBSTITUTED - inputs to the pose-dependent gates.**  ``oks_filter`` and
  ``create_id`` re-run PGPT's pose network on every candidate box; here the
  keypoints and scores already attached to the detection are used instead.
* Hungarian assignment uses :func:`scipy.optimize.linear_sum_assignment` in
  place of the removed sklearn ``linear_assignment``; both return an
  optimal assignment, and they can differ only in how ties are broken.
* Upstream's ``embedding_match_thresh=2`` is a cosine-distance gate.  Real
  PoseGCN distances are ~0.005, so as published the gate never rejects
  anything; it is kept at the upstream default and documented rather than
  silently retuned.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from mmengine.structures import InstanceData

from mmpose.structures import PoseDataSample

from ..base import BaseFilter
from ..matchers import (COCO17_TO_PGPT15, PGPT15_SIGMAS,
                        build_appearance_embedder, convert_keypoints,
                        convert_scores)
from ..matchers.pgpt_embedder import pgpt_area
from ..registry import POST_PROCESS_FILTERS


def oks_iou(
    kpts_g: np.ndarray,
    kpts_d: np.ndarray,
    area_g: float,
    areas_d: np.ndarray,
    sigmas: np.ndarray,
    scores_g: Optional[np.ndarray] = None,
    scores_d: Optional[np.ndarray] = None,
    in_vis_thre: Optional[float] = None,
) -> np.ndarray:
    """OKS between one pose and a set of poses.

    Port of ``Matcher.oks_iou``; note that upstream normalises by the mean
    of the two areas, not just the reference one.

    Args:
        kpts_g: Reference pose ``(K, 2)``.
        kpts_d: Candidate poses ``(N, K, 2)``.
        area_g: Reference area.
        areas_d: Candidate areas ``(N,)``.
        sigmas: Per-keypoint sigmas ``(K,)``.
        scores_g: Reference per-keypoint scores ``(K,)``, used only when
            ``in_vis_thre`` is set.
        scores_d: Candidate per-keypoint scores ``(N, K)``.
        in_vis_thre: Restrict the mean to joints visible in both poses.

    Returns:
        ``(N,)`` OKS values.
    """
    if len(kpts_d) == 0:
        return np.zeros(0, dtype=np.float32)

    variances = (sigmas * 2)**2
    out = np.zeros(len(kpts_d), dtype=np.float32)
    for n in range(len(kpts_d)):
        d = kpts_d[n] - kpts_g
        e = ((d**2).sum(axis=-1) / variances
             / ((area_g + areas_d[n]) / 2 + np.spacing(1)) / 2)
        if in_vis_thre is not None and scores_g is not None:
            # Upstream writes `list(vg > t) and list(vd > t)`, which in
            # Python evaluates to the second list whenever the first is
            # non-empty - i.e. only the candidate's visibility is used.
            ind = scores_d[n] > in_vis_thre
            e = e[ind]
        out[n] = float(np.sum(np.exp(-e)) / e.shape[0]) if e.size else 0.0
    return out


def bbox_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``xyxy`` boxes.

    Port of ``Matcher.iou``, vectorised over both sets.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    iw = (np.minimum(a[:, None, 2], b[None, :, 2])
          - np.maximum(a[:, None, 0], b[None, :, 0])).clip(min=0)
    ih = (np.minimum(a[:, None, 3], b[None, :, 3])
          - np.maximum(a[:, None, 1], b[None, :, 1])).clip(min=0)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(union > 0, inter / union, 0.0).astype(np.float32)


@dataclass
class _Candidate:
    """One detection considered for association this frame."""

    bbox: np.ndarray          # xyxy
    bbox_score: float
    keypoints: np.ndarray     # (17, 2), source layout
    keypoint_scores: np.ndarray
    kpts15: np.ndarray        # (15, 2), PGPT layout
    scores15: np.ndarray
    area: float
    pose_score: float         # oks_filter's re-scored confidence
    index: int                # index into the frame's pred_instances


@dataclass
class _Track:
    """A live identity."""

    track_id: int
    bbox: np.ndarray
    score: float
    feature: Optional[np.ndarray] = field(default=None)


@POST_PROCESS_FILTERS.register_module()
class PGPTTracker(BaseFilter):
    """Associate detections to tracks by OKS-NMS, then IoU, then appearance.

    Per frame, following upstream:

    1. Re-score every detection as ``mean(joint scores >= keypoint_score_thr)
       * box score`` and run OKS-NMS over the result (``oks_filter``).
    2. Hungarian-assign survivors to live tracks on box IoU, discarding
       pairs below ``iou_match_thresh``.
    3. Hungarian-assign the leftovers on appearance cosine distance,
       discarding pairs above ``embedding_match_thresh``.
    4. A detection still unmatched starts a track only if it clears the
       ``create_id`` admission rule; a track still unmatched is deleted
       immediately.

    Args:
        appearance_embedder: Config for the
            :class:`~mmpose.postprocessing.matchers.BaseAppearanceEmbedder`
            backing stage 3, normally
            :class:`~mmpose.postprocessing.matchers.PGPTPoseGCNEmbedder`.
            ``None`` disables stage 3 (and with it the need for images),
            leaving the geometric skeleton of the cascade.
        iou_match_thresh: Stage-2 gate, upstream ``0.5``.
        embedding_match_thresh: Stage-3 gate on cosine distance, upstream
            ``2`` - see the module docstring, this never rejects.
        oks_thresh: OKS-NMS suppression threshold, upstream ``0.8``.
        oks_in_vis_thre: Visibility threshold inside OKS, upstream ``0.2``.
        effective_detection_thresh: Minimum box score to start a track,
            upstream ``0.5``.
        effective_keypoints_thresh: Joint-score threshold in the admission
            rule, upstream ``0.6``.
        effective_keypoints_number: How many joints must clear it, upstream
            ``8``.
        keypoint_score_thr: Joint-score threshold for the re-scoring in
            ``oks_filter``, upstream ``0.2``.
        sigmas: OKS sigmas in PGPT's 15-joint order.  Defaults to
            upstream's PoseTrack sigmas.
        score_mode: How to read detection confidence, as in
            :class:`~mmpose.postprocessing.filters.OKSNMS`.
    """

    online = True

    def __init__(
        self,
        appearance_embedder: Optional[dict] = None,
        iou_match_thresh: float = 0.5,
        embedding_match_thresh: float = 2.0,
        oks_thresh: float = 0.8,
        oks_in_vis_thre: float = 0.2,
        effective_detection_thresh: float = 0.5,
        effective_keypoints_thresh: float = 0.6,
        effective_keypoints_number: int = 8,
        keypoint_score_thr: float = 0.2,
        sigmas: Optional[List[float]] = None,
        score_mode: str = 'auto',
    ) -> None:
        if score_mode not in ('bbox', 'keypoint', 'auto'):
            raise ValueError(
                "score_mode must be one of 'bbox', 'keypoint', 'auto', "
                f'got {score_mode!r}')

        self.iou_match_thresh = float(iou_match_thresh)
        self.embedding_match_thresh = float(embedding_match_thresh)
        self.oks_thresh = float(oks_thresh)
        self.oks_in_vis_thre = float(oks_in_vis_thre)
        self.effective_detection_thresh = float(effective_detection_thresh)
        self.effective_keypoints_thresh = float(effective_keypoints_thresh)
        self.effective_keypoints_number = int(effective_keypoints_number)
        self.keypoint_score_thr = float(keypoint_score_thr)
        self.score_mode = score_mode
        self.sigmas = (
            np.asarray(sigmas, dtype=np.float32)
            if sigmas is not None else PGPT15_SIGMAS)

        if appearance_embedder is not None:
            self.embedder = build_appearance_embedder(appearance_embedder)
        else:
            self.embedder = None
        self.requires_images = self.embedder is not None

        self._warned_missing_image = False
        self.reset()

    def reset(self) -> None:
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 0
        if self.embedder is not None:
            self.embedder.reset()

    # ------------------------------------------------------------------
    # Candidate preparation
    # ------------------------------------------------------------------

    def _pose_score(
        self,
        kpt_scores: np.ndarray,
        bbox_score: float,
    ) -> float:
        """PGPT's candidate score from ``oks_filter``.

        ``mean(joint scores >= keypoint_score_thr) * box score``, and ``0``
        when no joint clears the threshold.
        """
        valid = kpt_scores[kpt_scores >= self.keypoint_score_thr]
        if valid.size == 0:
            return 0.0
        return float(valid.mean() * bbox_score)

    def _detections(self, ds: PoseDataSample) -> List[_Candidate]:
        """Current frame's detections, as candidates."""
        instances = ds.pred_instances
        n = 0 if instances is None else len(instances)
        if n == 0:
            return []

        bboxes = np.asarray(
            instances.bboxes, dtype=np.float32).reshape(-1, 4)
        kpts = np.asarray(instances.keypoints, dtype=np.float32)
        kpt_scores = np.asarray(
            instances.keypoint_scores, dtype=np.float32)

        has_bbox = getattr(instances, 'bbox_scores', None) is not None
        if self.score_mode == 'keypoint' or (self.score_mode == 'auto'
                                             and not has_bbox):
            bbox_scores = kpt_scores.mean(axis=1)
        else:
            if not has_bbox:
                raise RuntimeError(
                    "score_mode='bbox' requires pred_instances.bbox_scores.")
            bbox_scores = np.asarray(
                instances.bbox_scores, dtype=np.float32).reshape(n)

        kpts15 = convert_keypoints(kpts, COCO17_TO_PGPT15)
        scores15 = convert_scores(kpt_scores, COCO17_TO_PGPT15)

        return [
            _Candidate(
                bbox=bboxes[i],
                bbox_score=float(bbox_scores[i]),
                keypoints=kpts[i],
                keypoint_scores=kpt_scores[i],
                kpts15=kpts15[i],
                scores15=scores15[i],
                area=pgpt_area(bboxes[i]),
                pose_score=self._pose_score(scores15[i], float(bbox_scores[i])),
                index=i,
            ) for i in range(n)
        ]

    def _oks_nms(self, cands: List[_Candidate]) -> List[_Candidate]:
        """Port of ``Matcher.oks_nms`` over the candidate list."""
        if not cands:
            return []

        scores = np.array([c.pose_score for c in cands], dtype=np.float32)
        kpts = np.stack([c.kpts15 for c in cands])
        areas = np.array([c.area for c in cands], dtype=np.float32)
        kpt_scores = np.stack([c.scores15 for c in cands])

        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            rest = order[1:]
            if rest.size == 0:
                break
            ovr = oks_iou(
                kpts[i], kpts[rest], areas[i], areas[rest], self.sigmas,
                kpt_scores[i], kpt_scores[rest], self.oks_in_vis_thre)
            order = rest[np.where(ovr <= self.oks_thresh)[0]]

        return [cands[i] for i in keep]

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _assign(
        cost: np.ndarray,
        gate: float,
        maximize: bool,
    ) -> List[Tuple[int, int]]:
        """Hungarian assignment with a per-pair gate.

        Mirrors ``associate_detections_to_trackers_*``, which run
        ``linear_assignment`` and then discard pairs on the wrong side of
        the threshold.

        Args:
            cost: ``(N_det, N_track)`` score (``maximize``) or distance.
            gate: Pairs below (``maximize``) or above the gate are dropped.
            maximize: Whether higher is better.

        Returns:
            Accepted ``(det_idx, track_idx)`` pairs.
        """
        if cost.size == 0:
            return []
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-cost if maximize else cost)
        out = []
        for r, c in zip(rows, cols):
            value = cost[r, c]
            if (value < gate) if maximize else (value > gate):
                continue
            out.append((int(r), int(c)))
        return out

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        ds: PoseDataSample,
        seq_key: str,
    ) -> PoseDataSample:
        cands = self._oks_nms(self._detections(ds))

        track_ids = sorted(self._tracks)
        matched_det: Dict[int, int] = {}   # candidate index -> track id

        # ── Stage 1: box IoU ────────────────────────────────────────────
        unmatched_dets = list(range(len(cands)))
        unmatched_tracks = list(track_ids)
        if cands and track_ids:
            iou = bbox_iou_matrix(
                np.stack([c.bbox for c in cands]),
                np.stack([self._tracks[t].bbox for t in track_ids]))
            pairs = self._assign(iou, self.iou_match_thresh, maximize=True)
            for di, ti in pairs:
                matched_det[di] = track_ids[ti]
            unmatched_dets = [
                i for i in range(len(cands)) if i not in matched_det]
            claimed = set(matched_det.values())
            unmatched_tracks = [t for t in track_ids if t not in claimed]

        # ── Stage 2: appearance ─────────────────────────────────────────
        image = ds.get('img', None)
        if (self.embedder is not None and unmatched_dets and unmatched_tracks
                and image is None and not self._warned_missing_image):
            print('Warning: PGPTTracker has an appearance embedder but a '
                  'frame arrived without pixels; the embedding stage is '
                  'inactive for such frames.')
            self._warned_missing_image = True

        det_feats: Dict[int, np.ndarray] = {}
        if (self.embedder is not None and image is not None
                and unmatched_dets):
            feats = self.embedder.embed(
                image, np.stack([cands[i].bbox for i in unmatched_dets]))
            det_feats = dict(zip(unmatched_dets, feats))

            usable_tracks = [
                t for t in unmatched_tracks
                if self._tracks[t].feature is not None
            ]
            if usable_tracks:
                dist = self.embedder.distance_matrix(
                    feats,
                    np.stack([self._tracks[t].feature for t in usable_tracks]))
                pairs = self._assign(
                    dist, self.embedding_match_thresh, maximize=False)
                for di, ti in pairs:
                    matched_det[unmatched_dets[di]] = usable_tracks[ti]
                claimed = set(matched_det.values())
                unmatched_dets = [
                    i for i in range(len(cands)) if i not in matched_det]
                unmatched_tracks = [
                    t for t in unmatched_tracks if t not in claimed]

        # ── Admission and deletion ──────────────────────────────────────
        for i in unmatched_dets:
            cand = cands[i]
            enough_joints = int(np.sum(
                cand.scores15 >= self.effective_keypoints_thresh))
            if (cand.bbox_score >= self.effective_detection_thresh
                    and enough_joints >= self.effective_keypoints_number):
                matched_det[i] = self._next_id
                self._next_id += 1

        for track_id in unmatched_tracks:
            # Upstream deletes an unmatched track immediately.
            self._tracks.pop(track_id, None)

        # ── Commit ──────────────────────────────────────────────────────
        emitted: List[Tuple[int, int]] = []   # (pred index, track id)
        for i, track_id in sorted(matched_det.items()):
            cand = cands[i]
            self._tracks[track_id] = _Track(
                track_id=track_id,
                bbox=cand.bbox.copy(),
                score=cand.pose_score,
                feature=det_feats.get(i, self._tracks[track_id].feature
                                      if track_id in self._tracks else None),
            )
            emitted.append((cand.index, track_id))

        return _build_output(ds, emitted)


def _build_output(
    ds: PoseDataSample,
    emitted: List[Tuple[int, int]],
) -> PoseDataSample:
    """Emit the surviving detections with their track ids."""
    new_ds = ds.new()
    new_ds.set_metainfo(ds.metainfo)
    if hasattr(ds, 'gt_instances'):
        new_ds.gt_instances = ds.gt_instances

    if ds.pred_instances is None:
        return new_ds

    if not emitted:
        empty = deepcopy(ds.pred_instances[np.zeros(0, dtype=np.int64)])
        empty.track_ids = np.zeros(0, dtype=np.int32)
        new_ds.pred_instances = empty
        return new_ds

    emitted.sort(key=lambda e: e[1])
    keep = np.array([e[0] for e in emitted], dtype=np.int64)
    ids = np.array([e[1] for e in emitted], dtype=np.int32)

    new_inst = deepcopy(ds.pred_instances[keep])
    new_inst.track_ids = ids
    new_ds.pred_instances = new_inst
    return new_ds
