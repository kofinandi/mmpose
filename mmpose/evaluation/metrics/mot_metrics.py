# Copyright (c) OpenMMLab. All rights reserved.
"""Standard multi-object-tracking metrics: MOTA, IDF1 and HOTA.

The three metrics answer different questions about the same track ids that
:class:`~mmpose.evaluation.metrics.tracking_metrics.IDSwitch` already
inspects, and they are the numbers the tracking literature reports:

* **MOTA** (CLEAR-MOT, Bernardin & Stiefelhagen 2008) - a *detection*-
  centric error rate: one penalty per missed GT, per false positive and
  per identity switch, all divided by the number of GT detections.  It is
  unbounded below and is dominated by detection errors.
* **IDF1** (Ristani et al., ECCVW 2016) - an *identity*-centric F1: GT and
  predicted trajectories are matched one-to-one over the whole sequence,
  and the score is the F1 over correctly identified detections.  It rewards
  keeping one id on one person for as long as possible.
* **HOTA** (Luiten et al., IJCV 2021) - ``sqrt(DetA * AssA)`` averaged over
  localisation thresholds, explicitly balancing detection against
  association instead of letting one dominate.

All three are implemented to match the reference implementation in
`TrackEval <https://github.com/JonathonLuiten/TrackEval>`_
(``trackeval/metrics/clear.py``, ``identity.py``, ``hota.py``), including
its cross-sequence combination rules: count fields are summed over
sequences, and HOTA's association scores are recombined as a
``HOTA_TP``-weighted average.

Pose-tracking adaptation
------------------------
TrackEval scores boxes and uses IoU as the similarity.  These metrics
default to **OKS** instead, so they agree with the rest of this package
(:class:`~mmpose.evaluation.metrics.IDSwitch` and the temporal metrics all
match on OKS) and actually measure pose tracking; set ``similarity='iou'``
for the box-based MOT convention.

There is no equivalent of TrackEval's dataset-specific preprocessing, which
on MOTChallenge removes predictions that match "distractor" annotation
classes.  On a dataset that annotates only some of the people in frame
(EMDB annotates a single subject), every correctly-detected bystander
counts as a false positive.  That mostly hurts MOTA, which is a raw error
rate over detections; ``score_thr`` is exposed so those can be traded off
against recall, but the honest reading is that MOTA on such a dataset
measures the detector's agreement with the annotation policy at least as
much as it measures tracking.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.logging import MMLogger
from scipy.optimize import linear_sum_assignment

from mmpose.registry import METRICS
from ..functional.frame_metrics import compute_oks_pairs
from .temporal_keypoint_metrics import _TemporalBaseMetric
from .tracking_metrics import _frame_order_key, _sequence_key_from_path

#: HOTA's localisation thresholds, ``np.arange(0.05, 0.99, 0.05)`` in
#: TrackEval (19 values from 0.05 to 0.95).
HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)

_EPS = float(np.finfo('float').eps)


def bbox_iou_matrix(gt_bboxes: np.ndarray, pred_bboxes: np.ndarray
                    ) -> np.ndarray:
    """Pairwise IoU between two sets of ``xyxy`` boxes.

    Args:
        gt_bboxes: ``(M, 4)`` ground-truth boxes.
        pred_bboxes: ``(N, 4)`` predicted boxes.

    Returns:
        ``(M, N)`` IoU matrix.
    """
    m, n = gt_bboxes.shape[0], pred_bboxes.shape[0]
    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float64)

    inter_w = (np.minimum(gt_bboxes[:, None, 2], pred_bboxes[None, :, 2]) -
               np.maximum(gt_bboxes[:, None, 0], pred_bboxes[None, :, 0]))
    inter_h = (np.minimum(gt_bboxes[:, None, 3], pred_bboxes[None, :, 3]) -
               np.maximum(gt_bboxes[:, None, 1], pred_bboxes[None, :, 1]))
    inter = inter_w.clip(min=0.0) * inter_h.clip(min=0.0)

    area_gt = ((gt_bboxes[:, 2] - gt_bboxes[:, 0]) *
               (gt_bboxes[:, 3] - gt_bboxes[:, 1]))
    area_pred = ((pred_bboxes[:, 2] - pred_bboxes[:, 0]) *
                 (pred_bboxes[:, 3] - pred_bboxes[:, 1]))
    union = area_gt[:, None] + area_pred[None, :] - inter
    return inter / np.maximum(union, _EPS)


@dataclass
class SequenceData:
    """One video's tracking data, in the layout TrackEval's metrics expect.

    Ids are re-indexed to ``0..num_*_ids - 1`` *within the sequence*, which
    is what lets the metrics use them directly as array indices.

    Attributes:
        gt_ids: Per frame, the ``(M_t,)`` dense GT ids present.
        pred_ids: Per frame, the ``(N_t,)`` dense predicted ids present.
        similarity: Per frame, the ``(M_t, N_t)`` similarity matrix.
        num_gt_ids: Number of distinct GT ids in the sequence.
        num_pred_ids: Number of distinct predicted ids in the sequence.
        num_gt_dets: Total GT detections over the sequence.
        num_pred_dets: Total predicted detections over the sequence.
    """

    gt_ids: List[np.ndarray]
    pred_ids: List[np.ndarray]
    similarity: List[np.ndarray]
    num_gt_ids: int
    num_pred_ids: int
    num_gt_dets: int
    num_pred_dets: int


class _MOTBaseMetric(_TemporalBaseMetric):
    """Shared frame accumulation for the MOT-style tracking metrics.

    :meth:`process` records, per frame, the GT/predicted track ids present
    and their pairwise similarity; :meth:`group_sequences` then reassembles
    those records into per-video :class:`SequenceData`, which is the form
    every metric below consumes.

    Frames whose predictions or ground truth lack ``track_ids`` contribute
    nothing but are still counted, so a pipeline with no tracker evaluates
    to zero rather than crashing - the same graceful degradation as
    :class:`~mmpose.evaluation.metrics.IDSwitch`.

    Args:
        match_thr (float): Similarity threshold for a true positive.  Used
            by MOTA and IDF1; HOTA sweeps its own thresholds and ignores
            this.  Default: ``0.5``.
        similarity (str): ``'oks'`` (default) or ``'iou'``; see the module
            docstring.
        score_thr (float): Drop predicted instances scoring below this
            before matching.  Default: ``0.0`` (keep everything).
        score_mode (str): Where the prediction score comes from:
            ``'bbox'``, ``'keypoint'`` (mean over joints) or ``'auto'``
            (prefer bbox, fall back to keypoints).  Default: ``'auto'``.
        collect_device (str): Device for distributed result collection.
            Default: ``'cpu'``.
        prefix (str, optional): Metric name prefix.  Default: ``None``
            (uses ``'tracking'``).
    """

    default_prefix: Optional[str] = 'tracking'

    def __init__(self,
                 match_thr: float = 0.5,
                 similarity: str = 'oks',
                 score_thr: float = 0.0,
                 score_mode: str = 'auto',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        if similarity not in ('oks', 'iou'):
            raise ValueError(
                f"similarity must be 'oks' or 'iou', got {similarity!r}")
        if score_mode not in ('bbox', 'keypoint', 'auto'):
            raise ValueError(
                "score_mode must be one of 'bbox', 'keypoint', 'auto', got "
                f'{score_mode!r}')
        super().__init__(
            match_thr=match_thr, collect_device=collect_device, prefix=prefix)
        self.similarity = similarity
        self.score_thr = float(score_thr)
        self.score_mode = score_mode
        self._frame_counter = 0

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def _pred_scores(self, pred: dict, n_pred: int) -> Optional[np.ndarray]:
        """Per-instance prediction score, or ``None`` if unavailable."""
        bbox_scores = pred.get('bbox_scores')
        kpt_scores = pred.get('keypoint_scores')
        if self.score_mode in ('bbox', 'auto') and bbox_scores is not None:
            return np.asarray(bbox_scores, dtype=np.float64).reshape(n_pred)
        if self.score_mode in ('keypoint', 'auto') and kpt_scores is not None:
            return np.asarray(kpt_scores, dtype=np.float64).mean(axis=1)
        return None

    def _similarity_matrix(
        self,
        gt: dict,
        pred: dict,
        gt_idx: List[int],
        pred_idx: List[int],
        pairs: List,
    ) -> np.ndarray:
        """Dense ``(len(gt_idx), len(pred_idx))`` similarity matrix.

        For ``'oks'`` the values come straight from the pairs produced by
        :func:`~mmpose.evaluation.functional.frame_metrics.compute_oks_pairs`;
        for ``'iou'`` they are recomputed from the boxes.
        """
        gt_pos = {g: i for i, g in enumerate(gt_idx)}
        pred_pos = {p: i for i, p in enumerate(pred_idx)}
        sim = np.zeros((len(gt_idx), len(pred_idx)), dtype=np.float64)

        if self.similarity == 'oks':
            for oks, gi, pi in pairs:
                if gi in gt_pos and pi in pred_pos:
                    sim[gt_pos[gi], pred_pos[pi]] = float(oks)
            return sim

        gt_bboxes = gt.get('bboxes')
        pred_bboxes = pred.get('bboxes')
        if gt_bboxes is None or pred_bboxes is None:
            return sim
        gt_b = np.asarray(gt_bboxes, dtype=np.float64).reshape(-1, 4)[gt_idx]
        pred_b = np.asarray(
            pred_bboxes, dtype=np.float64).reshape(-1, 4)[pred_idx]
        return bbox_iou_matrix(gt_b, pred_b)

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Record one entry per frame: ids present plus their similarities.

        Nothing is matched here.  MOTA needs the previous frame's
        assignment, and IDF1/HOTA need whole-sequence statistics, so all
        three resolve their matching in ``compute_metrics`` once the frames
        have been sorted back into chronological order per video.
        """
        sigmas = np.asarray(
            self.dataset_meta.get('sigmas', []), dtype=np.float32)

        for data_sample in data_samples:
            self._frame_counter += 1

            img_path = data_sample.get('img_path', '')
            record = {
                'seq_key': _sequence_key_from_path(img_path),
                'frame_order': _frame_order_key(img_path,
                                                self._frame_counter),
                'gt_ids': np.zeros(0, dtype=np.int64),
                'pred_ids': np.zeros(0, dtype=np.int64),
                'similarity': np.zeros((0, 0), dtype=np.float64),
            }

            pred = data_sample.get('pred_instances')
            gt = data_sample.get('gt_instances')
            if pred is None or gt is None or len(sigmas) == 0:
                self.results.append(record)
                continue

            pred_track_ids = pred.get('track_ids')
            gt_track_ids = gt.get('track_ids')
            if pred_track_ids is None or gt_track_ids is None:
                self.results.append(record)
                continue

            pred_kpts = np.asarray(pred.get('keypoints'))
            gt_kpts = np.asarray(gt.get('keypoints'))
            if pred_kpts.ndim == 2:
                pred_kpts = pred_kpts[None]
            if gt_kpts.ndim == 2:
                gt_kpts = gt_kpts[None]
            n_pred, n_gt = pred_kpts.shape[0], gt_kpts.shape[0]

            pred_track_ids = np.asarray(pred_track_ids).reshape(-1)
            gt_track_ids = np.asarray(gt_track_ids).reshape(-1)
            if (len(pred_track_ids) != n_pred
                    or len(gt_track_ids) != n_gt):
                self.results.append(record)
                continue

            # Confidence filter, before any matching happens.
            keep_pred = np.arange(n_pred)
            if self.score_thr > 0.0 and n_pred > 0:
                scores = self._pred_scores(pred, n_pred)
                if scores is not None:
                    keep_pred = np.where(scores >= self.score_thr)[0]

            gt_list = self._build_gt_list(gt, n_gt)
            pred_list = self._build_pred_list(
                {'keypoints': pred_kpts[keep_pred]}, len(keep_pred))

            pairs, valid_gt_idx, valid_pred_idx = compute_oks_pairs(
                gt_list, pred_list, sigmas)

            # Explicit int dtype: indexing with an empty Python list yields
            # a float array on older numpy, which would then fail as an
            # index into the track-id arrays.
            gt_sel = np.asarray(valid_gt_idx, dtype=np.int64)
            pred_sel = keep_pred[np.asarray(valid_pred_idx, dtype=np.int64)]
            record['gt_ids'] = gt_track_ids[gt_sel].astype(np.int64)
            record['pred_ids'] = pred_track_ids[pred_sel].astype(np.int64)
            record['similarity'] = self._similarity_matrix(
                gt,
                {'bboxes': (np.asarray(pred['bboxes'])[keep_pred]
                            if pred.get('bboxes') is not None else None)},
                valid_gt_idx, valid_pred_idx, pairs)

            self.results.append(record)

    # ------------------------------------------------------------------
    # Regrouping
    # ------------------------------------------------------------------

    @staticmethod
    def group_sequences(results: List[dict]) -> List[SequenceData]:
        """Reassemble per-frame records into per-video :class:`SequenceData`.

        Frames are sorted chronologically inside each sequence and track
        ids are re-indexed densely per sequence, mirroring TrackEval, which
        evaluates one sequence at a time and combines afterwards.
        """
        by_seq: Dict[str, List[dict]] = defaultdict(list)
        for r in results:
            by_seq[r['seq_key']].append(r)

        sequences: List[SequenceData] = []
        for frames in by_seq.values():
            frames.sort(key=lambda r: r['frame_order'])

            gt_map: Dict[int, int] = {}
            pred_map: Dict[int, int] = {}
            gt_ids, pred_ids, similarity = [], [], []
            num_gt_dets = num_pred_dets = 0

            for r in frames:
                dense_gt = np.array(
                    [gt_map.setdefault(int(i), len(gt_map))
                     for i in r['gt_ids']], dtype=np.int64)
                dense_pred = np.array(
                    [pred_map.setdefault(int(i), len(pred_map))
                     for i in r['pred_ids']], dtype=np.int64)
                gt_ids.append(dense_gt)
                pred_ids.append(dense_pred)
                similarity.append(r['similarity'])
                num_gt_dets += len(dense_gt)
                num_pred_dets += len(dense_pred)

            sequences.append(
                SequenceData(
                    gt_ids=gt_ids,
                    pred_ids=pred_ids,
                    similarity=similarity,
                    num_gt_ids=len(gt_map),
                    num_pred_ids=len(pred_map),
                    num_gt_dets=num_gt_dets,
                    num_pred_dets=num_pred_dets))
        return sequences


@METRICS.register_module()
class MOTA(_MOTBaseMetric):
    """CLEAR-MOT accuracy (MOTA) and precision (MOTP).

    Reimplementation of TrackEval's ``trackeval/metrics/clear.py``.  Each
    frame is matched with the Hungarian algorithm on a score matrix that
    adds a large bonus (``1000``) to pairs reproducing the *previous
    frame's* assignment, so identity is preserved wherever it legitimately
    can be and switches are only counted when the tracker really changed
    its mind; pairs below ``match_thr`` are zeroed out and never matched.
    An identity switch is counted when a GT track's matched prediction
    differs from the last prediction it was *ever* matched to, not merely
    the last frame's, so a track that is lost and recovered under a new id
    counts once.

    ``MOTA = (TP - FP - IDSW) / (TP + FN)``, which is at most 1 and
    unbounded below.  ``MOTP`` is the mean similarity over true positives -
    with the default OKS similarity it is a pose-localisation quality, not
    the box-overlap quality MOTP usually denotes.

    Returned metric names: ``'MOTA'``, ``'MOTP'``, ``'CLR_Re'``,
    ``'CLR_Pr'``, ``'CLR_TP'``, ``'CLR_FN'``, ``'CLR_FP'``, ``'CLR_IDSW'``.

    Args:
        idsw_reference (str): Which previous assignment the matching bonus
            reproduces, the one place TrackEval and ``py-motmetrics``
            genuinely disagree.  ``'prev_frame'`` (default) follows
            TrackEval: only the *immediately previous* frame's assignment
            is worth a bonus, so a track recovered after a gap may legally
            land on a different prediction and score a switch.
            ``'last_seen'`` follows ``py-motmetrics`` and the original
            MOTChallenge devkit, which re-establish a GT track onto the
            last prediction it was *ever* matched to whenever that
            prediction is present and in range, reporting fewer switches
            after gaps.  Both count a switch against the last-ever match;
            only the matching preference differs.  Use ``'last_seen'`` to
            compare against numbers produced by ``py-motmetrics``.
        See :class:`_MOTBaseMetric` for the remaining arguments.
    """

    def __init__(self, idsw_reference: str = 'prev_frame', **kwargs) -> None:
        if idsw_reference not in ('prev_frame', 'last_seen'):
            raise ValueError(
                "idsw_reference must be 'prev_frame' or 'last_seen', got "
                f'{idsw_reference!r}')
        super().__init__(**kwargs)
        self.idsw_reference = idsw_reference

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Match every frame under the CLEAR rules and accumulate counts."""
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info('Evaluating MOTA (CLEAR-MOT)...')

        tp = fn = fp = idsw = 0
        motp_sum = 0.0

        for seq in self.group_sequences(results):
            # Last prediction each GT id was matched to, ever (for IDSW)
            # and in the immediately preceding frame (for the sticky bonus).
            prev_id = np.full(seq.num_gt_ids, np.nan)
            prev_frame_id = np.full(seq.num_gt_ids, np.nan)

            for gt_t, pred_t, sim in zip(seq.gt_ids, seq.pred_ids,
                                         seq.similarity):
                # As TrackEval, an empty frame leaves the sticky state
                # alone: the bonus still refers to the last frame that had
                # something to match.
                if len(gt_t) == 0:
                    fp += len(pred_t)
                    continue
                if len(pred_t) == 0:
                    fn += len(gt_t)
                    continue

                # Prefer reproducing the previous assignment, then maximise
                # similarity; gate out pairs below the threshold.
                reference = (prev_frame_id
                             if self.idsw_reference == 'prev_frame'
                             else prev_id)
                sticky = (pred_t[np.newaxis, :] ==
                          reference[gt_t[:, np.newaxis]])
                score = 1000.0 * sticky + sim
                score[sim < self.match_thr - _EPS] = 0.0

                rows, cols = linear_sum_assignment(-score)
                accepted = score[rows, cols] > _EPS
                rows, cols = rows[accepted], cols[accepted]

                matched_gt = gt_t[rows]
                matched_pred = pred_t[cols]

                previously = prev_id[matched_gt]
                idsw += int(np.sum(
                    ~np.isnan(previously) &
                    np.not_equal(matched_pred, previously)))

                n_matches = len(rows)
                tp += n_matches
                fn += len(gt_t) - n_matches
                fp += len(pred_t) - n_matches
                motp_sum += float(np.sum(sim[rows, cols]))

                prev_id[matched_gt] = matched_pred
                prev_frame_id[:] = np.nan
                prev_frame_id[matched_gt] = matched_pred

        return {
            'MOTA': (tp - fp - idsw) / max(1.0, tp + fn),
            'MOTP': motp_sum / max(1.0, tp),
            'CLR_Re': tp / max(1.0, tp + fn),
            'CLR_Pr': tp / max(1.0, tp + fp),
            'CLR_TP': float(tp),
            'CLR_FN': float(fn),
            'CLR_FP': float(fp),
            'CLR_IDSW': float(idsw),
        }


@METRICS.register_module()
class IDF1(_MOTBaseMetric):
    """Identity F1 (IDF1), with its recall and precision.

    Reimplementation of TrackEval's ``trackeval/metrics/identity.py``.
    Unlike MOTA, no per-frame matching happens: GT and predicted
    trajectories are matched **one-to-one over the whole sequence** by
    minimising the total identity error, and the score is the F1 over the
    detections that end up correctly identified.  A tracker that fragments
    a person into several ids is penalised for every frame it spends on the
    wrong one, however good its per-frame detections are.

    The assignment runs on the standard ``(num_gt + num_pred)`` square cost
    matrix whose off-diagonal blocks are blocked out at ``1e10``, so any
    trajectory may also go unmatched (paying its full length as IDFN or
    IDFP).

    Returned metric names: ``'IDF1'``, ``'IDR'``, ``'IDP'``, ``'IDTP'``,
    ``'IDFN'``, ``'IDFP'``.

    Args:
        See :class:`_MOTBaseMetric`.
    """

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Solve the global identity assignment per sequence, then sum."""
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info('Evaluating IDF1 (identity F1)...')

        idtp = idfn = idfp = 0

        for seq in self.group_sequences(results):
            num_gt, num_pred = seq.num_gt_ids, seq.num_pred_ids
            if num_gt == 0 and num_pred == 0:
                continue

            # How many frames each (gt, pred) pair could be matched on, and
            # how long each trajectory is.
            potential = np.zeros((num_gt, num_pred))
            gt_count = np.zeros(num_gt)
            pred_count = np.zeros(num_pred)

            for gt_t, pred_t, sim in zip(seq.gt_ids, seq.pred_ids,
                                         seq.similarity):
                if len(gt_t) and len(pred_t):
                    rows, cols = np.nonzero(sim >= self.match_thr - _EPS)
                    np.add.at(potential, (gt_t[rows], pred_t[cols]), 1)
                gt_count[gt_t] += 1
                pred_count[pred_t] += 1

            # Square cost matrix over (gt + pred) slots: the top-left block
            # pairs a GT with a prediction, the diagonals of the other two
            # blocks leave a trajectory unmatched, and the remaining blocks
            # are forbidden.
            size = num_gt + num_pred
            fn_mat = np.zeros((size, size))
            fp_mat = np.zeros((size, size))
            fp_mat[num_gt:, :num_pred] = 1e10
            fn_mat[:num_gt, num_pred:] = 1e10
            for gt_id in range(num_gt):
                fn_mat[gt_id, :num_pred] = gt_count[gt_id]
                fn_mat[gt_id, num_pred + gt_id] = gt_count[gt_id]
            for pred_id in range(num_pred):
                fp_mat[:num_gt, pred_id] = pred_count[pred_id]
                fp_mat[num_gt + pred_id, pred_id] = pred_count[pred_id]
            fn_mat[:num_gt, :num_pred] -= potential
            fp_mat[:num_gt, :num_pred] -= potential

            rows, cols = linear_sum_assignment(fn_mat + fp_mat)
            seq_idfn = int(fn_mat[rows, cols].sum())
            seq_idfp = int(fp_mat[rows, cols].sum())

            idfn += seq_idfn
            idfp += seq_idfp
            idtp += int(gt_count.sum()) - seq_idfn

        return {
            'IDF1': idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn),
            'IDR': idtp / max(1.0, idtp + idfn),
            'IDP': idtp / max(1.0, idtp + idfp),
            'IDTP': float(idtp),
            'IDFN': float(idfn),
            'IDFP': float(idfp),
        }


@METRICS.register_module()
class HOTA(_MOTBaseMetric):
    """Higher Order Tracking Accuracy (HOTA) and its components.

    Reimplementation of TrackEval's ``trackeval/metrics/hota.py``.  HOTA
    deliberately refuses to let detection quality mask association quality:
    at each localisation threshold ``alpha`` it computes a detection
    accuracy ``DetA = TP / (TP + FN + FP)`` and an association accuracy
    ``AssA`` - the average, over true positives, of how well that
    detection's GT and predicted trajectories overlap across the whole
    sequence - and reports their geometric mean.  The final score is the
    average over ``alpha``.

    Per-frame matching maximises ``global_alignment_score * similarity``,
    where the alignment score is a whole-sequence Jaccard between the two
    trajectories.  That makes the matching itself identity-aware, which is
    why HOTA does not need MOTA's sticky bonus and does not count identity
    switches separately.

    ``match_thr`` is unused: HOTA sweeps ``alpha`` over
    :data:`HOTA_ALPHAS` instead.

    Returned metric names: ``'HOTA'``, ``'DetA'``, ``'AssA'``, ``'DetRe'``,
    ``'DetPr'``, ``'AssRe'``, ``'AssPr'``, ``'LocA'`` (all averaged over
    ``alpha``), plus ``'HOTA(0.5)'`` at the single ``alpha = 0.5``
    threshold.

    Note that ``LocA`` is TrackEval's, which evaluates to exactly ``1`` at
    any ``alpha`` that produced no true positives (``max(1e-10, 0) /
    max(1e-10, 0)``).  Averaging those in pulls the reported ``LocA``
    *upwards* for a poorly-localised tracker, so read it alongside
    ``DetA`` rather than on its own.

    Args:
        See :class:`_MOTBaseMetric`.
    """

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Accumulate per-alpha statistics per sequence, then combine."""
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info('Evaluating HOTA...')

        n_alpha = len(HOTA_ALPHAS)
        tp = np.zeros(n_alpha)
        fn = np.zeros(n_alpha)
        fp = np.zeros(n_alpha)
        # Association scores combine across sequences as a TP-weighted
        # average, exactly as TrackEval's combine_sequences does.
        ass_a_sum = np.zeros(n_alpha)
        ass_re_sum = np.zeros(n_alpha)
        ass_pr_sum = np.zeros(n_alpha)
        loc_a_sum = np.zeros(n_alpha)

        for seq in self.group_sequences(results):
            seq_tp, seq_fn, seq_fp, seq_loc, matches = self._score_sequence(
                seq)
            tp += seq_tp
            fn += seq_fn
            fp += seq_fp
            loc_a_sum += seq_loc

            gt_count, pred_count = self._id_counts(seq)
            for a in range(n_alpha):
                counts = matches[a]
                union = gt_count + pred_count - counts
                ass_a_sum[a] += np.sum(
                    counts * (counts / np.maximum(1.0, union)))
                ass_re_sum[a] += np.sum(
                    counts * (counts / np.maximum(1.0, gt_count)))
                ass_pr_sum[a] += np.sum(
                    counts * (counts / np.maximum(1.0, pred_count)))

        safe_tp = np.maximum(1.0, tp)
        ass_a = ass_a_sum / safe_tp
        det_a = tp / np.maximum(1.0, tp + fn + fp)
        hota = np.sqrt(det_a * ass_a)

        half = int(np.argmin(np.abs(HOTA_ALPHAS - 0.5)))
        return {
            'HOTA': float(np.mean(hota)),
            'DetA': float(np.mean(det_a)),
            'AssA': float(np.mean(ass_a)),
            'DetRe': float(np.mean(tp / np.maximum(1.0, tp + fn))),
            'DetPr': float(np.mean(tp / np.maximum(1.0, tp + fp))),
            'AssRe': float(np.mean(ass_re_sum / safe_tp)),
            'AssPr': float(np.mean(ass_pr_sum / safe_tp)),
            'LocA': float(np.mean(
                np.maximum(1e-10, loc_a_sum) / np.maximum(1e-10, tp))),
            'HOTA(0.5)': float(hota[half]),
        }

    @staticmethod
    def _id_counts(seq: SequenceData):
        """Detections per GT id ``(M, 1)`` and per predicted id ``(1, N)``."""
        gt_count = np.zeros((seq.num_gt_ids, 1))
        pred_count = np.zeros((1, seq.num_pred_ids))
        for gt_t, pred_t in zip(seq.gt_ids, seq.pred_ids):
            gt_count[gt_t] += 1
            pred_count[0, pred_t] += 1
        return gt_count, pred_count

    def _global_alignment(self, seq: SequenceData) -> np.ndarray:
        """Whole-sequence Jaccard alignment between every GT/pred id pair.

        This is the first of HOTA's two passes: accumulate a
        similarity-weighted count of how often each pair *could* be
        matched, then normalise by how long the two trajectories are.  The
        result biases the per-frame matching towards pairs that agree over
        the whole sequence rather than just this frame.
        """
        potential = np.zeros((seq.num_gt_ids, seq.num_pred_ids))
        for gt_t, pred_t, sim in zip(seq.gt_ids, seq.pred_ids,
                                     seq.similarity):
            if len(gt_t) == 0 or len(pred_t) == 0:
                continue
            denom = (sim.sum(0)[np.newaxis, :] + sim.sum(1)[:, np.newaxis] -
                     sim)
            sim_iou = np.zeros_like(sim)
            mask = denom > _EPS
            sim_iou[mask] = sim[mask] / denom[mask]
            np.add.at(potential, (gt_t[:, np.newaxis], pred_t[np.newaxis, :]),
                      sim_iou)

        gt_count, pred_count = self._id_counts(seq)
        return potential / np.maximum(
            _EPS, gt_count + pred_count - potential)

    def _score_sequence(self, seq: SequenceData):
        """Per-alpha TP/FN/FP, localisation sum and matches, one video."""
        n_alpha = len(HOTA_ALPHAS)
        tp = np.zeros(n_alpha)
        fn = np.zeros(n_alpha)
        fp = np.zeros(n_alpha)
        loc = np.zeros(n_alpha)
        matches = [
            np.zeros((seq.num_gt_ids, seq.num_pred_ids))
            for _ in range(n_alpha)
        ]

        alignment = self._global_alignment(seq)

        for gt_t, pred_t, sim in zip(seq.gt_ids, seq.pred_ids,
                                     seq.similarity):
            if len(gt_t) == 0:
                fp += len(pred_t)
                continue
            if len(pred_t) == 0:
                fn += len(gt_t)
                continue

            score = alignment[gt_t[:, np.newaxis],
                              pred_t[np.newaxis, :]] * sim
            rows, cols = linear_sum_assignment(-score)

            for a, alpha in enumerate(HOTA_ALPHAS):
                accepted = sim[rows, cols] >= alpha - _EPS
                a_rows, a_cols = rows[accepted], cols[accepted]
                n_matches = len(a_rows)
                tp[a] += n_matches
                fn[a] += len(gt_t) - n_matches
                fp[a] += len(pred_t) - n_matches
                if n_matches:
                    loc[a] += float(np.sum(sim[a_rows, a_cols]))
                    np.add.at(matches[a],
                              (gt_t[a_rows], pred_t[a_cols]), 1)

        return tp, fn, fp, loc, matches
