# Copyright (c) OpenMMLab. All rights reserved.
"""Tracking-quality metrics for pipelines that assign instance track IDs."""

import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from ..functional.frame_metrics import compute_oks_pairs
from .temporal_keypoint_metrics import _TemporalBaseMetric


def _sequence_key_from_path(img_path: str) -> str:
    """Derive a sequence identifier from an image path.

    Duplicated from ``mmpose/postprocessing/base.py`` (kept local to avoid
    a cross-package import between ``mmpose.evaluation`` and
    ``mmpose.postprocessing``); see that module for the canonical version.
    """
    if not img_path:
        return ''
    parts = img_path.replace('\\', '/').split('/')
    dirs = parts[:-1]
    if dirs and dirs[-1] == 'images':
        dirs = dirs[:-1]
    return '/'.join(dirs) if dirs else ''


def _frame_order_key(img_path: str, fallback: int) -> int:
    """Best-effort numeric frame index parsed from ``img_path``.

    Supports the same conventions as
    :func:`~mmpose.evaluation.metrics.temporal_keypoint_metrics._parse_frame_id`
    (``image_XXXXX.jpg`` / ``XXXXX.jpg``), but never raises: when the
    filename does not encode a plain integer (e.g. non-video datasets with
    hashed or alphanumeric names), ``fallback`` -- the running per-metric
    processing order -- is used instead, so this metric can be evaluated on
    any dataset without crashing.
    """
    basename = osp.splitext(osp.basename(img_path))[0]
    if basename.startswith('image_'):
        basename = basename.split('_')[-1]
    try:
        return int(basename)
    except (TypeError, ValueError):
        return fallback


@METRICS.register_module()
class IDSwitch(_TemporalBaseMetric):
    """Average number of track-ID switches per 100 frames.

    Measures the stability of a tracker's identity assignment
    (``pred_instances.track_ids``, e.g. produced by
    :class:`~mmpose.postprocessing.filters.oks_tracker.OKSTracker`) against
    ground-truth instance identity (``gt_instances.track_ids``).

    For every frame, predicted instances are matched to GT instances using
    an *identity-aware* variant of the greedy OKS matching used elsewhere
    (e.g. :class:`~mmpose.evaluation.metrics.temporal_keypoint_metrics.MPJVE`
    / :class:`.MPJAE`): each GT track first tries to keep the predicted
    ``track_id`` it was matched to in the previous observed frame, as long
    as that same predicted instance is still present and its OKS with the
    GT is still above ``match_thr``.  Only GT tracks without a valid
    "sticky" candidate (new tracks, or ones whose previous match
    disappeared / dropped below threshold) fall back to plain greedy
    best-OKS matching.  This avoids spurious ID switches when a short-lived
    duplicate/false-positive detection happens to score marginally higher
    than the correctly-tracked instance for a few frames -- without the
    sticky preference, greedy-best-OKS-only matching would "switch" onto
    the duplicate and immediately count it as an identity switch, even
    though the original track was never actually lost.

    For each GT track, the resulting sequence of matched predicted
    ``track_id`` values (ordered by frame) is inspected: whenever this
    predicted id changes from one matched observation to the next, an *ID
    switch* has occurred (the standard MOT / CLEAR-MOT identity-switch
    definition).  Switches are counted independently per video sequence
    (sequence boundaries are derived from ``img_path`` the same way as
    :mod:`mmpose.postprocessing`), so unrelated single-image datasets
    (e.g. COCO, where every image is its own one-frame "sequence") never
    contribute spurious switches.

    The metric degrades gracefully (contributes 0 switches, but still
    counts frames) whenever ``track_ids`` are absent from either
    ``pred_instances`` or ``gt_instances`` -- e.g. when no tracker was
    used in the post-processing pipeline -- so it is safe to include by
    default regardless of whether the evaluated predictions were tracked.

    Returned metric names:

    - ``'IDSwitches'``: total raw switch count across the whole dataset.
    - ``'IDSwitchesPer100Frames'``: ``100 * total_switches / total_frames``,
      the length-independent rate.

    Args:
        match_thr (float): OKS threshold for pred-to-GT matching (also used
            to validate "sticky" candidates).  Default: ``0.5``.
        collect_device (str): Device for distributed result collection.
            Default: ``'cpu'``.
        prefix (str, optional): Metric name prefix.  Default: ``None``
            (uses ``'tracking'``).
    """

    default_prefix: Optional[str] = 'tracking'

    def __init__(self,
                 match_thr: float = 0.5,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(
            match_thr=match_thr, collect_device=collect_device, prefix=prefix)
        self._frame_counter = 0

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Accumulate per-frame OKS pairs and GT/pred track-id maps.

        The final identity-aware matching depends on the predicted
        ``track_id`` each GT track was matched to in the *previous* frame,
        so it cannot be resolved per-frame in isolation here.  Instead, the
        full pairwise OKS scores (not just the greedily-selected matches)
        and the track-id lookups are stored, and the actual sticky
        assignment + switch counting happens once in :meth:`compute_metrics`
        after sorting every frame into chronological order per sequence.

        One record is appended per frame regardless of whether any match
        was found, so the total frame count used for the per-100-frames
        normalisation is exact.
        """
        sigmas = np.asarray(
            self.dataset_meta.get('sigmas', []), dtype=np.float32)

        for data_sample in data_samples:
            self._frame_counter += 1

            img_path = data_sample.get('img_path', '')
            seq_key = _sequence_key_from_path(img_path)
            frame_order = _frame_order_key(img_path, self._frame_counter)

            record = {
                'seq_key': seq_key,
                'frame_order': frame_order,
                'pairs': [],       # (oks, gt_idx, pred_idx)
                'gt_track_id': {},   # gt_idx -> gt track id
                'pred_track_id': {},  # pred_idx -> pred track id
            }

            pred = data_sample.get('pred_instances')
            gt = data_sample.get('gt_instances')

            if len(sigmas) > 0 and pred is not None and gt is not None:
                pred_track_ids = pred.get('track_ids')
                gt_track_ids = gt.get('track_ids')

                if pred_track_ids is not None and gt_track_ids is not None:
                    pred_kpts = np.asarray(pred['keypoints'])
                    gt_kpts = np.asarray(gt['keypoints'])
                    if pred_kpts.ndim == 2:
                        pred_kpts = pred_kpts[None]
                    if gt_kpts.ndim == 2:
                        gt_kpts = gt_kpts[None]

                    n_pred = pred_kpts.shape[0]
                    n_gt = gt_kpts.shape[0]
                    pred_track_ids = np.asarray(pred_track_ids).reshape(-1)
                    gt_track_ids = np.asarray(gt_track_ids).reshape(-1)

                    if (n_pred > 0 and n_gt > 0
                            and len(pred_track_ids) == n_pred
                            and len(gt_track_ids) == n_gt):
                        gt_list = self._build_gt_list(gt, n_gt)
                        pred_list = self._build_pred_list(
                            {'keypoints': pred_kpts}, n_pred)

                        pairs, valid_gt_idx, valid_pred_idx = \
                            compute_oks_pairs(gt_list, pred_list, sigmas)

                        record['pairs'] = pairs
                        record['gt_track_id'] = {
                            gi: int(gt_track_ids[gi]) for gi in valid_gt_idx}
                        record['pred_track_id'] = {
                            pi: int(pred_track_ids[pi])
                            for pi in valid_pred_idx
                        }

            self.results.append(record)

    @staticmethod
    def _match_frame_sticky(
        pairs: List[Tuple[float, int, int]],
        gt_track_id: Dict[int, int],
        pred_track_id: Dict[int, int],
        last_pred_id: Dict[int, int],
        match_thr: float,
    ) -> List[Tuple[int, int]]:
        """Match one frame's GT/pred instances, preferring track continuity.

        Every GT track first tries to keep the predicted instance carrying
        the ``track_id`` it was matched to previously (``last_pred_id``),
        provided that instance is present this frame and its OKS with the
        GT is still >= ``match_thr``.  Remaining (unmatched) GT/pred
        instances are then matched by plain greedy best-OKS, exactly as
        :func:`~mmpose.evaluation.functional.frame_metrics.match_instances_oks`
        would.

        Returns:
            List of ``(gt_idx, pred_idx)`` matches for this frame.
        """
        if not pairs:
            return []

        oks_lookup: Dict[Tuple[int, int], float] = {
            (gi, pi): oks for oks, gi, pi in pairs
        }

        used_gi: set = set()
        used_pi: set = set()
        frame_matches: List[Tuple[int, int]] = []

        # 1) Sticky pass: keep each GT track on its previous predicted
        #    identity when that instance is still a valid (>= match_thr)
        #    match this frame, even if some other instance (e.g. a
        #    duplicate / false-positive detection) now scores higher.
        sticky_candidates: List[Tuple[float, int, int]] = []
        for gi, gt_tid in gt_track_id.items():
            sticky_ptid = last_pred_id.get(gt_tid)
            if sticky_ptid is None:
                continue
            for pi, ptid in pred_track_id.items():
                if ptid != sticky_ptid:
                    continue
                oks_val = oks_lookup.get((gi, pi))
                if oks_val is not None and oks_val >= match_thr:
                    sticky_candidates.append((oks_val, gi, pi))
                break  # track ids are unique among preds within a frame

        sticky_candidates.sort(key=lambda x: x[0], reverse=True)
        for _, gi, pi in sticky_candidates:
            if gi in used_gi or pi in used_pi:
                continue
            used_gi.add(gi)
            used_pi.add(pi)
            frame_matches.append((gi, pi))

        # 2) Greedy pass over the remaining pairs fills in GT tracks with
        #    no (valid) sticky candidate -- new tracks, or ones whose
        #    previous match disappeared or dropped below threshold.
        for oks_val, gi, pi in pairs:
            if oks_val < match_thr:
                break
            if gi in used_gi or pi in used_pi:
                continue
            used_gi.add(gi)
            used_pi.add(pi)
            frame_matches.append((gi, pi))

        return frame_matches

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Run identity-aware matching and count ID switches per GT track.

        Returns:
            Dict[str, float]: ``'IDSwitches'`` (raw count) and
            ``'IDSwitchesPer100Frames'`` (length-normalised rate).
        """
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info('Evaluating IDSwitch (per-100-frames)...')

        by_seq: Dict[str, List[dict]] = defaultdict(list)
        for r in results:
            by_seq[r['seq_key']].append(r)

        total_switches = 0
        for frames in by_seq.values():
            frames.sort(key=lambda r: r['frame_order'])
            last_pred_id: Dict[int, int] = {}  # gt_track_id -> pred_track_id

            for r in frames:
                frame_matches = self._match_frame_sticky(
                    r['pairs'], r['gt_track_id'], r['pred_track_id'],
                    last_pred_id, self.match_thr)

                for gi, pi in frame_matches:
                    gt_tid = r['gt_track_id'][gi]
                    pred_tid = r['pred_track_id'][pi]
                    prev = last_pred_id.get(gt_tid)
                    if prev is not None and prev != pred_tid:
                        total_switches += 1
                    last_pred_id[gt_tid] = pred_tid

        total_frames = len(results)
        rate = (100.0 * total_switches / total_frames
                if total_frames else 0.0)

        return {
            'IDSwitches': float(total_switches),
            'IDSwitchesPer100Frames': rate,
        }
