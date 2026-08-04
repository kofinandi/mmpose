# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for the MOTA / IDF1 / HOTA tracking metrics.

The expected values here are all derived by hand from the published
definitions, so they pin the implementation to the metrics themselves
rather than to whatever it currently produces.

The metrics were additionally cross-checked against ``py-motmetrics``
1.4.0 on 30 randomised scenarios (exact agreement on MOTA, IDF1, IDSW, FP
and FN once :class:`MOTA`'s ``idsw_reference`` is set to ``'last_seen'``,
which is the convention ``py-motmetrics`` uses).  That check is not
reproduced here because ``motmetrics`` is not a dependency of this project.
"""

from typing import List
from unittest import TestCase

import numpy as np

from mmpose.evaluation.metrics import HOTA, IDF1, MOTA


def _records(gt_per_frame, pred_per_frame, sim_per_frame,
             seq: str = 'seq') -> List[dict]:
    """Build the per-frame records ``compute_metrics`` consumes.

    This bypasses ``process()`` -- which only exists to turn
    ``PoseDataSample``\\ s into exactly these records -- so the tests can
    state the tracking scenario directly.
    """
    records = []
    for t, (gt, pred, sim) in enumerate(
            zip(gt_per_frame, pred_per_frame, sim_per_frame)):
        records.append({
            'seq_key': seq,
            'frame_order': t,
            'gt_ids': np.asarray(gt, dtype=np.int64),
            'pred_ids': np.asarray(pred, dtype=np.int64),
            'similarity': np.asarray(sim, dtype=np.float64).reshape(
                len(gt), len(pred)),
        })
    return records


def _metrics(match_thr: float = 0.5):
    """Build the three metrics with dummy keypoint metadata."""
    built = (MOTA(match_thr=match_thr), IDF1(match_thr=match_thr), HOTA())
    for metric in built:
        metric.dataset_meta = {'sigmas': np.ones(17, dtype=np.float32)}
    return built


def _data_sample(gt_keypoints,
                  gt_track_ids,
                  pred_keypoints,
                  pred_track_ids,
                  gt_iscrowd=None,
                  gt_bboxes=None,
                  pred_bboxes=None,
                  img_path: str = 'seq/000001.jpg') -> dict:
    """Build a single-frame ``data_sample`` dict for ``process()``.

    Unlike :func:`_records`, this goes through ``process()`` itself, which
    is where ignore-region suppression lives.  ``gt_keypoints`` /
    ``pred_keypoints`` are ``(N, K, 2)``; every GT keypoint is visible
    unless the instance is marked ``iscrowd`` (which drops it from
    matching regardless of visibility, as in ``compute_oks_pairs``).
    """
    gt_kpts = np.asarray(gt_keypoints, dtype=np.float32)
    pred_kpts = np.asarray(pred_keypoints, dtype=np.float32)
    n_gt = gt_kpts.shape[0]

    gt_instances = {
        'keypoints': gt_kpts,
        'keypoints_visible': np.ones(gt_kpts.shape[:2], dtype=np.float32),
        'track_ids': np.asarray(gt_track_ids, dtype=np.int64),
        'iscrowd': (np.asarray(gt_iscrowd, dtype=np.int64)
                    if gt_iscrowd is not None else
                    np.zeros(n_gt, dtype=np.int64)),
    }
    if gt_bboxes is not None:
        gt_instances['bboxes'] = np.asarray(gt_bboxes, dtype=np.float64)

    pred_instances = {
        'keypoints': pred_kpts,
        'track_ids': np.asarray(pred_track_ids, dtype=np.int64),
    }
    if pred_bboxes is not None:
        pred_instances['bboxes'] = np.asarray(pred_bboxes, dtype=np.float64)

    return {
        'img_path': img_path,
        'gt_instances': gt_instances,
        'pred_instances': pred_instances,
    }


def _mota(**kwargs) -> MOTA:
    """Build a lone :class:`MOTA` with dummy single-keypoint metadata."""
    metric = MOTA(**kwargs)
    metric.dataset_meta = {'sigmas': np.ones(1, dtype=np.float32)}
    return metric


class TestMOTMetrics(TestCase):

    def test_perfect_tracking(self):
        """One GT track followed perfectly scores 1 on all three."""
        records = _records([[0]] * 4, [[0]] * 4, [[[1.0]]] * 4)
        mota, idf1, hota = _metrics()

        self.assertAlmostEqual(mota.compute_metrics(records)['MOTA'], 1.0)
        self.assertAlmostEqual(idf1.compute_metrics(records)['IDF1'], 1.0)

        res = hota.compute_metrics(records)
        self.assertAlmostEqual(res['HOTA'], 1.0)
        self.assertAlmostEqual(res['DetA'], 1.0)
        self.assertAlmostEqual(res['AssA'], 1.0)

    def test_single_identity_switch(self):
        """A track split in half: detection is perfect, association is not.

        Four frames of one GT track, predicted as id 0 then id 1:

        * MOTA loses exactly one point of its four -- ``(4 - 0 - 1) / 4``.
        * IDF1 can only claim one of the two halves -- ``IDTP = IDFN =
          IDFP = 2`` -- so ``2 / (2 + 1 + 1)``.
        * HOTA keeps ``DetA = 1`` but halves ``AssA``, which is the whole
          point of separating the two.
        """
        records = _records([[0]] * 4, [[0], [0], [1], [1]], [[[1.0]]] * 4)
        mota, idf1, hota = _metrics()

        res = mota.compute_metrics(records)
        self.assertAlmostEqual(res['MOTA'], 0.75)
        self.assertAlmostEqual(res['CLR_IDSW'], 1.0)

        self.assertAlmostEqual(idf1.compute_metrics(records)['IDF1'], 0.5)

        res = hota.compute_metrics(records)
        self.assertAlmostEqual(res['DetA'], 1.0)
        self.assertAlmostEqual(res['AssA'], 0.5)
        self.assertAlmostEqual(res['HOTA'], np.sqrt(0.5))

    def test_miss_and_false_positive(self):
        """One missed frame and one spurious detection.

        ``TP = 3``, ``FN = 1``, ``FP = 1``, no switch, so
        ``MOTA = (3 - 1 - 0) / 4``.  IDF1 pays half a point for each of the
        miss and the spurious track: ``3 / (3 + 0.5 + 0.5)``.
        """
        records = _records(
            [[0], [0], [0], [0]],
            [[0], [], [0, 1], [0]],
            [[[1.0]], np.zeros((1, 0)), [[1.0, 0.0]], [[1.0]]])
        mota, idf1, _ = _metrics()

        res = mota.compute_metrics(records)
        self.assertAlmostEqual(res['MOTA'], 0.5)
        self.assertAlmostEqual(res['CLR_TP'], 3.0)
        self.assertAlmostEqual(res['CLR_FN'], 1.0)
        self.assertAlmostEqual(res['CLR_FP'], 1.0)
        self.assertAlmostEqual(res['CLR_IDSW'], 0.0)

        self.assertAlmostEqual(
            idf1.compute_metrics(records)['IDF1'], 3 / 4)

    def test_similarity_below_threshold_is_not_a_match(self):
        """A prediction under ``match_thr`` is a miss *and* a false alarm."""
        records = _records([[0]] * 2, [[0]] * 2, [[[0.4]]] * 2)
        mota, idf1, _ = _metrics(match_thr=0.5)

        res = mota.compute_metrics(records)
        self.assertAlmostEqual(res['CLR_TP'], 0.0)
        self.assertAlmostEqual(res['CLR_FN'], 2.0)
        self.assertAlmostEqual(res['CLR_FP'], 2.0)
        self.assertAlmostEqual(res['MOTA'], -1.0)

        self.assertAlmostEqual(idf1.compute_metrics(records)['IDF1'], 0.0)

    def test_hota_alpha_sweep_penalises_loose_localisation(self):
        """HOTA drops when similarity only clears the low thresholds.

        With every match at similarity 0.5, the 10 thresholds at or below
        0.5 count a TP and the 9 above it do not, so ``DetA`` averages
        ``10 / 19``.

        ``LocA`` averages ``14 / 19``, not 0.5: TrackEval defines
        per-threshold ``LocA`` as ``max(1e-10, sum_sim) / max(1e-10, TP)``,
        which is exactly 1 at a threshold that produced no TPs at all, and
        those 9 ones are averaged in.  See the note in :class:`HOTA`.
        """
        records = _records([[0]] * 4, [[0]] * 4, [[[0.5]]] * 4)
        _, _, hota = _metrics()

        res = hota.compute_metrics(records)
        self.assertAlmostEqual(res['DetA'], 10 / 19)
        self.assertAlmostEqual(res['LocA'], (10 * 0.5 + 9 * 1.0) / 19)
        self.assertLess(res['HOTA'], 1.0)

    def test_sequences_are_scored_independently(self):
        """Ids repeat across videos without being conflated.

        The same id numbers in two different sequences must not be joined
        into one trajectory; scoring both videos perfectly must still give
        a perfect score.
        """
        records = (_records([[0]] * 3, [[0]] * 3, [[[1.0]]] * 3, seq='a') +
                   _records([[0]] * 3, [[0]] * 3, [[[1.0]]] * 3, seq='b'))
        mota, idf1, hota = _metrics()

        self.assertAlmostEqual(mota.compute_metrics(records)['MOTA'], 1.0)
        self.assertAlmostEqual(idf1.compute_metrics(records)['IDF1'], 1.0)
        self.assertAlmostEqual(hota.compute_metrics(records)['HOTA'], 1.0)

    def test_swapped_ids_across_two_tracks(self):
        """Two tracks whose ids swap midway.

        Frames 0-1 have GT 0 -> pred 0 and GT 1 -> pred 1; frames 2-3 swap
        the predicted ids.  Both GT tracks switch once, so
        ``MOTA = (8 - 0 - 2) / 8``.

        ``AssA`` is ``1/3``, not ``1/2``: every (GT, pred) pair overlaps on
        2 frames while *both* trajectories are 4 frames long, so each
        pair's Jaccard is ``2 / (4 + 4 - 2)``.  A swap between two tracks
        costs more association accuracy than one track splitting in half
        (:meth:`test_single_identity_switch`), where the GT stays whole.
        """
        eye = [[1.0, 0.0], [0.0, 1.0]]
        records = _records(
            [[0, 1]] * 4,
            [[0, 1], [0, 1], [1, 0], [1, 0]],
            [eye, eye, eye, eye])
        mota, idf1, hota = _metrics()

        res = mota.compute_metrics(records)
        self.assertAlmostEqual(res['CLR_IDSW'], 2.0)
        self.assertAlmostEqual(res['MOTA'], 0.75)

        self.assertAlmostEqual(idf1.compute_metrics(records)['IDF1'], 0.5)

        res = hota.compute_metrics(records)
        self.assertAlmostEqual(res['DetA'], 1.0)
        self.assertAlmostEqual(res['AssA'], 1 / 3)

    def test_missing_track_ids_degrade_gracefully(self):
        """Frames with no ids at all contribute nothing and do not raise."""
        records = _records([[]] * 3, [[]] * 3,
                           [np.zeros((0, 0))] * 3)
        mota, idf1, hota = _metrics()

        self.assertAlmostEqual(mota.compute_metrics(records)['MOTA'], 0.0)
        self.assertAlmostEqual(idf1.compute_metrics(records)['IDF1'], 0.0)
        self.assertAlmostEqual(hota.compute_metrics(records)['HOTA'], 0.0)

    def test_idsw_reference_conventions(self):
        """``'last_seen'`` recovers a track after a gap, ``'prev_frame'`` not.

        Frame 0 matches GT 0 to prediction 0.  Frame 1 offers only a
        prediction too far away to match, so GT 0 goes unmatched and the
        previous-frame reference is cleared.  Frame 2 brings prediction 0
        back alongside a better-scoring prediction 1.

        Under TrackEval's previous-frame rule there is nothing to be sticky
        about, so the higher-scoring prediction 1 wins and that is a switch
        away from prediction 0.  Under py-motmetrics' last-seen rule
        prediction 0 is re-established and no switch is counted.
        """
        gt = [[0], [0], [0]]
        pred = [[0], [1], [0, 1]]
        sim = [[[1.0]], [[0.2]], [[0.6, 0.9]]]

        prev_frame = MOTA(match_thr=0.5, idsw_reference='prev_frame')
        last_seen = MOTA(match_thr=0.5, idsw_reference='last_seen')
        for metric in (prev_frame, last_seen):
            metric.dataset_meta = {'sigmas': np.ones(17, dtype=np.float32)}

        self.assertAlmostEqual(
            prev_frame.compute_metrics(_records(gt, pred, sim))['CLR_IDSW'],
            1.0)
        self.assertAlmostEqual(
            last_seen.compute_metrics(_records(gt, pred, sim))['CLR_IDSW'],
            0.0)

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            MOTA(similarity='cosine')
        with self.assertRaises(ValueError):
            MOTA(idsw_reference='nonsense')
        with self.assertRaises(ValueError):
            IDF1(score_mode='nope')
        with self.assertRaises(ValueError):
            MOTA(ignore_mode='nonsense')


class TestMOTIgnoreRegions(TestCase):
    """Ignore-region suppression in ``_MOTBaseMetric.process()``.

    Unlike the scenarios above, these go through ``process()`` itself
    (via :func:`_data_sample`), since that is where the suppression is
    applied -- ``compute_metrics`` never sees an ignore region, only
    whatever ``process()`` decided to keep.
    """

    def test_ioa_drops_prediction_fully_inside_large_ignore_region(self):
        """Reproduces the CoMotion bug figure (arXiv:2504.12186, Appx A.1):
        a box entirely inside a large ignore region scores a low IoU
        (``0.0025`` here) but the full ``1.0`` under the IoA convention.

        The frame has no real GT at all, only the ignore region: with no
        suppression the lone prediction is a pure false positive
        (``MOTA = (0 - 1 - 0) / 1``); suppressing it recovers ``MOTA = 0``.
        This exercises ``compute_oks_pairs`` reporting the prediction as
        valid even though ``valid_gt_idx`` is empty -- see the fix in
        :func:`~mmpose.evaluation.functional.frame_metrics.compute_oks_pairs`.
        """
        sample = _data_sample(
            gt_keypoints=[[[0.0, 0.0]]],
            gt_track_ids=[0],
            gt_iscrowd=[1],
            gt_bboxes=[[0.0, 0.0, 1000.0, 1000.0]],
            pred_keypoints=[[[400.0, 400.0]]],
            pred_track_ids=[7],
            pred_bboxes=[[400.0, 400.0, 450.0, 450.0]],
        )

        ioa_metric = _mota(ignore_mode='ioa')
        ioa_metric.process(None, [sample])
        res = ioa_metric.compute_metrics(ioa_metric.results)
        self.assertAlmostEqual(res['CLR_FP'], 0.0)
        self.assertAlmostEqual(res['MOTA'], 0.0)

        iou_metric = _mota(ignore_mode='iou')
        iou_metric.process(None, [sample])
        res = iou_metric.compute_metrics(iou_metric.results)
        self.assertAlmostEqual(res['CLR_FP'], 1.0)
        self.assertAlmostEqual(res['MOTA'], -1.0)

        disabled_metric = _mota(ignore_regions=False)
        disabled_metric.process(None, [sample])
        res = disabled_metric.compute_metrics(disabled_metric.results)
        self.assertAlmostEqual(res['CLR_FP'], 1.0)
        self.assertAlmostEqual(res['MOTA'], -1.0)

    def test_zero_valid_gt_frame_counts_prediction_as_fp(self):
        """The underlying ``compute_oks_pairs`` fix, isolated from
        suppression: with ``ignore_regions`` off, an all-crowd frame's
        prediction is a plain false positive -- it is no longer silently
        dropped before ``process()`` can even attribute it to anything.
        """
        sample = _data_sample(
            gt_keypoints=[[[0.0, 0.0]]],
            gt_track_ids=[0],
            gt_iscrowd=[1],
            pred_keypoints=[[[0.0, 0.0]]],
            pred_track_ids=[7],
        )
        metric = _mota(ignore_regions=False)
        metric.process(None, [sample])
        res = metric.compute_metrics(metric.results)
        self.assertAlmostEqual(res['CLR_FP'], 1.0)
        self.assertAlmostEqual(res['CLR_TP'], 0.0)
        self.assertAlmostEqual(res['CLR_FN'], 0.0)
        self.assertAlmostEqual(res['MOTA'], -1.0)

    def test_matched_prediction_is_protected_even_inside_ignore_region(self):
        """A real GT/prediction pair is kept even when an ignore region
        happens to fully cover them too -- the "occasionally completely
        overlap with ground-truth annotations" case CoMotion notes.
        """
        sample = _data_sample(
            gt_keypoints=[[[0.0, 0.0]], [[0.0, 0.0]]],
            gt_track_ids=[1, 0],
            gt_iscrowd=[0, 1],
            gt_bboxes=[[495.0, 495.0, 505.0, 505.0],
                       [300.0, 300.0, 1300.0, 1300.0]],
            pred_keypoints=[[[0.0, 0.0]]],
            pred_track_ids=[1],
            pred_bboxes=[[495.0, 495.0, 505.0, 505.0]],
        )
        metric = _mota()
        metric.process(None, [sample])
        res = metric.compute_metrics(metric.results)
        self.assertAlmostEqual(res['CLR_TP'], 1.0)
        self.assertAlmostEqual(res['CLR_FP'], 0.0)
        self.assertAlmostEqual(res['MOTA'], 1.0)

    def test_unmatched_prediction_dropped_alongside_a_real_match(self):
        """A second, unmatched prediction inside an ignore region is
        dropped while a genuine match elsewhere in the same frame stands.
        """
        sample = _data_sample(
            gt_keypoints=[[[0.0, 0.0]], [[0.0, 0.0]]],
            gt_track_ids=[1, 0],
            gt_iscrowd=[0, 1],
            gt_bboxes=[[0.0, 0.0, 10.0, 10.0],
                       [300.0, 300.0, 1300.0, 1300.0]],
            pred_keypoints=[[[0.0, 0.0]], [[999.0, 999.0]]],
            pred_track_ids=[1, 2],
            pred_bboxes=[[0.0, 0.0, 10.0, 10.0],
                         [700.0, 700.0, 750.0, 750.0]],
        )
        metric = _mota()
        metric.process(None, [sample])
        res = metric.compute_metrics(metric.results)
        self.assertAlmostEqual(res['CLR_TP'], 1.0)
        self.assertAlmostEqual(res['CLR_FP'], 0.0)
        self.assertAlmostEqual(res['MOTA'], 1.0)

    def test_ignore_region_with_no_overlap_changes_nothing(self):
        """An ignore region far from every prediction is a no-op."""
        sample = _data_sample(
            gt_keypoints=[[[0.0, 0.0]], [[0.0, 0.0]]],
            gt_track_ids=[1, 0],
            gt_iscrowd=[0, 1],
            gt_bboxes=[[0.0, 0.0, 10.0, 10.0],
                       [900.0, 900.0, 950.0, 950.0]],
            pred_keypoints=[[[0.0, 0.0]]],
            pred_track_ids=[1],
            pred_bboxes=[[0.0, 0.0, 10.0, 10.0]],
        )

        enabled = _mota()
        enabled.process(None, [sample])
        res_enabled = enabled.compute_metrics(enabled.results)

        disabled = _mota(ignore_regions=False)
        disabled.process(None, [sample])
        res_disabled = disabled.compute_metrics(disabled.results)

        self.assertEqual(res_enabled, res_disabled)
