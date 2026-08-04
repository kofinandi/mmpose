# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for :func:`~mmpose.evaluation.functional.frame_metrics.compute_oks_pairs`.

Focused on ``valid_pred_idx``, the return value that was previously
(incorrectly) forced empty whenever ``valid_gt_idx`` was empty -- see the
docstring note in ``compute_oks_pairs`` for why that mattered to callers
such as :class:`~mmpose.evaluation.metrics.MOTA` that count unmatched
predictions as false positives.
"""

from unittest import TestCase

import numpy as np

from mmpose.evaluation.functional.frame_metrics import compute_oks_pairs

_SIGMAS = np.ones(1, dtype=np.float32)


class TestComputeOksPairs(TestCase):

    def test_valid_pred_idx_independent_of_all_crowd_gt(self):
        """A frame with only crowd GT still reports its predictions.

        ``valid_gt_idx`` is empty (the sole GT is crowd), but the
        prediction has keypoints and must still show up in
        ``valid_pred_idx`` -- that is what lets a caller correctly count
        it as an unmatched, present detection rather than silently losing
        track of it.
        """
        gt_list = [{
            'keypoints': np.zeros((1, 2)),
            'keypoints_visible': np.ones(1),
            'iscrowd': 1,
        }]
        pred_list = [{'keypoints': np.zeros((1, 2))}]

        pairs, valid_gt_idx, valid_pred_idx = compute_oks_pairs(
            gt_list, pred_list, _SIGMAS)

        self.assertEqual(pairs, [])
        self.assertEqual(valid_gt_idx, [])
        self.assertEqual(valid_pred_idx, [0])

    def test_valid_pred_idx_independent_of_no_gt_at_all(self):
        """Same, but for a frame with no GT instances whatsoever."""
        pairs, valid_gt_idx, valid_pred_idx = compute_oks_pairs(
            [], [{'keypoints': np.zeros((1, 2))}], _SIGMAS)

        self.assertEqual(pairs, [])
        self.assertEqual(valid_gt_idx, [])
        self.assertEqual(valid_pred_idx, [0])

    def test_keypoint_free_predictions_are_never_valid(self):
        """A prediction with no keypoints is excluded regardless of GT."""
        gt_list = [{
            'keypoints': np.zeros((1, 2)),
            'keypoints_visible': np.ones(1),
            'iscrowd': 1,
        }]
        pred_list = [{'keypoints': None}, {'keypoints': np.zeros((1, 2))}]

        _, valid_gt_idx, valid_pred_idx = compute_oks_pairs(
            gt_list, pred_list, _SIGMAS)

        self.assertEqual(valid_gt_idx, [])
        self.assertEqual(valid_pred_idx, [1])

    def test_no_sigmas_reports_no_valid_predictions(self):
        """``num_kpts == 0`` degenerates to no valid predictions either."""
        _, valid_gt_idx, valid_pred_idx = compute_oks_pairs(
            [], [{'keypoints': np.zeros((1, 2))}], np.zeros(0))

        self.assertEqual(valid_gt_idx, [])
        self.assertEqual(valid_pred_idx, [])

    def test_normal_matching_is_unaffected(self):
        """A real GT/prediction pair still scores and matches as before."""
        gt_list = [{
            'keypoints': np.zeros((1, 2)),
            'keypoints_visible': np.ones(1),
            'iscrowd': 0,
            'bbox': [0.0, 0.0, 10.0, 10.0],
        }]
        pred_list = [{'keypoints': np.zeros((1, 2))}]

        pairs, valid_gt_idx, valid_pred_idx = compute_oks_pairs(
            gt_list, pred_list, _SIGMAS)

        self.assertEqual(valid_gt_idx, [0])
        self.assertEqual(valid_pred_idx, [0])
        self.assertEqual(len(pairs), 1)
        oks, gi, pi = pairs[0]
        self.assertAlmostEqual(oks, 1.0)
        self.assertEqual((gi, pi), (0, 0))
