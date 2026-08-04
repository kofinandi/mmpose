# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for the ``IDSwitch`` tracking metric."""

from unittest import TestCase

import numpy as np

from mmpose.evaluation.metrics import IDSwitch


def _data_sample(gt_keypoints,
                  gt_track_ids,
                  pred_keypoints,
                  pred_track_ids,
                  gt_iscrowd=None,
                  img_path: str = 'seq/000001.jpg') -> dict:
    """Build a single-frame ``data_sample`` dict for ``process()``."""
    gt_kpts = np.asarray(gt_keypoints, dtype=np.float32)
    n_gt = gt_kpts.shape[0]
    return {
        'img_path': img_path,
        'gt_instances': {
            'keypoints': gt_kpts,
            'keypoints_visible': np.ones(gt_kpts.shape[:2], dtype=np.float32),
            'track_ids': np.asarray(gt_track_ids, dtype=np.int64),
            'iscrowd': (np.asarray(gt_iscrowd, dtype=np.int64)
                        if gt_iscrowd is not None else
                        np.zeros(n_gt, dtype=np.int64)),
        },
        'pred_instances': {
            'keypoints': np.asarray(pred_keypoints, dtype=np.float32),
            'track_ids': np.asarray(pred_track_ids, dtype=np.int64),
        },
    }


class TestIDSwitch(TestCase):

    def test_all_crowd_frame_with_a_prediction_contributes_no_switch(self):
        """An all-crowd GT frame reports no matches no matter how many
        predictions are present.

        This pins ``IDSwitch`` against the fix to ``compute_oks_pairs``
        that made ``valid_pred_idx`` independent of ``valid_gt_idx`` (see
        the ``mmpose.evaluation.metrics.MOTA`` ignore-region tests for the
        metric that fix *does* change): matching here still requires a GT
        track (``gt_track_id``, built from ``valid_gt_idx``) to iterate
        over, so an all-crowd frame contributes nothing regardless of how
        many predictions ``valid_pred_idx`` now reports.
        """
        sample = _data_sample(
            gt_keypoints=[[[0.0, 0.0]]],
            gt_track_ids=[0],
            gt_iscrowd=[1],
            pred_keypoints=[[[0.0, 0.0]]],
            pred_track_ids=[7],
        )
        metric = IDSwitch()
        metric.dataset_meta = {'sigmas': np.ones(1, dtype=np.float32)}
        metric.process(None, [sample])

        res = metric.compute_metrics(metric.results)
        self.assertAlmostEqual(res['IDSwitches'], 0.0)
        self.assertAlmostEqual(res['IDSwitchesPer100Frames'], 0.0)

    def test_normal_switch_counting_is_unaffected(self):
        """Sanity check: ordinary switch counting still works.

        One GT track followed by prediction 0, then a switch to
        prediction 1.
        """
        metric = IDSwitch()
        metric.dataset_meta = {'sigmas': np.ones(1, dtype=np.float32)}
        metric.process(None, [
            _data_sample(
                gt_keypoints=[[[0.0, 0.0]]],
                gt_track_ids=[0],
                pred_keypoints=[[[0.0, 0.0]]],
                pred_track_ids=[1]),
            _data_sample(
                gt_keypoints=[[[0.0, 0.0]]],
                gt_track_ids=[0],
                pred_keypoints=[[[0.0, 0.0]]],
                pred_track_ids=[2]),
        ])

        res = metric.compute_metrics(metric.results)
        self.assertAlmostEqual(res['IDSwitches'], 1.0)
