# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for :mod:`mmpose.evaluation.functional.good_frames`."""

from unittest import TestCase

from mmpose.evaluation.functional.good_frames import (
    BAD_FRAME_DATASETS,
    frame_record_is_good,
    partition_frame_records,
)


def _frame(*, good_frame=None, instances=None):
    """Build a minimal frame-record dict for testing."""
    frame = {
        'ground_truth': {
            'instances': [] if instances is None else instances,
        },
    }
    if good_frame is not None:
        frame['good_frame'] = good_frame
    return frame


class TestFrameRecordIsGood(TestCase):

    def test_explicit_true(self):
        self.assertTrue(
            frame_record_is_good(_frame(good_frame=True),
                                 allow_heuristic=False))

    def test_explicit_false(self):
        self.assertFalse(
            frame_record_is_good(_frame(good_frame=False),
                                 allow_heuristic=False))

    def test_explicit_false_wins_over_heuristic(self):
        """An explicit False is trusted even when non-crowd GT is present."""
        frame = _frame(
            good_frame=False,
            instances=[{'iscrowd': 0, 'keypoints': [[0, 0]]}],
        )
        self.assertFalse(
            frame_record_is_good(frame, allow_heuristic=True))

    def test_missing_key_heuristic_off_is_good(self):
        """Legacy frames without the key stay evaluable by default."""
        self.assertTrue(
            frame_record_is_good(_frame(), allow_heuristic=False))

    def test_missing_key_heuristic_on_zero_gt_is_bad(self):
        self.assertFalse(
            frame_record_is_good(_frame(instances=[]),
                                 allow_heuristic=True))

    def test_missing_key_heuristic_on_non_crowd_gt_is_good(self):
        frame = _frame(instances=[{'iscrowd': 0}])
        self.assertTrue(
            frame_record_is_good(frame, allow_heuristic=True))

    def test_posetrack_ignore_only_is_bad_under_heuristic(self):
        """PoseTrack21 unlabeled frames may carry only iscrowd=1 regions."""
        frame = _frame(instances=[{'iscrowd': 1}, {'iscrowd': 1}])
        self.assertFalse(
            frame_record_is_good(frame, allow_heuristic=True))


class TestPartitionFrameRecords(TestCase):

    def test_partition_reports_heuristic_use(self):
        frames = [
            _frame(good_frame=True),
            _frame(instances=[]),  # missing key → heuristic
            _frame(good_frame=False),
            _frame(instances=[{'iscrowd': 0}]),  # missing key → heuristic
        ]
        good_idxs, used_heuristic = partition_frame_records(
            frames, allow_heuristic=True)
        self.assertEqual(good_idxs, [0, 3])
        self.assertTrue(used_heuristic)

    def test_partition_no_heuristic_keeps_missing_keys(self):
        frames = [
            _frame(good_frame=True),
            _frame(instances=[]),
            _frame(good_frame=False),
        ]
        good_idxs, used_heuristic = partition_frame_records(
            frames, allow_heuristic=False)
        self.assertEqual(good_idxs, [0, 1])
        self.assertFalse(used_heuristic)

    def test_bad_frame_datasets_contains_expected(self):
        self.assertIn('emdb', BAD_FRAME_DATASETS)
        self.assertIn('emdb-mini', BAD_FRAME_DATASETS)
        self.assertIn('3dpw', BAD_FRAME_DATASETS)
        self.assertIn('posetrack21', BAD_FRAME_DATASETS)
        self.assertNotIn('coco', BAD_FRAME_DATASETS)
