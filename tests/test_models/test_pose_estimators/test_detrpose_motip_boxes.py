# Copyright (c) OpenMMLab. All rights reserved.
"""Keypoint-extent boxes and the PoseTrack21-train expansion."""

from unittest import TestCase

import numpy as np

from mmpose.models.pose_estimators.detrpose_motip_wrapper import (
    _PT21_BOX_EXPAND, _bboxes_from_keypoints, _expand_keypoint_boxes)


class TestKeypointBoxes(TestCase):

    def test_extent_is_min_max(self):
        keypoints = np.array([[[0, 10], [4, 2], [1, 8]]], dtype=np.float32)
        boxes = _bboxes_from_keypoints(keypoints)
        np.testing.assert_allclose(boxes, [[0, 2, 4, 10]])

    def test_expansion_uses_the_train_fit(self):
        # width 20, height 40.  0.17 / 0.03 / 0.11 of those.
        extent = np.array([[10, 20, 30, 60]], dtype=np.float32)
        left_right, top, bottom = _PT21_BOX_EXPAND
        out = _expand_keypoint_boxes(extent, left_right, top, bottom)
        np.testing.assert_allclose(
            out[0], [10 - 3.4, 20 - 1.2, 30 + 3.4, 60 + 4.4], atol=1e-5)
        self.assertEqual(_PT21_BOX_EXPAND, (0.17, 0.03, 0.11))

    def test_empty_extent_stays_empty(self):
        empty = np.zeros((0, 4), dtype=np.float32)
        out = _expand_keypoint_boxes(empty, *_PT21_BOX_EXPAND)
        self.assertEqual(out.shape, (0, 4))
        self.assertEqual(_bboxes_from_keypoints(
            np.zeros((0, 15, 2), dtype=np.float32)).shape, (0, 4))
