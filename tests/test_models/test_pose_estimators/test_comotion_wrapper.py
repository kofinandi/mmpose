# Copyright (c) OpenMMLab. All rights reserved.
"""PoseTrack-17 layout for CoMotion's COCO regressor, including the head fill."""

from unittest import TestCase

import numpy as np

from mmpose.models.pose_estimators.comotion_wrapper import (
    _HEAD_FILL_A_BOTTOM, _HEAD_FILL_B_TOP, _HEAD_FILL_NUM_PERSONS,
    posetrack17_from_coco)


class TestCoMotionKeypoints(TestCase):

    def test_head_fill_uses_the_train_fit(self):
        coco = np.zeros((1, 17, 2), dtype=np.float32)
        coco[0, 0] = [10, 20]  # nose
        coco[0, 1] = [99, 99]  # left eye, not a PoseTrack joint
        coco[0, 5] = [0, 10]  # left shoulder
        coco[0, 6] = [20, 10]  # right shoulder
        kpts, scores = posetrack17_from_coco(coco, fill_head=True)
        # shoulder mid (10, 10), delta nose-mid (0, 10)
        self.assertAlmostEqual(
            float(kpts[0, 1, 1]), 10 + _HEAD_FILL_A_BOTTOM * 10, places=5)
        self.assertAlmostEqual(
            float(kpts[0, 2, 1]), 20 + _HEAD_FILL_B_TOP * 10, places=5)
        self.assertEqual(float(scores[0, 1]), 1.0)
        self.assertEqual(float(scores[0, 2]), 1.0)
        self.assertFalse(np.any(kpts == 99))
        self.assertEqual(_HEAD_FILL_NUM_PERSONS, 87069)

    def test_head_joints_stay_empty_without_the_fill(self):
        coco = np.zeros((1, 17, 2), dtype=np.float32)
        coco[0, 0] = [10, 20]
        coco[0, 5] = [0, 10]
        coco[0, 6] = [20, 10]
        kpts, scores = posetrack17_from_coco(coco, fill_head=False)
        self.assertTrue(np.all(kpts[0, 1] == 0))
        self.assertTrue(np.all(kpts[0, 2] == 0))
        self.assertEqual(float(scores[0, 1]), 0.0)
        self.assertEqual(float(scores[0, 2]), 0.0)
        # left shoulder is PoseTrack index 5, copied from COCO.
        np.testing.assert_allclose(kpts[0, 5], [0, 10])
        self.assertEqual(float(scores[0, 5]), 1.0)
