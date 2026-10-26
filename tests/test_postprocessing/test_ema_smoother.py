# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for the exponential moving average pose smoother."""

from unittest import TestCase

import numpy as np
from mmengine.structures import InstanceData

from mmpose.postprocessing import build_post_processor
from mmpose.postprocessing.filters.ema_smoother import EMASmoother
from mmpose.structures import PoseDataSample


def _frame(xy, track_ids, img_path='seq/a/000.jpg', extra=None):
    """Build one prediction frame.

    Args:
        xy: Keypoint coordinates, shape ``(N, K, 2)``.
        track_ids: Per-instance ids, length ``N``.  ``None`` omits the field.
        extra: Optional third keypoint channel, shape ``(N, K)``.
    """
    xy = np.asarray(xy, dtype=np.float32)
    if extra is not None:
        kpts = np.concatenate(
            [xy, np.asarray(extra, dtype=np.float32)[..., None]], axis=-1)
    else:
        kpts = xy
    n, k = kpts.shape[:2]
    inst = InstanceData()
    inst.keypoints = kpts
    inst.keypoint_scores = np.full((n, k), 0.8, dtype=np.float32)
    inst.bboxes = np.zeros((n, 4), dtype=np.float32)
    if track_ids is not None:
        inst.track_ids = np.asarray(track_ids, dtype=np.int32)
    ds = PoseDataSample()
    ds.set_metainfo(dict(img_path=img_path, ori_shape=(480, 640)))
    ds.pred_instances = inst
    return ds


class TestEMASmoother(TestCase):

    def test_step_response(self):
        """A jump from 0 to 10 follows the closed-form EMA."""
        alpha = 0.5
        smoother = EMASmoother(alpha=alpha)
        observations = [0.0, 10.0, 10.0, 10.0]
        expected = 0.0
        for t, x in enumerate(observations):
            xy = np.full((1, 3, 2), x, dtype=np.float32)
            out = smoother.process_frame(_frame(xy, [7]), 'seq/a')
            got = np.asarray(out.pred_instances.keypoints)
            if t == 0:
                expected = x
            else:
                expected = alpha * x + (1.0 - alpha) * expected
            np.testing.assert_allclose(got, expected, atol=1e-6)
            np.testing.assert_allclose(
                out.pred_instances.keypoint_scores, 0.8)

    def test_tracks_are_independent(self):
        smoother = EMASmoother(alpha=0.5)
        still = np.zeros((1, 2, 2), dtype=np.float32)
        moving = np.full((1, 2, 2), 10.0, dtype=np.float32)
        smoother.process_frame(_frame(still, [1]), 'seq/a')
        smoother.process_frame(_frame(moving, [2]), 'seq/a')

        both = np.concatenate(
            [still, np.full_like(moving, 10.0)], axis=0)
        out = smoother.process_frame(_frame(both, [1, 2]), 'seq/a')
        kpts = np.asarray(out.pred_instances.keypoints)
        np.testing.assert_allclose(kpts[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(kpts[1], 10.0, atol=1e-6)

    def test_reset_restarts_from_the_observation(self):
        smoother = EMASmoother(alpha=0.25)
        origin = np.zeros((1, 1, 2), dtype=np.float32)
        jumped = np.full((1, 1, 2), 8.0, dtype=np.float32)
        smoother.process_frame(_frame(origin, [1]), 'seq/a')
        lagged = smoother.process_frame(_frame(jumped, [1]), 'seq/a')
        self.assertLess(float(np.asarray(lagged.pred_instances.keypoints).max()),
                        8.0)
        smoother.reset()
        fresh = smoother.process_frame(_frame(jumped, [1]), 'seq/b')
        np.testing.assert_allclose(
            fresh.pred_instances.keypoints, jumped, atol=1e-6)

    def test_preserves_extra_keypoint_channel(self):
        smoother = EMASmoother(alpha=0.5)
        xy = np.zeros((1, 2, 2), dtype=np.float32)
        extra = np.array([[0.2, 0.9]], dtype=np.float32)
        smoother.process_frame(_frame(xy, [1], extra=extra), 'seq/a')
        moved = np.ones((1, 2, 2), dtype=np.float32)
        out = smoother.process_frame(
            _frame(moved, [1], extra=extra), 'seq/a')
        kpts = np.asarray(out.pred_instances.keypoints)
        self.assertEqual(kpts.shape[-1], 3)
        np.testing.assert_allclose(kpts[0, :, 2], extra[0])
        np.testing.assert_allclose(kpts[0, :, :2], 0.5)

    def test_requires_track_ids(self):
        smoother = EMASmoother()
        xy = np.zeros((1, 2, 2), dtype=np.float32)
        with self.assertRaises(RuntimeError):
            smoother.process_frame(_frame(xy, None), 'seq/a')

    def test_empty_frame_is_unchanged(self):
        smoother = EMASmoother()
        ds = _frame(np.zeros((0, 2, 2), dtype=np.float32), [])
        self.assertIs(smoother.process_frame(ds, 'seq/a'), ds)

    def test_rejects_alpha_outside_unit_interval(self):
        with self.assertRaises(ValueError):
            EMASmoother(alpha=0.0)
        with self.assertRaises(ValueError):
            EMASmoother(alpha=1.5)

    def test_config_builds(self):
        pipeline = build_post_processor(
            'configs/post_processing/oks_track_ema.py')
        self.assertEqual(
            [type(f).__name__ for f in pipeline.filters],
            ['OKSTracker', 'EMASmoother'])
        self.assertTrue(pipeline.filters[-1].online)
        self.assertAlmostEqual(pipeline.filters[-1].alpha, 0.4)
