# Copyright (c) OpenMMLab. All rights reserved.
"""Keypoint-space conversion and torso-index resolution."""

from pathlib import Path
from unittest import TestCase

import numpy as np

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.datasets.transforms.keypoint_registry import get_keypoints
from mmpose.evaluation.functional.keypoint_space import (
    COCO_TORSO_PAIRS, convert_keypoint_arrays, eval_convention,
    make_gt_converter, torso_index_pairs)
from mmpose.models.pose_estimators.detrpose_motip_wrapper import (
    DETRPoseMOTIPEstimator, PT21_15_KEYPOINT_NAMES)

_REPO = Path(__file__).resolve().parents[3]


def _meta(name: str) -> dict:
    return parse_pose_metainfo(
        dict(from_file=str(_REPO / 'configs' / '_base_' / 'datasets' / name)))


class TestKeypointSpace(TestCase):

    def test_convention_switch(self):
        self.assertEqual(eval_convention('posetrack18', 'coco'), 'coco')
        self.assertEqual(
            eval_convention('posetrack18', 'dataset'), 'posetrack18')
        self.assertIsNone(make_gt_converter('posetrack18', 'dataset'))
        converter = make_gt_converter('posetrack18', 'coco')
        self.assertEqual(converter.dst, 'coco')
        with self.assertRaises(ValueError):
            eval_convention('coco', 'native')

    def test_posetrack_to_coco_drops_head_joints(self):
        src = get_keypoints('posetrack18')
        dst = get_keypoints('coco')
        coords = np.zeros((1, len(src), 2), dtype=np.float32)
        coords[0, src.index('nose')] = [1.0, 2.0]
        coords[0, src.index('head_bottom')] = [12.0, 34.0]
        coords[0, src.index('left_shoulder')] = [3.0, 4.0]
        scores = np.ones((1, len(src)), dtype=np.float32)
        out, mapped = convert_keypoint_arrays(coords, scores, src, dst)
        self.assertEqual(out.shape, (1, 17, 2))
        np.testing.assert_allclose(out[0, dst.index('nose')], [1.0, 2.0])
        np.testing.assert_allclose(
            out[0, dst.index('left_shoulder')], [3.0, 4.0])
        self.assertFalse(np.any(np.all(np.isclose(out[0], [12.0, 34.0]),
                                        axis=-1)))
        self.assertEqual(mapped[0, dst.index('left_eye')], 0.0)

    def test_coco_to_posetrack_does_not_invent_head_joints(self):
        src = get_keypoints('coco')
        dst = get_keypoints('posetrack18')
        coords = np.zeros((1, len(src), 2), dtype=np.float32)
        coords[0, src.index('nose')] = [5.0, 6.0]
        coords[0, src.index('left_shoulder')] = [7.0, 8.0]
        scores = np.ones((1, len(src)), dtype=np.float32)
        out, mapped = convert_keypoint_arrays(coords, scores, src, dst)
        np.testing.assert_allclose(out[0, dst.index('nose')], [5.0, 6.0])
        np.testing.assert_allclose(
            out[0, dst.index('left_shoulder')], [7.0, 8.0])
        np.testing.assert_allclose(out[0, dst.index('head_bottom')], [0, 0])
        np.testing.assert_allclose(out[0, dst.index('head_top')], [0, 0])
        self.assertEqual(mapped[0, dst.index('head_bottom')], 0.0)
        self.assertEqual(mapped[0, dst.index('head_top')], 0.0)

    def test_native_15_keeps_heads_and_drops_ears(self):
        dst = get_keypoints('posetrack18')
        coords = np.arange(
            len(PT21_15_KEYPOINT_NAMES) * 2, dtype=np.float32).reshape(
                1, len(PT21_15_KEYPOINT_NAMES), 2)
        scores = np.full((1, len(PT21_15_KEYPOINT_NAMES)), 0.5)
        out, mapped = convert_keypoint_arrays(
            coords, scores, PT21_15_KEYPOINT_NAMES, dst)
        for name in ('nose', 'head_bottom', 'head_top', 'left_ankle'):
            src_i = PT21_15_KEYPOINT_NAMES.index(name)
            np.testing.assert_allclose(
                out[0, dst.index(name)], coords[0, src_i])
            self.assertAlmostEqual(mapped[0, dst.index(name)], 0.5)
        self.assertEqual(mapped[0, dst.index('left_ear')], 0.0)
        self.assertEqual(mapped[0, dst.index('right_ear')], 0.0)

    def test_identical_layout_is_a_copy(self):
        names = get_keypoints('posetrack18')
        coords = np.arange(34, dtype=np.float32).reshape(1, 17, 2)
        scores = np.linspace(0.1, 0.9, 17, dtype=np.float32).reshape(1, 17)
        out, mapped = convert_keypoint_arrays(coords, scores, names, names)
        np.testing.assert_allclose(out, coords)
        np.testing.assert_allclose(mapped, scores)

    def test_motip_output_names(self):
        coco = type('M', (), {'map_to_coco': True})()
        native = type('M', (), {'map_to_coco': False})()
        coco_names = DETRPoseMOTIPEstimator.output_keypoint_names.fget(coco)
        native_names = DETRPoseMOTIPEstimator.output_keypoint_names.fget(
            native)
        self.assertEqual(coco_names, get_keypoints('coco'))
        self.assertEqual(list(native_names), list(PT21_15_KEYPOINT_NAMES))
        self.assertIn('head_bottom', native_names)
        self.assertNotIn('left_ear', native_names)


class TestTorsoPairs(TestCase):

    def test_coco_and_fallback(self):
        self.assertEqual(torso_index_pairs(None), COCO_TORSO_PAIRS)
        self.assertEqual(torso_index_pairs({}), COCO_TORSO_PAIRS)
        self.assertEqual(torso_index_pairs(_meta('coco.py')), ((5, 12), (6, 11)))

    def test_crowdpose_uses_its_own_indices(self):
        # left_shoulder=0, right_shoulder=1, left_hip=6, right_hip=7.
        self.assertEqual(
            torso_index_pairs(_meta('crowdpose.py')), ((0, 7), (1, 6)))
