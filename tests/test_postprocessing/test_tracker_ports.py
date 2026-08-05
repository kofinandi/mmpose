# Copyright (c) OpenMMLab. All rights reserved.
"""Equivalence tests for the ported tracker components.

Each of the three trackers in ``mmpose/postprocessing`` reimplements
algorithms that live in an unimportable upstream module (Python 2, Caffe2,
compiled extensions, module-scope side effects).  These tests pin the ports
to independent transcriptions of the upstream source, so a refactor that
silently changes the algorithm fails here rather than showing up as a
slightly different tracking metric months later.
"""

from unittest import TestCase

import numpy as np
import torch

from mmpose.postprocessing.filters.detect_and_track_linker import (
    bbox_overlaps, bipartite_matching_greedy, compute_head_size, pck_distance)
from mmpose.postprocessing.filters.lighttrack_tracker import (
    bbox_invalid, enlarge_bbox, iou)
from mmpose.postprocessing.filters.pgpt_tracker import bbox_iou_matrix
from mmpose.postprocessing.matchers import (COCO17_TO_PGPT15, PGPT15_SIGMAS,
                                            convert_keypoints,
                                            to_lighttrack15)
from mmpose.postprocessing.matchers.pgpt_embedder import (
    _align_features, _align_features_reference, pgpt_adjacency)


def _upstream_greedy(cost):
    """Verbatim ``bipartite_matching_greedy`` from DetectAndTrack."""
    cost = cost.copy()
    prev_ids, cur_ids = [], []
    row_ids = np.arange(cost.shape[0])
    col_ids = np.arange(cost.shape[1])
    while cost.size > 0:
        i, j = np.unravel_index(cost.argmin(), cost.shape)
        prev_ids.append(row_ids[i])
        cur_ids.append(col_ids[j])
        cost = np.delete(cost, i, 0)
        cost = np.delete(cost, j, 1)
        row_ids = np.delete(row_ids, i, 0)
        col_ids = np.delete(col_ids, j, 0)
    return prev_ids, cur_ids


def _upstream_enlarge(bbox, scale):
    """Verbatim ``enlarge_bbox`` from lighttrack (xyxy in, xyxy out)."""
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_x < 0:
        margin_x = 2
    if margin_y < 0:
        margin_y = 2
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y
    width, height = max_x - min_x, max_y - min_y
    if (max_y < 0 or max_x < 0 or width <= 0 or height <= 0
            or width > 2000 or height > 2000):
        return [0, 0, 2, 2]
    return [min_x, min_y, max_x, max_y]


def _upstream_iou(box_a, box_b):
    """Verbatim ``iou`` from lighttrack, including the +1 inflation."""
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    return inter / float(area_a + area_b - inter)


def _random_boxes(rng, n, scale=200.0):
    xy = rng.random((n, 2)) * scale
    wh = 1.0 + rng.random((n, 2)) * scale
    return np.hstack([xy, xy + wh]).astype(np.float32)


class TestDetectAndTrackPorts(TestCase):
    """Detect-and-Track (CVPR'18) linking primitives."""

    def test_greedy_matching_matches_upstream(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            m, n = int(rng.integers(0, 6)), int(rng.integers(0, 6))
            cost = rng.random((m, n))
            ours = bipartite_matching_greedy(cost)
            theirs = _upstream_greedy(cost)
            self.assertEqual([list(map(int, x)) for x in ours],
                             [list(map(int, x)) for x in theirs])

    def test_bbox_overlaps_is_inclusive(self):
        # Cython bbox_overlaps measures extents with a +1 on each side.
        rng = np.random.default_rng(1)
        a, b = _random_boxes(rng, 5), _random_boxes(rng, 4)
        expected = np.array([[_upstream_iou(x, y) for y in b] for x in a])
        np.testing.assert_allclose(bbox_overlaps(a, b), expected, atol=1e-6)

    def test_bbox_overlaps_edge_cases(self):
        box = np.array([[0., 0., 10., 10.]], dtype=np.float32)
        far = np.array([[100., 100., 110., 110.]], dtype=np.float32)
        self.assertAlmostEqual(float(bbox_overlaps(box, box)[0, 0]), 1.0)
        self.assertEqual(float(bbox_overlaps(box, far)[0, 0]), 0.0)
        self.assertEqual(bbox_overlaps(np.zeros((0, 4)), box).shape, (0, 1))

    def test_head_size_floor(self):
        kpts = np.zeros((17, 2), dtype=np.float32)
        kpts[3], kpts[4] = [0., 0.], [10., 0.]      # ears 10 px apart
        # Upstream adds 1 to avoid zero-division.
        self.assertAlmostEqual(compute_head_size(kpts, (3, 4)), 11.0)
        # A collapsed ear pair would make every joint a PCK match; the
        # floor keeps the normaliser tied to the person's size.
        kpts[3] = kpts[4]
        bbox = np.array([0., 0., 100., 100.], dtype=np.float32)
        self.assertAlmostEqual(
            compute_head_size(kpts, (3, 4), bbox, 0.1),
            0.1 * float(np.hypot(100., 100.)), places=4)

    def test_pck_distance_bounds(self):
        kpts = np.random.default_rng(2).random((17, 2)).astype(np.float32)
        self.assertEqual(pck_distance(kpts, kpts, 11.0), 0.0)
        self.assertEqual(pck_distance(kpts, kpts + 1e4, 11.0), 1.0)


class TestLightTrackPorts(TestCase):
    """LightTrack (CVPRW'20) box helpers."""

    def test_enlarge_bbox_matches_upstream(self):
        rng = np.random.default_rng(3)
        for box in _random_boxes(rng, 200, scale=600.0):
            np.testing.assert_allclose(
                enlarge_bbox(box, 0.2),
                np.array(_upstream_enlarge([float(v) for v in box], 0.2),
                         dtype=np.float32))

    def test_oversized_box_collapses_to_sentinel(self):
        huge = np.array([0., 0., 2500., 2500.], dtype=np.float32)
        np.testing.assert_array_equal(
            enlarge_bbox(huge, 0.2), np.array([0., 0., 2., 2.]))
        self.assertTrue(bbox_invalid(enlarge_bbox(huge, 0.2)))

    def test_bbox_invalid(self):
        self.assertTrue(bbox_invalid(np.array([0., 0., 2., 2.])))     # sentinel
        self.assertTrue(bbox_invalid(np.array([10., 10., 5., 20.])))  # w <= 0
        self.assertTrue(bbox_invalid(np.array([0., 0., 3000., 10.])))  # too wide
        self.assertFalse(bbox_invalid(np.array([10., 10., 110., 210.])))

    def test_iou_matches_upstream(self):
        rng = np.random.default_rng(4)
        boxes_a, boxes_b = _random_boxes(rng, 50), _random_boxes(rng, 50)
        for a, b in zip(boxes_a, boxes_b):
            self.assertAlmostEqual(
                iou(a, b), _upstream_iou(a, b), places=6)


class TestKeypointLayouts(TestCase):
    """COCO-17 conversions to the two 15-joint layouts."""

    def test_pgpt15_is_coco_without_ears(self):
        kpts = np.arange(34, dtype=np.float32).reshape(17, 2)
        np.testing.assert_array_equal(
            convert_keypoints(kpts, COCO17_TO_PGPT15),
            np.delete(kpts, [3, 4], axis=0))

    def test_pgpt_sigmas_are_coco_without_ears(self):
        coco = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,
                         .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
        np.testing.assert_allclose(
            PGPT15_SIGMAS, np.delete(coco, [3, 4]), atol=1e-8)

    def test_lighttrack15_ordering_and_synthesis(self):
        kpts = np.arange(34, dtype=np.float32).reshape(17, 2)
        out = to_lighttrack15(kpts)
        self.assertEqual(out.shape, (15, 2))
        np.testing.assert_array_equal(out[0], kpts[16])   # r_ankle
        np.testing.assert_array_equal(out[8], kpts[6])    # r_shoulder
        np.testing.assert_array_equal(out[13], kpts[0])   # nose
        # head_bottom is the shoulder midpoint...
        np.testing.assert_allclose(out[12], 0.5 * (kpts[5] + kpts[6]))
        # ...and head_top extrapolates past the nose away from it.
        np.testing.assert_allclose(
            out[14], kpts[0] + 0.55 * (kpts[0] - out[12]), atol=1e-5)

    def test_lighttrack15_reads_only_shared_indices(self):
        # COCO-17 and PoseTrack-17 agree on the nose (0) and every limb
        # joint (5-16) but disagree at 1-4 (eyes+ears vs head joints+ears).
        # The conversion must read none of 1-4, so that the SGCN sees the
        # same representation whether it is fed PoseTrack GT during
        # training or COCO predictions at inference.
        base = np.arange(34, dtype=np.float32).reshape(17, 2)
        perturbed = base.copy()
        perturbed[1:5] = -999.0
        np.testing.assert_array_equal(
            to_lighttrack15(perturbed), to_lighttrack15(base))

    def test_batched_conversion(self):
        kpts = np.zeros((4, 17, 2), dtype=np.float32)
        self.assertEqual(to_lighttrack15(kpts).shape, (4, 15, 2))
        self.assertEqual(
            convert_keypoints(kpts, COCO17_TO_PGPT15).shape, (4, 15, 2))


class TestPGPTPorts(TestCase):
    """PGPT (TMM'20) matching and feature-alignment primitives."""

    def test_bbox_iou_matrix(self):
        a = np.array([[0., 0., 10., 10.]], dtype=np.float32)
        b = np.array([[0., 0., 10., 10.], [5., 5., 15., 15.],
                      [100., 100., 110., 110.]], dtype=np.float32)
        got = bbox_iou_matrix(a, b)[0]
        self.assertAlmostEqual(float(got[0]), 1.0)
        self.assertAlmostEqual(float(got[1]), 25. / 175., places=5)
        self.assertEqual(float(got[2]), 0.0)

    def test_adjacency_values(self):
        adj = pgpt_adjacency()
        self.assertEqual(tuple(adj.shape), (15, 15))
        self.assertAlmostEqual(float(adj[0, 0]), 0.9)    # self-connection
        self.assertAlmostEqual(float(adj[0, 1]), 1.0)    # skeleton edge
        self.assertAlmostEqual(float(adj[0, 14]), 0.2)   # non-edge

    def test_vectorised_alignment_matches_literal_port(self):
        torch.manual_seed(0)
        for _ in range(3):
            hp = torch.randn(2, 15, 12, 9)
            emb = torch.randn(2, 256, 12, 9)
            torch.testing.assert_close(
                _align_features(hp, emb), _align_features_reference(hp, emb))

    def test_vectorised_alignment_on_clamped_borders(self):
        # Peaks on every edge and corner exercise the dedup + mean-padding
        # path, where the mask has fewer than 9 unique positions.
        hp = torch.full((1, 15, 12, 9), -10.0)
        spots = [(0, 0), (0, 4), (0, 8), (11, 0), (11, 8), (5, 0), (5, 8),
                 (11, 4), (6, 3), (1, 1), (10, 7), (0, 1), (1, 0), (11, 7),
                 (7, 8)]
        for j, (y, x) in enumerate(spots):
            hp[0, j, y, x] = 5.0
        emb = torch.randn(1, 256, 12, 9)
        torch.testing.assert_close(
            _align_features(hp, emb), _align_features_reference(hp, emb))
