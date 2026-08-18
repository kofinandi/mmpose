# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for the BoxMOT post-processor wrapper.

Skipped when ``boxmot`` is not installed.  These pin the wrapper contract
(id stability, ``det_ind`` keypoint alignment, sequence reset,
``keep_untracked``) rather than BoxMOT's association itself.
"""

from unittest import TestCase, skipUnless

import numpy as np
from mmengine.structures import InstanceData

from mmpose.structures import PoseDataSample

try:
    import boxmot  # noqa: F401
    _HAS_BOXMOT = True
except ImportError:
    _HAS_BOXMOT = False

if _HAS_BOXMOT:
    from mmpose.postprocessing.filters.boxmot_tracker import BoxMOTTracker


def _make_ds(bboxes, scores, tags, img_path='seq/a/000.jpg', shape=(480, 640)):
    """One frame: each instance's keypoints are tagged with a unique x."""
    n = len(bboxes)
    kpts = np.zeros((n, 17, 2), dtype=np.float32)
    kpts[:, :, 0] = np.asarray(tags, dtype=np.float32)[:, None]
    kpts[:, :, 1] = 10.0
    inst = InstanceData()
    inst.keypoints = kpts
    inst.keypoint_scores = np.ones((n, 17), dtype=np.float32)
    inst.bboxes = np.asarray(bboxes, dtype=np.float32)
    inst.bbox_scores = np.asarray(scores, dtype=np.float32)
    ds = PoseDataSample()
    ds.set_metainfo(dict(img_path=img_path, ori_shape=shape, img_id=0))
    ds.pred_instances = inst
    return ds


@skipUnless(_HAS_BOXMOT, 'boxmot is not installed')
class TestBoxMOTTracker(TestCase):

    def _tracker(self, **kwargs):
        cfg = dict(tracker='bytetrack', device='cpu',
                   tracker_kwargs=dict(track_thresh=0.3, min_conf=0.1))
        cfg.update(kwargs)
        return BoxMOTTracker(**cfg)

    def test_requires_images_auto_motion_only(self):
        trk = self._tracker()
        self.assertFalse(trk.requires_images)

    def test_id_stability_and_det_ind_alignment(self):
        """Two people move horizontally, vertically separated so IoU is 0.

        Person A is tagged with keypoint x=100, person B with x=200.
        After tracking, each track id must keep the same tag, which
        fails if det_ind is mapped to the wrong input row.
        """
        trk = self._tracker()
        # A starts left and moves right; B starts right and moves left.
        # Different y so the boxes never overlap.
        id_to_tag = None
        for t in range(12):
            ax1 = 20.0 + 8.0 * t
            bx1 = 400.0 - 8.0 * t
            ds = _make_ds(
                bboxes=[
                    [ax1, 10, ax1 + 40, 80],
                    [bx1, 200, bx1 + 40, 270],
                ],
                scores=[0.95, 0.92],
                tags=[100.0, 200.0],
                img_path=f'seq/a/{t:03d}.jpg',
            )
            out = trk.process_frame(ds, 'seq/a')
            tids = np.asarray(out.pred_instances.track_ids)
            tags = np.asarray(out.pred_instances.keypoints)[:, 0, 0]
            self.assertEqual(len(tids), 2, f'frame {t} dropped a detection')
            mapping = {int(tid): float(tag) for tid, tag in zip(tids, tags)}
            if id_to_tag is None:
                id_to_tag = mapping
                self.assertEqual(set(id_to_tag.values()), {100.0, 200.0})
            else:
                self.assertEqual(mapping, id_to_tag)

    def test_reset_restarts_ids(self):
        trk = self._tracker()
        ds = _make_ds(
            bboxes=[[10, 10, 50, 80], [200, 10, 240, 80]],
            scores=[0.9, 0.9],
            tags=[1.0, 2.0],
        )
        out1 = trk.process_frame(ds, 'seq/a')
        ids1 = set(int(i) for i in out1.pred_instances.track_ids)
        trk.reset()
        out2 = trk.process_frame(ds, 'seq/b')
        ids2 = set(int(i) for i in out2.pred_instances.track_ids)
        self.assertEqual(ids1, ids2)
        self.assertEqual(ids1, {1, 2})

    def test_keep_untracked_emits_every_passed_detection(self):
        """A very low-score box is ignored by ByteTrack; keep_untracked
        still emits it with a fresh id."""
        trk = self._tracker(keep_untracked=True)
        # First frame: two high-score people, so ByteTrack confirms them.
        ds0 = _make_ds(
            bboxes=[[10, 10, 50, 80], [200, 10, 240, 80]],
            scores=[0.95, 0.95],
            tags=[1.0, 2.0],
            img_path='seq/a/000.jpg',
        )
        trk.process_frame(ds0, 'seq/a')
        # Second frame: same two plus a 0.05-score stray that ByteTrack
        # discards (min_conf=0.1).
        ds1 = _make_ds(
            bboxes=[[18, 10, 58, 80], [208, 10, 248, 80], [300, 10, 330, 40]],
            scores=[0.95, 0.95, 0.05],
            tags=[1.0, 2.0, 9.0],
            img_path='seq/a/001.jpg',
        )
        out = trk.process_frame(ds1, 'seq/a')
        tags = np.asarray(out.pred_instances.keypoints)[:, 0, 0]
        self.assertEqual(len(out.pred_instances), 3)
        self.assertIn(9.0, set(float(t) for t in tags))

    def test_min_bbox_score_drops_before_tracker(self):
        trk = self._tracker(min_bbox_score=0.5, keep_untracked=True)
        ds = _make_ds(
            bboxes=[[10, 10, 50, 80], [200, 10, 240, 80]],
            scores=[0.95, 0.1],
            tags=[1.0, 9.0],
        )
        out = trk.process_frame(ds, 'seq/a')
        tags = set(float(t) for t in out.pred_instances.keypoints[:, 0, 0])
        self.assertEqual(tags, {1.0})
