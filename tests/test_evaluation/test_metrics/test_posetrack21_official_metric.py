# Copyright (c) OpenMMLab. All rights reserved.
"""PoseTrack21 official-kit translation and the ignore-region bug.

These tests do not import the eval kit. They check the overlap rule
CoMotion and DETRAM fixed, and that benchmark keypoints land in the
kit's raw-17 JSON without invented joints.
"""

import json
from pathlib import Path
from unittest import TestCase

import numpy as np
from shapely.geometry import Polygon

from mmengine.evaluator import Evaluator
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData

from mmpose.datasets.transforms.keypoint_registry import get_keypoints
from mmpose.evaluation.functional.keypoint_space import evaluation_metainfo
from mmpose.evaluation.functional.posetrack21_kit import (
    POSETRACK_RAW_NAMES, VARIANTS, _compact_track_ids, hota_remove_indices,
    instances_to_raw17, prediction_counts_as_fp, region_overlap,
    write_sequence_files)
from mmpose.evaluation.metrics.posetrack21_official_metric import (
    PoseTrack21OfficialMetric)
from mmpose.structures import PoseDataSample


def _square(x1, y1, x2, y2) -> Polygon:
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


class TestIgnoreRegionBug(TestCase):

    def test_box_inside_large_region_has_tiny_iou(self):
        region = _square(0, 0, 400, 400)
        box = _square(10, 10, 30, 30)
        self.assertLess(region_overlap(region, box, 'iou'), 0.1)
        self.assertAlmostEqual(region_overlap(region, box, 'box_fraction'), 1.0)

    def test_unmatched_box_counts_only_under_official(self):
        region = _square(0, 0, 400, 400)
        # A ground-truth box far away, so the prediction is unmatched.
        gt = np.array([[300, 300, 320, 320]], dtype=np.float64)
        pred = [10.0, 10.0, 30.0, 30.0]
        self.assertTrue(prediction_counts_as_fp(
            pred, gt, [region], VARIANTS['official']))
        self.assertFalse(prediction_counts_as_fp(
            pred, gt, [region], VARIANTS['fixed']))

    def test_frame_without_gt_is_a_false_positive_only_when_official(self):
        pred = [10.0, 10.0, 30.0, 30.0]
        gt = np.zeros((0, 4), dtype=np.float64)
        self.assertTrue(prediction_counts_as_fp(
            pred, gt, [], VARIANTS['official']))
        self.assertFalse(prediction_counts_as_fp(
            pred, gt, [], VARIANTS['fixed']))

    def test_matched_box_inside_region_is_kept(self):
        region = _square(0, 0, 400, 400)
        box = np.array([[10.0, 10.0, 30.0, 30.0]])
        for rule in (VARIANTS['official'], VARIANTS['fixed']):
            removed = hota_remove_indices(box, box, [region], rule)
            self.assertEqual(len(removed), 0, rule.name)


class TestKitTranslation(TestCase):

    def test_coco_raw17_leaves_heads_unscored(self):
        src = get_keypoints('coco')
        coords = np.zeros((1, 17, 2), dtype=np.float32)
        coords[0, src.index('nose')] = [5.0, 6.0]
        coords[0, src.index('left_shoulder')] = [7.0, 8.0]
        scores = np.ones((1, 17), dtype=np.float32)
        raw = instances_to_raw17(coords, scores, src)
        self.assertEqual(tuple(POSETRACK_RAW_NAMES),
                         tuple(get_keypoints('posetrack18')))
        np.testing.assert_allclose(raw[0, POSETRACK_RAW_NAMES.index('nose'), :2],
                                    [5.0, 6.0])
        self.assertEqual(raw[0, POSETRACK_RAW_NAMES.index('head_bottom'), 2], 0)
        self.assertEqual(raw[0, POSETRACK_RAW_NAMES.index('head_top'), 2], 0)
        np.testing.assert_allclose(
            raw[0, POSETRACK_RAW_NAMES.index('head_bottom'), :2], [0, 0])

    def test_write_sequence_drops_frames_that_were_not_run(self):
        with self._tmpdir() as root:
            root = Path(root)
            self._write_official(root)
            coco = get_keypoints('coco')
            coords = np.zeros((1, 17, 2), dtype=np.float32)
            coords[0, 0] = [5.0, 6.0]
            scores = np.zeros((1, 17), dtype=np.float32)
            scores[0, 0] = 0.9
            records = [{
                'seq_id': '000342_mpii_test',
                'img_id': 10,
                'track_ids': np.array([-1], dtype=np.int64),
                'keypoints': coords,
                'scores': scores,
                'boxes': np.array([[1.0, 2.0, 11.0, 12.0]], dtype=np.float32),
                'box_scores': np.array([0.8], dtype=np.float32),
            }]
            pose_dir, mot_dir, gt_pose, gt_mot, synthetic = write_sequence_files(
                records, coco, root / 'pose', root / 'mot', root / 'out')
            self.assertTrue(synthetic)
            written = json.loads(
                (pose_dir / '000342_mpii_test.json').read_text())
            self.assertEqual([img['id'] for img in written['images']], [10])
            ann = written['annotations'][0]
            flat = np.asarray(ann['keypoints']).reshape(17, 3)
            self.assertEqual(ann['track_id'], 1)
            self.assertAlmostEqual(flat[0, 0], 5.0)
            self.assertEqual(flat[1, 2], 0.0)  # head_bottom unscored
            gt_written = json.loads(
                (gt_pose / '000342_mpii_test.json').read_text())
            self.assertEqual([img['id'] for img in gt_written['images']], [10])
            mot = (mot_dir / '000342_mpii_test.txt').read_text().strip()
            self.assertTrue(mot.startswith('1,2,'))
            info = json.loads(
                (gt_mot / '000342_mpii_test' / 'image_info.json').read_text())
            self.assertEqual(info[0]['frame_index'], 1)
            self.assertEqual(len(info), 1)
            gt_row = (gt_mot / '000342_mpii_test' / 'gt' / 'gt.txt').read_text()
            self.assertTrue(gt_row.startswith('1,'))
            self.assertNotIn('\n2,', '\n' + gt_row)

    def test_evaluator_dict_records_track_ids_and_scores(self):
        """The crash: Evaluator.process hands metrics a dict, not InstanceData.

        Also runs the kit on two real frames so a missing field or a bad
        file layout fails here instead of after a full val inference.
        """
        gt = Path('data/posetrack21/PoseTrack21/posetrack_data/val'
                  '/000342_mpii_test.json')
        if not gt.is_file():
            self.skipTest('PoseTrack21 val annotations are not on disk')
        images = json.loads(gt.read_text())['images']
        chosen = [img for img in images if img.get('is_labeled')][:3]
        MMLogger.get_instance('test_pt21_official')
        metric = PoseTrack21OfficialMetric(num_cores=1)
        metric.dataset_meta = evaluation_metainfo('posetrack21', 'coco')
        evaluator = Evaluator([metric])
        samples = []
        for i, img in enumerate(chosen):
            ds = PoseDataSample()
            ds.set_metainfo({
                'img_id': int(img['id']),
                'seq_id': '000342_mpii_test',
                'img_path': img['file_name'],
            })
            if i == 0:
                # A frame the tracker left empty. bboxes are (0, 4); reshaping
                # that with ``-1`` is what crashed the full val run.
                pred = InstanceData()
                pred.keypoints = np.zeros((0, 17, 2), dtype=np.float32)
                pred.keypoint_scores = np.zeros((0, 17), dtype=np.float32)
                pred.bboxes = np.zeros((0, 4), dtype=np.float32)
                pred.bbox_scores = np.zeros((0, ), dtype=np.float32)
                pred.track_ids = np.zeros((0, ), dtype=np.int32)
            else:
                pred = InstanceData(
                    keypoints=np.zeros((1, 17, 2), dtype=np.float32),
                    keypoint_scores=np.zeros((1, 17), dtype=np.float32),
                    bboxes=np.array([[10, 10, 40, 80]], dtype=np.float32),
                    bbox_scores=np.array([0.9], dtype=np.float32),
                    track_ids=np.array([7], dtype=np.int32))
                pred.keypoints[0, 0] = [20, 30]
                pred.keypoint_scores[0, 0] = 0.8
            ds.pred_instances = pred
            samples.append(ds)
        evaluator.process(samples[:1], None)
        # Unlabeled-frame path calls the metric directly, still a PoseDataSample.
        metric.process(None, samples[1:])
        self.assertEqual(len(metric.results), 3)
        self.assertEqual(metric.results[0]['boxes'].shape, (0, 4))
        self.assertEqual(int(metric.results[1]['track_ids'][0]), 7)
        metrics = metric.compute_metrics(metric.results)
        for key in ('kp_HOTA', 'kp_mAP_Total', 'kp_MOTA',
                    'box_MOTA_official', 'box_MOTA_fixed',
                    'box_HOTA_official', 'box_HOTA_fixed'):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))
        self.assertEqual(metrics['partial'], 1.0)

    def test_compact_ids_keep_a_track_across_frames(self):
        frames, synthetic = _compact_track_ids([
            np.array([7, -1]),
            np.array([7, 7]),
        ])
        self.assertTrue(synthetic)
        self.assertEqual(frames[0].tolist(), [1, 2])
        # The second 7 in frame 2 is a duplicate, so it is a new track.
        self.assertEqual(frames[1][0], 1)
        self.assertNotEqual(frames[1][1], 1)

    def _tmpdir(self):
        import tempfile
        return tempfile.TemporaryDirectory()

    def _write_official(self, root: Path) -> None:
        pose = root / 'pose'
        mot = root / 'mot' / '000342_mpii_test' / 'gt'
        pose.mkdir(parents=True)
        mot.mkdir(parents=True)
        images = [
            {
                'id': 10, 'is_labeled': True, 'file_name': 'a.jpg',
                'ignore_regions_x': [], 'ignore_regions_y': [],
            },
            {
                'id': 11, 'is_labeled': True, 'file_name': 'b.jpg',
                'ignore_regions_x': [], 'ignore_regions_y': [],
            },
        ]
        (pose / '000342_mpii_test.json').write_text(json.dumps({
            'images': images,
            'annotations': [{
                'image_id': 10, 'track_id': 3, 'keypoints': [0] * 51,
                'bbox': [0, 0, 1, 1], 'bbox_head': [0, 0, 1, 1],
            }, {
                'image_id': 11, 'track_id': 3, 'keypoints': [0] * 51,
                'bbox': [0, 0, 1, 1], 'bbox_head': [0, 0, 1, 1],
            }],
            'categories': [{'id': 1, 'name': 'person'}],
        }))
        (root / 'mot' / '000342_mpii_test' / 'image_info.json').write_text(
            json.dumps([
                {'id': 10, 'frame_index': 1, 'ignore_regions_x': [],
                 'ignore_regions_y': []},
                {'id': 11, 'frame_index': 2, 'ignore_regions_x': [],
                 'ignore_regions_y': []},
            ]))
        (mot / 'gt.txt').write_text(
            '1,3,0,0,10,20,1,-1,-1,-1\n'
            '2,3,1,1,10,20,1,-1,-1,-1\n')
