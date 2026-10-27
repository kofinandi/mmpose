# Copyright (c) OpenMMLab. All rights reserved.
"""Official PoseTrack21 metrics for ``tools/benchmark_e2e.py``.

Opt-in (``--official-posetrack21``). The in-repo TrackEval port stays the
default. This metric writes the kit's JSON / MOT files from the frames the
benchmark actually ran and scores them with the official keypoint kit plus
box MOTA/IDF1/HOTA. Box scores are reported twice: ``official`` (the
ignore-region IoU test) and ``fixed`` (the CoMotion/DETRAM overlap).

``requires_all_frames`` is true. ``--eval-good-frames-only`` still feeds
this metric the unlabeled frames, because the official box scorer counts
predictions there as false positives and the fix is defined by skipping
those frames. Keypoint metrics drop unlabeled frames themselves.
"""

from __future__ import annotations

import os.path as osp
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from ..functional import posetrack21_kit as kit
from ..functional.keypoint_space import names_from_metainfo

_DEFAULT_GT_POSE = 'data/posetrack21/PoseTrack21/posetrack_data/val'
_DEFAULT_GT_MOT = 'data/posetrack21/PoseTrack21/posetrack_mot/mot/val'


def _as_2d_scores(scores, n: int, n_kpts: int) -> np.ndarray:
    """``(n, n_kpts)`` scores. Empty frames stay ``(0, n_kpts)``.

    ``reshape(0, -1)`` is illegal: a length-0 array does not determine
    the missing dimension, which is what a frame with no detections hits.
    """
    if n == 0 or scores is None:
        return np.zeros((n, n_kpts), dtype=np.float32)
    arr = np.asarray(scores, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim == 1:
        arr = arr.reshape(n, -1)
    if arr.shape[0] != n or arr.shape[-1] != n_kpts:
        return np.zeros((n, n_kpts), dtype=np.float32)
    return arr.reshape(n, n_kpts)


def _boxes_from_prediction(pred: dict, keypoints: np.ndarray,
                           scores: np.ndarray) -> tuple:
    """xyxy boxes and one score per instance.

    Uses ``bboxes`` / ``bbox_scores`` when they line up with the
    instances. Otherwise the box is the min/max of joints with a positive
    score, which is all a pose-only model has.
    """
    n = int(keypoints.shape[0]) if keypoints.ndim == 3 else 0
    if n == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, ), dtype=np.float32))
    boxes = pred.get('bboxes')
    if boxes is not None:
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.size == 0:
            boxes = None
        elif boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        if (boxes is not None and boxes.shape[0] == n
                and boxes.shape[-1] >= 4):
            boxes = np.ascontiguousarray(boxes.reshape(n, boxes.shape[-1])[:, :4])
        else:
            boxes = None
    if boxes is None:
        boxes = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            if scores.shape[1] == 0:
                continue
            keep = scores[i] > 0
            if not np.any(keep):
                continue
            xy = keypoints[i][keep]
            boxes[i, :2] = xy.min(axis=0)
            boxes[i, 2:] = xy.max(axis=0)
    box_scores = pred.get('bbox_scores')
    if box_scores is not None:
        box_scores = np.asarray(box_scores, dtype=np.float32).reshape(-1)
        if len(box_scores) != n:
            box_scores = None
    if box_scores is None:
        if scores.size:
            counted = (scores > 0).sum(axis=1)
            summed = np.where(scores > 0, scores, 0).sum(axis=1)
            box_scores = np.divide(
                summed, counted, out=np.zeros(n, dtype=np.float32),
                where=counted > 0).astype(np.float32)
        else:
            box_scores = np.zeros(n, dtype=np.float32)
    return boxes, box_scores


@METRICS.register_module()
class PoseTrack21OfficialMetric(BaseMetric):
    """Official PoseTrack21 keypoint and box metrics.

    Args:
        gt_pose_dir: Official per-sequence keypoint JSON directory.
            Default: ``data/posetrack21/PoseTrack21/posetrack_data/val``.
        gt_mot_dir: Official MOT sequence directory.
            Default: ``data/posetrack21/PoseTrack21/posetrack_mot/mot/val``.
        eval_root: Kit ``eval/`` directory (contains ``posetrack21/`` and
            ``mot/``). Default: ``--pt21-eval-root`` / ``POSETRACK21_EVAL_ROOT``
            / a known checkout.
        num_cores: Parallelism for the keypoint kit and box HOTA.
            Default: 8.
        collect_device: Device for result collection. Default: ``'cpu'``.
        prefix: Metric prefix. Default: ``'pt21'``.
    """

    default_prefix: Optional[str] = 'pt21'

    # Unlabeled frames change the official/fixed box false-positive term.
    # benchmark_e2e feeds them to this metric even with
    # --eval-good-frames-only.
    requires_all_frames = True

    def __init__(self,
                 gt_pose_dir: Optional[str] = None,
                 gt_mot_dir: Optional[str] = None,
                 eval_root: Optional[str] = None,
                 num_cores: int = 8,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.gt_pose_dir = gt_pose_dir or _DEFAULT_GT_POSE
        self.gt_mot_dir = gt_mot_dir or _DEFAULT_GT_MOT
        self.eval_root = eval_root
        self.num_cores = int(num_cores)

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Record one frame's predictions. Ground truth comes from the kit."""
        for data_sample in data_samples:
            pred = data_sample.get('pred_instances', None)
            if pred is None:
                continue
            keypoints = pred.get('keypoints')
            if keypoints is None:
                continue
            keypoints = np.asarray(keypoints)
            if keypoints.ndim == 2:
                keypoints = keypoints[None]
            n = int(keypoints.shape[0])
            n_kpts = int(keypoints.shape[1]) if keypoints.ndim >= 2 else 0
            scores = _as_2d_scores(pred.get('keypoint_scores'), n, n_kpts)
            boxes, box_scores = _boxes_from_prediction(
                pred, keypoints[..., :2], scores)
            # Evaluator.process converts PoseDataSample to a dict first, so
            # pred_instances is a dict here. The unlabeled-frame path calls
            # process directly and pred_instances is still an InstanceData.
            # .get covers both; attribute access does not.
            track_ids = pred.get('track_ids')
            if track_ids is None:
                track_ids = np.full(n, -1, dtype=np.int64)
            else:
                track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
                if len(track_ids) != n:
                    track_ids = np.full(n, -1, dtype=np.int64)
            seq_id = data_sample.get('seq_id')
            if not seq_id:
                img_path = data_sample.get('img_path') or ''
                seq_id = osp.basename(osp.dirname(img_path))
            self.results.append({
                'seq_id': str(seq_id),
                'img_id': int(data_sample.get('img_id')),
                'track_ids': track_ids,
                'keypoints': np.asarray(keypoints[..., :2], dtype=np.float32),
                'scores': scores.astype(np.float32),
                'boxes': boxes.astype(np.float32),
                'box_scores': box_scores.astype(np.float32),
            })

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Write a temp run and score it with the official kit."""
        logger = MMLogger.get_current_instance()
        if not results:
            logger.warning(
                'PoseTrack21OfficialMetric received no frames '
                '(--official-posetrack21).')
            return {'partial': 1.0, 'synthetic_track_ids': 0.0}

        gt_pose = Path(self.gt_pose_dir)
        gt_mot = Path(self.gt_mot_dir)
        if not gt_pose.is_dir() or not gt_mot.is_dir():
            raise FileNotFoundError(
                'Official PoseTrack21 ground truth not found at '
                f'{gt_pose} and {gt_mot}. Required by '
                '--official-posetrack21.')
        eval_root = kit.resolve_eval_root(self.eval_root)
        src_names = names_from_metainfo(self.dataset_meta)
        if not src_names:
            raise ValueError(
                'PoseTrack21OfficialMetric needs dataset_meta keypoint '
                'names so it can map predictions into PoseTrack raw-17.')

        evaluated: Dict[str, set] = {}
        for rec in results:
            evaluated.setdefault(rec['seq_id'], set()).add(int(rec['img_id']))
        labeled = kit.labeled_image_ids(gt_pose)
        unlabeled = kit.unlabeled_image_ids(gt_pose)
        missing_labeled = 0
        for seq, ids in labeled.items():
            have = evaluated.get(seq, set())
            missing_labeled += sum(1 for i in ids if i not in have)
        missing_unlabeled = 0
        for seq, ids in unlabeled.items():
            have = evaluated.get(seq, set())
            missing_unlabeled += sum(1 for i in ids if i not in have)
        if missing_unlabeled:
            logger.warning(
                'PoseTrack21OfficialMetric did not see '
                f'{missing_unlabeled} unlabeled frames. The official box '
                'false-positive count omits predictions the tracker never '
                'ran. Pass --include-bad-frames so the tracker sees them.')

        with tempfile.TemporaryDirectory(prefix='pt21_official_') as tmp:
            (pose_dir, mot_dir, gt_pose_sub, gt_mot_sub,
             synthetic) = kit.write_sequence_files(
                 results, src_names, gt_pose, gt_mot, Path(tmp))
            metrics: Dict[str, float] = {}
            metrics.update(kit.keypoint_hota(
                pose_dir, gt_pose_sub, eval_root, self.num_cores))
            metrics.update(kit.keypoint_map(
                pose_dir, gt_pose_sub, eval_root, self.num_cores))
            metrics.update(kit.keypoint_mota(pose_dir, gt_pose_sub))
            metrics.update(kit.box_clear(
                mot_dir, gt_mot_sub, eval_root))
            metrics.update(kit.box_hota(
                mot_dir, gt_mot_sub, eval_root, self.num_cores))

        metrics['partial'] = 1.0 if missing_labeled else 0.0
        metrics['synthetic_track_ids'] = 1.0 if synthetic else 0.0
        return metrics
