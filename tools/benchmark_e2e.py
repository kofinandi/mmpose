# Copyright (c) OpenMMLab. All rights reserved.
"""End-to-end pose estimation benchmarking pipeline.

Supports two modes:
  - Bottomup: single model (e.g. YOLOX-pose) processes whole images.
  - Topdown: detector + keypoint localizer pipeline with an async
    producer/consumer queue between the two stages.

Quality metrics are the same as tools/test.py (CocoMetric, PCK, etc.).
Performance metrics cover FPS and per-frame/per-location latency at
whole-pipeline and per-stage granularity.
"""

from mmpose.compat.transformers_v5 import install_transformers_v5_shims

install_transformers_v5_shims()

import argparse
import collections
import json
import os
import os.path as osp
import queue
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import mmcv
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate
from mmengine.evaluator import Evaluator
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope

# Register all mmpose modules
import mmpose.datasets       # noqa: F401
import mmpose.evaluation     # noqa: F401
import mmpose.models         # noqa: F401
from mmengine.registry import DATASETS, METRICS
from mmpose.apis.det_inference import inference_det_model, init_det_model
from mmpose.apis import init_model
from mmpose.evaluation.functional import nms
from mmpose.evaluation.benchmark_datasets import (
    BENCHMARK_TEST_DATASET_NAMES,
    apply_benchmark_test_dataset,
)
from mmpose.evaluation.functional.frame_metrics import (
    build_frame_record,
    save_prediction_bundle,
    serialize_gt_instances,
    serialize_pred_instances,
    sanitize_dataset_meta,
)
from mmpose.evaluation.functional.pose_gt_to_coco_det import (
    load_pose_gt_per_image,
    resolve_det_ann_file,
)
from mmpose.structures import PoseDataSample, merge_data_samples

try:
    import mmdet  # noqa: F401
    HAS_MMDET = True
except (ImportError, ModuleNotFoundError):
    HAS_MMDET = False

# Named tuple carried through the producer->consumer queue
BBoxItem = collections.namedtuple(
    'BBoxItem',
    ['frame_id', 'img_id', 'img', 'bbox_xyxy', 'det_score',
     'n_bboxes_in_frame', 'frame_start_wall'],
)
_SENTINEL = object()   # signals producer is done


# ── Timing ────────────────────────────────────────────────────────────────

class _CudaTimer:
    """Context-manager timer: CUDA events on GPU, perf_counter on CPU."""

    def __init__(self):
        self.elapsed_s = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self._start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if torch.cuda.is_available():
            self._end.record()
            torch.cuda.synchronize()
            self.elapsed_s = self._start.elapsed_time(self._end) / 1000.0
        else:
            self.elapsed_s = time.perf_counter() - self._t0


# ── Dataset helpers ────────────────────────────────────────────────────────

def _init_scope(cfg: Config) -> None:
    scope = cfg.get('default_scope', 'mmpose')
    if scope:
        init_default_scope(scope)


def build_unique_image_list(
    pose_cfg: Config,
    num_frames: Optional[int] = None,
) -> List[Tuple[int, str]]:
    """Return [(img_id, img_path), ...] with one entry per unique image.

    For bottomup configs the dataset is already per-image; for topdown
    configs (one item per GT instance) we deduplicate by img_id.
    """
    _init_scope(pose_cfg)
    ds_cfg = pose_cfg.test_dataloader.dataset.to_dict()
    ds_cfg['pipeline'] = []
    dataset = DATASETS.build(ds_cfg)

    seen: dict = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        img_id = item['img_id']
        if img_id not in seen:
            seen[img_id] = item['img_path']
        if num_frames is not None and len(seen) >= num_frames:
            break

    return list(seen.items())


def _evaluator_needs_gt_from_samples(pose_cfg: Config) -> bool:
    """Return True when the pose evaluator synthesizes GT from samples."""
    ev_cfg = pose_cfg.test_evaluator
    if isinstance(ev_cfg, dict):
        ev_cfg = [ev_cfg]
    return any(m.get('gt_from_samples', False) for m in ev_cfg)


def build_gt_by_img_id(
    pose_cfg: Config,
    num_frames: Optional[int] = None,
) -> Dict[int, list]:
    """Map COCO ``img_id`` to per-instance GT dicts in original image coords."""
    return {
        img_id: gt_insts
        for img_id, _, gt_insts in load_pose_gt_per_image(
            pose_cfg, num_frames)
    }


def build_gt_instances_per_frame(
    pose_cfg: Config,
    img_list: List[Tuple[int, str]],
) -> Dict[int, list]:
    """Map frame index to per-instance GT dicts for metric evaluation.

    Used in real-detector mode when ``gt_from_samples=True``: predictions
    come from the detector, but CocoMetric still needs GT keypoints/area
    attached to each sample.
    """
    gt_data = build_gt_bbox_image_list(pose_cfg, num_frames=len(img_list))
    gt_by_img_id = {iid: insts for iid, _, _, _, insts in gt_data}
    return {
        fid: gt_by_img_id[img_id]
        for fid, (img_id, _) in enumerate(img_list)
        if img_id in gt_by_img_id
    }


def build_gt_bbox_image_list(
    pose_cfg: Config,
    num_frames: Optional[int] = None,
) -> List[Tuple[int, str, np.ndarray, np.ndarray, list]]:
    """Return GT bbox info from the test dataset, one entry per unique image.

    Returns a list of
    ``(img_id, img_path, bboxes_xyxy, bbox_scores, gt_instances_list)``
    where:

    - ``bboxes_xyxy`` has shape ``(N, 4)``
    - ``bbox_scores`` has shape ``(N,)`` with all values set to 1.0
    - ``gt_instances_list`` is a list of per-instance dicts with keys
      ``keypoints``, ``keypoints_visible``, ``bbox_scale``, ``area``,
      ``category_id``, and ``id``.  If the test pipeline contains a
      ``KeypointConverter``, it is applied to each raw item so keypoints
      arrive in the target order (e.g. COCO) for metric evaluation.
    """
    result = []
    for img_id, img_path, gt_insts in load_pose_gt_per_image(
            pose_cfg, num_frames):
        boxes = np.stack(
            [g['bbox'].reshape(4) for g in gt_insts],
            axis=0,
        ).astype(np.float32)
        scores = np.ones(len(boxes), dtype=np.float32)
        result.append((img_id, img_path, boxes, scores, gt_insts))
    return result


def prefetch_images(
    img_list: List[Tuple[int, str]],
) -> List[Tuple[int, np.ndarray]]:
    """Load all images into memory as BGR numpy arrays."""
    print(f'Prefetching {len(img_list)} images...')
    result = []
    for img_id, img_path in img_list:
        img = mmcv.imread(img_path)  # BGR, HWC
        result.append((img_id, img))
    print('Prefetch complete.')
    return result


def build_bottomup_batches(
    pose_cfg: Config,
    kp_batch_size: int,
    num_frames: Optional[int] = None,
) -> Tuple[list, int]:
    """Apply val pipeline and partition into batches for bottomup mode.

    Returns (batches, total_items) where each batch is
    (pseudo_collated_dict, list_of_img_ids).
    """
    _init_scope(pose_cfg)
    ds_cfg = pose_cfg.test_dataloader.dataset.to_dict()
    ds_cfg['pipeline'] = pose_cfg.test_dataloader.dataset.pipeline
    dataset = DATASETS.build(ds_cfg)

    total = len(dataset)
    if num_frames is not None:
        total = min(total, num_frames)

    batches = []
    for start in range(0, total, kp_batch_size):
        end = min(start + kp_batch_size, total)
        items = [dataset[i] for i in range(start, end)]
        img_ids = [it['data_samples'].metainfo['img_id'] for it in items]
        batch = pseudo_collate(items)
        batches.append((batch, img_ids))

    return batches, total


# ── Evaluator helpers ──────────────────────────────────────────────────────

def build_evaluator(pose_cfg: Config, dataset_meta: dict) -> Evaluator:
    """Build mmengine Evaluator from the config's test_evaluator section."""
    _init_scope(pose_cfg)
    ev_cfg = pose_cfg.test_evaluator
    if isinstance(ev_cfg, dict):
        ev_cfg = [ev_cfg]

    metrics = []
    for m_cfg in ev_cfg:
        m = METRICS.build(m_cfg)
        m.dataset_meta = dataset_meta
        metrics.append(m)

    return Evaluator(metrics)


def build_det_evaluator(
    pose_cfg: Config,
    num_frames: Optional[int] = None,
) -> Evaluator:
    """COCO bbox AP/AR for the person (human) class only.

    Uses the annotation file referenced by the pose config when it is already
    in COCO bbox-detection format; otherwise converts pose-dataset GT on the
    fly.  Restricts evaluation to the images that were actually processed
    (important when --num-frames is set).
    """
    if not HAS_MMDET:
        raise ImportError(
            'mmdet is required for --det-metrics. '
            'Install it with: pip install mmdet')

    from mmdet.evaluation.metrics import CocoMetric as _CocoMetric

    class _SubsetCocoMetric(_CocoMetric):
        """CocoMetric that evaluates only on the images seen during process().

        Standard CocoMetric uses every image in the annotation file as the
        denominator for recall.  When only a subset of frames is benchmarked
        (--num-frames), that would unfairly depress AR.  This subclass tracks
        which img_ids were actually processed and restricts COCOeval to those.
        """

        def process(self, data_batch, data_samples):
            if not hasattr(self, '_seen_img_ids'):
                self._seen_img_ids: set = set()
            for ds in data_samples:
                self._seen_img_ids.add(ds['img_id'])
            super().process(data_batch, data_samples)

        def compute_metrics(self, results):
            if getattr(self, '_seen_img_ids', None):
                self.img_ids = sorted(self._seen_img_ids)
            out = super().compute_metrics(results)
            # CocoMetric hardcodes key names using the default proposal_nums
            # (100, 300, 1000).  Build an explicit rename table so that keys
            # reflect the actual maxDets values used.  String-replacement is
            # avoided because '@1000' contains '@100' as a substring.
            nums = self.proposal_nums  # e.g. [5, 10, 100]
            ar_renames = {
                f'bbox_AR@100':    f'bbox_AR@{nums[0]}',
                f'bbox_AR@300':    f'bbox_AR@{nums[1]}',
                f'bbox_AR@1000':   f'bbox_AR@{nums[2]}',
                f'bbox_AR_s@1000': f'bbox_AR_s@{nums[2]}',
                f'bbox_AR_m@1000': f'bbox_AR_m@{nums[2]}',
                f'bbox_AR_l@1000': f'bbox_AR_l@{nums[2]}',
            }
            return {ar_renames.get(k, k): v for k, v in out.items()}

    ann_file = resolve_det_ann_file(pose_cfg, num_frames=num_frames)

    metric = _SubsetCocoMetric(
        ann_file=ann_file,
        metric='bbox',
        classwise=False,
        prefix='det',
        proposal_nums=(5, 10, 100),
        metric_items=[
            'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l',
            'AR@100', 'AR@300', 'AR@1000',
            'AR_s@1000', 'AR_m@1000', 'AR_l@1000',
        ],
    )
    metric.dataset_meta = {'classes': ('person',)}
    return Evaluator([metric])


def _make_topdown_data_sample(
    img_id: int,
    bbox_results: list,
    det_scores: List[float],
    ori_shape: Tuple[int, int],
    gt_instances_data: Optional[list] = None,
) -> PoseDataSample:
    """Merge per-bbox PoseDataSamples into a single per-image PoseDataSample.

    Sets ``id`` as a list so ``CocoMetric._sort_and_unique_bboxes`` skips
    deduplication (matching bottomup convention) and all N detected persons
    are preserved.

    Args:
        img_id: Image identifier.
        bbox_results: Per-bbox ``PoseDataSample`` outputs from the keypoint
            model.
        det_scores: Detector confidence score for each bbox.
        ori_shape: ``(height, width)`` of the original image.
        gt_instances_data: Optional list of per-GT-instance dicts, each with
            keys ``keypoints``, ``keypoints_visible``, ``bbox_scale``,
            ``area``, ``category_id``, and ``id``.  When provided the GT
            data is written onto ``merged.gt_instances`` so that metrics
            with ``gt_from_samples=True`` (e.g. :class:`CocoMetric`) can
            synthesize ground-truth annotations without an ``ann_file``.
    """
    from mmengine.structures import InstanceData

    n = len(bbox_results)

    if n == 0:
        ds = PoseDataSample()
        num_kpts = 17  # COCO default; will be overridden if model differs
        ds.set_metainfo({
            'img_id': img_id,
            'id': [img_id],
            'category_id': 1,
            'ori_shape': ori_shape,
        })
        pred_inst = InstanceData()
        pred_inst.keypoints = np.zeros((0, num_kpts, 2), dtype=np.float32)
        pred_inst.keypoint_scores = np.zeros((0, num_kpts), dtype=np.float32)
        pred_inst.bboxes = np.zeros((0, 4), dtype=np.float32)
        pred_inst.bbox_scores = np.zeros(0, dtype=np.float32)
        ds.pred_instances = pred_inst
        gt_inst = InstanceData()
        gt_inst.bbox_scores = np.zeros(0, dtype=np.float32)
        ds.gt_instances = gt_inst
        if gt_instances_data:
            _attach_gt_instances(ds, gt_instances_data, img_id)
        return ds

    # Inject detector scores into each result's gt_instances so that
    # CocoMetric (score_mode='bbox_keypoint') uses the detector confidence.
    for res, score in zip(bbox_results, det_scores):
        res.gt_instances.bbox_scores = np.array([score], dtype=np.float32)

    merged = merge_data_samples(bbox_results)

    # Detector scores belong on pred_instances; gt_instances may be replaced
    # below with dataset GT that has a different instance count.
    merged.pred_instances.bbox_scores = np.array(det_scores, dtype=np.float32)

    # Use a list of unique synthetic IDs to trigger the bottomup code path
    # inside CocoMetric._sort_and_unique_bboxes (which early-returns when
    # the 'id' value is a Sequence).
    ids = list(range(img_id * 10000, img_id * 10000 + n))
    merged.set_metainfo({
        'img_id': img_id,
        'id': ids,
        'category_id': 1,
        'ori_shape': ori_shape,
    })

    # Attach GT instances so that CocoMetric with gt_from_samples=True can
    # synthesize COCO-format ground-truth annotations.
    if gt_instances_data:
        _attach_gt_instances(merged, gt_instances_data, img_id)

    return merged


def _attach_gt_instances(
    data_sample: PoseDataSample,
    gt_instances_data: list,
    img_id: int,
) -> None:
    """Write stacked GT keypoint/bbox arrays into data_sample.gt_instances.

    Args:
        data_sample: The merged per-image ``PoseDataSample``.  Its
            ``gt_instances`` is updated in-place.
        gt_instances_data: List of per-GT-instance dicts as produced by
            :func:`build_gt_bbox_image_list`.
        img_id: Image identifier used as a fallback for generating unique
            annotation ids when the GT dicts don't carry an ``id`` field.
    """
    from mmengine.structures import InstanceData

    n = len(gt_instances_data)
    if n == 0:
        return

    def _stack(key):
        arrays = [g.get(key) for g in gt_instances_data]
        if any(a is None for a in arrays):
            return None
        return np.concatenate([np.atleast_2d(a) for a in arrays], axis=0)

    # Replace gt_instances entirely.  After merge_data_samples the existing
    # gt_instances has one entry per *detection*; GT may have a different
    # count (real-detector mode), so we must not update it in-place.
    gt_inst = InstanceData()
    data_sample.gt_instances = gt_inst

    # keypoints: each item is (1, K, 2) → stack to (N, K, 2)
    kpts = _stack('keypoints')
    if kpts is not None:
        gt_inst.keypoints = kpts.astype(np.float32)

    # keypoints_visible: each item is (1, K) or (1, K, 2) → (N, K) or (N, K, 2)
    kv = _stack('keypoints_visible')
    if kv is not None:
        gt_inst.keypoints_visible = kv.astype(np.float32)

    # bbox_scale: each item is (1, 2) → (N, 2)
    bs = _stack('bbox_scale')
    if bs is not None:
        gt_inst.bbox_scales = bs.astype(np.float32)

    # bbox: each item is (1, 4) xyxy → (N, 4).  Required when detector
    # bboxes differ from GT (real-detector mode); mock mode gets these from
    # the inference pipeline via merge_data_samples.
    bb = _stack('bbox')
    if bb is not None:
        gt_inst.bboxes = bb.astype(np.float32)

    # Original per-instance annotation area (mask-based for COCO, heuristic
    # for MPII).  Store as a (N,) float array so the metric can use the
    # correct area for OKS computation instead of the padded bbox_scale area.
    orig_areas = [g.get('orig_area') for g in gt_instances_data]
    if all(a is not None for a in orig_areas):
        gt_inst.orig_areas = np.array(orig_areas, dtype=np.float32)

    # Override the sample's id with the GT instance ids so that
    # CocoMetric._synthesize_raw_ann_info produces unique annotation ids that
    # match the original dataset's per-instance identifiers.
    gt_ids = [g.get('id') for g in gt_instances_data]
    if all(x is not None for x in gt_ids):
        data_sample.set_metainfo({'id': [int(x) for x in gt_ids]})
    else:
        data_sample.set_metainfo(
            {'id': list(range(img_id * 10000, img_id * 10000 + n))})


# ── Bottomup pipeline ──────────────────────────────────────────────────────

def run_bottomup(
    model,
    batches: list,
    total_frames: int,
    evaluator: Evaluator,
    device: str,
    warmup_batches: int = 3,
    log_interval: int = 100,
    frame_records: Optional[List[dict]] = None,
    dataset_meta: Optional[dict] = None,
    data_root: str = '',
    gt_by_img_id: Optional[Dict[int, list]] = None,
) -> Tuple[dict, dict]:
    """Bottomup benchmark loop with CUDA-event timing."""
    batch_latencies: List[Tuple[float, int]] = []  # (elapsed_s, n_samples)
    n_batches = len(batches)

    for i, (batch, _img_ids) in enumerate(batches):
        # Move tensor inputs to device
        batch['inputs'] = [t.to(device) for t in batch['inputs']]

        with torch.no_grad():
            with _CudaTimer() as timer:
                results = model.test_step(batch)

        if i >= warmup_batches:
            batch_latencies.append((timer.elapsed_s, len(results)))

        evaluator.process(data_samples=results, data_batch=batch)

        if frame_records is not None:
            for ds in results:
                meta = ds.metainfo
                ori_shape = meta.get('ori_shape', (0, 0))
                img_id = int(meta['img_id'])
                gt_src = (gt_by_img_id or {}).get(img_id, [])
                frame_records.append(
                    build_frame_record(
                        img_id=img_id,
                        frame_id=len(frame_records),
                        img_path=meta.get('img_path', ''),
                        data_root=data_root,
                        ori_shape=(int(ori_shape[0]), int(ori_shape[1])),
                        pred_instances=serialize_pred_instances(
                            ds.pred_instances),
                        gt_instances=serialize_gt_instances(gt_src),
                        dataset_meta=dataset_meta,
                    ))

        if (i + 1) % log_interval == 0 or (i + 1) == n_batches:
            frames_done = sum(n for _, n in batch_latencies)
            elapsed = sum(t for t, _ in batch_latencies)
            fps_so_far = frames_done / elapsed if elapsed > 0 else 0.0
            print(f'  [bottomup] batch {i + 1}/{n_batches} '
                  f'| frames processed: {frames_done} '
                  f'| running FPS: {fps_so_far:.1f}')

    quality = evaluator.evaluate(total_frames)

    timed_time = sum(t for t, _ in batch_latencies)
    timed_frames = sum(n for _, n in batch_latencies)
    per_frame = [t / n for t, n in batch_latencies]
    batch_times = [t for t, _ in batch_latencies]

    perf = {
        'e2e/fps': timed_frames / timed_time if timed_time > 0 else 0.0,
        'e2e/latency_ms_per_batch': (
            1000.0 * sum(batch_times) / len(batch_times) if batch_times else 0.0
        ),
        'e2e/latency_ms_per_frame': (
            1000.0 * sum(per_frame) / len(per_frame) if per_frame else 0.0
        ),
    }
    return quality, perf


# ── Topdown pipeline ───────────────────────────────────────────────────────

def _mock_detector_producer(
    prefetched_images: List[Tuple[int, np.ndarray]],
    gt_bboxes_per_frame: Dict[int, Tuple[np.ndarray, np.ndarray]],
    det_batch_size: int,
    bbox_queue: queue.Queue,
    log_interval: int,
    # shared output containers (written only by this thread)
    frame_start_times: dict,
    frame_end_times: dict,
    zero_det_frames: dict,
    det_predictions: dict,
) -> None:
    """Producer: push GT BBoxItems into bbox_queue without running a detector.

    Mirrors the structure of _detector_producer so the downstream consumer
    and timing containers work identically, but det_timings is never written
    (detector timing metrics will therefore report 0 in the output).
    """
    n = len(prefetched_images)
    n_batches = -(-n // det_batch_size)

    for batch_start in range(0, n, det_batch_size):
        batch_end = min(batch_start + det_batch_size, n)
        batch = prefetched_images[batch_start:batch_end]

        wall_start = time.perf_counter()
        for fid in range(batch_start, batch_end):
            frame_start_times[fid] = wall_start

        batch_idx = batch_start // det_batch_size
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == n_batches:
            print(f'  [mock-detector] batch {batch_idx + 1}/{n_batches} '
                  f'| frames processed: {batch_end}')

        wall_after = time.perf_counter()

        for rel, (img_id, img) in enumerate(batch):
            frame_id = batch_start + rel
            h, w = img.shape[:2]

            bboxes_xyxy, scores = gt_bboxes_per_frame.get(
                frame_id, (np.zeros((0, 4), dtype=np.float32),
                            np.zeros(0, dtype=np.float32)))

            det_predictions[frame_id] = (bboxes_xyxy, scores)
            n_bboxes = len(bboxes_xyxy)

            if n_bboxes == 0:
                zero_det_frames[frame_id] = (img_id, (h, w))
                frame_end_times[frame_id] = wall_after
            else:
                for bbox, score in zip(bboxes_xyxy, scores):
                    bbox_queue.put(BBoxItem(
                        frame_id=frame_id,
                        img_id=img_id,
                        img=img,
                        bbox_xyxy=bbox,
                        det_score=float(score),
                        n_bboxes_in_frame=n_bboxes,
                        frame_start_wall=wall_start,
                    ))

    bbox_queue.put(_SENTINEL)


def _detector_producer(
    detector,
    prefetched_images: List[Tuple[int, np.ndarray]],
    det_batch_size: int,
    bbox_thr: float,
    nms_thr: float,
    det_cat_id: int,
    bbox_queue: queue.Queue,
    warmup_batches: int,
    log_interval: int,
    # shared output containers (written only by this thread)
    det_timings: list,
    frame_start_times: dict,
    frame_end_times: dict,
    zero_det_frames: dict,
    det_predictions: dict,
) -> None:
    """Producer: run detector and push BBoxItems into bbox_queue.

    For frames with zero detections the pipeline ends at the detector, so
    frame_end_times is set here (to the wall-clock time after the detector
    batch returns) rather than in the consumer.
    """
    n = len(prefetched_images)
    n_batches = -(-n // det_batch_size)  # ceil division

    for batch_start in range(0, n, det_batch_size):
        batch_end = min(batch_start + det_batch_size, n)
        batch = prefetched_images[batch_start:batch_end]

        wall_start = time.perf_counter()
        for fid in range(batch_start, batch_end):
            frame_start_times[fid] = wall_start

        imgs = [img for _, img in batch]
        with torch.no_grad():
            with _CudaTimer() as timer:
                det_results = inference_det_model(detector, imgs)

        wall_after_det = time.perf_counter()

        batch_idx = batch_start // det_batch_size
        if batch_idx >= warmup_batches:
            det_timings.append((timer.elapsed_s, len(batch)))

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == n_batches:
            timed_frames = sum(k for _, k in det_timings)
            timed_time = sum(t for t, _ in det_timings)
            fps_so_far = timed_frames / timed_time if timed_time > 0 else 0.0
            print(f'  [detector] batch {batch_idx + 1}/{n_batches} '
                  f'| frames processed: {batch_end} '
                  f'| running FPS: {fps_so_far:.1f}')

        for rel, ((img_id, img), det_result) in enumerate(
                zip(batch, det_results)):
            frame_id = batch_start + rel
            pred = det_result.pred_instances.cpu().numpy()

            # Class-filter + score-threshold
            mask = np.logical_and(pred.labels == det_cat_id,
                                  pred.scores > bbox_thr)
            bboxes_s = np.concatenate(
                [pred.bboxes[mask], pred.scores[mask, None]], axis=1)

            if bboxes_s.shape[0] > 0:
                keep = nms(bboxes_s, nms_thr)
                bboxes_xyxy = bboxes_s[keep, :4]
                scores = bboxes_s[keep, 4]
            else:
                bboxes_xyxy = bboxes_s[:, :4]
                scores = np.array([], dtype=np.float32)

            det_predictions[frame_id] = (bboxes_xyxy, scores)
            n_bboxes = len(bboxes_xyxy)
            h, w = img.shape[:2]

            if n_bboxes == 0:
                # Pipeline ends here for this frame; record end time now.
                zero_det_frames[frame_id] = (img_id, (h, w))
                frame_end_times[frame_id] = wall_after_det
            else:
                for bbox, score in zip(bboxes_xyxy, scores):
                    bbox_queue.put(BBoxItem(
                        frame_id=frame_id,
                        img_id=img_id,
                        img=img,
                        bbox_xyxy=bbox,
                        det_score=float(score),
                        n_bboxes_in_frame=n_bboxes,
                        frame_start_wall=wall_start,
                    ))

    bbox_queue.put(_SENTINEL)


def _keypoint_consumer(
    kp_model,
    kp_pipeline: Compose,
    kp_batch_size: int,
    queue_strategy: str,
    bbox_queue: queue.Queue,
    warmup_batches: int,
    log_interval: int,
    # shared output containers (written only by this thread)
    kp_timings: list,
    frame_results: dict,
    frame_end_times: dict,
    frame_counts: dict,
    dataset_meta: dict,
) -> None:
    """Consumer: collect BBoxItems, run keypoint model, record timings."""
    batch_counter = 0

    def flush(buf: list) -> None:
        nonlocal batch_counter
        if not buf:
            return

        data_list = []
        for item in buf:
            di = dict(img=item.img)
            di['bbox'] = item.bbox_xyxy[None]       # shape (1, 4)
            di['bbox_score'] = np.array(
                [item.det_score], dtype=np.float32)  # shape (1,)
            di.update(dataset_meta)
            data_list.append(kp_pipeline(di))

        batch = pseudo_collate(data_list)

        with torch.no_grad():
            with _CudaTimer() as timer:
                results = kp_model.test_step(batch)

        if batch_counter >= warmup_batches:
            kp_timings.append((timer.elapsed_s, len(buf)))
        batch_counter += 1

        if batch_counter % log_interval == 0:
            timed_locs = sum(k for _, k in kp_timings)
            timed_time = sum(t for t, _ in kp_timings)
            fps_so_far = timed_locs / timed_time if timed_time > 0 else 0.0
            print(f'  [keypoint] batch {batch_counter} '
                  f'| locations processed: {timed_locs} '
                  f'| running FPS: {fps_so_far:.1f}')

        t_end = time.perf_counter()

        for item, result in zip(buf, results):
            fid = item.frame_id
            if fid not in frame_results:
                frame_results[fid] = []
                frame_counts[fid] = item.n_bboxes_in_frame
            frame_results[fid].append((result, item.det_score))
            if len(frame_results[fid]) >= frame_counts[fid]:
                frame_end_times[fid] = t_end

    buffer: List[BBoxItem] = []
    done = False

    while not done:
        # Ensure at least one item is in the buffer before strategy logic
        if not buffer:
            item = bbox_queue.get()
            if item is _SENTINEL:
                break
            buffer.append(item)

        if queue_strategy == 'full_batch':
            # Block until buffer has a full batch or producer is done
            while len(buffer) < kp_batch_size:
                item = bbox_queue.get()
                if item is _SENTINEL:
                    done = True
                    break
                buffer.append(item)
            flush(buffer[:kp_batch_size])
            buffer = buffer[kp_batch_size:]

        elif queue_strategy == 'any':
            # Non-blocking drain up to kp_batch_size, then flush
            while len(buffer) < kp_batch_size:
                try:
                    item = bbox_queue.get_nowait()
                    if item is _SENTINEL:
                        done = True
                        break
                    buffer.append(item)
                except queue.Empty:
                    break
            flush(buffer[:kp_batch_size])
            buffer = buffer[kp_batch_size:]

        elif queue_strategy == 'same_frame':
            # Accumulate only items that share the current frame_id.
            # If a different frame_id arrives, flush immediately and start fresh.
            current_fid = buffer[0].frame_id
            pending: Optional[BBoxItem] = None

            while len(buffer) < kp_batch_size:
                try:
                    item = bbox_queue.get_nowait()
                    if item is _SENTINEL:
                        done = True
                        break
                    if item.frame_id == current_fid:
                        buffer.append(item)
                    else:
                        # Different frame – stop accumulating for current_fid
                        pending = item
                        break
                except queue.Empty:
                    break

            flush(buffer[:kp_batch_size])
            buffer = buffer[kp_batch_size:]

            # Remaining same-frame items (if any) stay at front of buffer;
            # inject the next-frame item right after them.
            if pending is not None:
                buffer.append(pending)

    # Flush anything left over
    if buffer:
        flush(buffer)


def run_topdown(
    det_model,
    kp_model,
    prefetched_images: List[Tuple[int, np.ndarray]],
    evaluator: Evaluator,
    device: str,
    det_batch_size: int,
    kp_batch_size: int,
    queue_strategy: str,
    bbox_thr: float,
    nms_thr: float,
    det_cat_id: int,
    warmup_batches: int,
    pose_cfg: Config,
    log_interval: int = 100,
    gt_bboxes_per_frame: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    det_evaluator: Optional[Evaluator] = None,
    gt_instances_per_frame: Optional[Dict[int, list]] = None,
    frame_records: Optional[List[dict]] = None,
    dataset_meta: Optional[dict] = None,
    data_root: str = '',
    img_paths: Optional[List[str]] = None,
    gt_by_img_id: Optional[Dict[int, list]] = None,
) -> Tuple[dict, dict]:
    """Run topdown async producer/consumer benchmark.

    When gt_bboxes_per_frame is provided the real detector is bypassed and
    GT bboxes are fed directly into the queue (mock-detector mode).
    """
    _init_scope(pose_cfg)
    # Filter out GT-only transforms that require keypoints in the input dict
    # (e.g. KeypointConverter).  During inference the consumer only provides
    # image + bbox, so these steps would raise KeyError on 'keypoints'.
    _GT_ONLY_TYPES = {'KeypointConverter'}
    kp_pipeline_cfg = [
        s for s in pose_cfg.test_dataloader.dataset.pipeline
        if not (isinstance(s, dict) and s.get('type') in _GT_ONLY_TYPES)
    ]
    kp_pipeline = Compose(kp_pipeline_cfg)
    dataset_meta = kp_model.dataset_meta

    # Bounded queue provides back-pressure so the producer does not run too
    # far ahead of the consumer.
    bbox_queue: queue.Queue = queue.Queue(maxsize=kp_batch_size * 8)

    det_timings: list = []
    kp_timings: list = []
    frame_start_times: dict = {}
    frame_end_times: dict = {}
    frame_results: dict = {}
    frame_counts: dict = {}
    zero_det_frames: dict = {}
    det_predictions: dict = {}  # frame_id -> (bboxes_xyxy, scores)

    if gt_bboxes_per_frame is not None:
        producer = threading.Thread(
            target=_mock_detector_producer,
            args=(prefetched_images, gt_bboxes_per_frame, det_batch_size,
                  bbox_queue, log_interval,
                  frame_start_times, frame_end_times, zero_det_frames,
                  det_predictions),
            daemon=True,
        )
    else:
        producer = threading.Thread(
            target=_detector_producer,
            args=(det_model, prefetched_images, det_batch_size, bbox_thr,
                  nms_thr, det_cat_id, bbox_queue, warmup_batches,
                  log_interval, det_timings, frame_start_times,
                  frame_end_times, zero_det_frames, det_predictions),
            daemon=True,
        )
    consumer = threading.Thread(
        target=_keypoint_consumer,
        args=(kp_model, kp_pipeline, kp_batch_size, queue_strategy,
              bbox_queue, warmup_batches, log_interval, kp_timings,
              frame_results, frame_end_times, frame_counts, dataset_meta),
        daemon=True,
    )

    producer.start()
    consumer.start()
    producer.join()
    consumer.join()

    # Build per-frame PoseDataSamples for the evaluator
    n_images = len(prefetched_images)
    for frame_id, (img_id, img) in enumerate(prefetched_images):
        h, w = img.shape[:2]
        gt_inst_data = (gt_instances_per_frame or {}).get(frame_id)
        if frame_id in zero_det_frames:
            ds = _make_topdown_data_sample(
                img_id, [], [], (h, w), gt_inst_data)
        elif frame_id in frame_results:
            res_scores = frame_results[frame_id]
            ds = _make_topdown_data_sample(
                img_id,
                [r for r, _ in res_scores],
                [s for _, s in res_scores],
                (h, w),
                gt_inst_data,
            )
        else:
            # Frame was neither processed by detector nor had results (shouldn't
            # happen in normal operation but guard defensively).
            raise ValueError(f'Frame {frame_id} was neither processed by detector nor had results')

        evaluator.process(data_samples=[ds], data_batch=None)

        if frame_records is not None:
            img_path = ''
            if img_paths is not None and frame_id < len(img_paths):
                img_path = img_paths[frame_id]
            gt_src = (gt_by_img_id or {}).get(int(img_id), [])
            det_bb, det_sc = det_predictions.get(
                frame_id,
                (np.zeros((0, 4), dtype=np.float32),
                 np.zeros(0, dtype=np.float32)))
            frame_records.append(
                build_frame_record(
                    img_id=int(img_id),
                    frame_id=frame_id,
                    img_path=img_path,
                    data_root=data_root,
                    ori_shape=(int(h), int(w)),
                    pred_instances=serialize_pred_instances(
                        ds.pred_instances),
                    gt_instances=serialize_gt_instances(gt_src),
                    dataset_meta=dataset_meta,
                    det_bboxes=det_bb,
                    det_scores=det_sc,
                ))

    quality = evaluator.evaluate(n_images)

    # ── Detection AP/AR metrics (optional) ──────────────────────────────
    if det_evaluator is not None:
        for frame_id, (img_id, img) in enumerate(prefetched_images):
            h, w = img.shape[:2]
            bb, sc = det_predictions.get(
                frame_id,
                (np.zeros((0, 4), dtype=np.float32),
                 np.zeros(0, dtype=np.float32)))
            det_ds = dict(
                img_id=img_id,
                ori_shape=(h, w),
                pred_instances=dict(
                    bboxes=torch.from_numpy(bb),
                    scores=torch.from_numpy(sc),
                    labels=torch.zeros(len(bb), dtype=torch.long),
                ),
            )
            det_evaluator.process(data_samples=[det_ds], data_batch=None)
        det_quality = det_evaluator.evaluate(n_images)
        quality = {**quality, **det_quality}

    # ── Performance metrics ──────────────────────────────────────────────
    timed_det_time = sum(t for t, _ in det_timings)
    timed_det_frames = sum(n for _, n in det_timings)
    timed_kp_time = sum(t for t, _ in kp_timings)
    timed_kp_bboxes = sum(n for _, n in kp_timings)

    det_per_frame = [t / n for t, n in det_timings]
    det_batch_times = [t for t, _ in det_timings]
    kp_per_loc = [t / n for t, n in kp_timings]
    kp_batch_times = [t for t, _ in kp_timings]

    complete_fids = [
        fid for fid in range(n_images)
        if fid in frame_start_times and fid in frame_end_times
    ]
    if complete_fids:
        pipe_wall_start = min(frame_start_times[f] for f in complete_fids)
        pipe_wall_end = max(frame_end_times[f] for f in complete_fids)
        pipe_latencies = [
            frame_end_times[f] - frame_start_times[f] for f in complete_fids
        ]
    else:
        pipe_wall_start = pipe_wall_end = 0.0
        pipe_latencies = []

    pipe_total_wall = pipe_wall_end - pipe_wall_start

    perf = {
        'e2e/fps': (
            len(complete_fids) / pipe_total_wall
            if pipe_total_wall > 0 else 0.0
        ),
        'e2e/latency_ms_per_frame': (
            1000.0 * sum(pipe_latencies) / len(pipe_latencies)
            if pipe_latencies else 0.0
        ),
        'detector/fps': (
            timed_det_frames / timed_det_time if timed_det_time > 0 else 0.0
        ),
        'detector/latency_ms_per_batch': (
            1000.0 * sum(det_batch_times) / len(det_batch_times)
            if det_batch_times else 0.0
        ),
        'detector/latency_ms_per_frame': (
            1000.0 * sum(det_per_frame) / len(det_per_frame)
            if det_per_frame else 0.0
        ),
        'keypoint/fps': (
            timed_kp_bboxes / timed_kp_time if timed_kp_time > 0 else 0.0
        ),
        'keypoint/latency_ms_per_batch': (
            1000.0 * sum(kp_batch_times) / len(kp_batch_times)
            if kp_batch_times else 0.0
        ),
        'keypoint/latency_ms_per_location': (
            1000.0 * sum(kp_per_loc) / len(kp_per_loc)
            if kp_per_loc else 0.0
        ),
    }
    return quality, perf


# ── Output helpers ─────────────────────────────────────────────────────────

def _print_results(quality: dict, perf: dict, mode: str) -> None:
    sep = '=' * 62
    pose_metrics = {k: v for k, v in quality.items() if not k.startswith('det/')}
    det_metrics = {k: v for k, v in quality.items() if k.startswith('det/')}

    print(f'\n{sep}')
    print(f'  Benchmark results  ({mode})')
    print(sep)
    print('  Quality metrics:')
    for k, v in sorted(pose_metrics.items()):
        print(f'    {k}: {v:.4f}')
    if det_metrics:
        print('  Detector metrics:')
        for k, v in sorted(det_metrics.items()):
            print(f'    {k}: {v:.4f}')
    print('  Performance metrics:')
    for k, v in sorted(perf.items()):
        unit = 'ms' if 'latency' in k else 'fps/s'
        print(f'    {k}: {v:.2f}')
    print(f'{sep}\n')


def _save_out(quality: dict, perf: dict, mode: str, args) -> None:
    payload = {
        'mode': mode,
        'quality': quality,
        'perf': perf,
        'config': {
            'pose_config': osp.abspath(args.pose_config),
            'pose_checkpoint': osp.abspath(args.pose_checkpoint),
            'det_config': (
                osp.abspath(args.det_config) if args.det_config else None),
            'det_checkpoint': (
                osp.abspath(args.det_checkpoint)
                if args.det_checkpoint else None),
        },
        'test_dataset': args.test_dataset,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }
    out_dir = osp.dirname(osp.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Results saved to {args.out}')


def _prediction_mode_tag(is_topdown: bool) -> str:
    return 'topdown' if is_topdown else 'e2e'


def _prediction_out_dir(args, run_date: str, is_topdown: bool) -> str:
    if args.pred_dir:
        return osp.abspath(args.pred_dir)
    model_label = args.model_name
    if args.model_variant:
        model_label = f'{args.model_name}-{args.model_variant}'
    tag = f'{run_date}_{args.test_dataset}_{_prediction_mode_tag(is_topdown)}'
    return osp.join('benchmark', 'predictions', tag, model_label)


def _save_predictions(
    args,
    pose_cfg: Config,
    mode: str,
    is_topdown: bool,
    quality: dict,
    perf: dict,
    frame_records: List[dict],
    dataset_meta: dict,
    run_date: str,
) -> None:
    data_root = pose_cfg.test_dataloader.dataset.get('data_root', 'data/')
    manifest = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'mode': mode,
        'mode_tag': _prediction_mode_tag(is_topdown),
        'test_dataset': args.test_dataset,
        'model_name': args.model_name,
        'model_variant': args.model_variant,
        'pose_config': osp.abspath(args.pose_config),
        'pose_checkpoint': osp.abspath(args.pose_checkpoint),
        'det_config': (
            osp.abspath(args.det_config) if args.det_config else None),
        'det_checkpoint': (
            osp.abspath(args.det_checkpoint)
            if args.det_checkpoint else None),
        'quality': quality,
        'perf': perf,
        'data_root': data_root,
        'dataset_meta': sanitize_dataset_meta(dataset_meta),
        'badcase_defaults': {
            'metric_key': 'mean_oks',
            'metric_type': 'accuracy',
            'thr': 0.5,
        },
        'num_frames': len(frame_records),
    }
    out_dir = _prediction_out_dir(args, run_date, is_topdown)
    save_prediction_bundle(manifest, frame_records, out_dir)
    print(f'Predictions saved to {osp.abspath(out_dir)}')


def _append_to_results_file(quality: dict, perf: dict, args) -> None:
    """Append to --results-file in the same schema as test_tracked.py."""
    results_file = args.results_file
    if osp.isfile(results_file):
        with open(results_file, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    metrics = {**quality, **{f'perf/{k}': v for k, v in perf.items()}}
    entry = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'config': osp.abspath(args.pose_config),
        'checkpoint': osp.abspath(args.pose_checkpoint),
        'test_dataset': args.test_dataset,
        'metrics': metrics,
    }
    data.setdefault(args.model_name, {}).setdefault(
        args.model_variant, []).append(entry)

    os.makedirs(osp.dirname(osp.abspath(results_file)), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Results appended to {results_file} '
          f'({args.model_name} / {args.model_variant})')


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='End-to-end pose estimation benchmarking pipeline')
    p.add_argument('pose_config',
                   help='Pose model config file (bottomup or topdown)')
    p.add_argument('pose_checkpoint', help='Pose model checkpoint')

    grp = p.add_argument_group('Topdown mode')
    grp.add_argument('--det-config', default=None,
                     help='Detector config (topdown mode)')
    grp.add_argument('--det-checkpoint', default=None,
                     help='Detector checkpoint (topdown mode)')
    grp.add_argument('--mock-detector', action='store_true',
                     help='Skip the real detector and seed the queue with GT '
                          'bboxes from the test dataset (topdown mode only). '
                          'Mutually exclusive with --det-config/--det-checkpoint.')

    p.add_argument('--det-batch-size', type=int, default=4,
                   help='Detector batch size (topdown mode, default: 4)')
    p.add_argument('--kp-batch-size', type=int, default=32,
                   help='Keypoint model batch size (default: 32)')
    p.add_argument(
        '--queue-strategy',
        choices=['full_batch', 'any', 'same_frame'],
        default='full_batch',
        help='Queue batching strategy for the keypoint consumer '
             '(topdown mode, default: full_batch)')
    p.add_argument('--device', default='cuda:0',
                   help='Inference device (default: cuda:0)')
    p.add_argument('--num-frames', type=int, default=None,
                   help='Cap number of images evaluated (default: full val set)')
    p.add_argument('--warmup-batches', type=int, default=3,
                   help='Leading batches excluded from timing (default: 3)')
    p.add_argument('--bbox-thr', type=float, default=0.3,
                   help='Detector bbox score threshold (default: 0.3)')
    p.add_argument('--nms-thr', type=float, default=0.3,
                   help='NMS IoU threshold for detector bboxes (default: 0.3)')
    p.add_argument('--det-cat-id', type=int, default=0,
                   help='Detector category id for the person class (default: 0)')
    p.add_argument('--out', default=None,
                   help='Save JSON results to this file')
    p.add_argument('--results-file', default=None,
                   help='Append to tracked results JSON (test_tracked schema)')
    p.add_argument('--model-name', default=None,
                   help='Model group name (required with --results-file)')
    p.add_argument('--model-variant', default=None,
                   help='Model variant name (required with --results-file)')
    p.add_argument(
        '--pred-dir',
        default=None,
        help='Directory for prediction export (default: '
             'benchmark/predictions/{DATE}_{DATASET}_{MODE}/{MODELNAME})')
    p.add_argument('--det-metrics', action='store_true',
                   help='Compute COCO-style detection AP/AR for the person '
                        'class (topdown mode only).')
    p.add_argument('--log-interval', type=int, default=100,
                   help='Print a progress line every N batches (default: 100)')
    p.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default={},
        help='Override config options, e.g. model.backbone.depth=18')
    p.add_argument(
        '--test-dataset', default='coco',
        choices=list(BENCHMARK_TEST_DATASET_NAMES),
        help='Override test set and metric (default: use config as-is)')
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    run_date = datetime.now().strftime('%Y%m%d')
    frame_records: Optional[List[dict]] = [] if args.model_name else None
    gt_by_img_id: Optional[Dict[int, list]] = None
    dataset_meta = None

    # ── Config ────────────────────────────────────────────────────────────
    pose_cfg = Config.fromfile(args.pose_config)
    if args.cfg_options:
        pose_cfg.merge_from_dict(args.cfg_options)
    apply_benchmark_test_dataset(pose_cfg, args.test_dataset)

    _init_scope(pose_cfg)
    MMLogger.get_current_instance()  # ensure logger is initialised

    if frame_records is not None:
        print('Building GT lookup for prediction export...')
        gt_by_img_id = build_gt_by_img_id(pose_cfg, args.num_frames)

    use_real_detector = bool(args.det_config and args.det_checkpoint)
    is_topdown = args.mock_detector or use_real_detector
    model_type = pose_cfg.model.get('type', '')
    if 'Bottomup' in model_type and is_topdown:
        raise ValueError(
            'The pose config uses a BottomupPoseEstimator but --det-config '
            'or --mock-detector was also provided. '
            'Bottomup models do not need a detector.')
    if args.mock_detector and use_real_detector:
        raise ValueError(
            '--mock-detector and --det-config/--det-checkpoint are mutually '
            'exclusive.')
    if getattr(args, 'det_metrics', False) and not is_topdown:
        raise ValueError(
            '--det-metrics requires topdown mode '
            '(pass --det-config/--det-checkpoint or --mock-detector).')

    # ── Topdown mode ───────────────────────────────────────────────────────
    if is_topdown:
        print(f'\nMode: topdown  |  strategy: {args.queue_strategy}')

        detector = None
        gt_bboxes_per_frame = None
        gt_instances_per_frame = None

        if args.mock_detector:
            print('  Detector : MOCK (GT bboxes from dataset)')
            print(f'  KP model : {args.pose_config}\n')

            print('Building GT bbox image list...')
            img_list_with_bbox = build_gt_bbox_image_list(
                pose_cfg, args.num_frames)
            img_list = [(iid, p) for iid, p, _, _, _ in img_list_with_bbox]
            gt_bboxes_per_frame = {
                fid: (bb, sc)
                for fid, (_, _, bb, sc, _) in enumerate(img_list_with_bbox)
            }
            gt_instances_per_frame = {
                fid: gt_insts
                for fid, (_, _, _, _, gt_insts) in enumerate(img_list_with_bbox)
            }
        else:
            if not HAS_MMDET:
                raise ImportError(
                    'mmdet is required for topdown mode. '
                    'Install it with: pip install mmdet')

            print(f'  Detector : {args.det_config}')
            print(f'  KP model : {args.pose_config}\n')

            detector = init_det_model(
                args.det_config, args.det_checkpoint, device=args.device)

            print('Building image list...')
            img_list = build_unique_image_list(pose_cfg, args.num_frames)
            if (_evaluator_needs_gt_from_samples(pose_cfg)
                    or frame_records is not None):
                print('Building GT instance lookup for evaluation...')
                gt_instances_per_frame = build_gt_instances_per_frame(
                    pose_cfg, img_list)

        kp_model = init_model(
            args.pose_config, args.pose_checkpoint, device=args.device)
        dataset_meta = kp_model.dataset_meta
        evaluator = build_evaluator(pose_cfg, kp_model.dataset_meta)

        det_evaluator = None
        if getattr(args, 'det_metrics', False):
            det_evaluator = build_det_evaluator(
                pose_cfg, num_frames=args.num_frames)

        prefetched = prefetch_images(img_list)
        img_paths = [path for _, path in img_list]
        data_root = pose_cfg.test_dataloader.dataset.get('data_root', 'data/')

        quality, perf = run_topdown(
            det_model=detector,
            kp_model=kp_model,
            prefetched_images=prefetched,
            evaluator=evaluator,
            device=args.device,
            det_batch_size=args.det_batch_size,
            kp_batch_size=args.kp_batch_size,
            queue_strategy=args.queue_strategy,
            bbox_thr=args.bbox_thr,
            nms_thr=args.nms_thr,
            det_cat_id=args.det_cat_id,
            warmup_batches=args.warmup_batches,
            pose_cfg=pose_cfg,
            log_interval=args.log_interval,
            gt_bboxes_per_frame=gt_bboxes_per_frame,
            det_evaluator=det_evaluator,
            gt_instances_per_frame=gt_instances_per_frame,
            frame_records=frame_records,
            dataset_meta=kp_model.dataset_meta,
            data_root=data_root,
            img_paths=img_paths,
            gt_by_img_id=gt_by_img_id,
        )
        mode_suffix = 'mock-detector' if args.mock_detector else args.queue_strategy
        mode = f'topdown (strategy={mode_suffix})'

    # ── Bottomup mode ──────────────────────────────────────────────────────
    else:
        print(f'\nMode: bottomup')
        print(f'  Model : {args.pose_config}\n')

        model = init_model(
            args.pose_config, args.pose_checkpoint, device=args.device)
        dataset_meta = model.dataset_meta
        evaluator = build_evaluator(pose_cfg, model.dataset_meta)

        print('Building and prefetching batches...')
        batches, total_frames = build_bottomup_batches(
            pose_cfg, args.kp_batch_size, args.num_frames)
        data_root = pose_cfg.test_dataloader.dataset.get('data_root', 'data/')

        quality, perf = run_bottomup(
            model=model,
            batches=batches,
            total_frames=total_frames,
            evaluator=evaluator,
            device=args.device,
            warmup_batches=args.warmup_batches,
            log_interval=args.log_interval,
            frame_records=frame_records,
            dataset_meta=model.dataset_meta,
            data_root=data_root,
            gt_by_img_id=gt_by_img_id,
        )
        mode = 'bottomup'

    # ── Output ─────────────────────────────────────────────────────────────
    _print_results(quality, perf, mode)

    if args.out:
        _save_out(quality, perf, mode, args)

    if args.results_file:
        assert args.model_name and args.model_variant, (
            '--model-name and --model-variant are required when '
            '--results-file is specified')
        _append_to_results_file(quality, perf, args)

    if args.model_name and frame_records is not None:
        _save_predictions(
            args, pose_cfg, mode, is_topdown, quality, perf,
            frame_records, dataset_meta, run_date)


if __name__ == '__main__':
    main()
