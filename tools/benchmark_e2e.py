# Copyright (c) OpenMMLab. All rights reserved.
"""End-to-end pose estimation benchmarking pipeline.

Supports two modes:
  - Bottomup: single model (e.g. YOLOX-pose) processes whole images.
  - Topdown: detector + keypoint localizer pipeline with an async
    producer/consumer queue between the two stages.

Quality metrics are the same as tools/test.py (CocoMetric, PCK, etc.).
Performance metrics cover FPS and per-frame/per-location latency at
whole-pipeline and per-stage granularity.

Data loading is unified via :func:`load_unified_samples`: all datasets
(coco, crowdpose, mpii, aic, ochuman, emdb) go through one code path that
loads every annotation unfiltered, converts keypoints to COCO-17 format,
and prefetches images.  GT is never read from the model pipeline; it is
assembled from :class:`UnifiedSample` and attached to each
:class:`PoseDataSample` before metric evaluation.
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
from mmengine.structures import InstanceData

# Register all mmpose modules
import mmpose.datasets       # noqa: F401
import mmpose.evaluation     # noqa: F401
import mmpose.models         # noqa: F401
from mmengine.registry import METRICS
from mmpose.apis.det_inference import inference_det_model, init_det_model
from mmpose.apis import init_model
from mmpose.evaluation.benchmark_datasets import BENCHMARK_TEST_DATASETS
from mmpose.evaluation.functional import nms
from mmpose.evaluation.functional.benchmark_data import (
    GTInstance,
    UnifiedSample,
    build_det_ann_from_samples,
    is_valid_instance,
    load_unified_samples,
)
from mmpose.evaluation.functional.frame_metrics import (
    build_frame_record,
    save_prediction_bundle,
    sanitize_dataset_meta,
)
from mmpose.structures import PoseDataSample, merge_data_samples

try:
    import mmdet  # noqa: F401
    HAS_MMDET = True
except (ImportError, ModuleNotFoundError):
    HAS_MMDET = False

# Named tuple carried through the producer→consumer queue
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


# ── Helpers ───────────────────────────────────────────────────────────────

def _init_scope(cfg: Config) -> None:
    scope = cfg.get('default_scope', 'mmpose')
    if scope:
        init_default_scope(scope)


def _data_root_for_dataset(dataset_name: str) -> str:
    """Return the data root for the dataset being evaluated."""
    return BENCHMARK_TEST_DATASETS[dataset_name].data_root


# ── Evaluator helpers ──────────────────────────────────────────────────────

def build_evaluator(pose_cfg: Config, dataset_meta: dict) -> Evaluator:
    """Build an mmengine Evaluator from the config's test_evaluator section.

    Forces ``gt_from_samples=True`` and removes ``ann_file`` on every metric
    so that ground truth is always synthesised from the unified loader data
    rather than read from the dataset's annotation file.  This makes every
    dataset (including COCO) behave consistently.
    """
    _init_scope(pose_cfg)
    ev_cfg = pose_cfg.test_evaluator
    if isinstance(ev_cfg, dict):
        ev_cfg = [ev_cfg]

    metrics = []
    for m_cfg in ev_cfg:
        m_cfg = dict(m_cfg)
        m_cfg['gt_from_samples'] = True
        m_cfg.pop('ann_file', None)
        m = METRICS.build(m_cfg)
        m.dataset_meta = dataset_meta
        metrics.append(m)

    return Evaluator(metrics)


def build_det_evaluator(
    samples: List[UnifiedSample],
) -> 'Evaluator':
    """COCO bbox AP/AR for the person class only.

    Uses :func:`build_det_ann_from_samples` to build the annotation file
    from the already-loaded unified samples so that ``iscrowd`` values are
    preserved correctly.
    """
    if not HAS_MMDET:
        raise ImportError(
            'mmdet is required for --det-metrics. '
            'Install it with: pip install mmdet')

    from mmdet.evaluation.metrics import CocoMetric as _CocoMetric

    class _SubsetCocoMetric(_CocoMetric):
        """CocoMetric restricted to images actually processed."""

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
            nums = self.proposal_nums
            ar_renames = {
                'bbox_AR@100':    f'bbox_AR@{nums[0]}',
                'bbox_AR@300':    f'bbox_AR@{nums[1]}',
                'bbox_AR@1000':   f'bbox_AR@{nums[2]}',
                'bbox_AR_s@1000': f'bbox_AR_s@{nums[2]}',
                'bbox_AR_m@1000': f'bbox_AR_m@{nums[2]}',
                'bbox_AR_l@1000': f'bbox_AR_l@{nums[2]}',
            }
            return {ar_renames.get(k, k): v for k, v in out.items()}

    ann_file = build_det_ann_from_samples(samples)

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


# ── Sample → PoseDataSample helpers ──────────────────────────────────────

def _merge_pose_predictions_to_posedatasample(
    img_id: int,
    bbox_results: list,
    det_scores: List[float],
    ori_shape: Tuple[int, int],
    img_path: Optional[str] = None,
) -> PoseDataSample:
    """Merge per-bbox keypoint results into a single per-image PoseDataSample.

    Prediction-only; GT is attached separately by :func:`_attach_gt_to_sample`.
    """
    n = len(bbox_results)
    num_kpts = 17  # COCO default; may be overridden if model differs

    if n == 0:
        ds = PoseDataSample()
        meta = {
            'img_id': img_id,
            'id': [img_id],
            'category_id': 1,
            'ori_shape': ori_shape,
        }
        if img_path is not None:
            meta['img_path'] = img_path
        ds.set_metainfo(meta)
        pred_inst = InstanceData()
        pred_inst.keypoints = np.zeros((0, num_kpts, 2), dtype=np.float32)
        pred_inst.keypoint_scores = np.zeros((0, num_kpts), dtype=np.float32)
        pred_inst.bboxes = np.zeros((0, 4), dtype=np.float32)
        pred_inst.bbox_scores = np.zeros(0, dtype=np.float32)
        ds.pred_instances = pred_inst
        return ds

    # Inject detector scores into each result's gt_instances so that
    # CocoMetric (score_mode='bbox_keypoint') uses the detector confidence.
    for res, score in zip(bbox_results, det_scores):
        res.gt_instances.bbox_scores = np.array([score], dtype=np.float32)

    merged = merge_data_samples(bbox_results)
    merged.pred_instances.bbox_scores = np.array(det_scores, dtype=np.float32)

    # Use a list of unique synthetic IDs to trigger the bottomup code path
    # inside CocoMetric._sort_and_unique_bboxes (which early-returns when
    # the 'id' value is a Sequence).
    ids = list(range(img_id * 10000, img_id * 10000 + n))
    meta = {
        'img_id': img_id,
        'id': ids,
        'category_id': 1,
        'ori_shape': ori_shape,
    }
    if img_path is not None:
        meta['img_path'] = img_path
    merged.set_metainfo(meta)
    return merged


def _attach_gt_to_posedatasample(
    ds: PoseDataSample,
    sample: UnifiedSample,
) -> PoseDataSample:
    """Attach GT annotations from a :class:`UnifiedSample` to a prediction PoseDataSample.

    Builds ``gt_instances`` (with ``keypoints``, ``keypoints_visible``,
    ``bboxes``, ``orig_areas``, and ``iscrowd``) and updates the metainfo
    with the real annotation IDs and optional ``crowd_index``.
    """
    gt_insts = sample.gt_instances
    n = len(gt_insts)

    gt = InstanceData()
    if n > 0:
        gt.keypoints = np.stack(
            [g.keypoints for g in gt_insts], axis=0).astype(np.float32)
        gt.keypoints_visible = np.stack(
            [g.keypoints_visible for g in gt_insts], axis=0).astype(np.float32)
        gt.bboxes = np.stack(
            [g.bbox for g in gt_insts], axis=0).astype(np.float32)
        gt.orig_areas = np.array(
            [g.area for g in gt_insts], dtype=np.float32)
        gt.iscrowd = np.array(
            [g.iscrowd for g in gt_insts], dtype=np.int32)

    ds.gt_instances = gt

    # Use actual annotation IDs (list → CocoMetric skips deduplication and
    # uses these as annotation IDs when synthesising raw_ann_info).
    gt_ids = [g.id for g in gt_insts] if n > 0 else [sample.img_id]
    meta_update: dict = {'id': gt_ids}
    if sample.crowd_index is not None:
        meta_update['crowd_index'] = float(sample.crowd_index)
    ds.set_metainfo(meta_update)
    return ds


def _pose_dicts_from_unifiedsample(sample: UnifiedSample) -> List[dict]:
    """Build serialisation-friendly GT dicts from a :class:`UnifiedSample`."""
    result = []
    for g in sample.gt_instances:
        d: dict = {
            'keypoints': g.keypoints,
            'keypoints_visible': g.keypoints_visible,
            'bbox': g.bbox,
            'orig_area': g.area,
            'iscrowd': int(g.iscrowd),
        }
        if g.keypoints_visible_coco is not None:
            d['keypoints_visible_coco'] = g.keypoints_visible_coco
        result.append(d)
    return result


# ── Bottomup pipeline ──────────────────────────────────────────────────────

def run_bottomup(
    model,
    samples: List[UnifiedSample],
    val_pipeline: Compose,
    dataset_meta: dict,
    kp_batch_size: int,
    device: str,
    warmup_batches: int = 3,
    log_interval: int = 100,
) -> Tuple[Dict[int, PoseDataSample], dict]:
    """Run bottomup inference on unified samples.

    Pre-builds all batches (images are already prefetched) then runs the
    timed model loop.  GT and evaluation are handled in :func:`main`.

    Args:
        model: Bottomup pose estimator.
        samples: Loaded unified samples (one per image).
        val_pipeline: Val pipeline from config; first step handles
            pre-loaded images via ``LoadImage``.
        dataset_meta: Model dataset meta for pipeline augmentation.
        kp_batch_size: Images per batch.
        device: Target device string.
        warmup_batches: Leading batches excluded from timing.
        log_interval: Batches between progress prints.

    Returns:
        pred_by_img_id: Mapping ``img_id → PoseDataSample`` (prediction only).
        perf: Performance-metric dict.
    """
    print('Building bottomup batches ...')
    batches: List[Tuple[dict, List[int]]] = []
    n = len(samples)
    for start in range(0, n, kp_batch_size):
        end = min(start + kp_batch_size, n)
        items = []
        img_ids = []
        for sample in samples[start:end]:
            di: dict = {
                'img': sample.image,
                'img_path': sample.img_path,
                'img_id': sample.img_id,
                # Use GT annotation IDs so CocoMetric gets the right IDs
                # when synthesising raw_ann_info.  A list value also causes
                # CocoMetric._sort_and_unique_bboxes to skip deduplication.
                'id': [g.id for g in sample.gt_instances],
                'ori_shape': sample.ori_shape,
            }
            di.update(dataset_meta)
            items.append(val_pipeline(di))
            img_ids.append(sample.img_id)
        batches.append((pseudo_collate(items), img_ids))

    batch_latencies: List[Tuple[float, int]] = []
    n_batches = len(batches)
    pred_by_img_id: Dict[int, PoseDataSample] = {}

    for i, (batch, img_ids) in enumerate(batches):
        batch['inputs'] = [t.to(device) for t in batch['inputs']]

        with torch.no_grad():
            with _CudaTimer() as timer:
                results = model.test_step(batch)

        if i >= warmup_batches:
            batch_latencies.append((timer.elapsed_s, len(results)))

        for ds, img_id in zip(results, img_ids):
            pred_by_img_id[img_id] = ds

        if (i + 1) % log_interval == 0 or (i + 1) == n_batches:
            frames_done = sum(k for _, k in batch_latencies)
            elapsed = sum(t for t, _ in batch_latencies)
            fps_so_far = frames_done / elapsed if elapsed > 0 else 0.0
            print(f'  [bottomup] batch {i + 1}/{n_batches} '
                  f'| frames processed: {frames_done} '
                  f'| running FPS: {fps_so_far:.1f}')

    timed_time = sum(t for t, _ in batch_latencies)
    timed_frames = sum(k for _, k in batch_latencies)
    per_frame = [t / k for t, k in batch_latencies]
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
    return pred_by_img_id, perf


# ── Topdown pipeline ───────────────────────────────────────────────────────

def _mock_detector_producer(
    prefetched_images: List[Tuple[int, np.ndarray]],
    gt_bboxes_per_frame: Dict[int, Tuple[np.ndarray, np.ndarray]],
    det_batch_size: int,
    bbox_queue: queue.Queue,
    log_interval: int,
    frame_start_times: dict,
    frame_end_times: dict,
    zero_det_frames: dict,
    det_predictions: dict,
) -> None:
    """Producer: push GT BBoxItems into bbox_queue without running a detector."""
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
    det_timings: list,
    frame_start_times: dict,
    frame_end_times: dict,
    zero_det_frames: dict,
    det_predictions: dict,
) -> None:
    """Producer: run detector and push BBoxItems into bbox_queue."""
    n = len(prefetched_images)
    n_batches = -(-n // det_batch_size)

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
            di['bbox'] = item.bbox_xyxy[None]
            di['bbox_score'] = np.array(
                [item.det_score], dtype=np.float32)
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
        if not buffer:
            item = bbox_queue.get()
            if item is _SENTINEL:
                break
            buffer.append(item)

        if queue_strategy == 'full_batch':
            while len(buffer) < kp_batch_size:
                item = bbox_queue.get()
                if item is _SENTINEL:
                    done = True
                    break
                buffer.append(item)
            flush(buffer[:kp_batch_size])
            buffer = buffer[kp_batch_size:]

        elif queue_strategy == 'any':
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
                        pending = item
                        break
                except queue.Empty:
                    break

            flush(buffer[:kp_batch_size])
            buffer = buffer[kp_batch_size:]

            if pending is not None:
                buffer.append(pending)

    if buffer:
        flush(buffer)


def run_topdown(
    det_model,
    kp_model,
    samples: List[UnifiedSample],
    det_batch_size: int,
    kp_batch_size: int,
    queue_strategy: str,
    bbox_thr: float,
    nms_thr: float,
    det_cat_id: int,
    warmup_batches: int,
    pose_cfg: Config,
    log_interval: int = 100,
    use_mock_detector: bool = False,
) -> Tuple[Dict[int, PoseDataSample], dict]:
    """Run topdown async producer/consumer benchmark on unified samples.

    When ``use_mock_detector=True`` the real detector is bypassed and valid
    GT bboxes (filtered by :func:`is_valid_instance`) are fed directly into
    the queue.

    Raw detector bboxes/scores are stored on each returned
    :class:`PoseDataSample` under metainfo keys ``'det_bboxes'`` and
    ``'det_scores'`` so :func:`build_frame_record` can access them without
    additional arguments.

    Args:
        det_model: Detector model (ignored when ``use_mock_detector=True``).
        kp_model: Keypoint model.
        samples: Unified samples (one per image).
        det_batch_size: Detector batch size.
        kp_batch_size: Keypoint model batch size.
        queue_strategy: One of ``'full_batch'``, ``'any'``, ``'same_frame'``.
        bbox_thr: Detector bbox confidence threshold.
        nms_thr: NMS IoU threshold.
        det_cat_id: Category id for the person class in the detector.
        warmup_batches: Leading batches excluded from timing.
        pose_cfg: Pose config (used to build the keypoint pipeline).
        log_interval: Batches between progress prints.
        use_mock_detector: If True, skip real detector.

    Returns:
        pred_by_img_id: Mapping ``img_id → PoseDataSample`` (prediction only).
        perf: Performance-metric dict.
    """
    _init_scope(pose_cfg)

    # Filter GT-only transforms (KeypointConverter) from the keypoint pipeline.
    _GT_ONLY_TYPES = {'KeypointConverter'}
    kp_pipeline_cfg = [
        s for s in pose_cfg.test_dataloader.dataset.pipeline
        if not (isinstance(s, dict) and s.get('type') in _GT_ONLY_TYPES)
    ]
    kp_pipeline = Compose(kp_pipeline_cfg)
    dataset_meta = kp_model.dataset_meta

    # Convert samples to the legacy (img_id, img) tuple list for producers
    prefetched_images: List[Tuple[int, np.ndarray]] = [
        (s.img_id, s.image) for s in samples
    ]

    # Build GT bboxes for mock detector (valid instances only)
    gt_bboxes_per_frame: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if use_mock_detector:
        for fid, sample in enumerate(samples):
            valid = [g for g in sample.gt_instances if is_valid_instance(g)]
            if valid:
                bboxes = np.stack([g.bbox for g in valid], axis=0)
                scores = np.ones(len(valid), dtype=np.float32)
            else:
                bboxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros(0, dtype=np.float32)
            gt_bboxes_per_frame[fid] = (bboxes, scores)

    bbox_queue: queue.Queue = queue.Queue(maxsize=kp_batch_size * 8)

    det_timings: list = []
    kp_timings: list = []
    frame_start_times: dict = {}
    frame_end_times: dict = {}
    frame_results: dict = {}
    frame_counts: dict = {}
    zero_det_frames: dict = {}
    det_predictions: dict = {}

    if use_mock_detector:
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

    # ── Performance metrics ─────────────────────────────────────────────
    n_images = len(samples)
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

    timed_det_time = sum(t for t, _ in det_timings)
    timed_det_frames = sum(k for _, k in det_timings)
    timed_kp_time = sum(t for t, _ in kp_timings)
    timed_kp_bboxes = sum(k for _, k in kp_timings)

    det_per_frame = [t / k for t, k in det_timings]
    det_batch_times = [t for t, _ in det_timings]
    kp_per_loc = [t / k for t, k in kp_timings]
    kp_batch_times = [t for t, _ in kp_timings]

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

    # ── Assemble one PoseDataSample per image ─────────────────────────────
    # Store raw detector output in metainfo so build_frame_record can pick it
    # up without needing separate det_bboxes / det_scores arguments.
    pred_by_img_id: Dict[int, PoseDataSample] = {}
    for fid, sample in enumerate(samples):
        img_id = sample.img_id
        h, w = sample.ori_shape

        det_bb, det_sc = det_predictions.get(
            fid,
            (np.zeros((0, 4), dtype=np.float32),
             np.zeros(0, dtype=np.float32)),
        )

        if fid in frame_results:
            res_scores = frame_results[fid]
            pred_ds = _merge_pose_predictions_to_posedatasample(
                img_id,
                [r for r, _ in res_scores],
                [s for _, s in res_scores],
                (h, w),
                img_path=sample.img_path,
            )
        else:
            pred_ds = _merge_pose_predictions_to_posedatasample(
                img_id, [], [], (h, w), img_path=sample.img_path)

        if len(det_bb) > 0:
            pred_ds.set_metainfo({'det_bboxes': det_bb, 'det_scores': det_sc})

        pred_by_img_id[img_id] = pred_ds

    return pred_by_img_id, perf


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
    data_root: str,
    mode: str,
    is_topdown: bool,
    quality: dict,
    perf: dict,
    frame_records: List[dict],
    dataset_meta: dict,
    run_date: str,
) -> None:
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

    p.add_argument('--det-batch-size', type=int, default=32,
                   help='Detector batch size (topdown mode, default: 32)')
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
        choices=list(BENCHMARK_TEST_DATASETS),
        help='Override test set and metric (default: use config as-is)')
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    run_date = datetime.now().strftime('%Y%m%d')
    frame_records: Optional[List[dict]] = [] if args.model_name else None

    # ── Config ────────────────────────────────────────────────────────────
    pose_cfg = Config.fromfile(args.pose_config)
    if args.cfg_options:
        pose_cfg.merge_from_dict(args.cfg_options)

    _init_scope(pose_cfg)
    MMLogger.get_current_instance()

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

    # ── Unified data loading ───────────────────────────────────────────────
    print(f'\nLoading dataset: {args.test_dataset}')
    samples = load_unified_samples(args.test_dataset, args.num_frames)
    n_images = len(samples)
    data_root = _data_root_for_dataset(args.test_dataset)

    # Val pipeline from config with GT-only transforms stripped.
    # We never need KeypointConverter in the inference pipeline since GT
    # keypoint conversion is already handled in load_unified_samples.
    _GT_ONLY_TYPES = {'KeypointConverter'}
    raw_pipeline = list(pose_cfg.test_dataloader.dataset.pipeline)
    pipeline_cfg = [
        s for s in raw_pipeline
        if not (isinstance(s, dict) and s.get('type') in _GT_ONLY_TYPES)
    ]

    # ── Topdown mode ───────────────────────────────────────────────────────
    if is_topdown:
        print(f'\nMode: topdown  |  strategy: {args.queue_strategy}')

        detector = None
        if args.mock_detector:
            print('  Detector : MOCK (GT bboxes from dataset)')
            print(f'  KP model : {args.pose_config}\n')
        else:
            if not HAS_MMDET:
                raise ImportError(
                    'mmdet is required for topdown mode. '
                    'Install it with: pip install mmdet')
            print(f'  Detector : {args.det_config}')
            print(f'  KP model : {args.pose_config}\n')
            detector = init_det_model(
                args.det_config, args.det_checkpoint, device=args.device)

        kp_model = init_model(
            args.pose_config, args.pose_checkpoint, device=args.device)
        dataset_meta = kp_model.dataset_meta
        evaluator = build_evaluator(pose_cfg, dataset_meta)

        det_evaluator = (build_det_evaluator(samples)
                          if getattr(args, 'det_metrics', False) else None)

        pred_by_img_id, perf = run_topdown(
            det_model=detector,
            kp_model=kp_model,
            samples=samples,
            det_batch_size=args.det_batch_size,
            kp_batch_size=args.kp_batch_size,
            queue_strategy=args.queue_strategy,
            bbox_thr=args.bbox_thr,
            nms_thr=args.nms_thr,
            det_cat_id=args.det_cat_id,
            warmup_batches=args.warmup_batches,
            pose_cfg=pose_cfg,
            log_interval=args.log_interval,
            use_mock_detector=args.mock_detector,
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
        evaluator = build_evaluator(pose_cfg, dataset_meta)
        det_evaluator = None

        val_pipeline = Compose(pipeline_cfg)

        pred_by_img_id, perf = run_bottomup(
            model=model,
            samples=samples,
            val_pipeline=val_pipeline,
            dataset_meta=dataset_meta,
            kp_batch_size=args.kp_batch_size,
            device=args.device,
            warmup_batches=args.warmup_batches,
            log_interval=args.log_interval,
        )

        mode = 'bottomup'

    # ── Unified post-processing (both modes) ───────────────────────────────
    for fid, sample in enumerate(samples):
        img_id = sample.img_id
        pred_ds = pred_by_img_id.get(img_id)
        if pred_ds is None:
            continue

        ds = _attach_gt_to_posedatasample(pred_ds, sample)
        evaluator.process(data_samples=[ds], data_batch=None)

        if frame_records is not None:
            frame_records.append(build_frame_record(
                img_id=int(img_id),
                frame_id=fid,
                img_path=sample.img_path,
                data_root=data_root,
                pred_ds=ds,
                gt_instances=_pose_dicts_from_unifiedsample(sample),
                dataset_meta=dataset_meta,
            ))

    quality = evaluator.evaluate(n_images)

    # Optional detection AP/AR (topdown only — reads from pred_ds.metainfo)
    if det_evaluator is not None:
        for fid, sample in enumerate(samples):
            pred_ds = pred_by_img_id.get(sample.img_id)
            bb = (pred_ds.metainfo.get(
                'det_bboxes', np.zeros((0, 4), dtype=np.float32))
                  if pred_ds is not None
                  else np.zeros((0, 4), dtype=np.float32))
            sc = (pred_ds.metainfo.get(
                'det_scores', np.zeros(0, dtype=np.float32))
                  if pred_ds is not None
                  else np.zeros(0, dtype=np.float32))
            bb = np.asarray(bb, dtype=np.float32)
            sc = np.asarray(sc, dtype=np.float32)
            det_ds = dict(
                img_id=sample.img_id,
                ori_shape=sample.ori_shape,
                pred_instances=dict(
                    bboxes=torch.from_numpy(bb),
                    scores=torch.from_numpy(sc),
                    labels=torch.zeros(len(bb), dtype=torch.long),
                ),
            )
            det_evaluator.process(data_samples=[det_ds], data_batch=None)
        det_quality = det_evaluator.evaluate(n_images)
        quality = {**quality, **det_quality}

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
            args, data_root, mode, is_topdown, quality, perf,
            frame_records, dataset_meta, run_date)


if __name__ == '__main__':
    main()
