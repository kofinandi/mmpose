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
from mmpose.apis import init_model
from mmpose.evaluation.functional import nms
from mmpose.structures import PoseDataSample, merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
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


def _make_topdown_data_sample(
    img_id: int,
    bbox_results: list,
    det_scores: List[float],
    ori_shape: Tuple[int, int],
) -> PoseDataSample:
    """Merge per-bbox PoseDataSamples into a single per-image PoseDataSample.

    Sets ``id`` as a list so ``CocoMetric._sort_and_unique_bboxes`` skips
    deduplication (matching bottomup convention) and all N detected persons
    are preserved.
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
        return ds

    # Inject detector scores into each result's gt_instances so that
    # CocoMetric (score_mode='bbox_keypoint') uses the detector confidence.
    for res, score in zip(bbox_results, det_scores):
        res.gt_instances.bbox_scores = np.array([score], dtype=np.float32)

    merged = merge_data_samples(bbox_results)

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
    return merged


# ── Bottomup pipeline ──────────────────────────────────────────────────────

def run_bottomup(
    model,
    batches: list,
    total_frames: int,
    evaluator: Evaluator,
    device: str,
    warmup_batches: int = 3,
    log_interval: int = 100,
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
                det_results = inference_detector(detector, imgs)

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
) -> Tuple[dict, dict]:
    """Run topdown async producer/consumer benchmark."""
    _init_scope(pose_cfg)
    kp_pipeline = Compose(pose_cfg.test_dataloader.dataset.pipeline)
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

    producer = threading.Thread(
        target=_detector_producer,
        args=(det_model, prefetched_images, det_batch_size, bbox_thr, nms_thr,
              det_cat_id, bbox_queue, warmup_batches, log_interval,
              det_timings, frame_start_times, frame_end_times, zero_det_frames),
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
        if frame_id in zero_det_frames:
            ds = _make_topdown_data_sample(img_id, [], [], (h, w))
        elif frame_id in frame_results:
            res_scores = frame_results[frame_id]
            ds = _make_topdown_data_sample(
                img_id,
                [r for r, _ in res_scores],
                [s for _, s in res_scores],
                (h, w),
            )
        else:
            # Frame was neither processed by detector nor had results (shouldn't
            # happen in normal operation but guard defensively).
            raise ValueError(f'Frame {frame_id} was neither processed by detector nor had results')

        evaluator.process(data_samples=[ds], data_batch=None)

    quality = evaluator.evaluate(n_images)

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
    print(f'\n{sep}')
    print(f'  Benchmark results  ({mode})')
    print(sep)
    print('  Quality metrics:')
    for k, v in sorted(quality.items()):
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
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }
    out_dir = osp.dirname(osp.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Results saved to {args.out}')


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

    grp = p.add_argument_group('Topdown mode (requires both flags)')
    grp.add_argument('--det-config', default=None,
                     help='Detector config (topdown mode)')
    grp.add_argument('--det-checkpoint', default=None,
                     help='Detector checkpoint (topdown mode)')

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
    p.add_argument('--log-interval', type=int, default=100,
                   help='Print a progress line every N batches (default: 100)')
    p.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default={},
        help='Override config options, e.g. model.backbone.depth=18')
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    pose_cfg = Config.fromfile(args.pose_config)
    if args.cfg_options:
        pose_cfg.merge_from_dict(args.cfg_options)

    _init_scope(pose_cfg)
    MMLogger.get_current_instance()  # ensure logger is initialised

    is_topdown = bool(args.det_config and args.det_checkpoint)
    model_type = pose_cfg.model.get('type', '')
    if 'Bottomup' in model_type and is_topdown:
        raise ValueError(
            'The pose config uses a BottomupPoseEstimator but --det-config '
            'was also provided.  Bottomup models do not need a detector.')

    # ── Topdown mode ───────────────────────────────────────────────────────
    if is_topdown:
        if not HAS_MMDET:
            raise ImportError(
                'mmdet is required for topdown mode. '
                'Install it with: pip install mmdet')

        print(f'\nMode: topdown  |  strategy: {args.queue_strategy}')
        print(f'  Detector : {args.det_config}')
        print(f'  KP model : {args.pose_config}\n')

        detector = init_detector(
            args.det_config, args.det_checkpoint, device=args.device)
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)

        kp_model = init_model(
            args.pose_config, args.pose_checkpoint, device=args.device)
        evaluator = build_evaluator(pose_cfg, kp_model.dataset_meta)

        print('Building image list...')
        img_list = build_unique_image_list(pose_cfg, args.num_frames)
        prefetched = prefetch_images(img_list)

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
        )
        mode = f'topdown (strategy={args.queue_strategy})'

    # ── Bottomup mode ──────────────────────────────────────────────────────
    else:
        print(f'\nMode: bottomup')
        print(f'  Model : {args.pose_config}\n')

        model = init_model(
            args.pose_config, args.pose_checkpoint, device=args.device)
        evaluator = build_evaluator(pose_cfg, model.dataset_meta)

        print('Building and prefetching batches...')
        batches, total_frames = build_bottomup_batches(
            pose_cfg, args.kp_batch_size, args.num_frames)

        quality, perf = run_bottomup(
            model=model,
            batches=batches,
            total_frames=total_frames,
            evaluator=evaluator,
            device=args.device,
            warmup_batches=args.warmup_batches,
            log_interval=args.log_interval,
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


if __name__ == '__main__':
    main()
