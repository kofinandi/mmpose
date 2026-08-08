# Copyright (c) OpenMMLab. All rights reserved.
"""End-to-end pose estimation benchmarking pipeline.

Supports two modes:
  - Bottomup: single model (e.g. YOLOX-pose) processes whole images.
  - Topdown: detector + keypoint localizer pipeline with an async
    producer/consumer queue between the two stages.

Quality metrics are the same as tools/test.py (CocoMetric, PCK, etc.).
Performance metrics cover FPS and per-frame/per-location latency at
whole-pipeline and per-stage granularity.

Data loading is unified: all datasets (coco, crowdpose, mpii, aic, ochuman,
emdb, 3dpw, posetrack21) go through one code path that loads every
annotation unfiltered and converts keypoints to COCO-17 format.  EMDB and
3DPW downscale images (default ~0.33x) to reduce RAM; GT annotations are
scaled to match.  GT is never read from the model pipeline; it is assembled
from :class:`~mmpose.evaluation.functional.benchmark_data.UnifiedSample` and
attached to each :class:`PoseDataSample` before metric evaluation.

By default (:func:`~mmpose.evaluation.functional.benchmark_data.load_unified_samples`)
every image is decoded up front, which eliminates disk I/O from the timed
loop but does not fit large datasets (e.g. PoseTrack21) in memory all at
once. Pass ``--prefetch-chunk-size N`` to instead build samples with
:func:`~mmpose.evaluation.functional.benchmark_data.build_unified_samples`
(metadata only) and stream their pixel data in bounded chunks via
:class:`~mmpose.evaluation.functional.benchmark_data.SampleImageStream`,
decoding in the background while inference runs on already-decoded chunks.
This trades a bounded amount of RAM for the possibility that decoding
becomes the bottleneck; watch the reported ``dataload/stall_s`` perf key
(and the warning printed when it is significant) to tell whether measured
FPS is I/O-bound.
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
from typing import Dict, Iterator, List, Optional, Tuple

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
    SampleImageStream,
    UnifiedSample,
    build_det_ann_from_samples,
    build_unified_samples,
    is_valid_instance,
    load_unified_samples,
)
from mmpose.evaluation.functional.frame_metrics import (
    build_frame_record,
    save_prediction_bundle,
    sanitize_dataset_meta,
)
from mmpose.evaluation.functional.good_frames import BAD_FRAME_DATASETS
from mmpose.postprocessing import PostProcessingPipeline, build_post_processor
from mmpose.structures import PoseDataSample, merge_data_samples

try:
    import mmdet  # noqa: F401
    HAS_MMDET = True
except (ImportError, ModuleNotFoundError):
    HAS_MMDET = False

# Named tuple carried through the producer→consumer queue
BBoxItem = collections.namedtuple(
    'BBoxItem',
    ['frame_id', 'img_id', 'img', 'clip_imgs', 'bbox_xyxy', 'det_score',
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


def _blank_image(ori_shape: Tuple[int, int]) -> np.ndarray:
    """Build a black placeholder image for a sample whose decode failed.

    Used so a single unreadable frame (possible in chunked-prefetch mode,
    where every sample must still flow through the pipeline to preserve
    ``frame_id`` alignment) degrades to a zero-detection frame instead of
    crashing the run.
    """
    h, w = max(int(ori_shape[0]), 1), max(int(ori_shape[1]), 1)
    return np.zeros((h, w, 3), dtype=np.uint8)


FrameItem = collections.namedtuple(
    'FrameItem',
    ['frame_id', 'sample', 'image', 'clip'],
)


def _iter_stream_items(
    sample_stream,
) -> Iterator[Tuple[int, UnifiedSample, Optional[np.ndarray]]]:
    """Enumerate a sample stream as ``(frame_id, sample, image)`` triples.

    Each sample's ``image`` is captured the instant it is pulled from
    ``sample_stream``, rather than left to be read later from
    ``sample.image``. This matters for chunked prefetch, where
    :class:`SampleImageStream` releases each chunk's pixel data
    (``sample.image = None``) once the stream has advanced past it, and
    for clip windowing (see :func:`_with_clip_windows`), which must keep
    its own references to a handful of neighbouring frames while the
    stream continues to advance underneath it.
    """
    for frame_id, sample in enumerate(sample_stream):
        yield frame_id, sample, sample.image


def _clip_seq_key(sample: UnifiedSample):
    """Sequence-grouping key for clip-window assembly.

    Uses ``UnifiedSample.seq_id`` (populated for EMDB/3DPW/PoseTrack21 from
    the annotation's ``seq_name``) when available. Datasets without video
    metadata (COCO, CrowdPose, MPII, AIC, OCHuman) fall back to the
    sample's own ``img_id`` as a singleton "sequence" of one frame, so a
    clip requested around such a frame is just that frame repeated --
    no cross-image bleed, but no real temporal context either.
    """
    return sample.seq_id if sample.seq_id is not None else sample.img_id


def _with_clip_windows(
    item_iter: Iterator[Tuple[int, UnifiedSample, Optional[np.ndarray]]],
    num_input_frames: int,
) -> Iterator[Tuple[int, UnifiedSample, Optional[np.ndarray],
                    Optional[List[Optional[np.ndarray]]]]]:
    """Attach a temporal clip window to each ``(frame_id, sample, image)``.

    For ``num_input_frames > 1``, yields ``(frame_id, sample, image, clip)``
    where ``clip`` is a list of ``num_input_frames`` images
    ``[image[-half], ..., image, ..., image[+half]]`` centred on ``image``
    (``half = num_input_frames // 2``), matching the fixed odd-length
    temporal windows used by every published video-pose config (Poseidon,
    TAR-ViTPose: 5; PAVE-Net: 3).

    Windows are edge-clamped *within a sequence*: frames whose
    :func:`_clip_seq_key` differs from the current frame's (i.e. would
    cross a video boundary, or the stream start/end) are replaced with the
    current frame itself, replicating the reference implementations'
    "clip index to [0, nframes-1]" boundary behaviour without needing to
    know each sequence's length up front.

    For ``num_input_frames <= 1`` (the default, and every existing
    single-frame model), yields ``(frame_id, sample, image, None)`` --
    ``clip`` is ``None`` rather than ``[image]`` so callers can cheaply
    branch on ``clip is None`` and leave the single-frame code path (and
    its performance) completely unchanged.

    Holds at most ``num_input_frames`` decoded frames in memory beyond
    whatever :class:`SampleImageStream` itself buffers, since each
    ``image`` reference is captured by :func:`_iter_stream_items` the
    instant it is produced; this works for chunked prefetch as long as
    ``--prefetch-chunk-size`` (if used) is at least ``num_input_frames``,
    since otherwise a chunk boundary could release a frame's pixels before
    a later frame's window is assembled from it.
    """
    if num_input_frames <= 1:
        for frame_id, sample, image in item_iter:
            yield frame_id, sample, image, None
        return

    if num_input_frames % 2 == 0:
        raise ValueError(
            'num_input_frames must be odd (or 1); every published '
            f'video-pose config uses a centred, odd-length window. Got '
            f'{num_input_frames}.')

    half = num_input_frames // 2
    it = iter(item_iter)
    window: List[Tuple[int, UnifiedSample, Optional[np.ndarray]]] = []
    pos = 0
    exhausted = False

    def _extend_lookahead() -> None:
        nonlocal exhausted
        while not exhausted and (len(window) - pos - 1) < half:
            try:
                window.append(next(it))
            except StopIteration:
                exhausted = True

    _extend_lookahead()
    while pos < len(window):
        _extend_lookahead()
        frame_id, sample, image = window[pos]
        seq_key = _clip_seq_key(sample)

        # Walk outward from the centre in each direction, freezing at the
        # last in-sequence frame once a boundary (different seq_id, or the
        # stream's own start/end) is crossed -- since the stream visits
        # each sequence contiguously, every offset beyond that point is
        # equally out of range, so it repeats that same boundary frame
        # (matching the reference implementations' `clip(idx, 0, n-1)`).
        before = []
        last = image
        for offset in range(1, half + 1):
            idx = pos - offset
            if idx >= 0:
                _, nb_sample, nb_image = window[idx]
                if _clip_seq_key(nb_sample) == seq_key:
                    last = nb_image
            before.append(last)
        before.reverse()

        after = []
        last = image
        for offset in range(1, half + 1):
            idx = pos + offset
            if idx < len(window):
                _, nb_sample, nb_image = window[idx]
                if _clip_seq_key(nb_sample) == seq_key:
                    last = nb_image
            after.append(last)

        clip = before + [image] + after

        yield frame_id, sample, image, clip

        pos += 1
        if pos > half:
            drop = pos - half
            window = window[drop:]
            pos -= drop


def _batched_samples(
    sample_stream,
    batch_size: int,
    num_input_frames: int = 1,
) -> Iterator[List[FrameItem]]:
    """Group a sample stream into batches of :class:`FrameItem`.

    ``frame_id`` is the running index into the stream, i.e. the index into
    the original samples list, giving callers a stable id to key
    dictionaries by (e.g. ``frame_start_times`` / ``det_predictions`` in
    :func:`run_topdown`) -- matching the dataset-order semantics the rest
    of the pipeline relies on, regardless of whether the stream is eager
    or chunked.

    When ``num_input_frames > 1``, each item's ``clip`` field additionally
    carries the temporal window around it (see :func:`_with_clip_windows`);
    otherwise ``clip`` is ``None``. See :func:`_iter_stream_items` for how
    ``image`` capture interacts with chunked prefetch.
    """
    batch: List[FrameItem] = []
    stream_items = _with_clip_windows(
        _iter_stream_items(sample_stream), num_input_frames)
    for frame_id, sample, image, clip in stream_items:
        batch.append(FrameItem(frame_id, sample, image, clip))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _fill_blank_clip(
    clip: List[Optional[np.ndarray]],
    ori_shape: Tuple[int, int],
) -> List[np.ndarray]:
    """Replace any ``None`` frame in a clip (failed decode) with a blank."""
    return [im if im is not None else _blank_image(ori_shape) for im in clip]


# ── Evaluator helpers ──────────────────────────────────────────────────────

def build_evaluator(
    pose_cfg: Config,
    dataset_meta: dict,
    test_dataset: Optional[str] = None,
    include_tracking_metrics: bool = False,
    full_tracking_metrics: bool = False,
) -> Evaluator:
    """Build an mmengine Evaluator from the config's test_evaluator section.

    Forces ``gt_from_samples=True`` and removes ``ann_file`` on every metric
    so that ground truth is always synthesised from the unified loader data
    rather than read from the dataset's annotation file.  This makes every
    dataset (including COCO) behave consistently.

    When ``test_dataset`` is provided and the corresponding
    :class:`BenchmarkTestDataset` specifies ``extra_metrics``, those metrics
    are appended to the evaluator **without** ``gt_from_samples`` injection
    (temporal metrics manage their own state via ``process()``).

    Args:
        pose_cfg: Pose config providing ``test_evaluator``.
        dataset_meta: Dataset meta info (e.g. keypoint sigmas) assigned to
            every built metric.
        test_dataset: Optional benchmark dataset name used to look up
            ``extra_metrics``.
        include_tracking_metrics: When ``True``, unconditionally appends an
            :class:`~mmpose.evaluation.metrics.IDSwitch` metric, independent
            of ``test_dataset``.  It manages its own state via ``process()``
            and reports zero switches when the evaluated predictions carry
            no ``track_ids`` (e.g. no tracker in the post-processing
            pipeline), so it is safe to enable whenever post-processed
            (potentially tracked) predictions are being evaluated.
        full_tracking_metrics: When ``True``, additionally appends
            :class:`MOTA`, :class:`IDF1` and :class:`HOTA`.  Used for models
            that assign track IDs during inference (``emits_track_ids``),
            where the run would otherwise produce no association metrics at
            all -- the post-processing path can still reach these via
            ``tools/postprocess_predictions.py --metrics``.  Kept off by
            default so existing runs' metric sets are unchanged.
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

    if test_dataset is not None:
        spec = BENCHMARK_TEST_DATASETS.get(test_dataset)
        if spec is not None and spec.extra_metrics:
            for m_cfg in spec.extra_metrics:
                m = METRICS.build(dict(m_cfg))
                m.dataset_meta = dataset_meta
                metrics.append(m)

    if include_tracking_metrics:
        tracking_types = ['IDSwitch']
        if full_tracking_metrics:
            tracking_types += ['MOTA', 'IDF1', 'HOTA']
        for m_type in tracking_types:
            m = METRICS.build(dict(type=m_type))
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
    num_kpts = 17  # COCO default; may be overridden if model differs

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
        gt.track_ids = np.array(
            [g.track_id for g in gt_insts], dtype=np.int32)
        # Raw COCO visibility (0/1/2) used by OKS matching in temporal metrics
        kv_coco_list = [g.keypoints_visible_coco for g in gt_insts]
        if any(v is not None for v in kv_coco_list):
            gt.keypoints_visible_coco = np.stack(
                [v if v is not None else g.keypoints_visible
                 for v, g in zip(kv_coco_list, gt_insts)],
                axis=0).astype(np.float32)
    else:
        # No GT this frame (e.g. a "bad" EMDB/3DPW frame loaded via
        # --include-bad-frames), set explicit zero-length arrays.
        gt.keypoints = np.zeros((0, num_kpts, 2), dtype=np.float32)
        gt.keypoints_visible = np.zeros((0, num_kpts), dtype=np.float32)
        gt.bboxes = np.zeros((0, 4), dtype=np.float32)
        gt.orig_areas = np.zeros(0, dtype=np.float32)
        gt.iscrowd = np.zeros(0, dtype=np.int32)
        gt.track_ids = np.zeros(0, dtype=np.int32)

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
            'gt_ann_id': int(g.id),
            'keypoints': g.keypoints,
            'keypoints_visible': g.keypoints_visible,
            'bbox': g.bbox,
            'orig_area': g.area,
            'iscrowd': int(g.iscrowd),
            'track_id': int(g.track_id),
        }
        if g.keypoints_visible_coco is not None:
            d['keypoints_visible_coco'] = g.keypoints_visible_coco
        result.append(d)
    return result


# ── Post-processing helper ────────────────────────────────────────────────

def _run_post_processing(
    pipeline: PostProcessingPipeline,
    samples: List[UnifiedSample],
    pred_by_img_id: Dict[int, PoseDataSample],
) -> Tuple[Dict[int, PoseDataSample], dict]:
    """Apply a post-processing pipeline to per-image predictions.

    Iterates *samples* in order (preserving sequence order for temporal
    filters), feeds prediction-only :class:`PoseDataSample`s into the
    pipeline, and returns the post-processed map plus a timing perf dict.

    For an **online** pipeline each :meth:`~PostProcessingPipeline.process`
    call is individually timed; for an **offline** pipeline a single
    :meth:`~PostProcessingPipeline.evaluate` call is timed.

    Args:
        pipeline: Built :class:`PostProcessingPipeline`.
        samples: Ordered unified samples (same order as inference run).
        pred_by_img_id: Mapping ``img_id → PoseDataSample`` from inference.

    Returns:
        Tuple of the post-processed ``img_id → PoseDataSample`` mapping and
        a flat perf dict.
    """
    pipeline.reset()

    postproc_by_img_id: Dict[int, PoseDataSample] = {}

    if pipeline.is_online:
        for sample in samples:
            img_id = sample.img_id
            ds = pred_by_img_id.get(img_id)
            if ds is None:
                continue
            result = pipeline.process(ds)
            if result is not None:
                postproc_by_img_id[img_id] = result

        n = len(pipeline._frame_times)
        perf = {
            'postproc/latency_ms_per_frame': pipeline.per_frame_ms,
            'postproc/total_s': pipeline.total_s,
            'postproc/fps': n / pipeline.total_s if pipeline.total_s > 0 else 0.0,
        }
    else:
        for sample in samples:
            img_id = sample.img_id
            ds = pred_by_img_id.get(img_id)
            if ds is None:
                continue
            pipeline.process(ds)

        results = pipeline.evaluate()
        # Reassociate processed frames with img_ids in insertion order
        img_ids_ordered = [
            s.img_id for s in samples if s.img_id in pred_by_img_id
        ]
        for img_id, processed_ds in zip(img_ids_ordered, results):
            postproc_by_img_id[img_id] = processed_ds

        n = len(results)
        total_s = pipeline.total_s
        perf = {
            'postproc/latency_ms_per_frame': (
                1000.0 * total_s / n if n > 0 else 0.0),
            'postproc/total_s': total_s,
            'postproc/fps': n / total_s if total_s > 0 else 0.0,
        }

    return postproc_by_img_id, perf


# ── Bottomup pipeline ──────────────────────────────────────────────────────

def run_bottomup(
    model,
    samples: List[UnifiedSample],
    sample_stream: 'SampleImageStream',
    val_pipeline: Compose,
    dataset_meta: dict,
    kp_batch_size: int,
    device: str,
    warmup_batches: int = 3,
    log_interval: int = 100,
    num_input_frames: int = 1,
) -> Tuple[Dict[int, PoseDataSample], dict]:
    """Run bottomup inference on unified samples.

    Batches are pulled from ``sample_stream`` rather than by indexing
    ``samples`` directly.  In eager mode (``samples`` already carry pixel
    data) this only pays pipeline/collation cost per batch, not disk I/O.
    In chunked-prefetch mode, images are additionally decoded lazily in
    the background as the stream is consumed, so peak RAM is roughly one
    batch's worth of preprocessed tensors plus a few chunks of raw images,
    instead of the whole dataset at once.  GT and evaluation are handled
    in :func:`main`.

    Args:
        model: Bottomup pose estimator.
        samples: Loaded unified samples (one per image); only its length is
            used here, ``sample_stream`` is the actual source of pixels.
        sample_stream: Stream yielding each sample (with pixel data) once,
            in the same order as ``samples``.
        val_pipeline: Val pipeline from config; first step handles
            pre-loaded images via ``LoadImage``.
        dataset_meta: Model dataset meta for pipeline augmentation.
        kp_batch_size: Images per batch.
        device: Target device string.
        warmup_batches: Leading batches excluded from timing.
        log_interval: Batches between progress prints.
        num_input_frames: Temporal clip window size for multi-frame video
            models (see :func:`_with_clip_windows`). ``1`` (default) keeps
            the single-frame behaviour: ``di['img']`` is a single image.
            When ``> 1``, ``di['img']`` is instead a list of
            ``num_input_frames`` whole-frame images (the model's own
            ``data_preprocessor``/wrapper is responsible for stacking and
            normalising the clip).

    Returns:
        pred_by_img_id: Mapping ``img_id → PoseDataSample`` (prediction only).
        perf: Performance-metric dict.
    """
    n = len(samples)
    n_batches = (n + kp_batch_size - 1) // kp_batch_size

    batch_latencies: List[Tuple[float, int]] = []
    pred_by_img_id: Dict[int, PoseDataSample] = {}

    for i, batch_samples in enumerate(
            _batched_samples(sample_stream, kp_batch_size, num_input_frames)):
        items = []
        img_ids = []
        for _, sample, img, clip in batch_samples:
            if img is None:
                img = _blank_image(sample.ori_shape)
            model_img = (_fill_blank_clip(clip, sample.ori_shape)
                        if clip is not None else img)
            di: dict = {
                'img': model_img,
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
        batch = pseudo_collate(items)
        del items

        # Build a per-iteration batch with GPU inputs.  Do NOT write the GPU
        # tensors back into ``batch``, otherwise every batch's inputs stay
        # resident on the GPU for the whole loop and memory grows until it
        # OOMs.
        gpu_batch = dict(batch)
        gpu_batch['inputs'] = [t.to(device) for t in batch['inputs']]
        del batch

        with torch.no_grad():
            with _CudaTimer() as timer:
                results = model.test_step(gpu_batch)

        if i >= warmup_batches:
            batch_latencies.append((timer.elapsed_s, len(results)))

        for ds, img_id in zip(results, img_ids):
            pred_by_img_id[img_id] = ds

        # Release the GPU input tensors for this batch before moving on.
        del gpu_batch, results

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
        'dataload/stall_s': sample_stream.stall_s,
    }
    return pred_by_img_id, perf


# ── Topdown pipeline ───────────────────────────────────────────────────────

def _mock_detector_producer(
    sample_batches: Iterator[List[FrameItem]],
    n_images: int,
    det_batch_size: int,
    bbox_queue: queue.Queue,
    log_interval: int,
    frame_start_times: dict,
    frame_end_times: dict,
    zero_det_frames: dict,
    det_predictions: dict,
) -> None:
    """Producer: push GT BBoxItems into bbox_queue without running a detector.

    A sample whose image failed to decode (``image is None``, only
    possible in chunked-prefetch mode) is treated as a zero-detection
    frame regardless of its GT, since there is no image to run the
    keypoint model on.
    """
    n_batches = -(-n_images // det_batch_size)

    for batch_idx, batch in enumerate(sample_batches):
        batch_start = batch[0].frame_id
        batch_end = batch[-1].frame_id + 1

        wall_start = time.perf_counter()
        for item in batch:
            frame_start_times[item.frame_id] = wall_start

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == n_batches:
            print(f'  [mock-detector] batch {batch_idx + 1}/{n_batches} '
                  f'| frames processed: {batch_end}')

        wall_after = time.perf_counter()

        for frame_id, sample, img, clip in batch:
            img_id = sample.img_id
            h, w = sample.ori_shape

            if img is None:
                bboxes_xyxy = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros(0, dtype=np.float32)
            else:
                valid = [
                    g for g in sample.gt_instances if is_valid_instance(g)
                ]
                if valid:
                    bboxes_xyxy = np.stack(
                        [g.bbox for g in valid], axis=0)
                    scores = np.ones(len(valid), dtype=np.float32)
                else:
                    bboxes_xyxy = np.zeros((0, 4), dtype=np.float32)
                    scores = np.zeros(0, dtype=np.float32)

            det_predictions[frame_id] = (bboxes_xyxy, scores)
            n_bboxes = 0 if img is None else len(bboxes_xyxy)

            clip_imgs = (
                _fill_blank_clip(clip, sample.ori_shape)
                if clip is not None else None)

            if n_bboxes == 0:
                zero_det_frames[frame_id] = (img_id, (h, w))
                frame_end_times[frame_id] = wall_after
            else:
                for bbox, score in zip(bboxes_xyxy, scores):
                    bbox_queue.put(BBoxItem(
                        frame_id=frame_id,
                        img_id=img_id,
                        img=img,
                        clip_imgs=clip_imgs,
                        bbox_xyxy=bbox,
                        det_score=float(score),
                        n_bboxes_in_frame=n_bboxes,
                        frame_start_wall=wall_start,
                    ))

    bbox_queue.put(_SENTINEL)


def _detector_producer(
    detector,
    sample_batches: Iterator[List[FrameItem]],
    n_images: int,
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
    """Producer: run detector and push BBoxItems into bbox_queue.

    A sample whose image failed to decode (``image is None``, only
    possible in chunked-prefetch mode) is fed to the detector as a blank
    placeholder (to keep batch construction simple) but its result is
    always forced to a zero-detection frame, since there is no real image
    to run the keypoint model on either way.
    """
    n_batches = -(-n_images // det_batch_size)

    for batch_idx, batch in enumerate(sample_batches):
        batch_start = batch[0].frame_id
        batch_end = batch[-1].frame_id + 1

        wall_start = time.perf_counter()
        for item in batch:
            frame_start_times[item.frame_id] = wall_start

        imgs = [
            item.image if item.image is not None
            else _blank_image(item.sample.ori_shape)
            for item in batch
        ]
        with torch.no_grad():
            with _CudaTimer() as timer:
                det_results = inference_det_model(detector, imgs)

        wall_after_det = time.perf_counter()

        if batch_idx >= warmup_batches:
            det_timings.append((timer.elapsed_s, len(batch)))

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == n_batches:
            timed_frames = sum(k for _, k in det_timings)
            timed_time = sum(t for t, _ in det_timings)
            fps_so_far = timed_frames / timed_time if timed_time > 0 else 0.0
            print(f'  [detector] batch {batch_idx + 1}/{n_batches} '
                  f'| frames processed: {batch_end} '
                  f'| running FPS: {fps_so_far:.1f}')

        for (frame_id, sample, orig_img, clip), img, det_result in zip(
                batch, imgs, det_results):
            img_id = sample.img_id
            h, w = sample.ori_shape
            decode_failed = orig_img is None

            if decode_failed:
                bboxes_xyxy = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros(0, dtype=np.float32)
            else:
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
            n_bboxes = 0 if decode_failed else len(bboxes_xyxy)

            clip_imgs = (
                _fill_blank_clip(clip, sample.ori_shape)
                if clip is not None else None)

            if n_bboxes == 0:
                zero_det_frames[frame_id] = (img_id, (h, w))
                frame_end_times[frame_id] = wall_after_det
            else:
                for bbox, score in zip(bboxes_xyxy, scores):
                    bbox_queue.put(BBoxItem(
                        frame_id=frame_id,
                        img_id=img_id,
                        img=img,
                        clip_imgs=clip_imgs,
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
            img_for_model = (
                item.clip_imgs if item.clip_imgs is not None else item.img)
            di = dict(img=img_for_model)
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
    sample_stream: 'SampleImageStream',
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
    num_input_frames: int = 1,
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
        samples: Unified samples (one per image); only its length is used
            here, ``sample_stream`` is the actual source of pixels.
        sample_stream: Stream yielding each sample (with pixel data) once,
            in the same order as ``samples``. Feeds the detector producer;
            GT bboxes for the mock detector are computed per sample as the
            stream is consumed.
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
        num_input_frames: Temporal clip window size for multi-frame video
            models (see :func:`_with_clip_windows`). ``1`` (default) keeps
            the single-frame behaviour: each ``BBoxItem`` fed to the
            keypoint model carries a single crop image. When ``> 1``, it
            instead carries a list of ``num_input_frames`` crops (the same
            bbox applied to every frame in the window around it) for the
            keypoint pipeline/wrapper to stack into a clip tensor.

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

    n_images = len(samples)
    sample_batches = _batched_samples(
        sample_stream, det_batch_size, num_input_frames)

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
            args=(sample_batches, n_images, det_batch_size,
                  bbox_queue, log_interval,
                  frame_start_times, frame_end_times, zero_det_frames,
                  det_predictions),
            daemon=True,
        )
    else:
        producer = threading.Thread(
            target=_detector_producer,
            args=(det_model, sample_batches, n_images, det_batch_size,
                  bbox_thr, nms_thr, det_cat_id, bbox_queue, warmup_batches,
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
        'dataload/stall_s': sample_stream.stall_s,
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


# ── Self-tracking pipeline ─────────────────────────────────────────────────

def _detections_for_frame(
    detector,
    sample: UnifiedSample,
    image: Optional[np.ndarray],
    use_mock_detector: bool,
    bbox_thr: float,
    nms_thr: float,
    det_cat_id: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Detections for a single frame, filtered exactly as in topdown mode.

    Mirrors the per-frame body of :func:`_detector_producer` (real detector)
    and :func:`_mock_detector_producer` (GT bboxes) so that a self-tracking
    model sees the same boxes a plain topdown model would, but synchronously
    and one frame at a time.

    Returns ``(bboxes_xyxy, scores, det_time_s)``.  A frame whose image
    failed to decode yields zero detections, matching the topdown path.
    """
    if image is None:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.float32), 0.0)

    if use_mock_detector:
        valid = [g for g in sample.gt_instances if is_valid_instance(g)]
        if not valid:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros(0, dtype=np.float32), 0.0)
        bboxes_xyxy = np.stack([g.bbox for g in valid], axis=0).astype(np.float32)
        scores = np.ones(len(valid), dtype=np.float32)
        return bboxes_xyxy, scores, 0.0

    with torch.no_grad():
        with _CudaTimer() as timer:
            det_results = inference_det_model(detector, [image])

    pred = det_results[0].pred_instances.cpu().numpy()
    mask = np.logical_and(pred.labels == det_cat_id, pred.scores > bbox_thr)
    bboxes_s = np.concatenate([pred.bboxes[mask], pred.scores[mask, None]],
                              axis=1)
    if bboxes_s.shape[0] > 0:
        keep = nms(bboxes_s, nms_thr)
        return bboxes_s[keep, :4], bboxes_s[keep, 4], timer.elapsed_s
    return bboxes_s[:, :4], np.array([], dtype=np.float32), timer.elapsed_s


def _ensure_track_ids(ds: PoseDataSample) -> PoseDataSample:
    """Guarantee ``pred_instances.track_ids`` exists, defaulting to ``-1``.

    A zero-detection frame produces an empty ``pred_instances`` that never
    passed through the model, and a model may legitimately decline to assign
    an id to a low-confidence instance.  Both are represented as ``-1`` so
    downstream metrics and the saved bundle see a consistently shaped field
    on every frame rather than the field being absent on some.
    """
    pred = getattr(ds, 'pred_instances', None)
    if pred is None:
        return ds
    if 'track_ids' not in pred:
        n = len(pred.get('keypoints', []))
        pred.track_ids = np.full(n, -1, dtype=np.int32)
    else:
        pred.track_ids = np.asarray(pred.track_ids, dtype=np.int32).reshape(-1)
    return ds


def run_tracking(
    model,
    samples: List[UnifiedSample],
    sample_stream: 'SampleImageStream',
    kp_pipeline: Compose,
    dataset_meta: dict,
    device: str,
    *,
    det_model=None,
    is_topdown: bool = False,
    use_mock_detector: bool = False,
    bbox_thr: float = 0.3,
    nms_thr: float = 0.3,
    det_cat_id: int = 0,
    warmup_batches: int = 3,
    log_interval: int = 100,
    num_input_frames: int = 1,
) -> Tuple[Dict[int, PoseDataSample], dict]:
    """Run a model that assigns track IDs itself, strictly frame by frame.

    Neither :func:`run_topdown` nor :func:`run_bottomup` can host such a
    model.  ``run_topdown`` batches crops *across* frames in a
    producer/consumer pair, so a tracker never sees one frame's detections
    as a unit and never sees frames in order; ``run_bottomup`` batches N
    frames into one ``test_step``, which desynchronises any recurrent state
    the model carries between frames.  This path instead:

    * consumes the stream one frame at a time, in dataset order;
    * in topdown mode, runs the detector on that frame inline and hands the
      model **all** of the frame's boxes in a single ``test_step`` call, so
      the model's own association step sees the complete frame;
    * calls ``model.reset_tracking()`` whenever :func:`_clip_seq_key`
      changes, i.e. at every video boundary.

    The model is expected to return one :class:`PoseDataSample` per input
    element with ``pred_instances.track_ids`` already set (``np.int32``,
    one entry per instance).  In topdown mode the per-box samples are merged
    with :func:`_merge_pose_predictions_to_posedatasample`, whose
    ``InstanceData.cat`` carries ``track_ids`` through unchanged.

    Because the batch size is fixed at one frame, the reported FPS is a
    sequential, unbatched figure and is **not** comparable with the batched
    ``run_topdown`` / ``run_bottomup`` numbers for other models.

    Args:
        model: Pose estimator that emits its own track IDs.  Must expose
            ``reset_tracking()``.
        samples: Unified samples (one per image), used for ordering and to
            assemble the per-image results afterwards.
        sample_stream: Stream yielding each sample (with pixels) once.
        kp_pipeline: Inference pipeline from the config, GT-only transforms
            already stripped.
        dataset_meta: Model dataset meta, merged into each pipeline input.
        device: Target device string (bottomup inputs are moved explicitly,
            as in :func:`run_bottomup`).
        det_model: Detector, topdown mode with a real detector only.
        is_topdown: Whether the model consumes person crops.
        use_mock_detector: Feed GT bboxes instead of running a detector.
        bbox_thr: Detector bbox confidence threshold.
        nms_thr: NMS IoU threshold.
        det_cat_id: Person class id in the detector.
        warmup_batches: Leading frames excluded from timing.
        log_interval: Frames between progress prints.
        num_input_frames: Temporal clip window size (see
            :func:`_with_clip_windows`).

    Returns:
        pred_by_img_id: Mapping ``img_id → PoseDataSample`` (prediction only).
        perf: Performance-metric dict.
    """
    if not hasattr(model, 'reset_tracking'):
        raise TypeError(
            f'{type(model).__name__} declares emits_track_ids=True but does '
            'not implement reset_tracking(); the driver needs it to clear '
            'per-sequence tracker state at video boundaries.')

    n_images = len(samples)
    frame_latencies: List[float] = []
    det_timings: List[float] = []
    # Model-only GPU time per frame (the _CudaTimer also synchronises, which
    # is what makes the surrounding wall-clock frame latency meaningful).
    model_timings: List[float] = []
    pred_by_img_id: Dict[int, PoseDataSample] = {}
    det_predictions: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    current_seq = None
    n_done = 0

    for batch in _batched_samples(sample_stream, 1, num_input_frames):
        frame_id, sample, image, clip = batch[0]
        img_id = sample.img_id
        h, w = sample.ori_shape

        seq_key = _clip_seq_key(sample)
        if seq_key != current_seq:
            model.reset_tracking()
            current_seq = seq_key

        clip_imgs = (_fill_blank_clip(clip, sample.ori_shape)
                     if clip is not None else None)
        frame_img = image if image is not None else _blank_image((h, w))

        t_frame_start = time.perf_counter()

        if is_topdown:
            bboxes_xyxy, scores, det_s = _detections_for_frame(
                det_model, sample, image, use_mock_detector,
                bbox_thr, nms_thr, det_cat_id)
            det_predictions[frame_id] = (bboxes_xyxy, scores)

            if len(bboxes_xyxy) == 0:
                # Deliberately does not invoke the model.  Upstream
                # AlphaPose skips its tracker entirely on a zero-detection
                # frame (demo_inference.py `continue`s before track()), so
                # the tracker's frame counter and track ages do not advance
                # either; calling into the model here would diverge from the
                # published behaviour.
                pred_ds = _merge_pose_predictions_to_posedatasample(
                    img_id, [], [], (h, w), img_path=sample.img_path)
            else:
                data_list = []
                for bbox, score in zip(bboxes_xyxy, scores):
                    di = dict(
                        img=clip_imgs if clip_imgs is not None else frame_img)
                    di['bbox'] = bbox[None]
                    di['bbox_score'] = np.array([score], dtype=np.float32)
                    di['img_path'] = sample.img_path
                    di['img_id'] = img_id
                    di['ori_shape'] = sample.ori_shape
                    di.update(dataset_meta)
                    data_list.append(kp_pipeline(di))
                batch_in = pseudo_collate(data_list)

                with torch.no_grad():
                    with _CudaTimer() as timer:
                        results = model.test_step(batch_in)
                model_timings.append(timer.elapsed_s)

                pred_ds = _merge_pose_predictions_to_posedatasample(
                    img_id, list(results), [float(s) for s in scores],
                    (h, w), img_path=sample.img_path)
                del batch_in, results

            if len(bboxes_xyxy) > 0:
                pred_ds.set_metainfo(
                    {'det_bboxes': bboxes_xyxy, 'det_scores': scores})
            det_timings.append(det_s)
        else:
            di = {
                'img': (clip_imgs if clip_imgs is not None else frame_img),
                'img_path': sample.img_path,
                'img_id': img_id,
                'id': [g.id for g in sample.gt_instances],
                'ori_shape': sample.ori_shape,
            }
            di.update(dataset_meta)
            batch_in = pseudo_collate([kp_pipeline(di)])
            batch_in['inputs'] = [t.to(device) for t in batch_in['inputs']]

            with torch.no_grad():
                with _CudaTimer() as timer:
                    results = model.test_step(batch_in)
            model_timings.append(timer.elapsed_s)

            pred_ds = results[0]
            pred_ds.set_metainfo({
                'img_id': img_id,
                'ori_shape': sample.ori_shape,
                'img_path': sample.img_path,
            })
            del batch_in, results

        if n_done >= warmup_batches:
            frame_latencies.append(time.perf_counter() - t_frame_start)

        pred_by_img_id[img_id] = _ensure_track_ids(pred_ds)
        n_done += 1

        if n_done % log_interval == 0 or n_done == n_images:
            elapsed = sum(frame_latencies)
            fps_so_far = len(frame_latencies) / elapsed if elapsed > 0 else 0.0
            print(f'  [tracking] frame {n_done}/{n_images} '
                  f'| running FPS: {fps_so_far:.1f}')

    total_s = sum(frame_latencies)
    perf = {
        'e2e/fps': len(frame_latencies) / total_s if total_s > 0 else 0.0,
        'e2e/latency_ms_per_frame': (
            1000.0 * total_s / len(frame_latencies) if frame_latencies else 0.0
        ),
        'dataload/stall_s': sample_stream.stall_s,
    }
    timed_det = [t for t in det_timings if t > 0]
    if timed_det:
        perf['detector/fps'] = len(timed_det) / sum(timed_det)
        perf['detector/latency_ms_per_frame'] = (
            1000.0 * sum(timed_det) / len(timed_det))
    if model_timings:
        model_total = sum(model_timings)
        perf['keypoint/fps'] = (
            len(model_timings) / model_total if model_total > 0 else 0.0)
        perf['keypoint/latency_ms_per_frame'] = (
            1000.0 * model_total / len(model_timings))

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


def _save_out(
    quality: dict,
    perf: dict,
    mode: str,
    args,
    pp_quality: Optional[dict] = None,
    n_eval: Optional[int] = None,
    n_images: Optional[int] = None,
) -> None:
    post_config = getattr(args, 'post_config', None)
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
            'post_config': (
                osp.abspath(post_config) if post_config else None),
        },
        'test_dataset': args.test_dataset,
        'eval_good_frames_only': bool(
            getattr(args, 'eval_good_frames_only', False)),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }
    if n_eval is not None:
        payload['num_eval_frames'] = int(n_eval)
    if n_images is not None:
        payload['num_frames'] = int(n_images)
    if pp_quality is not None:
        payload['post_processed_quality'] = pp_quality
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


def _postproc_out_dir(base_dir: str, postproc_name: str) -> str:
    """Build the post-processed output dir next to the run folder.

    ``base_dir`` is the (raw) prediction bundle dir, e.g.
    ``benchmark/predictions/20260715_emdb_e2e/YOLO-Pose-tiny``. The
    post-processed bundle is saved as a sibling of the run folder
    (``20260715_emdb_e2e``) with ``postproc_name`` appended to its name,
    e.g. ``benchmark/predictions/20260715_emdb_e2e_smoothnetw8/YOLO-Pose-tiny``.
    """
    base_dir_abs = osp.abspath(base_dir)
    model_label = osp.basename(base_dir_abs)
    run_dir = osp.dirname(base_dir_abs)
    run_name = osp.basename(run_dir)
    predictions_dir = osp.dirname(run_dir)
    return osp.join(predictions_dir, f'{run_name}_{postproc_name}',
                    model_label)


def _filter_pred_for_saving(
    ds: PoseDataSample,
    score_thr: float,
) -> PoseDataSample:
    """Return a copy of *ds* with low-confidence predictions removed.

    Only ``pred_instances`` is filtered; all metainfo and GT fields are
    preserved unchanged.  The original *ds* is not modified, so COCO
    evaluation that already consumed it is unaffected.

    Filtering is based on ``bbox_scores`` when present, otherwise
    ``keypoint_scores`` mean is used as a fallback.
    """
    if score_thr <= 0:
        return ds

    instances = ds.pred_instances
    if instances is None or len(instances) == 0:
        return ds

    if hasattr(instances, 'bbox_scores') and instances.bbox_scores is not None:
        scores = np.asarray(instances.bbox_scores)
    elif (hasattr(instances, 'keypoint_scores')
          and instances.keypoint_scores is not None):
        scores = np.asarray(instances.keypoint_scores).mean(axis=-1)
    else:
        return ds

    keep = scores >= score_thr
    if keep.all():
        return ds

    filtered_ds = ds.new()
    filtered_ds.set_metainfo(ds.metainfo)
    filtered_ds.pred_instances = instances[keep]
    if hasattr(ds, 'gt_instances'):
        filtered_ds.gt_instances = ds.gt_instances
    return filtered_ds


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
    n_eval: Optional[int] = None,
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
        'save_score_thr': args.save_score_thr,
        'eval_good_frames_only': bool(
            getattr(args, 'eval_good_frames_only', False)),
        'badcase_defaults': {
            'metric_key': 'mean_oks',
            'metric_type': 'accuracy',
            'thr': 0.5,
        },
        'num_frames': len(frame_records),
        'num_eval_frames': (
            int(n_eval) if n_eval is not None else len(frame_records)),
    }
    out_dir = _prediction_out_dir(args, run_date, is_topdown)
    save_prediction_bundle(manifest, frame_records, out_dir)
    print(f'Predictions saved to {osp.abspath(out_dir)}')


def _save_predictions_postproc(
    args,
    data_root: str,
    mode: str,
    is_topdown: bool,
    quality: dict,
    perf: dict,
    frame_records: List[dict],
    dataset_meta: dict,
    run_date: str,
    post_config: Optional[str] = None,
    n_eval: Optional[int] = None,
) -> None:
    """Save the post-processed prediction bundle next to the run folder."""
    base_dir = _prediction_out_dir(args, run_date, is_topdown)
    out_dir = _postproc_out_dir(base_dir, args.postproc_name)

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
        'post_config': osp.abspath(post_config) if post_config else None,
        'quality': quality,
        'perf': perf,
        'data_root': data_root,
        'dataset_meta': sanitize_dataset_meta(dataset_meta),
        'save_score_thr': args.save_score_thr,
        'eval_good_frames_only': bool(
            getattr(args, 'eval_good_frames_only', False)),
        'badcase_defaults': {
            'metric_key': 'mean_oks',
            'metric_type': 'accuracy',
            'thr': 0.5,
        },
        'num_frames': len(frame_records),
        'num_eval_frames': (
            int(n_eval) if n_eval is not None else len(frame_records)),
    }
    save_prediction_bundle(manifest, frame_records, out_dir)
    print(f'Post-processed predictions saved to {osp.abspath(out_dir)}')


def _append_to_results_file(
    quality: dict,
    perf: dict,
    args,
    pp_quality: Optional[dict] = None,
) -> None:
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
        'post_processed': False,
        'post_config': None,
        'metrics': metrics,
    }
    entries = data.setdefault(args.model_name, {}).setdefault(
        args.model_variant, [])
    entries.append(entry)

    # Append a second entry for post-processed results in the same call when
    # both raw and pp quality are available (avoids a second file read/write).
    post_config = getattr(args, 'post_config', None)
    if pp_quality is not None:
        pp_metrics = {**pp_quality, **{f'perf/{k}': v for k, v in perf.items()}}
        pp_entry = {
            'timestamp': entry['timestamp'],
            'config': entry['config'],
            'checkpoint': entry['checkpoint'],
            'test_dataset': entry['test_dataset'],
            'post_processed': True,
            'post_config': osp.abspath(post_config) if post_config else None,
            'metrics': pp_metrics,
        }
        entries.append(pp_entry)

    os.makedirs(osp.dirname(osp.abspath(results_file)), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)
    suffix = ' + post-processed' if pp_quality is not None else ''
    print(f'Results appended to {results_file} '
          f'({args.model_name} / {args.model_variant}){suffix}')


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
    p.add_argument(
        '--num-input-frames', type=int, default=None,
        help='Temporal clip window size (number of consecutive frames) fed '
             'to the pose model per prediction, for multi-frame video '
             'models (e.g. Poseidon, TAR-ViTPose: 5; PAVE-Net: 3). Must be '
             'odd. Defaults to the pose config\'s `num_input_frames` field '
             '(itself defaulting to 1, i.e. single-frame) when not set.')
    p.add_argument(
        '--prefetch-chunk-size', type=int, default=0,
        help='If > 0, load images lazily in chunks of this many frames '
             'instead of decoding the whole dataset up front. Use this for '
             'large datasets (e.g. PoseTrack21) that do not fit in memory '
             'all at once. Default: 0 (eager, load everything up front).')
    p.add_argument(
        '--prefetch-queue-chunks', type=int, default=2,
        help='Number of decoded chunks buffered ahead of the consumer when '
             '--prefetch-chunk-size > 0 (default: 2).')
    p.add_argument(
        '--prefetch-workers', type=int, default=4,
        help='Parallel image-decode threads per chunk when '
             '--prefetch-chunk-size > 0 (default: 4).')
    p.add_argument(
        '--include-bad-frames', action='store_true',
        help='EMDB/3DPW/PoseTrack21: also load frames normally excluded for '
             'lacking reliable GT (e.g. EMDB invalid_idxs, 3DPW frames with '
             'no valid actor, PoseTrack21 unlabeled frames), so '
             'post-processing sees a temporally continuous frame sequence. '
             'These frames have no GT and may produce false-positive '
             'detections that affect detection/pose metrics; temporal '
             'metrics (MPJVE/MPJAE) still exclude them via frame_id-gap '
             'masking. Combine with --eval-good-frames-only to keep '
             'inference continuous while excluding bad frames from metric '
             'computation. No effect on other datasets.')
    p.add_argument(
        '--eval-good-frames-only', action='store_true',
        help='Exclude frames without reliable GT (EMDB/3DPW/PoseTrack21 '
             '``good_frame=False``) from all metric computation. Inference, '
             'post-processing, and prediction-bundle export still run over '
             'every loaded frame. Intended for use with '
             '--include-bad-frames. Warns (no-op) on datasets without a '
             'good_frame concept.')
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
        '--save-score-thr', type=float, default=0.0,
        help='Minimum bbox_score for a prediction to be included in the '
             'saved prediction bundle (does NOT affect COCO eval, '
             'default: 0.0)')
    p.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default={},
        help='Override config options, e.g. model.backbone.depth=18')
    p.add_argument(
        '--test-dataset', default='coco',
        choices=list(BENCHMARK_TEST_DATASETS),
        help='Override test set and metric (default: use config as-is)')
    p.add_argument(
        '--post-config', default=None,
        help='Path to a post-processing pipeline config '
             '(e.g. configs/post_processing/oks_track_one_euro.py). '
             'When provided both raw and post-processed outputs are '
             'evaluated and saved separately.')
    p.add_argument(
        '--postproc-name', default=None,
        help='Name of this post-processing run (e.g. "smoothnetw8"). '
             'Used to save the post-processed bundle next to the run '
             'folder, with "_<postproc-name>" appended to its name (e.g. '
             'benchmark/predictions/20260715_emdb_e2e/YOLO-Pose-tiny -> '
             'benchmark/predictions/20260715_emdb_e2e_smoothnetw8/'
             'YOLO-Pose-tiny). Required when --post-config and '
             '--model-name are both specified.')
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

    num_input_frames = args.num_input_frames
    if num_input_frames is None:
        num_input_frames = pose_cfg.get('num_input_frames', 1)
    if num_input_frames > 1:
        print(f'\nMulti-frame video model: num_input_frames={num_input_frames}')
        if args.prefetch_chunk_size and args.prefetch_chunk_size < num_input_frames:
            print(
                f'Warning: --prefetch-chunk-size ({args.prefetch_chunk_size}) '
                f'is smaller than --num-input-frames ({num_input_frames}); '
                f'clip windows near a chunk boundary may see frames whose '
                f'pixels were already released. Set --prefetch-chunk-size '
                f'>= --num-input-frames (or 0 for eager loading).')

    # ── Optional post-processing pipeline ────────────────────────────────
    post_pipeline: Optional[PostProcessingPipeline] = None
    if getattr(args, 'post_config', None):
        print(f'\nBuilding post-processing pipeline from: {args.post_config}')
        post_pipeline = build_post_processor(args.post_config)
        if post_pipeline.needs_images:
            raise ValueError(
                f'The post-processing config {args.post_config!r} declares '
                f'needs_images=True, but --post-config in benchmark_e2e '
                f'does not supply frame images (chunked prefetch releases '
                f'the pixels before post-processing runs). This includes '
                f'hybrid online-image + offline pipelines. Save the '
                f'predictions first and run tools/postprocess_predictions.py '
                f'on the bundle instead.')
        mode_label = 'online' if post_pipeline.is_online else 'offline'
        print(f'  Pipeline mode : {mode_label}')
        print(f'  Filters       : {[type(f).__name__ for f in post_pipeline.filters]}')

    use_real_detector = bool(args.det_config and args.det_checkpoint)
    is_topdown = args.mock_detector or use_real_detector
    model_type = pose_cfg.model.get('type', '')

    # Models that assign track IDs during inference (AlphaPose's re-ID
    # tracker, OpenPifPaf's spatio-temporal association) declare this at the
    # config top level, alongside num_input_frames.  It routes the run
    # through run_tracking(); every other config is untouched.
    emits_track_ids = bool(pose_cfg.get('emits_track_ids', False))

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
    if (getattr(args, 'post_config', None) and args.model_name
            and not getattr(args, 'postproc_name', None)):
        raise ValueError(
            '--postproc-name is required when both --post-config and '
            '--model-name are specified (used to name the post-processed '
            'output run folder).')
    if emits_track_ids and getattr(args, 'post_config', None):
        raise ValueError(
            f'{args.pose_config} declares emits_track_ids=True, so the model '
            f'assigns track IDs during inference. Adding a post-processing '
            f'tracker via --post-config would overwrite them, and the run '
            f'would report the post-processing tracker\'s association rather '
            f'than the model\'s. Drop --post-config to benchmark the model\'s '
            f'own tracking, or benchmark a non-tracking pose config with '
            f'--post-config to compare against it.')

    # ── Unified data loading ───────────────────────────────────────────────
    print(f'\nLoading dataset: {args.test_dataset}')
    if args.prefetch_chunk_size > 0:
        samples = build_unified_samples(
            args.test_dataset, args.num_frames,
            include_bad_frames=args.include_bad_frames)
        prefetch_scale = BENCHMARK_TEST_DATASETS[args.test_dataset].prefetch_scale
        print(f'  Chunked prefetch enabled: {len(samples)} images, '
              f'chunk_size={args.prefetch_chunk_size}, '
              f'queue_chunks={args.prefetch_queue_chunks}, '
              f'workers={args.prefetch_workers}')
        sample_stream = SampleImageStream(
            samples,
            prefetch_scale=prefetch_scale,
            chunk_size=args.prefetch_chunk_size,
            queue_chunks=args.prefetch_queue_chunks,
            workers=args.prefetch_workers,
        )
    else:
        samples = load_unified_samples(
            args.test_dataset, args.num_frames,
            include_bad_frames=args.include_bad_frames)
        sample_stream = SampleImageStream(samples)
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

    # ── Self-tracking mode ─────────────────────────────────────────────────
    if emits_track_ids:
        sub_mode = 'topdown' if is_topdown else 'bottomup'
        print(f'\nMode: tracking ({sub_mode})  |  model assigns its own '
              f'track IDs')
        print('  Frames are processed strictly sequentially, one per '
              'test_step; FPS is NOT comparable with batched runs.')

        detector = None
        if is_topdown and not args.mock_detector:
            if not HAS_MMDET:
                raise ImportError(
                    'mmdet is required for topdown mode. '
                    'Install it with: pip install mmdet')
            print(f'  Detector : {args.det_config}')
            detector = init_det_model(
                args.det_config, args.det_checkpoint, device=args.device)
        elif args.mock_detector:
            print('  Detector : MOCK (GT bboxes from dataset)')
        print(f'  Model    : {args.pose_config}\n')

        model = init_model(
            args.pose_config, args.pose_checkpoint, device=args.device)
        dataset_meta = model.dataset_meta
        evaluator = build_evaluator(
            pose_cfg, dataset_meta, args.test_dataset,
            include_tracking_metrics=True, full_tracking_metrics=True)

        t_infer_start = time.perf_counter()
        pred_by_img_id, perf = run_tracking(
            model=model,
            samples=samples,
            sample_stream=sample_stream,
            kp_pipeline=Compose(pipeline_cfg),
            dataset_meta=dataset_meta,
            device=args.device,
            det_model=detector,
            is_topdown=is_topdown,
            use_mock_detector=args.mock_detector,
            bbox_thr=args.bbox_thr,
            nms_thr=args.nms_thr,
            det_cat_id=args.det_cat_id,
            warmup_batches=args.warmup_batches,
            log_interval=args.log_interval,
            num_input_frames=num_input_frames,
        )
        infer_wall_s = time.perf_counter() - t_infer_start

        mode = f'tracking ({sub_mode})'

    # ── Topdown mode ───────────────────────────────────────────────────────
    elif is_topdown:
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
        evaluator = build_evaluator(pose_cfg, dataset_meta, args.test_dataset)

        t_infer_start = time.perf_counter()
        pred_by_img_id, perf = run_topdown(
            det_model=detector,
            kp_model=kp_model,
            samples=samples,
            sample_stream=sample_stream,
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
            num_input_frames=num_input_frames,
        )
        infer_wall_s = time.perf_counter() - t_infer_start

        mode_suffix = 'mock-detector' if args.mock_detector else args.queue_strategy
        mode = f'topdown (strategy={mode_suffix})'

    # ── Bottomup mode ──────────────────────────────────────────────────────
    else:
        print(f'\nMode: bottomup')
        print(f'  Model : {args.pose_config}\n')

        model = init_model(
            args.pose_config, args.pose_checkpoint, device=args.device)
        dataset_meta = model.dataset_meta
        evaluator = build_evaluator(pose_cfg, dataset_meta, args.test_dataset)

        val_pipeline = Compose(pipeline_cfg)

        t_infer_start = time.perf_counter()
        pred_by_img_id, perf = run_bottomup(
            model=model,
            samples=samples,
            sample_stream=sample_stream,
            val_pipeline=val_pipeline,
            dataset_meta=dataset_meta,
            kp_batch_size=args.kp_batch_size,
            device=args.device,
            warmup_batches=args.warmup_batches,
            log_interval=args.log_interval,
            num_input_frames=num_input_frames,
        )
        infer_wall_s = time.perf_counter() - t_infer_start

        mode = 'bottomup'

    # ── Warn if data loading meaningfully stalled the pipeline ──────────────
    stall_s = perf.get('dataload/stall_s', 0.0)
    if infer_wall_s > 0 and stall_s / infer_wall_s > 0.05:
        print(
            f'\nWarning: data loading stalled the pipeline for '
            f'{stall_s:.1f}s ({100.0 * stall_s / infer_wall_s:.0f}% of the '
            f'{infer_wall_s:.1f}s inference wall time). Measured FPS may be '
            f'I/O-bound rather than model-bound; consider raising '
            f'--prefetch-chunk-size, --prefetch-workers, or '
            f'--prefetch-queue-chunks.')

    # ── Post-processing (both modes) ────────────────────────────────────────
    postproc_by_img_id: Optional[Dict[int, PoseDataSample]] = None
    if post_pipeline is not None:
        postproc_by_img_id, pp_perf = _run_post_processing(
            post_pipeline, samples, pred_by_img_id)
        perf.update(pp_perf)

    # ── Unified evaluation (both modes) ────────────────────────────────────
    frame_records_pp: Optional[List[dict]] = (
        [] if (args.model_name and postproc_by_img_id is not None) else None
    )
    pp_evaluator = (
        build_evaluator(pose_cfg, dataset_meta, args.test_dataset,
                        include_tracking_metrics=True)
        if postproc_by_img_id is not None else None
    )

    eval_good_only = bool(getattr(args, 'eval_good_frames_only', False))
    if eval_good_only and args.test_dataset not in BAD_FRAME_DATASETS:
        print(
            f'Warning: --eval-good-frames-only has no effect on '
            f'{args.test_dataset!r} (no good_frame concept); evaluating '
            f'every loaded frame.')
        eval_good_only = False

    n_eval = 0
    for fid, sample in enumerate(samples):
        img_id = sample.img_id
        gt_dicts = _pose_dicts_from_unifiedsample(sample)
        include_in_eval = (not eval_good_only) or sample.good_frame
        if include_in_eval:
            n_eval += 1

        # ── Regular predictions ────────────────────────────────────────
        pred_ds = pred_by_img_id.get(img_id)
        if pred_ds is not None:
            ds = _attach_gt_to_posedatasample(pred_ds, sample)
            if include_in_eval:
                evaluator.process(data_samples=[ds], data_batch=None)

            if frame_records is not None:
                pred_ds_save = _filter_pred_for_saving(ds, args.save_score_thr)
                frame_records.append(build_frame_record(
                    img_id=int(img_id),
                    frame_id=fid,
                    img_path=sample.img_path,
                    data_root=data_root,
                    pred_ds=pred_ds_save,
                    gt_instances=gt_dicts,
                    dataset_meta=dataset_meta,
                    good_frame=sample.good_frame,
                ))

        # ── Post-processed predictions ─────────────────────────────────
        if postproc_by_img_id is not None:
            pp_ds = postproc_by_img_id.get(img_id)
            if pp_ds is not None:
                pp_ds_gt = _attach_gt_to_posedatasample(pp_ds, sample)
                if include_in_eval:
                    pp_evaluator.process(
                        data_samples=[pp_ds_gt], data_batch=None)

                if frame_records_pp is not None:
                    pp_ds_save = _filter_pred_for_saving(
                        pp_ds_gt, args.save_score_thr)
                    frame_records_pp.append(build_frame_record(
                        img_id=int(img_id),
                        frame_id=fid,
                        img_path=sample.img_path,
                        data_root=data_root,
                        pred_ds=pp_ds_save,
                        gt_instances=gt_dicts,
                        dataset_meta=dataset_meta,
                        good_frame=sample.good_frame,
                    ))

    n_bad = n_images - n_eval
    if eval_good_only:
        print(f'\nEvaluating on {n_eval}/{n_images} frames '
              f'({n_bad} bad frames excluded)')
    else:
        n_eval = n_images

    quality = evaluator.evaluate(n_eval)
    pp_quality: Optional[dict] = None
    if pp_evaluator is not None:
        pp_quality = pp_evaluator.evaluate(n_eval)

    # Optional detection AP/AR (topdown only — reads from pred_ds.metainfo).
    # Built here (after inference) rather than up front, since in chunked
    # prefetch mode GT bboxes/areas/ori_shape on `samples` are only final
    # once every image has been decoded.
    det_samples = (
        [s for s in samples if s.good_frame] if eval_good_only else samples)
    det_evaluator = (build_det_evaluator(det_samples)
                      if getattr(args, 'det_metrics', False) else None)
    if det_evaluator is not None:
        for fid, sample in enumerate(samples):
            if eval_good_only and not sample.good_frame:
                continue
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
        det_quality = det_evaluator.evaluate(n_eval)
        quality = {**quality, **det_quality}

    # ── Output ─────────────────────────────────────────────────────────────
    _print_results(quality, perf, mode)
    if pp_quality is not None:
        _print_results(pp_quality, perf, f'{mode} [post-processed]')

    if args.out:
        _save_out(
            quality, perf, mode, args, pp_quality=pp_quality,
            n_eval=n_eval, n_images=n_images)

    if args.results_file:
        assert args.model_name and args.model_variant, (
            '--model-name and --model-variant are required when '
            '--results-file is specified')
        _append_to_results_file(quality, perf, args, pp_quality=pp_quality)

    if args.model_name and frame_records is not None:
        _save_predictions(
            args, data_root, mode, is_topdown, quality, perf,
            frame_records, dataset_meta, run_date, n_eval=n_eval)

    # ── Post-processed bundle ───────────────────────────────────────────
    if args.model_name and frame_records_pp is not None and pp_quality is not None:
        _save_predictions_postproc(
            args, data_root, mode, is_topdown, pp_quality, perf,
            frame_records_pp, dataset_meta, run_date,
            post_config=getattr(args, 'post_config', None),
            n_eval=n_eval)


if __name__ == '__main__':
    main()
