# Copyright (c) OpenMMLab. All rights reserved.
"""Standalone post-processor for saved benchmark prediction bundles.

Loads a ``manifest.json`` + ``frames.json`` bundle produced by
``tools/benchmark_e2e.py``, runs a configurable post-processing pipeline,
evaluates chosen metrics on the post-processed predictions, and saves a new
sibling bundle to ``<pred_dir>__postproc`` (or ``--out-dir``).

Timing is measured the same way as in ``tools/benchmark_e2e.py``:

* **Online pipeline** – each ``process()`` call is timed; a per-frame average
  and FPS are reported.
* **Offline pipeline** – a single ``evaluate()`` call is timed; per-frame
  average is derived from total / num_frames.

Usage::

    python tools/postprocess_predictions.py \\
        benchmark/predictions/20260615_emdb-mini_e2e/YOLO-Pose-tiny \\
        --post-config configs/post_processing/oks_track_one_euro.py \\
        --metrics CocoMetric MPJVE MPJAE
"""

from mmpose.compat.transformers_v5 import install_transformers_v5_shims

install_transformers_v5_shims()

import argparse
import json
import os
import os.path as osp
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from mmengine.evaluator import Evaluator
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

# Register all mmpose modules (metrics, etc.)
import mmpose.datasets   # noqa: F401
import mmpose.evaluation  # noqa: F401
import mmpose.models      # noqa: F401
from mmengine.registry import METRICS
from mmpose.evaluation.functional.frame_metrics import (
    build_frame_record,
    save_prediction_bundle,
    sanitize_dataset_meta,
)
from mmpose.postprocessing import PostProcessingPipeline, build_post_processor
from mmpose.structures import PoseDataSample

_LOGGER_INIT = False


def _get_logger():
    global _LOGGER_INIT
    if not _LOGGER_INIT:
        MMLogger.get_current_instance()
        _LOGGER_INIT = True
    return MMLogger.get_current_instance()


# ── Bundle loading ─────────────────────────────────────────────────────────

def _sequence_key_from_path(img_path: str) -> str:
    """Derive sequence key from img_path (matches mmpose/postprocessing/base.py)."""
    if not img_path:
        return ''
    parts = img_path.replace('\\', '/').split('/')
    dirs = parts[:-1]
    if dirs and dirs[-1] == 'images':
        dirs = dirs[:-1]
    return '/'.join(dirs) if dirs else ''


def _flatten_bbox(bbox_raw) -> Optional[List[float]]:
    """Normalise the bbox field (may be [[x1,y1,x2,y2]] or [x1,y1,x2,y2])."""
    if bbox_raw is None:
        return None
    b = np.asarray(bbox_raw, dtype=np.float32).reshape(-1)
    return b[:4].tolist()


def _make_pred_ds(frame: dict, dataset_meta: dict) -> PoseDataSample:
    """Reconstruct a prediction-only :class:`PoseDataSample` from a frame record."""
    ds = PoseDataSample()
    ori_shape = tuple(frame.get('ori_shape', [0, 0]))
    img_path = frame.get('img_path', '')
    img_id = frame.get('img_id', 0)

    ds.set_metainfo({
        'img_id': img_id,
        'img_path': img_path,
        'ori_shape': ori_shape,
        'id': [img_id],
        'category_id': 1,
    })

    insts = frame.get('predictions', {}).get('instances', [])
    n = len(insts)
    num_kpts = dataset_meta.get('num_keypoints', 17)

    pred = InstanceData()
    if n == 0:
        pred.keypoints = np.zeros((0, num_kpts, 2), dtype=np.float32)
        pred.keypoint_scores = np.zeros((0, num_kpts), dtype=np.float32)
        pred.bboxes = np.zeros((0, 4), dtype=np.float32)
        pred.bbox_scores = np.zeros(0, dtype=np.float32)
    else:
        kpts = np.array([inst['keypoints'] for inst in insts],
                        dtype=np.float32)  # (N, K, 2)
        scores = np.array([inst['keypoint_scores'] for inst in insts],
                          dtype=np.float32)  # (N, K)
        bboxes = []
        bbox_scores = []
        for inst in insts:
            b = _flatten_bbox(inst.get('bbox'))
            bboxes.append(b if b is not None else [0., 0., 1., 1.])
            bbox_scores.append(float(inst.get('bbox_score', 1.0)))

        pred.keypoints = kpts
        pred.keypoint_scores = scores
        pred.bboxes = np.array(bboxes, dtype=np.float32)
        pred.bbox_scores = np.array(bbox_scores, dtype=np.float32)

    ds.pred_instances = pred
    return ds


def _make_gt_instances(
    frame: dict,
    seq_track_id_map: Dict[str, int],
    num_kpts: int,
    img_path: str,
) -> Tuple[InstanceData, List[dict]]:
    """Build GT InstanceData and raw GT dicts for evaluation."""
    gt_list = frame.get('ground_truth', {}).get('instances', [])
    n = len(gt_list)

    gt = InstanceData()
    if n == 0:
        gt.keypoints = np.zeros((0, num_kpts, 2), dtype=np.float32)
        gt.keypoints_visible = np.zeros((0, num_kpts), dtype=np.float32)
        gt.bboxes = np.zeros((0, 4), dtype=np.float32)
        gt.orig_areas = np.zeros(0, dtype=np.float32)
        gt.iscrowd = np.zeros(0, dtype=np.int32)
        gt.track_ids = np.zeros(0, dtype=np.int32)
        return gt, gt_list

    kpts = np.array([
        np.asarray(g['keypoints'], dtype=np.float32).reshape(-1, 2)[:num_kpts]
        for g in gt_list
    ], dtype=np.float32)
    vis = np.array([
        np.asarray(g['keypoints_visible'], dtype=np.float32).reshape(-1)[:num_kpts]
        for g in gt_list
    ], dtype=np.float32)
    bboxes = np.array([
        _flatten_bbox(g.get('bbox')) or [0., 0., 1., 1.]
        for g in gt_list
    ], dtype=np.float32)
    areas = np.array([float(g.get('orig_area', 1.0)) for g in gt_list],
                     dtype=np.float32)
    iscrowd = np.array([int(g.get('iscrowd', 0)) for g in gt_list],
                       dtype=np.int32)

    # Resolve track_ids: prefer stored value, else derive from sequence key
    seq_key = _sequence_key_from_path(img_path)
    track_ids = []
    for g in gt_list:
        if 'track_id' in g:
            track_ids.append(int(g['track_id']))
        else:
            # Assign a unique stable id per sequence key
            if seq_key not in seq_track_id_map:
                next_id = len(seq_track_id_map) + 1
                seq_track_id_map[seq_key] = next_id
            track_ids.append(seq_track_id_map[seq_key])

    gt.keypoints = kpts
    gt.keypoints_visible = vis
    gt.bboxes = bboxes
    gt.orig_areas = areas
    gt.iscrowd = iscrowd
    gt.track_ids = np.array(track_ids, dtype=np.int32)

    # keypoints_visible_coco
    kv_coco_list = [g.get('keypoints_visible_coco') for g in gt_list]
    if any(v is not None for v in kv_coco_list):
        gt.keypoints_visible_coco = np.array([
            np.asarray(v, dtype=np.float32).reshape(-1)[:num_kpts]
            if v is not None
            else vis[i]
            for i, v in enumerate(kv_coco_list)
        ], dtype=np.float32)

    return gt, gt_list


def load_bundle(
    pred_dir: str,
) -> Tuple[dict, List[dict]]:
    """Load manifest and frames from a prediction bundle directory."""
    manifest_path = osp.join(pred_dir, 'manifest.json')
    frames_path = osp.join(pred_dir, 'frames.json')
    if not osp.isfile(manifest_path):
        raise FileNotFoundError(f'manifest.json not found in {pred_dir}')
    if not osp.isfile(frames_path):
        raise FileNotFoundError(f'frames.json not found in {pred_dir}')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    with open(frames_path, 'r') as f:
        frames = json.load(f)
    return manifest, frames


# ── Evaluation builder ─────────────────────────────────────────────────────

def _build_metric(metric_type: str, dataset_meta: dict) -> object:
    """Build a single metric from a type string."""
    cfg: dict = {'gt_from_samples': True}
    if metric_type == 'CocoMetric':
        cfg['type'] = 'CocoMetric'
    elif metric_type in ('MPJVE', 'MPJAE'):
        cfg = {'type': metric_type}
    else:
        cfg['type'] = metric_type

    m = METRICS.build(cfg)
    m.dataset_meta = dataset_meta
    return m


def build_evaluator_from_types(
    metric_types: List[str],
    dataset_meta: dict,
) -> Evaluator:
    metrics = [_build_metric(t, dataset_meta) for t in metric_types]
    return Evaluator(metrics)


# ── Attach GT to ds ───────────────────────────────────────────────────────

def _attach_gt(
    ds: PoseDataSample,
    gt: InstanceData,
    frame: dict,
    dataset_meta: dict,
) -> PoseDataSample:
    """Attach GT to a prediction PoseDataSample for evaluation."""
    ds.gt_instances = gt

    gt_list = frame.get('ground_truth', {}).get('instances', [])
    gt_ids = [int(g.get('gt_ann_id', frame['img_id'])) for g in gt_list]
    if not gt_ids:
        gt_ids = [frame['img_id']]
    ds.set_metainfo({'id': gt_ids})
    return ds


# ── Reconstruction helper ─────────────────────────────────────────────────

def reconstruct_pose_data_samples(
    frames: List[dict],
    dataset_meta: dict,
) -> Tuple[List[PoseDataSample], List[PoseDataSample]]:
    """Rebuild prediction-only and full (pred+GT) PoseDataSamples.

    Returns:
        pred_samples: Prediction-only samples for feeding to the pipeline.
        full_samples: Same samples with GT attached, used only for evaluation.
    """
    num_kpts = dataset_meta.get('num_keypoints', 17)
    seq_track_id_map: Dict[str, int] = {}

    pred_samples: List[PoseDataSample] = []
    full_samples: List[PoseDataSample] = []

    for frame in frames:
        img_path = frame.get('img_path', '')
        pred_ds = _make_pred_ds(frame, dataset_meta)
        gt, gt_list = _make_gt_instances(
            frame, seq_track_id_map, num_kpts, img_path)

        full_ds = _make_pred_ds(frame, dataset_meta)
        _attach_gt(full_ds, gt, frame, dataset_meta)

        pred_samples.append(pred_ds)
        full_samples.append(full_ds)

    return pred_samples, full_samples


# ── Main ──────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Apply a post-processing pipeline to a saved prediction '
                    'bundle and evaluate the results.')
    p.add_argument('pred_dir',
                   help='Directory containing manifest.json and frames.json')
    p.add_argument('--post-config', required=True,
                   help='Path to post-processing pipeline config '
                        '(e.g. configs/post_processing/oks_track_one_euro.py)')
    p.add_argument(
        '--metrics', nargs='+',
        default=['CocoMetric', 'MPJVE', 'MPJAE'],
        help='Metric types to evaluate (default: CocoMetric MPJVE MPJAE)')
    p.add_argument(
        '--out-dir', default=None,
        help='Output directory for the post-processed bundle '
             '(default: <pred_dir>__postproc)')
    p.add_argument(
        '--num-frames', type=int, default=None,
        help='Limit to the first N frames (for quick tests)')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _get_logger()
    init_default_scope('mmpose')

    # ── Load bundle ────────────────────────────────────────────────────────
    print(f'\nLoading prediction bundle from: {args.pred_dir}')
    manifest, frames = load_bundle(args.pred_dir)
    dataset_meta: dict = dict(manifest.get('dataset_meta', {}))
    # JSON serialises numpy arrays as lists; restore numeric arrays so
    # metrics that pass them directly to numpy ops (e.g. oks_iou) work.
    for _key in ('sigmas', 'keypoint_colors', 'skeleton_link_colors',
                 'dataset_keypoint_weights', 'flip_indices', 'flip_pairs'):
        if _key in dataset_meta and isinstance(dataset_meta[_key], list):
            try:
                dataset_meta[_key] = np.array(dataset_meta[_key],
                                              dtype=np.float32)
            except (ValueError, TypeError):
                pass
    data_root: str = manifest.get('data_root', '')
    test_dataset: str = manifest.get('test_dataset', 'coco')

    if args.num_frames is not None:
        frames = frames[:args.num_frames]

    n_frames = len(frames)
    print(f'  Frames loaded : {n_frames}')
    print(f'  Dataset       : {test_dataset}')
    print(f'  Dataset meta  : {dataset_meta.get("dataset_name", "unknown")} '
          f'({dataset_meta.get("num_keypoints", "?")} kpts)')

    # ── Build pipeline ─────────────────────────────────────────────────────
    print(f'\nBuilding post-processing pipeline from: {args.post_config}')
    pipeline = build_post_processor(args.post_config)
    mode_label = 'online' if pipeline.is_online else 'offline'
    print(f'  Pipeline mode : {mode_label}')
    print(f'  Filters       : {[type(f).__name__ for f in pipeline.filters]}')

    # ── Reconstruct PoseDataSamples ────────────────────────────────────────
    print('\nReconstructing PoseDataSamples ...')
    pred_samples, full_samples = reconstruct_pose_data_samples(
        frames, dataset_meta)

    # ── Run pipeline ───────────────────────────────────────────────────────
    print(f'\nRunning post-processing pipeline ({mode_label}) ...')
    postproc_samples: List[PoseDataSample] = []

    if pipeline.is_online:
        for pred_ds in pred_samples:
            result = pipeline.process(pred_ds)
            if result is not None:
                postproc_samples.append(result)
    else:
        for pred_ds in pred_samples:
            pipeline.process(pred_ds)
        postproc_samples = pipeline.evaluate()

    total_s = pipeline.total_s
    per_frame_ms = 1000.0 * total_s / len(postproc_samples) if postproc_samples else 0.0
    fps = len(postproc_samples) / total_s if total_s > 0 else 0.0

    print(f'  Frames processed : {len(postproc_samples)}')
    print(f'  Total time       : {total_s:.3f} s')
    print(f'  Per-frame avg    : {per_frame_ms:.2f} ms')
    print(f'  FPS              : {fps:.1f}')

    # ── Evaluate ───────────────────────────────────────────────────────────
    print(f'\nEvaluating with metrics: {args.metrics}')
    evaluator = build_evaluator_from_types(args.metrics, dataset_meta)

    # Attach GT from the full_samples (which have GT from frame records)
    num_kpts = dataset_meta.get('num_keypoints', 17)
    seq_track_id_map: Dict[str, int] = {}
    for pp_ds, full_ds, frame in zip(postproc_samples, full_samples, frames):
        # Merge post-processed pred_instances with GT from full_ds
        img_path = frame.get('img_path', '')
        pp_ds_gt = pp_ds.new()
        pp_ds_gt.set_metainfo(full_ds.metainfo)
        pp_ds_gt.pred_instances = pp_ds.pred_instances
        pp_ds_gt.gt_instances = full_ds.gt_instances
        pp_ds_gt.set_metainfo({'id': full_ds.metainfo.get('id', [frame['img_id']])})
        evaluator.process(data_samples=[pp_ds_gt], data_batch=None)

    quality = evaluator.evaluate(n_frames)

    # ── Print results ──────────────────────────────────────────────────────
    sep = '=' * 60
    print(f'\n{sep}')
    print(f'  Post-processing results')
    print(sep)
    print(f'  Pipeline : {[type(f).__name__ for f in pipeline.filters]}')
    print(f'  Timing:')
    print(f'    Total      : {total_s:.3f} s')
    print(f'    Per-frame  : {per_frame_ms:.2f} ms')
    print(f'    FPS        : {fps:.1f}')
    print('  Quality metrics:')
    for k, v in sorted(quality.items()):
        print(f'    {k}: {v:.4f}')
    print(f'{sep}\n')

    # ── Save post-processed bundle ─────────────────────────────────────────
    out_dir = args.out_dir or (osp.abspath(args.pred_dir) + '__postproc')
    print(f'Saving post-processed bundle to: {out_dir}')

    pp_frame_records: List[dict] = []
    for pp_ds, full_ds, frame in zip(postproc_samples, full_samples, frames):
        fid = frame.get('frame_id', 0)
        img_id = frame.get('img_id', 0)
        img_path = frame.get('img_path', '')

        pp_ds_gt = pp_ds.new()
        pp_ds_gt.set_metainfo(full_ds.metainfo)
        pp_ds_gt.pred_instances = pp_ds.pred_instances
        pp_ds_gt.gt_instances = full_ds.gt_instances

        gt_dicts = frame.get('ground_truth', {}).get('instances', [])
        pp_frame_records.append(build_frame_record(
            img_id=int(img_id),
            frame_id=int(fid),
            img_path=img_path,
            data_root=data_root,
            pred_ds=pp_ds_gt,
            gt_instances=gt_dicts,
            dataset_meta=dataset_meta,
        ))

    pp_manifest = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'source_pred_dir': osp.abspath(args.pred_dir),
        'post_config': osp.abspath(args.post_config),
        'mode': manifest.get('mode', 'unknown'),
        'mode_tag': manifest.get('mode_tag', 'unknown'),
        'test_dataset': test_dataset,
        'model_name': manifest.get('model_name', ''),
        'model_variant': manifest.get('model_variant', ''),
        'pose_config': manifest.get('pose_config', ''),
        'pose_checkpoint': manifest.get('pose_checkpoint', ''),
        'det_config': manifest.get('det_config'),
        'det_checkpoint': manifest.get('det_checkpoint'),
        'pipeline_filters': [type(f).__name__ for f in pipeline.filters],
        'quality': quality,
        'perf': {
            'postproc/latency_ms_per_frame': per_frame_ms,
            'postproc/total_s': total_s,
            'postproc/fps': fps,
        },
        'data_root': data_root,
        'dataset_meta': sanitize_dataset_meta(dataset_meta),
        'badcase_defaults': manifest.get('badcase_defaults', {
            'metric_key': 'mean_oks',
            'metric_type': 'accuracy',
            'thr': 0.5,
        }),
        'num_frames': len(pp_frame_records),
    }

    save_prediction_bundle(pp_manifest, pp_frame_records, out_dir)
    print(f'Bundle saved to {osp.abspath(out_dir)}')


if __name__ == '__main__':
    main()
