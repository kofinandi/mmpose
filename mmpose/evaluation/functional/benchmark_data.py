# Copyright (c) OpenMMLab. All rights reserved.
"""Unified data loading for benchmarking bottomup and topdown models.

Loads GT annotations unfiltered (preserving ``area``, ``iscrowd``, etc.) and
prefetches images, providing a single :class:`UnifiedSample` per image for
both bottomup and topdown inference pipelines.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import mmcv
import numpy as np
from mmengine.registry import DATASETS, TRANSFORMS, init_default_scope

from mmpose.evaluation.benchmark_datasets import BENCHMARK_TEST_DATASETS
from mmpose.structures.bbox import bbox_xyxy2xywh

# Dataset types that accept a `good_frame_mask` kwarg (see EmdbDataset /
# ThreeDPWDataset). Only these support --include-bad-frames.
_SUPPORTS_GOOD_FRAME_MASK = {
    'EmdbDataset', 'ThreeDPWDataset', 'PoseTrack21Dataset'}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GTInstance:
    """Per-instance ground-truth annotation in COCO-17 keypoint format."""

    keypoints: np.ndarray           # (K, 2) float32
    keypoints_visible: np.ndarray   # (K,)   float32, binary 0/1
    bbox: np.ndarray                # (4,)   float32, xyxy
    area: float
    id: int
    category_id: int = 1
    iscrowd: int = 0
    num_keypoints: int = 0          # visible kps in COCO-17 space
    # Raw COCO visibility (0=unlabeled, 1=occluded, 2=visible), when
    # available from raw_ann_info. None for non-COCO / converted datasets.
    keypoints_visible_coco: Optional[np.ndarray] = None
    # Video-track identifier; set from raw_ann_info['track_id'] when
    # available (e.g. EMDB, 3DPW).  Defaults to 0 for single-image datasets.
    track_id: int = 0


@dataclass
class UnifiedSample:
    """One image with its GT annotations and (optionally) pixel data.

    ``image`` is ``None`` until pixels have been loaded (see
    :func:`load_sample_image` / :class:`SampleImageStream`), and may be set
    back to ``None`` once a consumer is done with it to free memory in
    chunked-prefetch mode.
    """

    img_id: int
    img_path: str
    image: Optional[np.ndarray]         # BGR HWC uint8, or None if unloaded
    ori_shape: Tuple[int, int]          # (H, W)
    gt_instances: List[GTInstance] = field(default_factory=list)
    crowd_index: Optional[float] = None  # CrowdPose crowdIndex
    # Guards against double-scaling GT when load_sample_image is called more
    # than once for the same sample (e.g. re-fetched after being released).
    gt_scaled: bool = False
    # True when the frame has reliable GT (EMDB/3DPW/PoseTrack21
    # ``good_frame`` image field). Defaults to True for datasets without
    # that concept.
    good_frame: bool = True


# ---------------------------------------------------------------------------
# Validity helper (mirrors BaseCocoStyleDataset._is_valid_instance)
# ---------------------------------------------------------------------------

def is_valid_instance(gt: GTInstance) -> bool:
    """Return True when ``gt`` is a valid detection target.

    Used by the mock detector to select GT bboxes that represent real,
    labelled persons.  Replicates
    :meth:`BaseCocoStyleDataset._is_valid_instance`.
    """
    if gt.iscrowd:
        return False
    if gt.num_keypoints == 0:
        return False
    w = gt.bbox[2] - gt.bbox[0]
    h = gt.bbox[3] - gt.bbox[1]
    if w <= 0 or h <= 0:
        return False
    if np.max(gt.keypoints) <= 0:
        return False
    return True


def resize_to_ori_shape(img: np.ndarray, ori_shape) -> np.ndarray:
    """Resize *img* to ``ori_shape`` when it differs from the file on disk.

    Benchmark runs may prefetch some datasets (e.g. EMDB) at reduced
    resolution; predictions and GT in a saved bundle are stored in that
    inference coordinate space while ``img_path`` still points at the
    original file.  Consumers that reload images from disk (visualisation,
    image-consuming post-processing) use this to bring the pixels back into
    the bundle's coordinate space.

    Args:
        img: Image as read from disk, ``(H, W, 3)``.
        ori_shape: Target ``(height, width)`` from the bundle frame record.
            Falsy or malformed values leave the image untouched.

    Returns:
        The image at exactly ``ori_shape`` resolution.
    """
    if ori_shape is None or len(ori_shape) < 2:
        return img
    target_h, target_w = int(ori_shape[0]), int(ori_shape[1])
    if target_h <= 0 or target_w <= 0:
        return img
    h, w = img.shape[:2]
    if (h, w) == (target_h, target_w):
        return img
    return mmcv.imresize(img, (target_w, target_h))


def _resize_prefetch_image(
    image: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, float, float]:
    """Resize ``image`` by ``scale`` and return actual axis scale factors."""
    orig_h, orig_w = image.shape[:2]
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    sx = new_w / orig_w
    sy = new_h / orig_h
    resized = mmcv.imresize(image, (new_w, new_h))
    return resized, sx, sy


def _scale_gt_instances(
    gt_instances: List[GTInstance],
    sx: float,
    sy: float,
) -> None:
    """Scale GT bboxes/keypoints/areas in place to match a resized image."""
    area_scale = sx * sy
    for gt in gt_instances:
        gt.keypoints[:, 0] *= sx
        gt.keypoints[:, 1] *= sy
        gt.bbox[0] *= sx
        gt.bbox[1] *= sy
        gt.bbox[2] *= sx
        gt.bbox[3] *= sy
        gt.area *= area_scale


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def build_unified_samples(
    dataset_name: str,
    num_frames: Optional[int] = None,
    include_bad_frames: bool = False,
) -> List[UnifiedSample]:
    """Load GT annotations (metadata only, no pixel data) for all datasets.

    Returns one :class:`UnifiedSample` per unique image (including images
    with zero annotations so the denominator for recall is correct), with
    ``image=None``. Keypoints are always converted to COCO-17 format.

    The raw annotation data is obtained via
    :meth:`BaseCocoStyleDataset._load_annotations` **without** any
    ``_is_valid_instance`` or ``_get_topdown_data_infos`` filtering, so
    every annotation (including ``iscrowd=1``) is preserved.

    Use :func:`load_sample_image` (directly, or via :class:`SampleImageStream`)
    to populate ``image`` before running inference. For the common case of
    eagerly loading every image up front, use :func:`load_unified_samples`
    instead, which wraps this function.

    Args:
        dataset_name: One of ``'coco', 'crowdpose', 'mpii', 'aic',
            'ochuman', 'emdb', 'emdb-mini', '3dpw', 'posetrack21'``.
        num_frames: If set, cap the number of unique images loaded.
        include_bad_frames: Only meaningful for datasets whose loader
            supports a ``good_frame_mask`` kwarg (EMDB, 3DPW, PoseTrack21).
            When ``True``, overrides that kwarg to ``False`` so frames
            dropped for lacking reliable GT (e.g. EMDB ``invalid_idxs``, 3DPW
            frames with no valid actor, PoseTrack21 unlabeled frames) are
            still loaded -- with an image
            but no GT instances -- giving downstream consumers (e.g. the
            post-processing tracker) a temporally continuous frame
            sequence. These frames may produce false-positive detections
            that affect detection/pose metrics; temporal metrics
            (MPJVE/MPJAE) exclude them regardless, via frame_id-gap
            masking. Ignored (with a warning) for datasets that don't
            support the kwarg. Default: ``False``.

    Returns:
        List of :class:`UnifiedSample` (``image=None``) in dataset order.
    """
    init_default_scope('mmpose')

    spec = BENCHMARK_TEST_DATASETS[dataset_name]
    prefetch_scale = spec.prefetch_scale
    if prefetch_scale is not None and not (0.0 < prefetch_scale <= 1.0):
        raise ValueError(
            f'prefetch_scale for {dataset_name!r} must be in (0, 1], '
            f'got {prefetch_scale}.')

    # ── Resolve dataset build config ───────────────────────────────────────
    ds_cfg = dict(
        type=spec.dataset_type,
        data_root=spec.data_root,
        ann_file=spec.ann_file,
        data_prefix=spec.data_prefix,
        data_mode='topdown',
        pipeline=[],
        test_mode=True,
        lazy_init=True,
    )
    if spec.dataset_kwargs:
        ds_cfg.update(spec.dataset_kwargs)
    if include_bad_frames:
        if spec.dataset_type in _SUPPORTS_GOOD_FRAME_MASK:
            ds_cfg['good_frame_mask'] = False
        else:
            print(
                f'Warning: --include-bad-frames has no effect on '
                f'{dataset_name!r} ({spec.dataset_type!r} has no '
                f'good_frame_mask concept); ignoring.')
    keypoint_src = spec.keypoint_src

    # Build dataset with lazy_init=True so full_init (which calls
    # _get_topdown_data_infos filtering) is never triggered.
    dataset = DATASETS.build(ds_cfg)
    instance_list, image_list = dataset._load_annotations()

    # ── Image metadata lookup ──────────────────────────────────────────────
    img_info: Dict[int, dict] = {}
    for img in image_list:
        iid = int(img['img_id'])
        if iid not in img_info:
            img_info[iid] = {
                'img_path': img['img_path'],
                # CrowdPose stores crowdIndex in the raw COCO image dict
                'crowd_index': img.get('crowdIndex') or img.get('crowd_index'),
                # COCO-style raw image dicts carry width/height; used as a
                # placeholder ori_shape before pixels are loaded (e.g. MPII
                # has neither, so ori_shape stays (0, 0) until loaded).
                'width': img.get('width'),
                'height': img.get('height'),
                # EMDB / 3DPW / PoseTrack21 image field; default True so
                # datasets without the concept stay fully evaluable.
                'good_frame': bool(img.get('good_frame', True)),
            }

    # ── Ordered unique img_ids with optional num_frames cap ─────────────
    seen_img_ids: List[int] = []
    seen_set: set = set()
    for img in image_list:
        iid = int(img['img_id'])
        if iid not in seen_set:
            seen_set.add(iid)
            seen_img_ids.append(iid)
            if num_frames is not None and len(seen_img_ids) >= num_frames:
                break

    # ── Group instances by img_id ─────────────────────────────────────────
    img_to_instances: Dict[int, list] = {iid: [] for iid in seen_img_ids}
    for inst in instance_list:
        iid = int(inst['img_id'])
        if iid in img_to_instances:
            img_to_instances[iid].append(inst)

    # ── Keypoint converter ────────────────────────────────────────────────
    kp_converter = None
    if keypoint_src != 'coco':
        kp_converter = TRANSFORMS.build(
            dict(type='KeypointConverter', src=keypoint_src, dst='coco'))

    def _parse_instance(inst: dict) -> GTInstance:
        """Convert a raw dataset instance dict into a :class:`GTInstance`."""
        kpts = np.asarray(inst['keypoints'], dtype=np.float32)

        kv_raw = inst.get('keypoints_visible')
        if kv_raw is None:
            # Fall back to ones if visibility not stored
            kv = np.ones(kpts.shape[:-1], dtype=np.float32)
        else:
            kv = np.asarray(kv_raw, dtype=np.float32)

        # Normalise shapes for KeypointConverter: (N, K, 2) and (N, K)
        if kpts.ndim == 2:
            kpts = kpts[None]       # (1, K, 2)
        if kv.ndim == 1:
            kv = kv[None]           # (1, K)
        elif kv.ndim == 3:
            kv = kv[:, :, 0]       # (N, K, 2) → (N, K): take visibility

        if kp_converter is not None:
            tmp = kp_converter({'keypoints': kpts, 'keypoints_visible': kv})
            kpts_out = tmp['keypoints']          # (1, K_dst, 2)
            kv_out = tmp['keypoints_visible']    # (1, K_dst, 2): [vis, weight]
            kpts_final: np.ndarray = kpts_out[0]        # (K_dst, 2)
            kv_final: np.ndarray = kv_out[0, :, 0]     # (K_dst,)
        else:
            kpts_final = kpts[0]                        # (K, 2)
            kv_final = kv[0] if kv.ndim >= 2 else kv   # (K,)

        num_kpts = int(np.sum(kv_final > 0))

        bbox = np.asarray(inst['bbox'], dtype=np.float32).reshape(4)

        area_raw = inst.get('area')
        if area_raw is not None:
            area = float(np.asarray(area_raw).flat[0])
        else:
            area = 0.0
        if area <= 0:
            area = float(
                max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) * 0.53, 1.0))

        iscrowd = int(inst.get('iscrowd', 0))

        cat_id = inst.get('category_id', 1)
        try:
            cat_id = int(np.asarray(cat_id).flat[0])
        except (TypeError, ValueError):
            cat_id = 1

        inst_id = inst.get('id', 0)
        try:
            inst_id = int(np.asarray(inst_id).flat[0])
        except (TypeError, ValueError):
            inst_id = 0

        # Recover raw COCO visibility (0/1/2) from raw_ann_info when no
        # keypoint converter is applied.  _load_annotations clips v to [0,1]
        # via np.minimum(1, v), so we re-read from the original flat list.
        kv_coco: Optional[np.ndarray] = None
        if kp_converter is None:
            raw_kp_list = inst.get('raw_ann_info', {}).get('keypoints', [])
            if raw_kp_list:
                raw_arr = np.asarray(raw_kp_list, dtype=np.float32).reshape(-1, 3)
                if len(raw_arr) == len(kpts_final):
                    kv_coco = raw_arr[:, 2]  # raw v column: 0/1/2

        track_id = int(inst.get('raw_ann_info', {}).get('track_id', 0))

        return GTInstance(
            keypoints=kpts_final,
            keypoints_visible=kv_final,
            bbox=bbox,
            area=area,
            id=inst_id,
            category_id=cat_id,
            iscrowd=iscrowd,
            num_keypoints=num_kpts,
            keypoints_visible_coco=kv_coco,
            track_id=track_id,
        )

    # ── Build samples (metadata only; no pixel data yet) ────────────────────
    samples: List[UnifiedSample] = []
    for img_id in seen_img_ids:
        info = img_info.get(img_id)
        if info is None:
            insts = img_to_instances.get(img_id, [])
            if insts:
                img_path = insts[0]['img_path']
                crowd_index = None
                width = height = None
                good_frame = True
            else:
                continue  # cannot determine image path
        else:
            img_path = info['img_path']
            crowd_index = info.get('crowd_index')
            width = info.get('width')
            height = info.get('height')
            good_frame = bool(info.get('good_frame', True))

        gt_instances = [
            _parse_instance(inst) for inst in img_to_instances[img_id]
        ]

        # Placeholder ori_shape from raw annotation metadata (if available);
        # load_sample_image overwrites this with the true, possibly
        # downscaled, shape once pixels are read.
        ori_shape = (int(height), int(width)) if (width and height) else (0, 0)

        samples.append(UnifiedSample(
            img_id=img_id,
            img_path=img_path,
            image=None,
            ori_shape=ori_shape,
            gt_instances=gt_instances,
            crowd_index=crowd_index,
            good_frame=good_frame,
        ))

    return samples


def load_sample_image(
    sample: UnifiedSample,
    prefetch_scale: Optional[float] = None,
) -> None:
    """Load (and optionally downscale) pixel data for *sample*, in place.

    No-op if ``sample.image`` is already set. On a successful read, sets
    ``sample.image`` and the final ``sample.ori_shape``; on failure, prints
    a warning and leaves ``sample.image`` as ``None`` (the placeholder
    ``ori_shape`` from :func:`build_unified_samples` is left untouched).

    GT bboxes/keypoints/areas are scaled to match a downscaled image only
    once per sample (guarded by ``sample.gt_scaled``), so calling this
    again after ``sample.image`` has been released back to ``None`` (e.g.
    by :class:`SampleImageStream`) does not double-scale GT.

    Args:
        sample: Sample to populate in place.
        prefetch_scale: If set and not ``1.0``, downscale the image by this
            factor and scale GT to match (see ``prefetch_scale`` on
            :class:`~mmpose.evaluation.benchmark_datasets.BenchmarkTestDataset`).
    """
    if sample.image is not None:
        return

    image = mmcv.imread(sample.img_path)
    if image is None:
        print(f'Warning: could not read {sample.img_path}, skipping.')
        return

    if prefetch_scale is not None and prefetch_scale != 1.0:
        image, sx, sy = _resize_prefetch_image(image, prefetch_scale)
        if not sample.gt_scaled:
            _scale_gt_instances(sample.gt_instances, sx, sy)
            sample.gt_scaled = True

    h, w = image.shape[:2]
    sample.image = image
    sample.ori_shape = (h, w)


def load_unified_samples(
    dataset_name: str,
    num_frames: Optional[int] = None,
    include_bad_frames: bool = False,
) -> List[UnifiedSample]:
    """Load GT annotations and eagerly prefetch every image up front.

    Thin wrapper around :func:`build_unified_samples` +
    :func:`load_sample_image` that preserves the original all-at-once
    loading behaviour. See :func:`build_unified_samples` for argument docs.
    Prefer :class:`SampleImageStream` with ``chunk_size`` set for very large
    datasets (e.g. PoseTrack21) that don't fit in memory all at once.

    Returns:
        List of :class:`UnifiedSample` (with pixel data loaded) in dataset
        order. Samples whose image failed to load are dropped.
    """
    spec = BENCHMARK_TEST_DATASETS[dataset_name]
    prefetch_scale = spec.prefetch_scale

    samples = build_unified_samples(
        dataset_name, num_frames, include_bad_frames=include_bad_frames)

    if prefetch_scale is not None and prefetch_scale != 1.0:
        print(f'Prefetching {len(samples)} images '
              f'at scale {prefetch_scale} ...')
    else:
        print(f'Prefetching {len(samples)} images ...')

    loaded: List[UnifiedSample] = []
    for sample in samples:
        load_sample_image(sample, prefetch_scale)
        if sample.image is not None:
            loaded.append(sample)

    print(f'Prefetch complete. Loaded {len(loaded)} images.')
    return loaded


# ---------------------------------------------------------------------------
# Chunked image streaming (for large datasets that don't fit in RAM)
# ---------------------------------------------------------------------------

class SampleImageStream:
    """Yield :class:`UnifiedSample`\\ s with pixel data, in bounded chunks.

    Two modes:

    - ``chunk_size=None`` (default): eager pass-through. ``samples`` are
      assumed to already carry their pixel data (e.g. from
      :func:`load_unified_samples`); iteration just yields them in order
      with no extra memory or timing overhead.
    - ``chunk_size=N``: chunked prefetch. ``samples`` are assumed to carry
      *no* pixel data yet (e.g. from :func:`build_unified_samples`). A
      background thread walks the sample list in fixed-size chunks of
      ``N``, decoding each chunk's images in parallel via a
      ``ThreadPoolExecutor`` (OpenCV's decoder releases the GIL, so this
      parallelises real work) and buffering up to ``queue_chunks`` decoded
      chunks ahead of the consumer. Once a chunk has been fully consumed
      (i.e. the caller has moved on to the next chunk), every sample in it
      has ``image`` reset to ``None`` so it can be garbage collected --
      anyone holding their own reference to the pixel array (e.g. a
      producer/consumer queue item) keeps it alive regardless. Peak
      resident image memory is roughly ``chunk_size * (queue_chunks + 1)``
      samples instead of the whole dataset.

    Regardless of mode, iterating always yields exactly ``len(samples)``
    items, in order, including samples whose image failed to decode
    (yielded with ``image=None``) -- this keeps a 1:1 mapping between
    sample index and ``frame_id`` for callers that rely on it.
    """

    def __init__(
        self,
        samples: List[UnifiedSample],
        prefetch_scale: Optional[float] = None,
        chunk_size: Optional[int] = None,
        queue_chunks: int = 2,
        workers: int = 4,
    ) -> None:
        self.samples = samples
        self.prefetch_scale = prefetch_scale
        self.chunk_size = chunk_size
        self.queue_chunks = max(1, queue_chunks)
        self.workers = max(1, workers)
        self.stall_s = 0.0  # cumulative time spent blocked waiting for data

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[UnifiedSample]:
        self.stall_s = 0.0

        if self.chunk_size is None:
            yield from self.samples
            return

        chunk_queue: 'queue.Queue' = queue.Queue(maxsize=self.queue_chunks)
        _sentinel = object()
        errors: List[BaseException] = []

        def _produce() -> None:
            try:
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    for start in range(0, len(self.samples), self.chunk_size):
                        chunk = self.samples[start:start + self.chunk_size]
                        list(pool.map(
                            lambda s: load_sample_image(
                                s, self.prefetch_scale),
                            chunk))
                        chunk_queue.put(chunk)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                chunk_queue.put(_sentinel)

        producer = threading.Thread(target=_produce, daemon=True)
        producer.start()

        while True:
            t0 = time.perf_counter()
            chunk = chunk_queue.get()
            self.stall_s += time.perf_counter() - t0
            if chunk is _sentinel:
                break
            yield from chunk
            for sample in chunk:
                sample.image = None

        producer.join()
        if errors:
            raise errors[0]


# ---------------------------------------------------------------------------
# Detection annotation builder (for --det-metrics)
# ---------------------------------------------------------------------------

def build_det_ann_from_samples(
    samples: List[UnifiedSample],
    cache_dir: Optional[str] = None,
) -> str:
    """Build a COCO bbox-detection GT JSON from :class:`UnifiedSample` list.

    Replaces ``convert_pose_gt_to_coco_det_ann`` / ``resolve_det_ann_file``
    for the ``--det-metrics`` path.  Preserves the real ``iscrowd`` value
    from each annotation (previously hardcoded to 0).

    Args:
        samples: List of loaded unified samples.
        cache_dir: Directory for caching the JSON output.

    Returns:
        Path to the COCO-format annotation JSON file.
    """
    if cache_dir is None:
        cache_dir = osp.join('work_dirs', '.cache', 'coco_det_ann')
    os.makedirs(cache_dir, exist_ok=True)

    key = ','.join(f'{s.img_id}:{len(s.gt_instances)}' for s in samples)
    digest = hashlib.md5(key.encode()).hexdigest()
    out_path = osp.join(cache_dir, f'unified_{digest}.json')
    if osp.isfile(out_path):
        return out_path

    images = []
    annotations = []

    for sample in samples:
        h, w = sample.ori_shape
        images.append({
            'id': int(sample.img_id),
            'file_name': osp.basename(sample.img_path),
            'width': w,
            'height': h,
        })
        for inst in sample.gt_instances:
            bbox_xywh = bbox_xyxy2xywh(
                inst.bbox.reshape(1, 4).astype(np.float32))[0]
            annotations.append({
                'id': int(inst.id),
                'image_id': int(sample.img_id),
                'category_id': int(inst.category_id),
                'bbox': [float(x) for x in bbox_xywh],
                'area': float(inst.area),
                'iscrowd': int(inst.iscrowd),
            })

    coco_json = {
        'info': {'description': 'COCO det GT built by mmpose benchmark loader'},
        'images': images,
        'annotations': annotations,
        'categories': [
            {'supercategory': 'person', 'id': 1, 'name': 'person'}
        ],
    }
    with open(out_path, 'w') as f:
        json.dump(coco_json, f)
    return out_path
