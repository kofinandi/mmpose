# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from ..functional import keypoint_mpjae, keypoint_mpjve
from ..functional.frame_metrics import match_instances_oks

# COCO-17 keypoint indices used for torso-diameter normalisation.
# Torso diameter is the mean of the two cross-torso diagonals:
#   ||left_shoulder (5)  - right_hip (12)||
#   ||right_shoulder (6) - left_hip  (11)||
_TORSO_KP_PAIRS = [(5, 12), (6, 11)]

# Mapping from norm_item name to the metric-name prefix used in results.
_METRIC_PREFIX = {
    None: '',       # raw → 'MPJVE' / 'MPJAE'
    'bbox': 'b',    # bbox-normalised → 'bMPJVE' / 'bMPJAE'
    'head': 'h',    # head-normalised → 'hMPJVE' / 'hMPJAE'
    'torso': 't',   # torso-normalised → 'tMPJVE' / 'tMPJAE'
}


def _parse_frame_id(img_path: str) -> int:
    """Extract the zero-based frame index from a video image path.

    Supports:

    - 3DPW: ``image_XXXXX.jpg`` (e.g. ``image_00042`` → ``42``)
    - EMDB: ``XXXXX.jpg`` (e.g. ``00042`` → ``42``)
    """
    basename = osp.splitext(osp.basename(img_path))[0]
    if basename.startswith('image_'):
        return int(basename.split('_')[-1])
    return int(basename)


class _TemporalBaseMetric(BaseMetric):
    """Shared ``process()`` and initialisation logic for temporal metrics.

    Supports optional normalisation of the error by a body-size scale factor
    to make metrics comparable across subjects and viewing distances.

    The ``process()`` method handles two modes automatically:

    **Single-instance mode** (``tools/test.py`` path): ``pred_instances``
    and ``gt_instances`` each have exactly one instance after the dataset
    pipeline crops to the GT bbox.  OKS matching still runs but is trivially
    a 1-vs-1 comparison.

    **Multi-instance mode** (``tools/benchmark_e2e.py`` path): the frame
    carries *N* predicted poses and *M* GT tracks.  Greedy OKS matching
    assigns each predicted pose to at most one GT track; only matched pairs
    contribute to the temporal error for that frame.  ``gt_instances`` must
    carry ``track_ids`` (an integer array of shape ``[M]``).  Frames whose
    best match falls below ``match_thr`` are silently skipped, which is
    correct behaviour for a real detector that may miss a frame.

    Args:
        norm_item (str | Sequence[str] | None): Normalisation method(s) to
            apply.  Valid values:

            - ``None`` (default): no normalisation; returns raw error.
            - ``'bbox'``: normalise by the mean bbox diagonal across the track.
            - ``'head'``: normalise by the mean head size across the track.
              Requires ``head_size`` to be present in ``gt_instances``.
            - ``'torso'``: normalise by the mean torso diameter across the
              track, computed from COCO-17 shoulder/hip keypoints (indices
              5, 6, 11, 12).

            Multiple items may be given as a list; all normalised variants
            are reported together.

        match_thr (float): Minimum OKS score to accept a pred-to-GT match.
            Predicted poses whose best-match OKS is below this threshold are
            dropped for the current frame.  Default: ``0.5``.

        collect_device (str): Device for distributed result collection.
            Default: ``'cpu'``.
        prefix (str, optional): Metric name prefix.  Default: ``None``.
    """

    default_prefix: Optional[str] = 'temporal'

    _ALLOWED_NORM = ('bbox', 'head', 'torso')

    def __init__(self,
                 norm_item: Union[str, Sequence[str], None] = None,
                 match_thr: float = 0.5,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if norm_item is None:
            self.norm_item: List[Optional[str]] = [None]
        elif isinstance(norm_item, str):
            self.norm_item = [norm_item]
        else:
            self.norm_item = list(norm_item)

        for item in self.norm_item:
            if item is not None and item not in self._ALLOWED_NORM:
                raise KeyError(
                    f'`norm_item` {item!r} is not supported by '
                    f'{self.__class__.__name__}. '
                    f"Must be None, 'bbox', 'head', or 'torso'.")

        self.match_thr = float(match_thr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_gt_list(self, gt: dict, n_gt: int) -> List[dict]:
        """Build the list of GT dicts expected by :func:`match_instances_oks`.

        Each entry carries ``keypoints``, ``keypoints_visible``,
        optionally ``keypoints_visible_coco``, ``bbox`` (xyxy), ``orig_area``,
        and ``iscrowd``.
        """
        kpts = np.asarray(gt['keypoints'])      # (M, K, 2)
        kv = np.asarray(gt['keypoints_visible'])  # (M, K) or (M, K, 2)
        if kv.ndim == 3:
            kv = kv[:, :, 0]

        bboxes = gt.get('bboxes')
        areas = gt.get('orig_areas')
        iscrowd_arr = gt.get('iscrowd', np.zeros(n_gt, dtype=np.int32))
        kv_coco_arr = gt.get('keypoints_visible_coco')  # (M, K) or None
        track_ids_arr = gt.get('track_ids')              # (M,)   or None

        gt_list = []
        for i in range(n_gt):
            entry: dict = {
                'keypoints': kpts[i],        # (K, 2)
                'keypoints_visible': kv[i],  # (K,)
                'iscrowd': int(iscrowd_arr[i]) if iscrowd_arr is not None else 0,
            }
            if kv_coco_arr is not None:
                entry['keypoints_visible_coco'] = np.asarray(kv_coco_arr[i])

            if bboxes is not None and len(bboxes) > i:
                b = np.asarray(bboxes[i]).reshape(-1)[:4]
                entry['bbox'] = b
                w = float(b[2] - b[0])
                h = float(b[3] - b[1])
                entry['orig_area'] = float(
                    areas[i] if areas is not None and len(areas) > i
                    else max(w * h, 1.0))
            else:
                entry['orig_area'] = 1.0

            if track_ids_arr is not None:
                entry['_track_id'] = int(track_ids_arr[i])
            else:
                entry['_track_id'] = i  # fall back: use instance index

            gt_list.append(entry)
        return gt_list

    def _build_pred_list(self, pred: dict, n_pred: int) -> List[dict]:
        """Build the list of pred dicts expected by :func:`match_instances_oks`."""
        kpts = np.asarray(pred['keypoints'])  # (N, K, 2)
        pred_list = []
        for i in range(n_pred):
            pred_list.append({'keypoints': kpts[i]})
        return pred_list

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Accumulate per-sample prediction/GT pairs via OKS matching.

        Works for both the single-instance (``tools/test.py``) and
        multi-instance (``tools/benchmark_e2e.py``) cases.

        Args:
            data_batch (Sequence[dict]): Unused; present for API compatibility.
            data_samples (Sequence[dict]): Model outputs for a batch.
        """
        sigmas = np.asarray(
            self.dataset_meta.get('sigmas', []), dtype=np.float32)
        if len(sigmas) == 0:
            return  # no keypoint metadata → skip silently

        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            gt = data_sample['gt_instances']

            # `gt_instances`/`pred_instances` may have no `keypoints` field
            # at all (rather than a zero-length array) when a frame has zero
            # GT or zero detections -- e.g. a "bad" EMDB/3DPW frame loaded
            # via --include-bad-frames, which carries no GT annotations.
            # Treat a missing field the same as zero instances instead of
            # raising a KeyError.
            pred_kpts_raw = pred.get('keypoints')
            gt_kpts_raw = gt.get('keypoints')
            if pred_kpts_raw is None or gt_kpts_raw is None:
                continue

            pred_kpts = np.asarray(pred_kpts_raw)
            gt_kpts = np.asarray(gt_kpts_raw)

            # Normalise leading singleton dims (single-instance topdown path)
            if pred_kpts.ndim == 2:
                pred_kpts = pred_kpts[None]   # (1, K, 2)
            if gt_kpts.ndim == 2:
                gt_kpts = gt_kpts[None]       # (1, K, 2)

            n_pred = pred_kpts.shape[0]
            n_gt = gt_kpts.shape[0]

            if n_pred == 0 or n_gt == 0:
                continue

            img_path = data_sample.get('img_path', '')
            frame_id = _parse_frame_id(img_path)

            gt_list = self._build_gt_list(gt, n_gt)
            pred_list = self._build_pred_list(
                {'keypoints': pred_kpts}, n_pred)

            matches, _ = match_instances_oks(
                gt_list, pred_list, sigmas, match_thr=self.match_thr)

            gt_kv = np.asarray(gt['keypoints_visible'])
            if gt_kv.ndim == 3:
                gt_kv = gt_kv[:, :, 0]
            if gt_kv.ndim == 1:
                gt_kv = gt_kv[None]

            for m in matches:
                gi = m['gt_idx']
                pi = m['pred_idx']

                gt_entry = gt_list[gi]
                track_id = gt_entry['_track_id']

                pred_coords = pred_kpts[pi]      # (K, 2)
                gt_coords = gt_kpts[gi]          # (K, 2)
                mask = gt_kv[gi].astype(bool)    # (K,)

                result = dict(
                    track_id=track_id,
                    frame_id=frame_id,
                    pred_coords=pred_coords,
                    gt_coords=gt_coords,
                    mask=mask,
                )

                if 'bbox' in self.norm_item:
                    b = gt_entry.get('bbox')
                    if b is not None:
                        b = np.asarray(b).reshape(-1)[:4]
                        bbox_size = float(
                            np.max(b[2:] - b[:2]))
                    else:
                        # fall back to bboxes field in gt
                        bboxes = gt.get('bboxes')
                        if bboxes is not None and len(bboxes) > gi:
                            b2 = np.asarray(bboxes[gi]).reshape(-1)[:4]
                            bbox_size = float(np.max(b2[2:] - b2[:2]))
                        else:
                            bbox_size = 1.0
                    result['bbox_size'] = max(bbox_size, 1.0)

                if 'head' in self.norm_item:
                    assert 'head_size' in gt, (
                        'The ground truth data info does not have the expected '
                        'normalized_item ``"head_size"``.')
                    result['head_size'] = float(gt['head_size'])

                if 'torso' in self.norm_item:
                    sizes = []
                    for i_idx, j_idx in _TORSO_KP_PAIRS:
                        if mask[i_idx] and mask[j_idx]:
                            sizes.append(
                                float(np.linalg.norm(
                                    gt_coords[i_idx] - gt_coords[j_idx])))
                    if not sizes:
                        for i_idx, j_idx in _TORSO_KP_PAIRS:
                            sizes.append(
                                float(np.linalg.norm(
                                    pred_coords[i_idx] - pred_coords[j_idx])))
                        warnings.warn(
                            'Ground truth torso size < 1 for at least one '
                            'frame. Using predicted keypoints to estimate '
                            'torso size.')
                    torso_size = float(np.mean(sizes)) if sizes else 1.0
                    result['torso_size'] = max(torso_size, 1.0)

                self.results.append(result)

    def _build_tracks(
        self, results: List[dict]
    ) -> Dict[int, Dict[str, object]]:
        """Group per-sample results into per-track dicts, densified by
        ``frame_id`` so temporal gaps become explicit masked-out rows.

        A track's matched frames are not necessarily consecutive: a
        detector miss, an OKS mismatch, or an upstream dataset gap (e.g.
        EMDB ``invalid_idxs``, 3DPW frames with no valid actor) all leave
        a hole in ``frame_id``. Naively stacking only the matched frames
        would make frames on either side of such a hole appear adjacent to
        :func:`keypoint_mpjve`/:func:`keypoint_mpjae`, which assume a
        constant, unit frame spacing -- silently inflating the velocity /
        acceleration error across the gap.

        To avoid this, the track is expanded onto a contiguous
        ``frame_id`` grid spanning ``[min(frame_id), max(frame_id)]``; any
        missing ``frame_id`` becomes a row with an all-``False`` visibility
        mask. The existing pairwise/triplet masking in
        :func:`keypoint_mpjve`/:func:`keypoint_mpjae` then naturally
        excludes every pair or triplet touching such a row, so only truly
        consecutive frames contribute to the error.

        Returns:
            dict: Mapping from ``track_id`` to a dict with numpy arrays
            ``pred`` ``[T,K,D]``, ``gt`` ``[T,K,D]``, ``mask`` ``[T,K]``,
            and optional per-frame scale lists (``bbox_scale``,
            ``head_scale``, ``torso_scale``).
        """
        raw: dict = defaultdict(list)
        for r in results:
            raw[r['track_id']].append(r)

        tracks = {}
        for tid, frames in raw.items():
            frames.sort(key=lambda x: x['frame_id'])
            by_fid = {f['frame_id']: f for f in frames}
            first_fid = frames[0]['frame_id']
            last_fid = frames[-1]['frame_id']

            zero_kpts = np.zeros_like(frames[0]['pred_coords'])
            false_mask = np.zeros(zero_kpts.shape[0], dtype=bool)
            scale_keys = [
                k for k in ('bbox_size', 'head_size', 'torso_size')
                if k in frames[0]
            ]

            pred_rows, gt_rows, mask_rows = [], [], []
            scale_rows: Dict[str, list] = {k: [] for k in scale_keys}
            for fid in range(first_fid, last_fid + 1):
                f = by_fid.get(fid)
                if f is not None:
                    pred_rows.append(f['pred_coords'])
                    gt_rows.append(f['gt_coords'])
                    mask_rows.append(f['mask'])
                    for k in scale_keys:
                        scale_rows[k].append(f[k])
                else:
                    # Gap row: mask=False excludes every pair/triplet that
                    # touches it, so the fabricated zero coords never
                    # contribute to the error. `1.0` scale is a neutral
                    # placeholder (never used, but avoids a division by 0
                    # were a neighbouring valid pair to reference it).
                    pred_rows.append(zero_kpts)
                    gt_rows.append(zero_kpts)
                    mask_rows.append(false_mask)
                    for k in scale_keys:
                        scale_rows[k].append(1.0)

            entry: dict = dict(
                pred=np.stack(pred_rows),
                gt=np.stack(gt_rows),
                mask=np.stack(mask_rows),
            )
            if 'bbox_size' in scale_keys:
                entry['bbox_scale'] = np.array(scale_rows['bbox_size'])
            if 'head_size' in scale_keys:
                entry['head_scale'] = np.array(scale_rows['head_size'])
            if 'torso_size' in scale_keys:
                entry['torso_scale'] = np.array(scale_rows['torso_size'])
            tracks[tid] = entry
        return tracks

    def _pair_scale(self, scale_arr: np.ndarray) -> np.ndarray:
        """Average scale over adjacent frame pairs (for velocity).

        Args:
            scale_arr (np.ndarray): Per-frame scale with shape ``[T]``.

        Returns:
            np.ndarray: Shape ``[T-1]`` with mean of consecutive pairs.
        """
        return (scale_arr[:-1] + scale_arr[1:]) / 2.0

    def _triplet_scale(self, scale_arr: np.ndarray) -> np.ndarray:
        """Average scale over consecutive triplets (for acceleration).

        Args:
            scale_arr (np.ndarray): Per-frame scale with shape ``[T]``.

        Returns:
            np.ndarray: Shape ``[T-2]`` with mean of consecutive triplets.
        """
        return (scale_arr[:-2] + scale_arr[1:-1] + scale_arr[2:]) / 3.0


@METRICS.register_module()
class MPJVE(_TemporalBaseMetric):
    """Mean Per-Joint Velocity Error (MPJVE) evaluation metric.

    Measures temporal smoothness of predicted poses by comparing the
    first-order finite difference (velocity) of predicted keypoint
    trajectories against the ground-truth velocity.

    For a track of T frames the velocity at frame t is:
        ``v_t = P_t - P_{t-1}``

    The raw error is the mean Euclidean distance between predicted and
    ground-truth velocity vectors over all visible joints and all
    consecutive frame pairs across every person-track in the dataset.

    An OKS-based matching step (threshold ``match_thr``) is applied in
    ``process()`` before accumulating predictions, so the metric works
    correctly when the data sample carries multiple unmatched predicted
    instances alongside multiple GT tracks (e.g. from
    ``tools/benchmark_e2e.py``).  For the single-instance topdown path
    (``tools/test.py``) the matching is trivially 1-vs-1.

    Optionally, errors can be normalised by a body-size scale factor via
    ``norm_item`` (see :class:`_TemporalBaseMetric`).  When multiple
    ``norm_item`` values are given, all variants are reported together.

    Returned metric names:

    - ``norm_item=None``      → ``'MPJVE'``
    - ``norm_item='bbox'``    → ``'bMPJVE'``
    - ``norm_item='head'``    → ``'hMPJVE'``
    - ``norm_item='torso'``   → ``'tMPJVE'``

    .. note::
        Requires ``img_path`` following the naming convention
        ``*/<seq_name>/image_XXXXX.jpg`` (3DPW) or ``*/XXXXX.jpg`` (EMDB).
        In ``tools/test.py`` mode, ``track_id`` must be present in
        ``raw_ann_info``.  In ``tools/benchmark_e2e.py`` mode,
        ``track_ids`` must be set on ``gt_instances``.

    Args:
        norm_item (str | Sequence[str] | None): Normalisation method(s).
            Default: ``None`` (raw pixel-space error).
        match_thr (float): OKS threshold for pred-to-GT matching.
            Default: ``0.5``.
        collect_device (str): Device for distributed result collection.
            Default: ``'cpu'``.
        prefix (str, optional): Metric name prefix.  Default: ``None``.
    """

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute MPJVE (and normalised variants).

        Returns:
            Dict[str, float]: One entry per requested ``norm_item``.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        metrics: Dict[str, float] = {}

        err_sums: Dict[Optional[str], float] = {n: 0.0 for n in self.norm_item}
        err_counts: Dict[Optional[str], int] = {n: 0 for n in self.norm_item}

        for tid, track in self._build_tracks(results).items():
            pred, gt, mask = track['pred'], track['gt'], track['mask']
            for item in self.norm_item:
                nf = None if item is None else self._pair_scale(
                    track[item + '_scale'])
                s, c = keypoint_mpjve(pred, gt, mask, norm_factor=nf)
                err_sums[item] += s
                err_counts[item] += c

        for item in self.norm_item:
            key = _METRIC_PREFIX[item] + 'MPJVE'
            logger.info(f'Evaluating {key}...')
            metrics[key] = (float(err_sums[item] / err_counts[item])
                            if err_counts[item] else 0.0)

        return metrics


@METRICS.register_module()
class MPJAE(_TemporalBaseMetric):
    """Mean Per-Joint Acceleration Error (MPJAE) evaluation metric.

    Measures temporal consistency of predicted poses by comparing the
    second-order finite difference (acceleration) of predicted keypoint
    trajectories against the ground-truth acceleration.

    For a track of T frames the acceleration at frame t is:
        ``a_t = P_t - 2*P_{t-1} + P_{t-2}``

    The raw error is the mean Euclidean distance between predicted and
    ground-truth acceleration vectors over all visible joints and all
    valid frame triplets across every person-track in the dataset.

    An OKS-based matching step (threshold ``match_thr``) is applied in
    ``process()`` before accumulating predictions.  See :class:`MPJVE`
    for details.

    Optionally, errors can be normalised by a body-size scale factor via
    ``norm_item`` (see :class:`_TemporalBaseMetric`).  When multiple
    ``norm_item`` values are given, all variants are reported together.

    Returned metric names:

    - ``norm_item=None``      → ``'MPJAE'``
    - ``norm_item='bbox'``    → ``'bMPJAE'``
    - ``norm_item='head'``    → ``'hMPJAE'``
    - ``norm_item='torso'``   → ``'tMPJAE'``

    Args:
        norm_item (str | Sequence[str] | None): Normalisation method(s).
            Default: ``None`` (raw pixel-space error).
        match_thr (float): OKS threshold for pred-to-GT matching.
            Default: ``0.5``.
        collect_device (str): Device for distributed result collection.
            Default: ``'cpu'``.
        prefix (str, optional): Metric name prefix.  Default: ``None``.
    """

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute MPJAE (and normalised variants).

        Returns:
            Dict[str, float]: One entry per requested ``norm_item``.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        metrics: Dict[str, float] = {}

        err_sums: Dict[Optional[str], float] = {n: 0.0 for n in self.norm_item}
        err_counts: Dict[Optional[str], int] = {n: 0 for n in self.norm_item}

        for tid, track in self._build_tracks(results).items():
            pred, gt, mask = track['pred'], track['gt'], track['mask']
            for item in self.norm_item:
                nf = None if item is None else self._triplet_scale(
                    track[item + '_scale'])
                s, c = keypoint_mpjae(pred, gt, mask, norm_factor=nf)
                err_sums[item] += s
                err_counts[item] += c

        for item in self.norm_item:
            key = _METRIC_PREFIX[item] + 'MPJAE'
            logger.info(f'Evaluating {key}...')
            metrics[key] = (float(err_sums[item] / err_counts[item])
                            if err_counts[item] else 0.0)

        return metrics
