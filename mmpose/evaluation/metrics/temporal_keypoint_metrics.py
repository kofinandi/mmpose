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

        collect_device (str): Device for distributed result collection.
            Default: ``'cpu'``.
        prefix (str, optional): Metric name prefix.  Default: ``None``.
    """

    default_prefix: Optional[str] = 'temporal'

    _ALLOWED_NORM = ('bbox', 'head', 'torso')

    def __init__(self,
                 norm_item: Union[str, Sequence[str], None] = None,
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

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Accumulate per-sample prediction/GT pairs.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
                Unused; present for API compatibility.
            data_samples (Sequence[dict]): Model outputs for a batch.
        """
        for data_sample in data_samples:
            pred_coords = data_sample['pred_instances']['keypoints']
            if pred_coords.ndim == 3 and pred_coords.shape[0] == 1:
                pred_coords = pred_coords[0]

            gt = data_sample['gt_instances']
            gt_coords = gt['keypoints']
            if gt_coords.ndim == 3 and gt_coords.shape[0] == 1:
                gt_coords = gt_coords[0]

            mask = gt['keypoints_visible'].astype(bool)
            if mask.ndim == 2 and mask.shape[0] == 1:
                mask = mask[0]

            img_path = data_sample['img_path']
            frame_id = _parse_frame_id(img_path)
            track_id = data_sample['raw_ann_info']['track_id']

            result = dict(
                track_id=track_id,
                frame_id=frame_id,
                pred_coords=pred_coords,
                gt_coords=gt_coords,
                mask=mask,
            )

            if 'bbox' in self.norm_item:
                assert 'bboxes' in gt, (
                    'The ground truth data info does not have the expected '
                    'normalized_item ``"bbox"``.')
                bbox_size = float(
                    np.max(gt['bboxes'][0][2:] - gt['bboxes'][0][:2]))
                result['bbox_size'] = bbox_size

            if 'head' in self.norm_item:
                assert 'head_size' in gt, (
                    'The ground truth data info does not have the expected '
                    'normalized_item ``"head_size"``.')
                result['head_size'] = float(gt['head_size'])

            if 'torso' in self.norm_item:
                # Per-frame torso size: mean of cross-torso diagonals.
                sizes = []
                for i, j in _TORSO_KP_PAIRS:
                    if mask[i] and mask[j]:
                        sizes.append(
                            float(np.linalg.norm(gt_coords[i] - gt_coords[j])))
                if not sizes:
                    # Fall back to predicted keypoints (mirrors PCKAccuracy).
                    for i, j in _TORSO_KP_PAIRS:
                        sizes.append(
                            float(
                                np.linalg.norm(pred_coords[i] -
                                               pred_coords[j])))
                    warnings.warn(
                        'Ground truth torso size < 1 for at least one frame. '
                        'Using predicted keypoints to estimate torso size.')
                torso_size = float(np.mean(sizes)) if sizes else 1.0
                result['torso_size'] = max(torso_size, 1.0)

            self.results.append(result)

    def _build_tracks(
        self, results: List[dict]
    ) -> Dict[int, Dict[str, object]]:
        """Group and sort per-sample results into per-track dicts.

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
            entry: dict = dict(
                pred=np.stack([f['pred_coords'] for f in frames]),
                gt=np.stack([f['gt_coords'] for f in frames]),
                mask=np.stack([f['mask'] for f in frames]),
            )
            if 'bbox_size' in frames[0]:
                entry['bbox_scale'] = np.array(
                    [f['bbox_size'] for f in frames])
            if 'head_size' in frames[0]:
                entry['head_scale'] = np.array(
                    [f['head_size'] for f in frames])
            if 'torso_size' in frames[0]:
                entry['torso_scale'] = np.array(
                    [f['torso_size'] for f in frames])
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

    Optionally, errors can be normalised by a body-size scale factor via
    ``norm_item`` (see :class:`_TemporalBaseMetric`).  When multiple
    ``norm_item`` values are given, all variants are reported together.

    Returned metric names:

    - ``norm_item=None``      → ``'MPJVE'``
    - ``norm_item='bbox'``    → ``'bMPJVE'``
    - ``norm_item='head'``    → ``'hMPJVE'``
    - ``norm_item='torso'``   → ``'tMPJVE'``

    .. note::
        Requires ``track_id`` in ``raw_ann_info`` and ``img_path`` following
        the 3DPW naming convention ``*/<seq_name>/image_XXXXX.jpg``.

    Args:
        norm_item (str | Sequence[str] | None): Normalisation method(s).
            Default: ``None`` (raw pixel-space error).
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

    Optionally, errors can be normalised by a body-size scale factor via
    ``norm_item`` (see :class:`_TemporalBaseMetric`).  When multiple
    ``norm_item`` values are given, all variants are reported together.

    Returned metric names:

    - ``norm_item=None``      → ``'MPJAE'``
    - ``norm_item='bbox'``    → ``'bMPJAE'``
    - ``norm_item='head'``    → ``'hMPJAE'``
    - ``norm_item='torso'``   → ``'tMPJAE'``

    .. note::
        Requires ``track_id`` in ``raw_ann_info`` and ``img_path`` following
        the 3DPW naming convention ``*/<seq_name>/image_XXXXX.jpg``.

    Args:
        norm_item (str | Sequence[str] | None): Normalisation method(s).
            Default: ``None`` (raw pixel-space error).
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
