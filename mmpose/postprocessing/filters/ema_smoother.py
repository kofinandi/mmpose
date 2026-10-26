# Copyright (c) OpenMMLab. All rights reserved.
"""Per-joint exponential moving average smoother for tracked pose sequences.

The update is the causal EMA

    y_t = alpha * x_t + (1 - alpha) * y_{t-1}

with ``y_0 = x_0``.  ``alpha`` closer to 1 tracks the observation more
closely; ``alpha`` closer to 0 smooths more and lags more.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict

import numpy as np

from mmpose.structures import PoseDataSample

from ..base import BaseFilter
from ..registry import POST_PROCESS_FILTERS


class _EMAState:
    """Per-track exponential moving average on arrays of shape ``(K, 2)``.

    Args:
        x0: Initial position ``(K, 2)``.  Stored as the first filtered value.
        alpha: Observation weight in ``(0, 1]``.
    """

    def __init__(self, x0: np.ndarray, alpha: float) -> None:
        self.alpha = float(alpha)
        self.y_prev = x0.astype(np.float64)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply one EMA step.

        Args:
            x: Current keypoint positions ``(K, 2)``.

        Returns:
            Filtered positions ``(K, 2)`` (float32).
        """
        x = x.astype(np.float64)
        y = self.alpha * x + (1.0 - self.alpha) * self.y_prev
        self.y_prev = y
        return y.astype(np.float32)


@POST_PROCESS_FILTERS.register_module()
class EMASmoother(BaseFilter):
    """Per-joint exponential moving average smoother.

    Applies a fixed-weight exponential moving average to the ``(x, y)``
    coordinates of each keypoint, independently for each tracked instance.
    A fresh :class:`_EMAState` is created the first time a ``track_id`` is
    seen; state is reset at sequence boundaries.

    Requires ``pred_instances.track_ids`` to be set (i.e. the output of
    :class:`~mmpose.postprocessing.filters.OKSTracker`).  Raises
    ``RuntimeError`` on the first frame that lacks ``track_ids`` if that
    frame has at least one instance.

    Only ``keypoints[..., :2]`` (coordinates) are modified; scores, bboxes,
    and all other fields are left intact.  The first observation of a track
    is returned unchanged.

    Args:
        alpha (float): Weight on the current observation, in ``(0, 1]``.
            ``1`` disables smoothing.  Smaller values smooth more and lag
            more on fast motion.  Default: ``0.5``.
    """

    online = True

    def __init__(self, alpha: float = 0.5) -> None:
        alpha = float(alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(
                f'EMASmoother alpha must be in (0, 1], got {alpha}.')
        self.alpha = alpha
        self._track_states: Dict[int, _EMAState] = {}

    def reset(self) -> None:
        self._track_states.clear()

    def process_frame(
        self,
        ds: PoseDataSample,
        seq_key: str,
    ) -> PoseDataSample:
        instances = ds.pred_instances
        if instances is None or len(instances) == 0:
            return ds

        if not hasattr(instances, 'track_ids') or instances.track_ids is None:
            raise RuntimeError(
                'EMASmoother requires pred_instances.track_ids. '
                'Make sure OKSTracker (or another tracker) runs before it.')

        track_ids = np.asarray(instances.track_ids, dtype=np.int32)
        kpts = np.asarray(instances.keypoints, dtype=np.float32).copy()

        for i in range(len(instances)):
            tid = int(track_ids[i])
            xy = kpts[i, :, :2]

            if tid not in self._track_states:
                self._track_states[tid] = _EMAState(x0=xy, alpha=self.alpha)
            else:
                kpts[i, :, :2] = self._track_states[tid](xy)

        return self._replace_keypoints(ds, kpts)

    @staticmethod
    def _replace_keypoints(
        ds: PoseDataSample,
        new_kpts: np.ndarray,
    ) -> PoseDataSample:
        """Return a copy of *ds* with updated keypoints."""
        new_ds = ds.new()
        new_ds.set_metainfo(ds.metainfo)
        if hasattr(ds, 'gt_instances'):
            new_ds.gt_instances = ds.gt_instances
        # deepcopy keeps _data_fields independent from the original ds
        new_inst = deepcopy(ds.pred_instances)
        new_inst.keypoints = new_kpts
        new_ds.pred_instances = new_inst
        return new_ds
