# Copyright (c) OpenMMLab. All rights reserved.
"""Per-joint One-Euro filter smoother for tracked pose sequences.

Ported from the historical MMPose v0.x implementation at commit 137e8648
(mmpose/core/post_processing/temporal_filters/one_euro_filter.py).
"""

from __future__ import annotations

import copy
import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

from mmpose.structures import PoseDataSample

from ..base import BaseFilter
from ..registry import POST_PROCESS_FILTERS


# ---------------------------------------------------------------------------
# One-Euro filter math
# ---------------------------------------------------------------------------

def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


def _exp_smooth(
    alpha: np.ndarray,
    x: np.ndarray,
    x_prev: np.ndarray,
) -> np.ndarray:
    return alpha * x + (1.0 - alpha) * x_prev


class _OneEuroState:
    """Per-track, per-joint One-Euro filter state.

    Operates on arrays of shape ``(K, 2)`` — one entry per keypoint.

    Args:
        x0: Initial position ``(K, 2)``.
        min_cutoff: Minimum cutoff frequency (reduces slow jitter).
        beta: Speed coefficient (reduces lag on fast motion).
        d_cutoff: Fixed derivative cutoff frequency.  Default: ``1.0``.
    """

    def __init__(
        self,
        x0: np.ndarray,
        min_cutoff: float = 0.004,
        beta: float = 0.7,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_prev = x0.astype(np.float64)
        self.dx_prev = np.zeros_like(self.x_prev)
        self.t_prev: float = 0.0  # frame counter

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the filter to a new observation.

        Args:
            x: Current keypoint positions ``(K, 2)``.

        Returns:
            Filtered positions ``(K, 2)`` (float32).
        """
        x = x.astype(np.float64)
        t = self.t_prev + 1.0
        t_e = t - self.t_prev

        # Derivative estimate
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = _exp_smooth(a_d, dx, self.dx_prev)

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exp_smooth(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat.astype(np.float32)


# ---------------------------------------------------------------------------
# Filter class
# ---------------------------------------------------------------------------

@POST_PROCESS_FILTERS.register_module()
class OneEuroSmoother(BaseFilter):
    """Per-joint One-Euro filter smoother.

    Applies an adaptive exponential smoothing filter to the ``(x, y)``
    coordinates of each keypoint for each tracked instance independently.
    A fresh :class:`_OneEuroState` is created the first time a ``track_id``
    is seen; state is reset at sequence boundaries.

    Requires ``pred_instances.track_ids`` to be set (i.e. the output of
    :class:`~mmpose.postprocessing.filters.OKSTracker`).  Raises
    ``RuntimeError`` on the first frame that lacks ``track_ids`` if that frame
    has at least one instance.

    Only ``keypoints[:, :, :2]`` (coordinates) are modified; scores, bboxes,
    and all other fields are left intact.

    Args:
        min_cutoff (float): Lower cutoff → more smoothing at slow speeds.
            Default: ``0.004``.
        beta (float): Speed coefficient → larger → less lag at high speeds.
            Default: ``0.7``.
        d_cutoff (float): Fixed derivative cutoff frequency.  Default: ``1.0``.
    """

    online = True

    def __init__(
        self,
        min_cutoff: float = 0.004,
        beta: float = 0.7,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._track_states: Dict[int, _OneEuroState] = {}

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

        N = len(instances)

        # Validate that track_ids exist
        if not hasattr(instances, 'track_ids') or instances.track_ids is None:
            raise RuntimeError(
                'OneEuroSmoother requires pred_instances.track_ids. '
                'Make sure OKSTracker (or another tracker) runs before it.')

        track_ids = np.asarray(instances.track_ids, dtype=np.int32)
        kpts = np.asarray(instances.keypoints, dtype=np.float32).copy()  # (N, K, 2)

        for i in range(N):
            tid = int(track_ids[i])
            x_i = kpts[i]  # (K, 2)

            if tid not in self._track_states:
                self._track_states[tid] = _OneEuroState(
                    x0=x_i,
                    min_cutoff=self.min_cutoff,
                    beta=self.beta,
                    d_cutoff=self.d_cutoff,
                )
                # First observation: filter returns the initialisation value
                kpts[i] = x_i
            else:
                kpts[i] = self._track_states[tid](x_i)

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
