# Copyright (c) OpenMMLab. All rights reserved.
"""Box-normalised keypoint L2 distance - a learning-free pose matcher."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..registry import POST_PROCESS_POSE_MATCHERS
from .base import BasePoseMatcher


@POST_PROCESS_POSE_MATCHERS.register_module()
class NormalizedL2PoseMatcher(BasePoseMatcher):
    """Mean L2 distance between two poses in box-relative coordinates.

    Each pose is expressed in ``[0, 1]`` coordinates relative to its own
    box, which removes translation and scale, so what remains is a pure
    comparison of pose *shape*.  Distances are therefore roughly in
    ``[0, 1]`` and comparable across people at different depths.

    This is a geometric stand-in wherever a learned pose-similarity model is
    unavailable (see :class:`SGCNPoseMatcher`).  It has no identity signal:
    two different people in the same posture look identical to it.

    Args:
        keypoint_score_thr: Keypoints scoring below this in *either* pose
            are excluded from the mean.  Set ``0`` to use all keypoints.
        min_valid_keypoints: If fewer than this many keypoints survive the
            score threshold, the pair is reported as
            :attr:`~BasePoseMatcher.INVALID_DISTANCE`.
    """

    def __init__(
        self,
        keypoint_score_thr: float = 0.3,
        min_valid_keypoints: int = 3,
    ) -> None:
        self.keypoint_score_thr = float(keypoint_score_thr)
        self.min_valid_keypoints = int(min_valid_keypoints)

    @staticmethod
    def _normalize(
        kpts: np.ndarray,
        bbox: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Express ``kpts`` in ``[0, 1]`` box-relative coordinates.

        Returns ``None`` for a degenerate box, which the caller turns into
        :attr:`~BasePoseMatcher.INVALID_DISTANCE`.
        """
        x1, y1, x2, y2 = (float(v) for v in np.asarray(bbox).reshape(-1)[:4])
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return None
        out = np.empty_like(np.asarray(kpts, dtype=np.float32))
        out[:, 0] = (kpts[:, 0] - x1) / w
        out[:, 1] = (kpts[:, 1] - y1) / h
        return out

    def distance(
        self,
        kpts_a: np.ndarray,
        bbox_a: np.ndarray,
        kpts_b: np.ndarray,
        bbox_b: np.ndarray,
        scores_a: Optional[np.ndarray] = None,
        scores_b: Optional[np.ndarray] = None,
    ) -> float:
        norm_a = self._normalize(np.asarray(kpts_a, dtype=np.float32), bbox_a)
        norm_b = self._normalize(np.asarray(kpts_b, dtype=np.float32), bbox_b)
        if norm_a is None or norm_b is None:
            return self.INVALID_DISTANCE

        valid = np.ones(len(norm_a), dtype=bool)
        if self.keypoint_score_thr > 0:
            if scores_a is not None:
                valid &= np.asarray(scores_a) >= self.keypoint_score_thr
            if scores_b is not None:
                valid &= np.asarray(scores_b) >= self.keypoint_score_thr

        if int(valid.sum()) < self.min_valid_keypoints:
            return self.INVALID_DISTANCE

        diff = norm_a[valid] - norm_b[valid]
        return float(np.mean(np.linalg.norm(diff, axis=-1)))
