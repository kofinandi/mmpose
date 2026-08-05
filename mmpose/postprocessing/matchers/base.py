# Copyright (c) OpenMMLab. All rights reserved.
"""Base interfaces for the two similarity components trackers plug in.

Data association needs to answer "are these two detections the same person?".
Published trackers answer it in two distinct ways, which are kept as separate
swappable components here:

* :class:`BasePoseMatcher` - judges *pose shape* alone (keypoints + box),
  never sees pixels.  LightTrack's Siamese GCN and a box-normalised L2
  baseline are implementations.
* :class:`BaseAppearanceEmbedder` - judges *appearance*, so it needs the
  frame's pixels.  Detect-and-Track's ResNet crop features and PGPT's
  PoseGCN embedding are implementations.

The split matters for memory and for the pipeline's image contract: an
embedder only touches pixels inside :meth:`BaseAppearanceEmbedder.embed`,
which callers invoke once per frame.  Trackers then cache the returned
*feature vectors* across frames and never hold on to an image (see
:class:`~mmpose.postprocessing.base.BaseFilter` for why).

Both mirror the swappable-submodule pattern already used for
:class:`~mmpose.postprocessing.predictors.BasePredictor` and
:class:`~mmpose.postprocessing.measurement.BaseMeasurementModel`.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
from mmengine.config import Config


class BasePoseMatcher(metaclass=ABCMeta):
    """Distance between two poses, judged on pose shape alone.

    Implementations deliberately know nothing about track lifecycle,
    thresholds, or the order in which candidates are tried - that is the job
    of the filter that owns the matcher.
    """

    #: Returned when a pose pair cannot be compared at all (e.g. a
    #: degenerate box).  Callers treat it as "never match".
    INVALID_DISTANCE: float = float('inf')

    def reset(self) -> None:
        """Drop any cached state.  Called at sequence boundaries."""

    @abstractmethod
    def distance(
        self,
        kpts_a: np.ndarray,
        bbox_a: np.ndarray,
        kpts_b: np.ndarray,
        bbox_b: np.ndarray,
        scores_a: Optional[np.ndarray] = None,
        scores_b: Optional[np.ndarray] = None,
    ) -> float:
        """Distance between two poses.

        Args:
            kpts_a: First pose, shape ``(K, 2)``, in pixel coordinates.
            bbox_a: ``xyxy`` box of the first pose, shape ``(4,)``.
            kpts_b: Second pose, shape ``(K, 2)``.
            bbox_b: ``xyxy`` box of the second pose, shape ``(4,)``.
            scores_a: Optional per-keypoint confidence ``(K,)`` for the
                first pose.  Implementations may ignore it.
            scores_b: Optional per-keypoint confidence ``(K,)`` for the
                second pose.

        Returns:
            Non-negative distance; lower means more similar.
            :attr:`INVALID_DISTANCE` when the pair cannot be compared.
        """

    def distance_matrix(
        self,
        kpts_a: np.ndarray,
        bboxes_a: np.ndarray,
        kpts_b: np.ndarray,
        bboxes_b: np.ndarray,
        scores_a: Optional[np.ndarray] = None,
        scores_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Pairwise distances between two sets of poses.

        The default implementation loops over :meth:`distance`.  Override
        when an implementation can batch the work (e.g. embed each pose once
        and then compare embeddings, instead of embedding every pair).

        Args:
            kpts_a: ``(M, K, 2)`` poses.
            bboxes_a: ``(M, 4)`` boxes.
            kpts_b: ``(N, K, 2)`` poses.
            bboxes_b: ``(N, 4)`` boxes.
            scores_a: Optional ``(M, K)`` confidences.
            scores_b: Optional ``(N, K)`` confidences.

        Returns:
            ``(M, N)`` distance matrix.
        """
        m = len(kpts_a)
        n = len(kpts_b)
        out = np.full((m, n), self.INVALID_DISTANCE, dtype=np.float32)
        for i in range(m):
            sa = None if scores_a is None else scores_a[i]
            for j in range(n):
                sb = None if scores_b is None else scores_b[j]
                out[i, j] = self.distance(
                    kpts_a[i], bboxes_a[i], kpts_b[j], bboxes_b[j], sa, sb)
        return out


class BaseAppearanceEmbedder(metaclass=ABCMeta):
    """Maps image crops of detections to appearance feature vectors.

    Implementations are stateless with respect to pixels: :meth:`embed` is
    the only method that sees an image, and the caller keeps the returned
    vectors.  This is what lets an image-consuming tracker satisfy the
    pipeline's "never retain ``ds.img``" rule.
    """

    def reset(self) -> None:
        """Drop any cached state.  Called at sequence boundaries."""

    @abstractmethod
    def embed(self, image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """Embed every box's crop of ``image``.

        Args:
            image: Frame pixels, BGR ``(H, W, 3)`` ``uint8``, in the same
                coordinate space as ``bboxes``.
            bboxes: ``(N, 4)`` boxes in ``xyxy`` format.

        Returns:
            ``(N, D)`` float32 feature matrix, row ``i`` describing
            ``bboxes[i]``.  Returns shape ``(0, D)`` when ``N == 0``.
        """

    def distance_matrix(
        self,
        feats_a: np.ndarray,
        feats_b: np.ndarray,
    ) -> np.ndarray:
        """Pairwise distances between two sets of feature vectors.

        Defaults to cosine distance, which both Detect-and-Track and PGPT
        use over their respective features.

        Args:
            feats_a: ``(M, D)`` features.
            feats_b: ``(N, D)`` features.

        Returns:
            ``(M, N)`` distance matrix.
        """
        from scipy.spatial.distance import cdist

        if len(feats_a) == 0 or len(feats_b) == 0:
            return np.zeros((len(feats_a), len(feats_b)), dtype=np.float32)
        return cdist(feats_a, feats_b, metric='cosine').astype(np.float32)


def build_pose_matcher(cfg) -> BasePoseMatcher:
    """Build a :class:`BasePoseMatcher` from a config dict.

    Args:
        cfg: A ``dict`` (or :class:`~mmengine.config.ConfigDict`) with a
            ``type`` key naming a registered pose matcher, plus its
            constructor kwargs.

    Returns:
        The built pose matcher instance.
    """
    from ..registry import POST_PROCESS_POSE_MATCHERS

    if isinstance(cfg, Config):
        cfg = cfg.to_dict()
    cfg = dict(cfg)
    return POST_PROCESS_POSE_MATCHERS.build(cfg)


def build_appearance_embedder(cfg) -> BaseAppearanceEmbedder:
    """Build a :class:`BaseAppearanceEmbedder` from a config dict.

    Args:
        cfg: A ``dict`` (or :class:`~mmengine.config.ConfigDict`) with a
            ``type`` key naming a registered appearance embedder, plus its
            constructor kwargs.

    Returns:
        The built appearance embedder instance.
    """
    from ..registry import POST_PROCESS_APPEARANCE_EMBEDDERS

    if isinstance(cfg, Config):
        cfg = cfg.to_dict()
    cfg = dict(cfg)
    return POST_PROCESS_APPEARANCE_EMBEDDERS.build(cfg)
