# Copyright (c) OpenMMLab. All rights reserved.
"""Base interface for pluggable measurement-noise models used by
:class:`~mmpose.postprocessing.filters.PredictiveTracker`.

A measurement model answers two closely related questions the tracker needs
in order to fuse a detection into a track's motion-model prediction:

1. How much do we trust a detection with a given keypoint confidence score?
   (:meth:`BaseMeasurementModel.variance`)
2. Given that base trust level, should it be revised because the detection
   disagrees with the prediction (innovation) - e.g. by a lot, or in a
   direction that flip-flops frame to frame (oscillation)?
   (:meth:`BaseMeasurementModel.inflate`)

Both are inherently heuristic/experimental, and how a detection score maps
to trustworthiness varies a lot between detector architectures (score
calibration differs wildly), so this is made a swappable, registered
component - just like
:class:`~mmpose.postprocessing.predictors.BasePredictor` - instead of being
hard-coded in the tracker.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np
from mmengine.config import Config


class BaseMeasurementModel(metaclass=ABCMeta):
    """Converts detection confidence (and, optionally, prediction/detection
    disagreement) into a measurement-noise variance for Bayesian fusion.

    All methods are vectorised over the ``K`` keypoints of a single
    instance.
    """

    @abstractmethod
    def variance(self, scores: np.ndarray) -> np.ndarray:
        """Base measurement variance from detection confidence.

        Args:
            scores: Per-keypoint detection confidence, shape ``(K,)``.

        Returns:
            Per-keypoint measurement variance ``R``, shape ``(K,)``.
        """

    def inflate(
        self,
        var: np.ndarray,
        innov: np.ndarray,
        prev_innov: np.ndarray,
        observed: np.ndarray,
    ) -> np.ndarray:
        """Optionally revise ``var`` using the fusion innovation.

        Called for every keypoint of a matched track (including keypoints
        not observed this frame, for which ``innov`` is meaningless and the
        result is discarded by the caller - see ``observed``). The default
        implementation is a no-op; override to add innovation-based
        gating/inflation.

        Args:
            var: Base measurement variance from :meth:`variance`,
                shape ``(K,)``.
            innov: This frame's innovation (detection - prediction),
                shape ``(K, 2)``.
            prev_innov: Previous frame's innovation for the same track,
                shape ``(K, 2)``.
            observed: Boolean mask, shape ``(K,)``, selecting which
                keypoints were actually observed this frame (i.e. for
                which ``innov`` is meaningful).

        Returns:
            Revised per-keypoint measurement variance, shape ``(K,)``.
        """
        return var


def build_measurement_model(cfg) -> BaseMeasurementModel:
    """Build a :class:`BaseMeasurementModel` from a config dict.

    Args:
        cfg: A ``dict`` (or :class:`~mmengine.config.ConfigDict`) with a
            ``type`` key naming a registered measurement model, plus its
            constructor kwargs.

    Returns:
        The built measurement model instance.
    """
    from ..registry import POST_PROCESS_MEASUREMENT_MODELS

    if isinstance(cfg, Config):
        cfg = cfg.to_dict()
    cfg = dict(cfg)
    return POST_PROCESS_MEASUREMENT_MODELS.build(cfg)
