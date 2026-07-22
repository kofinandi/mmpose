# Copyright (c) OpenMMLab. All rights reserved.
"""Base interface for next-step pose predictors (motion models).

A predictor is a modular motion-model component owned by a post-processing
filter (e.g. a SORT-style tracker).  It is responsible for:

* Fitting/maintaining a per-track, per-keypoint model of motion.
* Producing a next-step prediction (:class:`Prediction`) for every currently
  tracked instance.
* Updating its internal state once the post-processor has decided which
  detections (if any) matched which tracks.

The predictor knows nothing about data association, OKS, output fusion, or
track lifecycle policy (max age, variance thresholds, etc.) - all of that is
the responsibility of the post-processing filter that owns it.  This keeps
the predictor swappable: any alternative motion model (e.g. a constant
-velocity Kalman filter) can implement the same interface.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from mmengine.config import Config


@dataclass
class Prediction:
    """Next-step prediction for a single tracked instance.

    Attributes:
        mean: Predicted keypoint coordinates, shape ``(K, 2)``.  Expressed
            in whatever coordinate system the predictor was fed (see
            :class:`BasePredictor`) - callers such as
            :class:`~mmpose.postprocessing.filters.PredictiveTracker` that
            normalize coordinates before calling the predictor scale this
            back to their own units before use.
        var: Predicted variance per keypoint, shape ``(K,)``.  When the
            underlying model predicts per-coordinate variances they are
            combined (e.g. via ``max``) into a single scalar per keypoint,
            since keypoint-level lifecycle decisions (age/variance
            thresholds) operate per keypoint, not per coordinate.  Whether
            this is expressed in the same (normalized) coordinate units as
            ``mean``, and thus needs rescaling alongside it, is declared by
            :attr:`BasePredictor.var_is_normalized`.
        age: Number of frames since each keypoint was last updated with an
            observed measurement, shape ``(K,)``.  ``0`` means the keypoint
            was updated on the most recent call to :meth:`BasePredictor.update`.
    """

    mean: np.ndarray
    var: np.ndarray
    age: np.ndarray


class BasePredictor(metaclass=ABCMeta):
    """Abstract next-step predictor (motion model) for pose tracking.

    Implementations maintain one internal state per track id, keyed by an
    externally-assigned integer ``track_id``.  All methods are keypoint
    -vectorised: an implementation tracks ``num_keypoints`` independent
    (but not necessarily identically parameterised) 2-D signals per track.

    Coordinate units are whatever the caller passes in: callers such as
    :class:`~mmpose.postprocessing.filters.PredictiveTracker` normalize
    keypoint coordinates to ``[0, 1]`` (dividing by image width/height)
    before calling :meth:`add_track`/:meth:`update`, and scale
    :attr:`Prediction.mean` back to their own units after :meth:`predict`,
    so implementations are automatically resolution-agnostic without having
    to know about image size themselves.

    Args:
        num_keypoints (int): Number of keypoints tracked per instance.
            Default: ``17`` (COCO).
    """

    #: Whether :attr:`Prediction.var` (and the ``variances`` arguments to
    #: :meth:`add_track`/:meth:`update`) is expressed in the same
    #: normalized coordinate units as keypoint positions - i.e. scales
    #: like coordinate\ :sup:`2` (length\ :sup:`2`) - and must therefore be
    #: rescaled by ``scale**2`` alongside ``mean`` whenever the caller
    #: converts between normalized and pixel coordinates.
    #:
    #: This is the correct default for most motion models (e.g. a Kalman
    #: filter whose covariance is propagated in the same units as its
    #: state).  Override to ``False`` for a predictor whose variance is
    #: intrinsically decoupled from the coordinate values it was fed (e.g.
    #: a purely kernel-relative confidence that never actually depends on
    #: the buffered coordinate values.
    var_is_normalized: bool = True

    def __init__(self, num_keypoints: int = 17) -> None:
        self.num_keypoints = int(num_keypoints)

    def reset(self) -> None:
        """Drop all track state.  Called at sequence boundaries."""

    @abstractmethod
    def add_track(
        self,
        track_id: int,
        keypoints: np.ndarray,
        variances: np.ndarray,
    ) -> None:
        """Initialise state for a brand-new track.

        Args:
            track_id: Unique id assigned by the post-processor.
            keypoints: Initial keypoint coordinates, shape ``(K, 2)``.
            variances: Initial per-keypoint measurement variance,
                shape ``(K,)``.
        """

    @abstractmethod
    def predict(
        self,
        track_ids: list,
    ) -> Dict[int, Prediction]:
        """Advance one time step and predict for the given tracks.

        Args:
            track_ids: Track ids to predict for.  Ids without existing state
                are ignored (the caller is expected to only pass ids that
                were previously registered via :meth:`add_track`).

        Returns:
            Mapping from track id to :class:`Prediction`.
        """

    @abstractmethod
    def update(
        self,
        track_id: int,
        keypoints: Optional[np.ndarray],
        variances: Optional[np.ndarray],
        valid_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Fuse a new observation into a track's motion model.

        Args:
            track_id: Track to update.
            keypoints: Observed keypoint coordinates ``(K, 2)``, or ``None``
                if the track had no matching detection this frame (fully
                lost -- every keypoint is treated as unobserved).
            variances: Observed per-keypoint measurement variance ``(K,)``.
                Ignored when ``keypoints`` is ``None``.
            valid_mask: Boolean mask ``(K,)`` selecting which keypoints were
                actually observed this frame (e.g. ``False`` for keypoints
                occluded below a confidence threshold).  ``None`` means all
                keypoints in ``keypoints`` are observed.  Unobserved
                keypoints keep their buffered state frozen and have their
                age incremented, exactly like a fully lost track.
        """

    @abstractmethod
    def remove_track(self, track_id: int) -> None:
        """Discard all state for ``track_id``."""

    @abstractmethod
    def active_ids(self) -> list:
        """Return the list of track ids currently held by the predictor."""


def build_predictor(cfg) -> BasePredictor:
    """Build a :class:`BasePredictor` from a config dict.

    Args:
        cfg: A ``dict`` (or :class:`~mmengine.config.ConfigDict`) with a
            ``type`` key naming a registered predictor, plus its
            constructor kwargs.

    Returns:
        The built predictor instance.
    """
    from ..registry import POST_PROCESS_PREDICTORS

    if isinstance(cfg, Config):
        cfg = cfg.to_dict()
    cfg = dict(cfg)
    return POST_PROCESS_PREDICTORS.build(cfg)
