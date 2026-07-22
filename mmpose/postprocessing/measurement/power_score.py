# Copyright (c) OpenMMLab. All rights reserved.
"""Default measurement model: a power-law score->variance curve with
innovation-based gating/inflation."""

from __future__ import annotations

import numpy as np

from ..registry import POST_PROCESS_MEASUREMENT_MODELS
from .base import BaseMeasurementModel


@POST_PROCESS_MEASUREMENT_MODELS.register_module()
class PowerScoreMeasurementModel(BaseMeasurementModel):
    """Power-law score->variance model with innovation-based inflation.

    ``R = (1 - score)^score_exp * pixel_scale + min_r``, further inflated
    by ``(1 + innovation^2 / inflation_factor)``, and by an extra
    ``osc_inflate`` factor when the innovation flips sign relative to the
    previous frame (oscillatory jitter).

    Args:
        pixel_scale (float): Scale of the score->variance curve.
        min_r (float): Noise floor of the score->variance curve.
        score_exp (float): Exponent of the score->variance curve.
        inflation_factor (float): Innovation-gating scale: large
            prediction/detection disagreement inflates ``R``.
        osc_inflate (float): Extra ``R`` inflation when the innovation
            flips sign relative to the previous frame (oscillation).
    """

    def __init__(
        self,
        pixel_scale: float = 1.0,
        min_r: float = 3e-4,
        score_exp: float = 8.0,
        inflation_factor: float = 8.0,
        osc_inflate: float = 2.0,
    ) -> None:
        self.pixel_scale = float(pixel_scale)
        self.min_r = float(min_r)
        self.score_exp = float(score_exp)
        self.inflation_factor = float(inflation_factor)
        self.osc_inflate = float(osc_inflate)

    def variance(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        return (1.0 - scores)**self.score_exp * self.pixel_scale + self.min_r

    def inflate(
        self,
        var: np.ndarray,
        innov: np.ndarray,
        prev_innov: np.ndarray,
        observed: np.ndarray,
    ) -> np.ndarray:
        innov = np.asarray(innov, dtype=np.float64)
        prev_innov = np.asarray(prev_innov, dtype=np.float64)

        nu2 = np.sum(innov**2, axis=-1)
        var = var * (1.0 + nu2 / self.inflation_factor)

        osc = np.sum(prev_innov * innov, axis=-1) < 0
        return np.where(osc, var * self.osc_inflate, var)
