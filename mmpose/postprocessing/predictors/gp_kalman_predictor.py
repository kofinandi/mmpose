# Copyright (c) OpenMMLab. All rights reserved.
"""GP-Kalman next-step predictor.

Implements a sliding-window, heteroscedastic Matern(nu=1.5) Gaussian
Process regression. The predictor operates on whole poses: every tracked
instance carries ``num_keypoints`` independent ``(x, y)`` signals, each
with its own sliding window and its own "age" (frames since the keypoint
was last observed).  This lets a track survive partial occlusion (e.g.
legs behind an object) while only the occluded keypoints go stale.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from ..registry import POST_PROCESS_PREDICTORS
from .base import BasePredictor, Prediction

_MATERN_SQRT3 = 3.0 ** 0.5


def _matern32(dist: np.ndarray, length_scale: float) -> np.ndarray:
    """Matern(nu=1.5) kernel, unit signal variance."""
    z = _MATERN_SQRT3 * dist / length_scale
    return (1.0 + z) * np.exp(-z)


def _gp_fit(
    t_buf: List[float],
    y_buf: List[float],
    v_buf: List[float],
    length_scale: float,
) -> tuple:
    """Fit a heteroscedastic Matern-3/2 GP to ``(t_buf, y_buf)`` with
    per-point noise ``v_buf``.

    Direct closed-form GP regression via Cholesky factorisation of
    ``K + diag(V)``, equivalent to a ``GaussianProcessRegressor(alpha=v_buf)``
    call but without per-call estimator overhead.

    Returns:
        A ``(cho_factor, alpha, y_mean)`` tuple that :func:`_gp_eval` can
        evaluate at any query point without refactorising.  Callers should
        cache and reuse this as long as ``(t_buf, y_buf, v_buf)`` is
        unchanged (e.g. a keypoint whose buffer is frozen while lost or
        occluded), and recompute it whenever the buffer is appended to.
    """
    n = len(t_buf)
    t_arr = np.asarray(t_buf, dtype=np.float64)
    y_arr = np.asarray(y_buf, dtype=np.float64)
    v_arr = np.asarray(v_buf, dtype=np.float64)
    y_mean = float(np.mean(y_arr))

    K = _matern32(np.abs(t_arr[:, None] - t_arr[None, :]), length_scale)
    K[np.diag_indices(n)] += v_arr

    jitter = 0.0
    for _ in range(5):
        try:
            c, lower = cho_factor(K + jitter * np.eye(n), lower=True)
            break
        except LinAlgError:
            jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
    else:
        raise LinAlgError('Cholesky factorization failed to stabilize')

    alpha = cho_solve((c, lower), y_arr - y_mean)
    return (c, lower), alpha, y_mean


def _gp_eval(
    t_buf: List[float],
    fit: tuple,
    t_query: float,
    length_scale: float,
) -> tuple:
    """Evaluate a GP posterior (from :func:`_gp_fit`) at ``t_query``.

    Returns:
        ``(mean, var)`` at ``t_query``, in the kernel's unit-signal-variance
        units.
    """
    (c, lower), alpha, y_mean = fit
    t_arr = np.asarray(t_buf, dtype=np.float64)

    k_star = _matern32(np.abs(t_query - t_arr), length_scale)  # (n,)
    mean = float(k_star @ alpha + y_mean)

    v = cho_solve((c, lower), k_star)  # (n,)
    var = 1.0 - float(np.dot(k_star, v))
    var = max(var, 1e-12)
    return mean, var


class _TrackState:
    """Per-track, per-keypoint sliding-window GP buffers."""

    def __init__(self, num_keypoints: int) -> None:
        K = num_keypoints
        # Buffers are shared between the x/y coordinates of a keypoint
        # (both observed/unobserved at the same times), but fit
        # independently as two separate 1-D GPs.
        self.t_buf: List[List[float]] = [[] for _ in range(K)]
        self.x_buf: List[List[float]] = [[] for _ in range(K)]
        self.y_buf: List[List[float]] = [[] for _ in range(K)]
        self.v_buf: List[List[float]] = [[] for _ in range(K)]
        self.age: np.ndarray = np.zeros(K, dtype=np.float32)
        self.last_mean: np.ndarray = np.zeros((K, 2), dtype=np.float32)
        # Cached (cho_factor, alpha, y_mean) per keypoint per coordinate.
        # `None` means dirty (must be refit on the next predict() call).
        # A keypoint's fit only changes when its buffer is appended to, so
        # a frozen (lost/occluded) keypoint reuses its cached factorisation
        # across consecutive frames instead of refitting an unchanged GP.
        self.fit_x: List[Optional[tuple]] = [None] * K
        self.fit_y: List[Optional[tuple]] = [None] * K


@POST_PROCESS_PREDICTORS.register_module()
class GPKalmanPredictor(BasePredictor):
    """Sliding-window GP next-step predictor, tracked per keypoint.

    Args:
        num_keypoints (int): Number of keypoints per instance.
            Default: ``17`` (COCO).
        window_size (int): Maximum number of buffered observations per
            keypoint. Default: ``20``.
        gp_length_scale (float): Matern-3/2 kernel length-scale, in frames.
            Default: ``16.0``.
        gp_signal_var (float): Prior variance used only when a keypoint has
            no buffered observations yet (should not normally occur once a
            track has been added, since :meth:`add_track` seeds one
            observation). Default: ``4000.0``.
    """

    # `Prediction.var` here is `1 - k_star^T (K + diag(v))^-1 k_star` (see
    # `_gp_eval`): a function of only the buffered *times* and injected
    # noise `v_buf`, never of the buffered x/y coordinate values (`alpha`,
    # which *is* linear in those values, only feeds into `mean`). It is
    # therefore already expressed in the kernel's own unit-signal-variance
    # scale and stays numerically identical no matter what coordinate units
    # (pixels, normalized [0, 1], ...) the caller's keypoints are in - it
    # must not be rescaled alongside `mean` when the caller
    # normalizes/denormalizes coordinates around this predictor.
    var_is_normalized = False

    def __init__(
        self,
        num_keypoints: int = 17,
        window_size: int = 20,
        gp_length_scale: float = 16.0,
        gp_signal_var: float = 4000.0,
    ) -> None:
        super().__init__(num_keypoints=num_keypoints)
        self.window_size = int(window_size)
        self.gp_length_scale = float(gp_length_scale)
        self.gp_signal_var = float(gp_signal_var)

        self._tracks: Dict[int, _TrackState] = {}
        self._t: int = 0

    def reset(self) -> None:
        self._tracks = {}
        self._t = 0

    def active_ids(self) -> list:
        return list(self._tracks.keys())

    def add_track(
        self,
        track_id: int,
        keypoints: np.ndarray,
        variances: np.ndarray,
    ) -> None:
        keypoints = np.asarray(keypoints, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)
        state = _TrackState(self.num_keypoints)
        for k in range(self.num_keypoints):
            state.t_buf[k] = [float(self._t)]
            state.x_buf[k] = [float(keypoints[k, 0])]
            state.y_buf[k] = [float(keypoints[k, 1])]
            state.v_buf[k] = [float(variances[k])]
            state.age[k] = 0.0
            state.last_mean[k] = keypoints[k]
        self._tracks[track_id] = state

    def predict(self, track_ids: list) -> Dict[int, Prediction]:
        self._t += 1
        out: Dict[int, Prediction] = {}
        for tid in track_ids:
            state = self._tracks.get(tid)
            if state is None:
                continue
            K = self.num_keypoints
            mean = np.zeros((K, 2), dtype=np.float32)
            var = np.zeros(K, dtype=np.float32)
            for k in range(K):
                if state.t_buf[k]:
                    if state.fit_x[k] is None:
                        state.fit_x[k] = _gp_fit(
                            state.t_buf[k], state.x_buf[k], state.v_buf[k],
                            self.gp_length_scale)
                    if state.fit_y[k] is None:
                        state.fit_y[k] = _gp_fit(
                            state.t_buf[k], state.y_buf[k], state.v_buf[k],
                            self.gp_length_scale)
                    mx, vx = _gp_eval(
                        state.t_buf[k], state.fit_x[k], float(self._t),
                        self.gp_length_scale)
                    my, vy = _gp_eval(
                        state.t_buf[k], state.fit_y[k], float(self._t),
                        self.gp_length_scale)
                    mean[k] = (mx, my)
                    var[k] = max(vx, vy)
                else:
                    mean[k] = state.last_mean[k]
                    var[k] = self.gp_signal_var
            out[tid] = Prediction(mean=mean, var=var, age=state.age.copy())
        return out

    def update(
        self,
        track_id: int,
        keypoints: Optional[np.ndarray],
        variances: Optional[np.ndarray],
        valid_mask: Optional[np.ndarray] = None,
    ) -> None:
        state = self._tracks.get(track_id)
        if state is None:
            return

        K = self.num_keypoints
        if keypoints is None:
            valid_mask = np.zeros(K, dtype=bool)
        elif valid_mask is None:
            valid_mask = np.ones(K, dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)

        keypoints = (
            np.asarray(keypoints, dtype=np.float64)
            if keypoints is not None else None)
        variances = (
            np.asarray(variances, dtype=np.float64)
            if variances is not None else None)

        for k in range(K):
            if valid_mask[k]:
                state.t_buf[k].append(float(self._t))
                state.x_buf[k].append(float(keypoints[k, 0]))
                state.y_buf[k].append(float(keypoints[k, 1]))
                state.v_buf[k].append(float(variances[k]))
                if len(state.t_buf[k]) > self.window_size:
                    state.t_buf[k].pop(0)
                    state.x_buf[k].pop(0)
                    state.y_buf[k].pop(0)
                    state.v_buf[k].pop(0)
                state.age[k] = 0.0
                state.last_mean[k] = keypoints[k]
                # Buffer changed: invalidate the cached factorisation.
                state.fit_x[k] = None
                state.fit_y[k] = None
            else:
                # Extrapolation (Step 4): freeze the buffer, age the
                # keypoint. Growing temporal distance to the frozen window
                # naturally inflates the GP's predicted variance.
                state.age[k] += 1.0

    def remove_track(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)
