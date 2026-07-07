"""
GP-Kalman Filter — core logic
==============================
Implements the Local Gaussian Process Regression Filter with Bayesian Fusion
described in GP_Kalman_Filter_Steps.md.

Public API
----------
    load_sequence(frames_json, sequence, joint_idx, coord)
        → (frame_ids, measurements, scores, gt_pos, gt_vis)

    run_filter(frame_ids, measurements, scores, *, window_size, n_forecast,
               pixel_scale, gp_length_scale, gp_signal_var,
               precompute_forecast=False)
        → list[dict]  (one record per time step)
"""

import bisect
import json
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Suppress the optimizer ConvergenceWarning – hyperparameters are always fixed
# (optimizer=None), so this warning should never fire; kept as a safety net.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

WINDOW_SIZE = 32
N_FORECAST  = 20
PIXEL_SCALE = 8.0
GP_LENGTH_SCALE = 12.0
GP_SIGNAL_VAR   = 2000.0
GATE_K = 12.0           # robust-outlier threshold, in units of scaled MAD
GATE_MAD_FLOOR = 5.0    # pixels; prevents the gate from tightening on a static buffer
GATE_INFLATE = 25.0     # variance inflation for suspected outliers (soft down-weight, not rejection)
MIN_SIGMA_P = 0.3       # floor on the GP predicted variance, prevents overconfidence stalling updates

# ── Noise model ────────────────────────────────────────────────────────────────

def measurement_variance(score: float) -> float:
    """
    Convert score to variance.
    """

    return (1.0 - score) ** 2 * PIXEL_SCALE


# ── GP helper ──────────────────────────────────────────────────────────────────

def gp_predict_at(
    T_buf: list,
    y_buf: list,
    V_buf: list,
    t_queries: list,
) -> tuple:
    """
    Fit a heteroscedastic GP to (T_buf, y_buf) with per-point noise V_buf,
    then evaluate at *t_queries*.

    The kernel is a fixed-hyperparameter RBF.  *V_buf* values are passed as
    ``alpha`` to GaussianProcessRegressor, implementing the diagonal noise
    matrix **V** from the filter specification.

    Manual mean-centering replaces ``normalize_y=True`` to avoid the std=0
    singularity when the buffer contains only one point.

    Returns
    -------
    means : np.ndarray  – posterior mean at each query point
    stds  : np.ndarray  – posterior std  at each query point
    """
    if not T_buf or not t_queries:
        n = len(t_queries)
        return np.zeros(n), np.full(n, GP_SIGNAL_VAR ** 0.5)

    T_arr = np.array(T_buf,     dtype=float).reshape(-1, 1)
    y_arr = np.array(y_buf,     dtype=float)
    V_arr = np.array(V_buf,     dtype=float)
    y_mean = float(np.mean(y_arr))

    kernel = Matern(GP_LENGTH_SCALE, length_scale_bounds="fixed", nu=1.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=V_arr,
        optimizer=None,       # hyperparameters are fixed; skip MLE optimisation
        normalize_y=False,
    )
    gp.fit(T_arr, y_arr - y_mean)

    t_q = np.array(t_queries, dtype=float).reshape(-1, 1)
    means, stds = gp.predict(t_q, return_std=True)
    return means + y_mean, stds


# ── Data loading ───────────────────────────────────────────────────────────────

def load_sequence(
    frames_json: str,
    sequence: str,
    joint_idx: int,
    coord: str,
) -> tuple:
    """
    Load per-frame predictions and ground truth for one sequence / joint /
    coordinate from a *frames.json* prediction file.

    Parameters
    ----------
    frames_json : path to the frames.json file produced by test_tracked.py
    sequence    : sequence prefix, e.g. ``"P1/14_outdoor_climb"``
    joint_idx   : 0-based COCO joint index
    coord       : ``"x"`` or ``"y"``

    Returns
    -------
    frame_ids  : list[int]
    measurements : list[float | None]   – predicted coordinate (None if no instance)
    scores     : list[float]            – keypoint confidence score
    gt_pos     : list[float | None]     – GT coordinate
    gt_vis     : list[float]            – GT visibility flag (0 or 1)
    """
    coord_idx = 0 if coord.lower() == "x" else 1

    print(f"Loading '{sequence}' …")
    with open(frames_json) as fh:
        raw = json.load(fh)

    seq_frames = [fr for fr in raw if fr["img_path"].startswith(sequence + "/")]
    seq_frames.sort(key=lambda fr: fr["frame_id"])

    frame_ids, meas, scores, gt_pos, gt_vis = [], [], [], [], []

    for fr in seq_frames:
        inst = fr["predictions"]["instances"]
        if inst:
            kp = inst[0]["keypoints"][joint_idx]
            sc = inst[0]["keypoint_scores"][joint_idx]
            meas.append(float(kp[coord_idx]))
            scores.append(float(sc))
        else:
            meas.append(None)
            scores.append(0.0)

        gt_inst = fr["ground_truth"]["instances"]
        if gt_inst:
            gt_pos.append(float(gt_inst[0]["keypoints"][joint_idx][coord_idx]))
            gt_vis.append(float(gt_inst[0]["keypoints_visible"][joint_idx]))
        else:
            gt_pos.append(None)
            gt_vis.append(0.0)

        frame_ids.append(fr["frame_id"])

    joint_name = KEYPOINT_NAMES[joint_idx] if joint_idx < len(KEYPOINT_NAMES) else str(joint_idx)
    print(f"  {len(frame_ids)} frames  |  joint {joint_idx} ({joint_name})  |  coord {coord.upper()}")
    return frame_ids, meas, scores, gt_pos, gt_vis


# ── Filter ─────────────────────────────────────────────────────────────────────

def run_filter(
    frame_ids: list,
    measurements: list,
    scores: list,
    *,
    precompute_forecast: bool = False,
    verbose: bool = False,
) -> list:
    """
    Run the GP-Kalman filter over the full sequence, using a symmetric
    (past + future) neighbor window for the GP prediction at each frame.
    This is an offline filter with the full sequence available up front, so
    unlike a causal running buffer, each frame's prediction can draw on
    nearby future measurements too — recovering information a forward-only
    filter loses to lag, in a single pass (one GP fit per frame, same as a
    causal window).

    Implements the four steps from GP_Kalman_Filter_Steps.md, with the
    buffer for Step 1 gathered symmetrically instead of causally:
      1. Heteroscedastic GP prediction
      2. Kalman gain
      3. Bayesian mean/variance update
      4. Extrapolation (no neighbors and no measurement at this frame)

    Parameters
    ----------
    frame_ids    : sorted list of integer frame identifiers
    measurements : per-frame predicted coordinate (float or None)
    scores       : per-frame keypoint confidence score
    precompute_forecast : if True, additionally evaluate the GP over the next
        *n_forecast* frames at every step (fwd_t, fwd_mus, fwd_stds), for the
        step-by-step viewer.  Default: False.

    Returns
    -------
    records : list of dicts, one per frame, with keys:
        t, z, R, score,
        mu_pred, sigma_pred,
        mu_post, sigma_post,
        T_post,
        fwd_t, fwd_mus, fwd_stds  (non-None only when precompute_forecast=True)
    """
    n_total = len(frame_ids)
    max_t = frame_ids[-1]
    half = WINDOW_SIZE // 2

    # Positions (indices into frame_ids/measurements) that have a real
    # detection, in ascending order — lets us binary-search for the nearest
    # past/future neighbors of any frame without rescanning the sequence.
    valid_positions = [i for i, z in enumerate(measurements) if z is not None]

    records: list = []

    for idx, t in enumerate(frame_ids):
        if verbose and idx % 100 == 0:
            print(f"  filtering … {idx}/{n_total}", end="\r", flush=True)

        z = measurements[idx]
        R = measurement_variance(scores[idx]) if z is not None else None

        pos = bisect.bisect_left(valid_positions, idx)
        left_positions  = valid_positions[max(0, pos - half):pos]
        right_start     = pos + 1 if (pos < len(valid_positions) and valid_positions[pos] == idx) else pos
        right_positions = valid_positions[right_start:right_start + half]
        neighbor_positions = left_positions + right_positions

        T_buf = [frame_ids[i] for i in neighbor_positions]
        z_buf = [measurements[i] for i in neighbor_positions]
        R_buf = [measurement_variance(scores[i]) for i in neighbor_positions]

        # ── Step 1: Prediction (heteroscedastic GP, symmetric neighbors) ─────
        if T_buf:
            (mu_p,), (std_p,) = gp_predict_at(T_buf, z_buf, R_buf, [t])
            sigma_p = max(float(std_p ** 2), MIN_SIGMA_P)
            mu_p    = float(mu_p)
        else:
            # No neighbors at all (isolated detection or empty sequence)
            mu_p    = float(z) if z is not None else 0.0
            sigma_p = GP_SIGNAL_VAR

        # ── Outlier soft-gate: down-weight (never reject) measurements that
        # fall far outside the neighbor buffer's own robust (MAD-based)
        # spread, in raw pixel units.
        if z is not None and len(z_buf) >= 3:
            arr = np.array(z_buf)
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            robust_std = max(mad * 1.4826, GATE_MAD_FLOOR)
            if abs(z - med) > GATE_K * robust_std:
                R = R * GATE_INFLATE

        # ── Steps 2 & 3: Bayesian update (measurement available) ─────────────
        if z is not None:
            K_gain  = sigma_p / (sigma_p + R)
            mu_t    = mu_p + K_gain * (z - mu_p)
            sigma_t = (1.0 - K_gain) * sigma_p
        else:
            # ── Step 4: Extrapolation (no measurement at this frame) ──────────
            mu_t    = mu_p
            sigma_t = sigma_p

        # ── GP forward projection (only when requested) ───────────────────────
        fwd_t: list = []
        fwd_mus = fwd_stds = None
        if precompute_forecast and T_buf:
            fwd_t = list(range(t, min(max_t + 1, t + N_FORECAST + 1)))
            fwd_mus, fwd_stds = gp_predict_at(T_buf, z_buf, R_buf, fwd_t)

        records.append({
            "t":          t,
            "z":          z,
            "R":          R,
            "score":      scores[idx],
            "mu_pred":    mu_p,
            "sigma_pred": sigma_p,
            "mu_post":    mu_t,
            "sigma_post": sigma_t,
            "T_post":     T_buf,   # neighbor snapshot for the viewer
            "fwd_t":      fwd_t,
            "fwd_mus":    fwd_mus,
            "fwd_stds":   fwd_stds,
        })

    if verbose:
        print(f"  filtering … {n_total}/{n_total}  ✓")
    return records
