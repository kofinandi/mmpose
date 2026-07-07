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

import json

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

WINDOW_SIZE = 20
N_FORECAST  = 20
PIXEL_SCALE = 1.0
GP_LENGTH_SCALE = 16.0
GP_SIGNAL_VAR   = 4000.0
INFLATION_FACTOR = 8.0
MIN_R = 0.0003

# ── Noise model ────────────────────────────────────────────────────────────────

def measurement_variance(score: float) -> float:
    """
    Convert score to variance.
    """

    # This exponent seems to provide a steep enough curve to separate low-confidence measurements from high-confidence ones.
    # MIN_R: typical one-step GP predictive variance (sigma_pred) is ~0.0026,
    # while R for high-score points is often <<1e-5 — the Kalman gain is then
    # ~1 (near-total pass-through of the raw, jittery detection). A small
    # floor well below sigma_pred nudges the gain down slightly for the most
    # "confident" points without collapsing trust the way a larger floor did.
    return (1.0 - score) ** 6 * PIXEL_SCALE + MIN_R


# ── GP helper ──────────────────────────────────────────────────────────────────

_MATERN_SQRT3 = 3.0 ** 0.5


def _matern32(dist: np.ndarray) -> np.ndarray:
    """Matérn(nu=1.5) kernel, unit signal variance, GP_LENGTH_SCALE lengthscale."""
    z = _MATERN_SQRT3 * dist / GP_LENGTH_SCALE
    return (1.0 + z) * np.exp(-z)


def gp_predict_at(
    T_buf: list,
    y_buf: list,
    V_buf: list,
    t_queries: list,
) -> tuple:
    """
    Fit a heteroscedastic GP to (T_buf, y_buf) with per-point noise V_buf,
    then evaluate at *t_queries*.

    Direct closed-form Matern(nu=1.5) GP regression via Cholesky, equivalent
    to the previous sklearn GaussianProcessRegressor(alpha=V_buf) call but
    without its per-call validation/construction overhead — the buffers here
    are at most WINDOW_SIZE long, so the linear algebra itself is cheap and
    sklearn's generic-estimator overhead dominated runtime.

    *V_buf* values sit on the kernel-matrix diagonal, implementing the
    heteroscedastic noise matrix **V** from the filter specification.

    Returns
    -------
    means : np.ndarray  – posterior mean at each query point
    stds  : np.ndarray  – posterior std  at each query point
    """
    if not T_buf or not t_queries:
        n = len(t_queries)
        return np.zeros(n), np.full(n, GP_SIGNAL_VAR ** 0.5)

    T_arr = np.array(T_buf, dtype=float)
    y_arr = np.array(y_buf, dtype=float)
    V_arr = np.array(V_buf, dtype=float)
    y_mean = float(np.mean(y_arr))
    n = T_arr.shape[0]

    K = _matern32(np.abs(T_arr[:, None] - T_arr[None, :]))
    K[np.diag_indices(n)] += V_arr

    jitter = 0.0
    for _ in range(5):
        try:
            c, lower = cho_factor(K + jitter * np.eye(n), lower=True)
            break
        except LinAlgError:
            jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
    else:
        raise LinAlgError("Cholesky factorization failed to stabilize")

    alpha = cho_solve((c, lower), y_arr - y_mean)

    t_q = np.array(t_queries, dtype=float)
    k_star = _matern32(np.abs(t_q[:, None] - T_arr[None, :]))  # (n_q, n)

    means = k_star @ alpha + y_mean

    v = cho_solve((c, lower), k_star.T)  # (n, n_q), K^-1 @ k_star.T
    var = 1.0 - np.sum(k_star.T * v, axis=0)
    stds = np.sqrt(np.maximum(var, 1e-12))
    return means, stds


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
    Run the GP-Kalman filter over the full sequence.

    Implements all four steps from GP_Kalman_Filter_Steps.md:
      1. Heteroscedastic GP prediction
      2. Kalman gain
      3. Bayesian mean/variance update
      4. Extrapolation (no-measurement case, buffer frozen)

    Parameters
    ----------
    frame_ids    : sorted list of integer frame identifiers
    measurements : per-frame predicted coordinate (float or None)
    scores       : per-frame keypoint confidence score
    precompute_forecast : if True, evaluate the GP over the next *n_forecast*
        frames at every step and store the results in each record (fwd_t,
        fwd_mus, fwd_stds).  Enables instant rendering in the step-by-step
        viewer at the cost of roughly doubling GP evaluations.  Default: False.

    Returns
    -------
    records : list of dicts, one per frame, with keys:
        t, z, R, score,
        mu_pred, sigma_pred,
        mu_post, sigma_post,
        T_post,
        fwd_t, fwd_mus, fwd_stds  (non-None only when precompute_forecast=True)
    """
    T_buf: list = []   # time steps of buffered measurements
    z_buf: list = []   # raw measurements  z_t  (NOT posteriors)
    R_buf: list = []   # measurement variances  R_t

    records: list = []
    max_t   = frame_ids[-1]
    n_total = len(frame_ids)

    for idx, t in enumerate(frame_ids):
        if verbose and idx % 100 == 0:
            print(f"  filtering … {idx}/{n_total}", end="\r", flush=True)

        z = measurements[idx]

        # ── Step 1: Prediction (heteroscedastic GP time update) ──────────────
        if T_buf:
            (mu_p,), (std_p,) = gp_predict_at(T_buf, z_buf, R_buf, [t])
            sigma_p = float(std_p ** 2)
            mu_p    = float(mu_p)
        else:
            # Bootstrap: no history yet → prior centred on the first measurement
            mu_p    = float(z) if z is not None else 0.0
            sigma_p = GP_SIGNAL_VAR

        # ── Steps 2 & 3: Bayesian update (measurement available) ─────────────
        if z is not None:
            innovation = z - mu_p
            R = measurement_variance(scores[idx])
            R = R * (1 + (innovation ** 2) / INFLATION_FACTOR)
            K_gain  = sigma_p / (sigma_p + R)
            mu_t    = mu_p + K_gain * innovation
            sigma_t = (1.0 - K_gain) * sigma_p

            # Store the raw measurement (not the posterior) to keep the GP
            # training data independent of its own predictions.
            T_buf.append(t);  z_buf.append(z);  R_buf.append(R)
            if len(T_buf) > WINDOW_SIZE:
                T_buf.pop(0);  z_buf.pop(0);  R_buf.pop(0)

        else:
            # ── Step 4: Extrapolation (no measurement, buffer frozen) ─────────
            mu_t    = mu_p
            sigma_t = sigma_p

        # ── GP forward projection (only when requested) ───────────────────────
        fwd_t: list = []
        fwd_mus = fwd_stds = None
        if precompute_forecast and T_buf:
            fwd_t = list(range(t, min(max_t + 1, t + N_FORECAST + 1)))
            fwd_mus, fwd_stds = gp_predict_at(
                list(T_buf), list(z_buf), list(R_buf), fwd_t
            )

        records.append({
            "t":          t,
            "z":          z,
            "R":          R,
            "score":      scores[idx],
            "innovation": innovation,
            "mu_pred":    mu_p,
            "sigma_pred": sigma_p,
            "mu_post":    mu_t,
            "sigma_post": sigma_t,
            "T_post":     list(T_buf),   # window snapshot for the viewer
            "fwd_t":      fwd_t,
            "fwd_mus":    fwd_mus,
            "fwd_stds":   fwd_stds,
        })

    if verbose:
        print(f"  filtering … {n_total}/{n_total}  ✓")
    return records
