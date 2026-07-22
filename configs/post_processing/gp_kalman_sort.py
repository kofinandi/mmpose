# Post-processing pipeline: SORT-style tracker/smoother driven by a
# swappable next-step predictor, currently a GP-Kalman motion model.
#
# Architecture
# ------------
# PredictiveTracker (mmpose/postprocessing/filters/predictive_tracker.py):
#   - Owns a `predictor` submodule (motion model only).
#   - Runs data association (Hungarian algorithm on OKS, over each track's
#     currently "alive" keypoints only).
#   - Computes output poses (Bayesian fusion of prediction + detection for
#     matched/observed keypoints; raw prediction for unmatched-but-alive
#     keypoints; raw detection for brand-new tracks).
#   - Manages track lifecycle: per-keypoint aging/variance (handles partial
#     occlusion, e.g. legs behind an object) and whole-instance aging.
#
# GPKalmanPredictor (mmpose/postprocessing/predictors/gp_kalman_predictor.py):
#   - Fits a sliding-window, heteroscedastic Matern(nu=1.5) GP per keypoint
#     per coordinate and reports (mean, variance, age) back to the tracker.
#   - Knows nothing about association, fusion policy, or thresholds.
#
# This filter is fully online (causal): it processes and returns one frame
# at a time.
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/gp_kalman_sort.py

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[
        dict(
            type='PredictiveTracker',
            predictor=dict(
                type='GPKalmanPredictor',
                # Number of keypoints tracked per instance (COCO-17).
                num_keypoints=17,
                # Sliding-window length (frames) the GP is conditioned on.
                window_size=20,
                # Matern-3/2 kernel length-scale, in frames.
                gp_length_scale=16.0,
                # Prior variance used only when a keypoint has no buffered
                # observations yet (bootstrap fallback).
                gp_signal_var=4000.0,
            ),

            # ── Data association ────────────────────────────────────────
            # Minimum OKS to accept a track <-> detection match.
            match_thr=0.5,
            # Per-keypoint OKS sigmas. `None` -> COCO-17 defaults.
            sigmas=None,

            # ── Per-keypoint lifecycle (partial-occlusion handling) ─────
            # Minimum detection confidence for a keypoint to count as
            # "observed" this frame (below this it is treated as occluded).
            keypoint_score_thr=0.3,
            # A keypoint is no longer predicted/matched/output with a real
            # score once its age (frames since last observed) exceeds this.
            keypoint_max_age=15,

            # ── Whole-instance lifecycle ─────────────────────────────────
            # The whole track is discarded once every keypoint's age
            # exceeds this value.
            instance_max_age=30,
            # A track must be matched to a detection for at least this many
            # frames before it is trusted enough to be "remembered"
            # (predicted/extrapolated) while lost. Filters out one-off
            # phantom detections that would otherwise linger, predicted,
            # for up to `instance_max_age` frames after a single spurious
            # detection.
            min_hits_to_remember=3,

            # ── Measurement noise model (score -> variance) ──────────────
            # Converts a detection's keypoint confidence into a measurement
            # variance R (and, at fusion time, revises it using the
            # prediction/detection innovation). This is heuristic and how a
            # score maps to trustworthiness varies a lot between detector
            # architectures, so it's a swappable component - see
            # `mmpose/postprocessing/measurement/`.
            measurement_model=dict(
                type='PowerScoreMeasurementModel',
                # R = (1 - score)^score_exp * pixel_scale + min_r
                pixel_scale=1.0,
                min_r=3e-4,
                score_exp=8.0,
                # Innovation gating: large prediction/detection disagreement
                # inflates R by (1 + innovation^2 / inflation_factor).
                inflation_factor=8.0,
                # Extra R inflation when the innovation flips sign relative
                # to the previous frame (oscillatory jitter).
                osc_inflate=2.0,
            ),
        ),
    ],
)
