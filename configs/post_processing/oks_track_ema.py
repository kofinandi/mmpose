# Post-processing pipeline: OKS tracker → EMA smoother
#
# Both filters are online (causal), so process() returns a result immediately
# for each frame with no buffering.
#
# alpha is the weight on the current keypoint.  Lower alpha smooths more
# and lags more; alpha=1 leaves the tracked poses unchanged.
#
# Usage:
#   python tools/benchmark_e2e.py CONFIG CKPT \
#       --test-dataset emdb-mini \
#       --post-config configs/post_processing/oks_track_ema.py
#
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/oks_track_ema.py

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[
        dict(
            type='OKSTracker',
            # Minimum OKS score to accept a match between consecutive frames.
            # Instances below this threshold are treated as new tracks.
            match_thr=0.5,
        ),
        dict(
            type='EMASmoother',
            alpha=0.4,
        ),
    ],
)
