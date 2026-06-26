# Post-processing pipeline: OKS tracker → One-Euro smoother
#
# Both filters are online (causal), so process() returns a result immediately
# for each frame with no buffering.
#
# Usage:
#   python tools/benchmark_e2e.py CONFIG CKPT \
#       --test-dataset emdb-mini \
#       --post-config configs/post_processing/oks_track_one_euro.py
#
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/oks_track_one_euro.py

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
            type='OneEuroSmoother',
            # Low min_cutoff → more smoothing on slow/static joints.
            min_cutoff=0.001,
            # Higher beta → less lag when joints move fast.
            beta=0.001,
        ),
    ],
)
