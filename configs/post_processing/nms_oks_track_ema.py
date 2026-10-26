# Post-processing pipeline: OKS-NMS → OKS tracker → EMA smoother
#
# All three filters are online (causal), so process() returns a result
# immediately for each frame with no buffering.
#
# alpha is the weight on the current keypoint.  Lower alpha smooths more
# and lags more; alpha=1 leaves the tracked poses unchanged.
#
# Usage:
#   python tools/benchmark_e2e.py CONFIG CKPT \
#       --test-dataset emdb-mini \
#       --post-config configs/post_processing/nms_oks_track_ema.py
#
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/nms_oks_track_ema.py

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[
        dict(
            type='OKSNMS',
            score_thr=0.3,
            oks_thr=0.9,
            score_mode='auto',
        ),
        dict(
            type='OKSTracker',
            match_thr=0.5,
            remember_frames=30,
        ),
        dict(
            type='EMASmoother',
            alpha=0.4,
        ),
    ],
)
