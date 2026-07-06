# Post-processing pipeline: OKS tracker → SmoothNet smoother (window size 32)
#
# SmoothNetSmoother is OFFLINE (non-causal): it needs the full sequence
# before it can process anything.  The pipeline therefore buffers all frames
# during process() calls and runs the full filter chain in a single
# evaluate() call.
#
# This config uses the ws=32 Human3.6M checkpoint.  Tracks shorter than 32
# frames are returned unchanged.  For a smaller window see
# oks_track_smoothnet.py (ws=8).
#
# Usage:
#   python tools/benchmark_e2e.py CONFIG CKPT \
#       --test-dataset emdb-mini \
#       --post-config configs/post_processing/oks_track_smoothnet_ws32.py
#
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/oks_track_smoothnet_ws32.py

_SMOOTHNET_BASE = (
    'https://download.openmmlab.com/mmpose/plugin/smoothnet/')

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[
        dict(
            type='OKSTracker',
            match_thr=0.5,
        ),
        dict(
            type='SmoothNetSmoother',
            window_size=32,
            output_size=32,
            checkpoint=_SMOOTHNET_BASE + 'smoothnet_ws32_h36m.pth',
            hidden_size=512,
            res_hidden_size=256,
            num_blocks=3,
            root_index=None,
            device='cpu',
        ),
    ],
)
