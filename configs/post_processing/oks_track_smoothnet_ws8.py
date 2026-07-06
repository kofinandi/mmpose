# Post-processing pipeline: OKS tracker → SmoothNet smoother
#
# SmoothNetSmoother is OFFLINE (non-causal): it needs the full sequence
# before it can process anything.  The pipeline therefore buffers all frames
# during process() calls and runs the full filter chain in a single
# evaluate() call.
#
# Checkpoints are the official MMPose Human3.6M weights.
# Alternatives (change window_size, output_size AND checkpoint together):
#   ws=16 → smoothnet_ws16_h36m.pth
#   ws=32 → smoothnet_ws32_h36m.pth
#   ws=64 → smoothnet_ws64_h36m.pth
#
# root_index is set to None because COCO-17 has no canonical pelvis joint
# at a fixed index.  The H36M checkpoints are channel-independent (the
# window-size linear layer acts on time, not channels), so they run on
# COCO-17 2D coordinates without modification.
#
# Usage:
#   python tools/benchmark_e2e.py CONFIG CKPT \
#       --test-dataset emdb-mini \
#       --post-config configs/post_processing/oks_track_smoothnet.py
#
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/oks_track_smoothnet.py

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
            window_size=8,
            output_size=8,
            checkpoint=_SMOOTHNET_BASE + 'smoothnet_ws8_h36m.pth',
            hidden_size=512,
            res_hidden_size=256,
            num_blocks=3,
            root_index=None,
            device='cpu',
        ),
    ],
)
