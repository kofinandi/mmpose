# BoxMOT BoT-SORT - bbox tracker with ReID and camera-motion compensation.
#
#   Broström, "BoxMOT: Pluggable SOTA tracking modules"
#   https://github.com/mikel-brostrom/boxmot
#   BoT-SORT: Aharon et al., "BoT-SORT: Robust Associations Multi-
#   Pedestrian Tracking", arXiv 2022.  https://arxiv.org/abs/2206.14651
#
# Association, CMC and ReID all run unmodified inside BoxMOT's Python
# BoT-SORT.  This is the published BoT-SORT pipeline wrapped as a
# post-processor: appearance embeddings come from BoxMOT's OSNet
# checkpoint, not a substitute.
#
# Detections and poses come from the prediction bundle rather than a
# BoxMOT-bundled YOLO.  Keypoints are carried through via det_ind and
# are never re-estimated.  Kalman-only tracks (no matching detection)
# are dropped because there is no pose to attach.
#
# The ReID weights are auto-downloaded by BoxMOT on first use if the
# file is missing (Google Drive id 1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF).
# Fetch them once with:
#
#   mkdir -p data/models/boxmot
#   python -m gdown 1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF \
#       -O data/models/boxmot/osnet_x0_25_msmt17.pt
#
# This config needs the source frames, hence `needs_images=True`.
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/boxmot_botsort.py \
#       --postproc-name boxmot_botsort

post_processor = dict(
    type='PostProcessingPipeline',
    needs_images=True,
    filters=[
        dict(
            type='BoxMOTTracker',
            tracker='botsort',
            reid_weights='data/models/boxmot/osnet_x0_25_msmt17.pt',
            device='cuda:0',
        ),
    ],
)
