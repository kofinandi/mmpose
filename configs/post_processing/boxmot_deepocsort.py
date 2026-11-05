# BoxMOT Deep OC-SORT - bbox tracker with adaptive ReID and camera motion.
#
#   Broström, "BoxMOT: Pluggable SOTA tracking modules"
#   https://github.com/mikel-brostrom/boxmot
#   Deep OC-SORT: Maggiolino et al., "Deep OC-SORT: Multi-Pedestrian
#   Tracking by Adaptive Re-Identification", ICIP 2023.
#   https://arxiv.org/abs/2302.11813
#
# Association, camera-motion compensation, and adaptive appearance
# weighting run inside BoxMOT's Python DeepOcSort.  BoxMOT's packaged
# YAML is used as-is (det_thresh=0.5, max_age=30, min_hits=3,
# w_association_emb=0.75, alpha_fixed_emb=0.95, aw_param=0.5;
# embedding, CMC, and adaptive weighting left on).
#
# Appearance embeddings are BoxMOT's OSNet-x0.25 MSMT17 checkpoint, the
# same file as configs/post_processing/boxmot_botsort.py.  That is not
# the official Deep OC-SORT appearance model: the release loads
# dataset-specific FastReID SBS-S50 weights (mot17_sbs_S50.pth /
# mot20_sbs_S50.pth).  Those files are not used here.
#
# Detections and poses come from the prediction bundle rather than
# YOLOX.  Keypoints are carried through via det_ind and are never
# re-estimated.  Kalman-only tracks (no matching detection) are dropped
# because there is no pose to attach.  The official linear-interpolation
# step (`--post`) is not applied.
#
# The ReID file is already at data/models/boxmot/osnet_x0_25_msmt17.pt.
# BoxMOT downloads it on first use if it is missing (Google Drive id
# 1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF):
#
#   mkdir -p data/models/boxmot
#   python -m gdown 1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF \
#       -O data/models/boxmot/osnet_x0_25_msmt17.pt
#
# This config needs the source frames, hence `needs_images=True`.
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/boxmot_deepocsort.py \
#       --postproc-name boxmot_deepocsort

post_processor = dict(
    type='PostProcessingPipeline',
    needs_images=True,
    filters=[
        dict(
            type='BoxMOTTracker',
            tracker='deepocsort',
            reid_weights='data/models/boxmot/osnet_x0_25_msmt17.pt',
            device='cuda:0',
        ),
    ],
)
