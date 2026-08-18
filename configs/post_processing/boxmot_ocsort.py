# BoxMOT OC-SORT - motion-only bbox tracker over pose-estimator boxes.
#
#   Broström, "BoxMOT: Pluggable SOTA tracking modules"
#   https://github.com/mikel-brostrom/boxmot
#   OC-SORT: Cao et al., "Observation-Centric SORT: Rethinking SORT
#   for Robust Multi-Object Tracking", CVPR 2023.
#   https://arxiv.org/abs/2203.14360
#
# Association runs unmodified inside BoxMOT's Python OC-SORT.  This
# config does not enable ReID or camera-motion compensation; OC-SORT
# as published is motion + IoU only, so a dummy image of ori_shape is
# passed and no frame pixels are read.  Nothing is substituted.
#
# Detections and poses come from the prediction bundle rather than a
# BoxMOT-bundled YOLO.  Keypoints are carried through via det_ind and
# are never re-estimated.  Kalman-only tracks (no matching detection)
# are dropped because there is no pose to attach.
#
# BoxMOT's packaged OC-SORT YAML is used as-is (min_hits=3,
# frame_rate is not a YAML key here; max_age=30).  Tracks are not
# emitted until they collect min_hits observations, so the first few
# frames of each sequence may drop detections.
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/boxmot_ocsort.py \
#       --postproc-name boxmot_ocsort

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[
        dict(
            type='BoxMOTTracker',
            tracker='ocsort',
        ),
    ],
)
