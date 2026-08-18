# BoxMOT ByteTrack - motion-only bbox tracker over pose-estimator boxes.
#
#   Broström, "BoxMOT: Pluggable SOTA tracking modules"
#   https://github.com/mikel-brostrom/boxmot
#   ByteTrack: Zhang et al., "ByteTrack: Multi-Object Tracking by
#   Associating Every Detection Box", ECCV 2022.
#   https://arxiv.org/abs/2110.06864
#
# Association runs unmodified inside BoxMOT's Python ByteTrack.  This
# config does not enable ReID or camera-motion compensation; ByteTrack
# as published is motion + IoU only, so a dummy image of ori_shape is
# passed and no frame pixels are read.  Nothing is substituted.
#
# Detections and poses come from the prediction bundle rather than a
# BoxMOT-bundled YOLO.  Keypoints are carried through via det_ind and
# are never re-estimated.  Kalman-only tracks (no matching detection)
# are dropped because there is no pose to attach.
#
# BoxMOT's packaged ByteTrack YAML is used as-is (track_thresh=0.6,
# frame_rate=30).  PoseTrack21 clips are ~25 fps; override with
# tracker_kwargs=dict(frame_rate=25) if you want buffer lifetimes in
# wall-clock seconds rather than BoxMOT's 30 fps assumption.
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/boxmot_bytetrack.py \
#       --postproc-name boxmot_bytetrack

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[
        dict(
            type='BoxMOTTracker',
            tracker='bytetrack',
        ),
    ],
)
