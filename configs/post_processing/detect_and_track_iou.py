# Detect-and-Track (CVPR 2018) - released tracking configuration.
#
#   Girdhar et al., "Detect-and-Track: Efficient Pose Estimation in Videos"
#   https://github.com/facebookresearch/DetectAndTrack
#
# This is the upstream *released* setting: `DISTANCE_METRICS` lists three
# costs but `DISTANCE_METRIC_WTS = (1.0, 0.0, 0.0)` zeroes all but box IoU,
# and `01_R101_best_hungarian.yaml` raises the detection filter to 0.95.
# Of the three configs shipped here this is the one to compare against the
# paper: no substituted component takes part in it.
#
# Detections and poses come from the prediction bundle rather than the
# paper's 3D Mask R-CNN, which is the integration point, not a substitution
# - upstream also tracks post-hoc over a saved detections.pkl.
#
# The LSTM matcher variant is not implemented: its weights were never
# released and the `lstm` package is absent from the upstream repository.
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/detect_and_track_iou.py \
#       --postproc-name dat_iou

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[
        dict(
            type='DetectAndTrackLinker',
            cost_types=('bbox-overlap', ),
            cost_weights=(1.0, ),
            bipart_match_algo='hungarian',
            conf_filter_initial_dets=0.95,
            min_box_area=50.0,
            max_track_ids=999,
        ),
    ],
)
