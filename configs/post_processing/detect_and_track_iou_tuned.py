# Detect-and-Track (CVPR 2018) - IoU matching at a detector-calibrated
# confidence threshold.
#
#   Girdhar et al., "Detect-and-Track: Efficient Pose Estimation in Videos"
#   https://github.com/facebookresearch/DetectAndTrack
#
# Identical to detect_and_track_iou.py except for
# `conf_filter_initial_dets`. Upstream's 0.95 was calibrated for the
# paper's Mask R-CNN; against the RF-DETR scores in these bundles it keeps
# only ~38% of EMDB and ~4% of PoseTrack21 detections, so the detection
# filter, not the matching stage, decides the result. 0.5 (this repo's
# usual working threshold) leaves enough detections for the linking stage
# to actually be measured.
#
# This is the control for detect_and_track_greedy_pck.py and
# detect_and_track_cnn.py, which use the same threshold: comparing against
# this config isolates the effect of the cost type / matching algorithm.
# It is a TUNED threshold, not the published one - see
# detect_and_track_iou.py for the released configuration.
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/detect_and_track_iou_tuned.py \
#       --postproc-name dat_iou_tuned

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[
        dict(
            type='DetectAndTrackLinker',
            cost_types=('bbox-overlap', ),
            cost_weights=(1.0, ),
            bipart_match_algo='hungarian',
            conf_filter_initial_dets=0.5,   # TUNED, upstream uses 0.95
            min_box_area=50.0,
            max_track_ids=999,
        ),
    ],
)
