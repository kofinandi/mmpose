# Detect-and-Track (CVPR 2018) - greedy matching with box IoU + pose PCKh.
#
#   Girdhar et al., "Detect-and-Track: Efficient Pose Estimation in Videos"
#   https://github.com/facebookresearch/DetectAndTrack
#
# Exercises the paper's other two matching options: the greedy bipartite
# matcher (a verbatim port of `bipartite_matching_greedy`) instead of the
# Hungarian algorithm, and the pose-similarity cost alongside box IoU.
#
# TUNED - `conf_filter_initial_dets` is 0.5 rather than upstream's 0.95,
# which was calibrated for the paper's Mask R-CNN and here would discard
# most detections before matching ever runs (see
# detect_and_track_iou_tuned.py, the matched-threshold IoU control for this
# config). detect_and_track_iou.py holds the released configuration.
#
# SUBSTITUTION - PCKh normalisation. Upstream normalises joint distances by
# the head size measured between the PoseTrack `head_top` and `head_bottom`
# joints. COCO-17 has neither, so `head_keypoint_ids=(3, 4)` uses the ears
# as the head extent, and `min_head_size_ratio` floors the result at 10% of
# the box diagonal - without a floor the ear pair collapses to ~0 px on a
# turned head, which would make every joint count as a PCK match. Set
# `min_head_size_ratio=0.0` to reproduce upstream's unfloored formula.
# The pose cost here is therefore not scaled the way the paper's is.
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/detect_and_track_greedy_pck.py \
#       --postproc-name dat_greedy_pck

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[
        dict(
            type='DetectAndTrackLinker',
            cost_types=('bbox-overlap', 'pose-pck'),
            cost_weights=(1.0, 1.0),
            bipart_match_algo='greedy',
            conf_filter_initial_dets=0.5,   # TUNED, upstream uses 0.95
            min_box_area=50.0,
            pck_dist_thresh=0.5,
            head_keypoint_ids=(3, 4),   # COCO ears, standing in for the head
            min_head_size_ratio=0.1,
            max_track_ids=999,
        ),
    ],
)
