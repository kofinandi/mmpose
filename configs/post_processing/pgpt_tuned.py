# PGPT (IEEE TMM 2020) - association cascade with a calibrated appearance gate.
#
#   Zhang et al., "Pose-Guided Tracking-by-Detection: Robust Multi-Person
#   Pose Tracking"
#   https://github.com/JDAI-CV/PGPT
#
# Identical to pgpt.py except for `embedding_match_thresh`. Everything in
# that config's header (which embedding variant runs, no SiamFC
# propagation, oks_filter/create_id reading the detection's existing
# keypoints) applies here unchanged.
#
# TUNED - `embedding_match_thresh` is 0.001 rather than upstream's 2.0.
# Upstream's value gates a cosine distance, but the released PoseGCN
# weights produce distances around 0.0002 (same person) and 0.0052
# (different person), so a gate of 2.0 never rejects anything: the
# Hungarian stage force-matches every leftover detection to some leftover
# track. That has two measurable effects, both of which this config undoes:
#
#   1. It bypasses the create_id admission rule. A force-matched detection
#      is emitted with the matched track's id without having to clear
#      `effective_detection_thresh` / `effective_keypoints_number`, so
#      ~12k extra (mostly false-positive) detections reach the output on
#      PoseTrack21 - raising AP/recall while lowering precision.
#   2. It contaminates track identities. ID switches drop (labels stay
#      alive) while association precision falls sharply - AssPr 0.49 ->
#      0.40 on PoseTrack21 - because tracks absorb detections belonging to
#      other people and to unlabeled background.
#
# Calibration: over 306 same-identity and 3378 different-identity pairs
# built from PoseTrack21 val GT boxes, the balanced-accuracy optimum is
# 0.00069 (80.7% of same-identity pairs accepted, 87.5% of different-identity
# pairs rejected). The distributions overlap substantially, so this is a
# genuinely tuned knob rather than a clean separation. 0.001 is shipped
# because it scored best on end-to-end tracking metrics over a 2000-frame
# PoseTrack21 slice (IDF1 0.5132 vs 0.5089 at 0.00069, HOTA tied at 0.380),
# beating both the published gate (IDF1 0.4988) and the appearance-free
# ablation (0.5072).
#
# Use pgpt.py when you want the published configuration, and this one when
# you want the appearance stage to actually discriminate. Neither is a
# reproduction of the paper's numbers - see pgpt.py on the unreleased graph
# branch and the absent SiamFC stage.
#
# This config needs the source frames, hence `needs_images=True`.
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/pgpt_tuned.py \
#       --postproc-name pgpt_tuned

post_processor = dict(
    type='PostProcessingPipeline',
    needs_images=True,
    filters=[
        dict(
            type='PGPTTracker',
            appearance_embedder=dict(
                type='PGPTPoseGCNEmbedder',
                checkpoint='data/models/pgpt_pose_gcn.pth.tar',
                variant='pose_gated',
                device='cuda:0',
                flip_test=True,
            ),
            iou_match_thresh=0.5,
            embedding_match_thresh=0.001,   # TUNED, upstream uses 2.0
            oks_thresh=0.8,
            oks_in_vis_thre=0.2,
            effective_detection_thresh=0.5,
            effective_keypoints_thresh=0.6,
            effective_keypoints_number=8,
            keypoint_score_thr=0.2,
        ),
    ],
)
