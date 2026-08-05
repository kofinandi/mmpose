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
# because it edged 0.00069 on end-to-end tracking metrics over a 2000-frame
# PoseTrack21 slice (IDF1 0.5132 vs 0.5089, HOTA tied at 0.380).
#
# Full-bundle results (PoseTrack21 val, 20161 frames):
#
#   config       AP      MOTA     IDF1    HOTA    IDSw
#   pgpt         0.3112  -0.2601  0.4766  0.3589   941   (published gate 2.0)
#   pgpt_tuned   0.2960  -0.2021  0.4867  0.3727   837   (this config)
#   pgpt_geom    0.2913  -0.1858  0.4876  0.3787  1140   (no appearance)
#
# So the calibrated gate clearly beats the published one on every metric
# except AP - and the AP drop is the point, since the published gate's AP
# came from bypassing the admission rule. Against the appearance-free
# ablation the picture is mixed: this config gives the fewest ID switches of
# any variant (837 vs 1140, a 27% reduction), but ties it on IDF1 and still
# trails slightly on HOTA and MOTA. On a 2000-frame slice the tuned gate did
# beat pgpt_geom on IDF1; that ordering does not hold over the full bundle,
# so read the appearance stage's value as identity *continuity* rather than
# an across-the-board gain.
#
# Use pgpt.py when you want the published configuration, this one when you
# want the appearance stage to discriminate, and pgpt_geom.py when you want
# the cascade's geometry alone. None is a reproduction of the paper's
# numbers - see pgpt.py on the unreleased graph branch and the absent
# SiamFC stage.
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
