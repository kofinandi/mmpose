# LightTrack (CVPRW 2020) - spatial consistency + geometric pose matching.
#
#   Ning et al., "LightTrack: A Generic Framework for Online Top-Down Human
#   Pose Tracking"
#   https://github.com/Guanghan/lighttrack
#
# LightTrack's cascade with its Siamese-GCN Re-ID replaced by a
# box-normalised keypoint L2 distance. Runs with no checkpoint at all, and
# serves as the ablation that shows what the learned matcher contributes
# over plain pose geometry.
#
# SUBSTITUTED - the pose-matching pass. NormalizedL2PoseMatcher compares
# pose shape only and carries no identity signal: two different people in
# the same posture are identical to it. This config therefore does NOT
# measure the paper's Re-ID module. Use lighttrack_sgcn.py for that.
#
# `pose_matching_threshold` is TUNED for the L2 distance scale (roughly
# [0, 1] in box-relative units) and bears no relation to upstream's 0.5,
# which applies to the SGCN's embedding distance.
#
# The scope notes in lighttrack_sgcn.py (no keyframe scheduling, no
# pose-guided box propagation, no re-detection trigger) apply here too.
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/lighttrack_l2.py \
#       --postproc-name lighttrack_l2

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[
        dict(
            type='LightTrackTracker',
            pose_matcher=dict(
                type='NormalizedL2PoseMatcher',
                keypoint_score_thr=0.3,
                min_valid_keypoints=3,
            ),
            pose_matching_threshold=0.2,   # TUNED for the L2 distance scale
            spatial_consistency_thr=0.3,   # upstream hard-codes 0.3
            enlarge_scale=0.2,
            score_thr=0.4,                 # upstream detector threshold
        ),
    ],
)
