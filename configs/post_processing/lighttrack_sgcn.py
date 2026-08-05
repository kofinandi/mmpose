# LightTrack (CVPRW 2020) - spatial consistency + Siamese-GCN pose matching.
#
#   Ning et al., "LightTrack: A Generic Framework for Online Top-Down Human
#   Pose Tracking"
#   https://github.com/Guanghan/lighttrack
#
# The paper's identity-association stage: greedy IoU against the previous
# frame, then its Siamese Graph Convolutional Network as the Re-ID fallback
# for whatever the IoU pass missed. The SGCN network is imported unmodified
# from the submodule; only the surrounding cascade is ported.
#
# RETRAINED WEIGHTS - the published checkpoint (weights/GCN/epoch210_model.pt
# from GCN.zip) is no longer downloadable: guanghan.info 404s, there is no
# Wayback snapshot, the training-pair tarball is gone too, and upstream
# issue #21 is still open. The checkpoint below is produced by
#
#   python tools/train_lighttrack_sgcn.py \
#       --out data/models/lighttrack_sgcn_posetrack21.pt
#
# which retrains the same architecture with upstream's recipe
# (graph/config/train.yaml) on PoseTrack21. Results from this config are
# therefore NOT a reproduction of the paper's numbers.
#
# APPROXIMATION - the SGCN graph has head_bottom and head_top nodes that
# COCO-17 does not provide. Both are synthesised from the nose and
# shoulders by matchers.synthesize_head_joints, and training uses the same
# rule, so the network is never asked to generalise across representations
# (see mmpose/postprocessing/matchers/keypoint_maps.py for the fit).
#
# ABSENT - keyframe scheduling, pose-guided box propagation and the
# is_target_lost re-detection trigger. All need to re-run a detector or
# pose estimator mid-sequence, which is out of scope: detections and poses
# already exist on every frame. That makes this upstream's
# keyframe_interval=1 limiting case.
#
# `pose_matching_threshold` is upstream's 0.5, but on the retrained
# embedding the distance scale is set by the contrastive margin, so treat
# it as a tunable rather than a published constant.
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/lighttrack_sgcn.py \
#       --postproc-name lighttrack_sgcn

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[
        dict(
            type='LightTrackTracker',
            pose_matcher=dict(
                type='SGCNPoseMatcher',
                checkpoint='data/models/lighttrack_sgcn_posetrack21.pt',
                device='cuda:0',
            ),
            pose_matching_threshold=0.5,
            spatial_consistency_thr=0.3,   # upstream hard-codes 0.3
            enlarge_scale=0.2,
            score_thr=0.4,                 # upstream detector threshold
        ),
    ],
)
