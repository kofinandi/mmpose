# PGPT (IEEE TMM 2020) - association cascade with PoseGCN appearance.
#
#   Zhang et al., "Pose-Guided Tracking-by-Detection: Robust Multi-Person
#   Pose Tracking"
#   https://github.com/JDAI-CV/PGPT
#
# The paper's identity-association cascade: re-score detections by their
# pose confidence and OKS-NMS them, Hungarian-match to live tracks on box
# IoU, fall back to the learned appearance embedding for the leftovers,
# then apply the create_id admission rule and delete unmatched tracks.
# All thresholds are upstream's class defaults on Track_And_Detect.
#
# The PoseGCN network is imported unmodified from the submodule and loaded
# with the published pose_gcn.pth.tar. Fetch it once with:
#
#   python -m gdown 1emHrW4OFFOndmR5OIUfHDq4xf8yhdPjR \
#       -O data/models/pgpt_pose_gcn.pth.tar
#
# WHICH EMBEDDING RUNS - the released checkpoint contains no weights for
# graph_layer1/2 or fc_feature_align, i.e. the graph-convolution branch
# that PoseNet.embedding nominally calls (flag=1) was never released.
# Upstream's loader leaves those layers randomly initialised and runs them
# anyway. This config uses variant='pose_gated' (upstream flag=2), the
# pose-conditioned embedding the checkpoint does cover: the appearance map
# is gated by the pose branch and pooled to 2048-D. Selecting
# variant='graph' raises unless real graph weights are supplied.
#
# ABSENT - SiamFC track propagation. Upstream merges each live track's
# single-object-tracker box into the candidate list (score decayed by 0.35)
# before association. Tracker-generated boxes are out of scope, so only
# real detections are associated. With upstream's immediate deletion of
# unmatched tracks, a person occluded for one frame gets a new id; expect
# id churn, and read it as a property of this configuration.
#
# SUBSTITUTED - oks_filter and create_id re-run PGPT's pose network on each
# candidate box upstream; here the detection's existing keypoints and
# scores are used.
#
# NOTE on embedding_match_thresh: upstream's default of 2 gates a cosine
# distance, but measured PoseGCN distances are ~0.005, so the gate never
# rejects a pair. It is left at the published value rather than silently
# retuned - lower it to make the appearance stage selective.
#
# This config needs the source frames, hence `needs_images=True`. Run it
# with tools/postprocess_predictions.py.
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/pgpt.py \
#       --postproc-name pgpt

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
            embedding_match_thresh=2.0,
            oks_thresh=0.8,
            oks_in_vis_thre=0.2,
            effective_detection_thresh=0.5,
            effective_keypoints_thresh=0.6,
            effective_keypoints_number=8,
            keypoint_score_thr=0.2,
        ),
    ],
)
