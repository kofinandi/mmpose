# PGPT (IEEE TMM 2020) - association cascade without the appearance stage.
#
#   Zhang et al., "Pose-Guided Tracking-by-Detection: Robust Multi-Person
#   Pose Tracking"
#   https://github.com/JDAI-CV/PGPT
#
# PGPT's cascade with the PoseGCN embedding stage switched off: OKS-NMS
# over pose-rescored detections, Hungarian IoU matching, the create_id
# admission rule and immediate deletion of unmatched tracks. Needs no
# checkpoint and no images.
#
# This measures the GEOMETRIC SKELETON of the cascade, not PGPT: the
# paper's learned Re-ID contribution is exactly the stage removed here.
# Use pgpt.py for the full method, and compare against this config to see
# what the appearance embedding is worth.
#
# The scope notes in pgpt.py (no SiamFC propagation; oks_filter/create_id
# read the detection's existing keypoints rather than re-running the pose
# network) apply here too.
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/pgpt_geom.py \
#       --postproc-name pgpt_geom

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[
        dict(
            type='PGPTTracker',
            appearance_embedder=None,
            iou_match_thresh=0.5,
            oks_thresh=0.8,
            oks_in_vis_thre=0.2,
            effective_detection_thresh=0.5,
            effective_keypoints_thresh=0.6,
            effective_keypoints_number=8,
            keypoint_score_thr=0.2,
        ),
    ],
)
