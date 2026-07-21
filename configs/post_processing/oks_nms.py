# Post-processing pipeline: confidence threshold + OKS-based NMS
#
# A single online (causal) filter that, per frame:
#   1. Drops instances with a confidence score below `score_thr`.
#   2. Greedily suppresses duplicate detections of the same person: sorts
#      surviving instances by score and removes any lower-scoring instance
#      whose OKS with an already-kept instance is >= `oks_thr`.
#
# Usage:
#   python tools/benchmark_e2e.py CONFIG CKPT \
#       --test-dataset emdb-mini \
#       --post-config configs/post_processing/oks_nms.py
#
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/oks_nms.py

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[
        dict(
            type='OKSNMS',
            # Minimum confidence score to keep an instance.
            score_thr=0.3,
            # OKS above which the lower-scoring of two instances is
            # suppressed as a duplicate detection.
            oks_thr=0.9,
            # 'bbox' -> bbox_scores, 'keypoint' -> mean keypoint_scores,
            # 'auto' -> prefer bbox_scores, fall back to keypoint_scores.
            score_mode='auto',
        ),
    ],
)
