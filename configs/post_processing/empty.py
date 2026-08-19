# Identity (empty) post-processing pipeline.
#
# No filters run: each frame is returned unchanged.  Use this to re-evaluate
# a saved prediction bundle with different metrics or evaluator settings
# without applying any tracker, NMS, or smoother.
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/empty.py \
#       --postproc-name empty \
#       --metrics CocoMetric MPJVE MPJAE
#
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/empty.py \
#       --postproc-name empty \
#       --eval-good-frames-only

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[],
)
