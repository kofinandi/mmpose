# GP-Kalman Filter post-processing config
#
# Applies the Local Gaussian Process Regression Filter with Bayesian Fusion
# (projects/gp_kalman_filter/GP_Kalman_Filter_Steps.md) to all joints and
# both x / y coordinates independently.
#
# Prototype simplifications:
#   - No tracker (always uses instance 0)
#   - Offline filter (processes each sequence as a batch)
#
# Usage:
#   python tools/postprocess_predictions.py \
#       benchmark/predictions/20260622_emdb_topdown/ViTPose-small-rfdetr \
#       --post-config configs/post_processing/gp_kalman.py

post_processor = dict(
    type='PostProcessingPipeline',
    filters=[
        dict(
            type='GPKalmanSmoother',
        ),
    ],
)
