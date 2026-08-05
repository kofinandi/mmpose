# Copyright (c) OpenMMLab. All rights reserved.
"""Post-processing pipeline for predicted poses.

Provides a configurable chain of online (causal) and offline filters that
transform :class:`~mmpose.structures.PoseDataSample` objects after model
inference.

Example usage::

    from mmpose.postprocessing import build_post_processor

    pipeline = build_post_processor('configs/post_processing/oks_track_one_euro.py')
    for frame_ds in frames:
        result = pipeline.process(frame_ds)   # None if any filter is offline
    # For an all-online pipeline, results were already returned above.
    # For an any-offline pipeline:
    results = pipeline.evaluate()
"""

from .base import BaseFilter, sequence_key_from_path
from .filters import (DetectAndTrackLinker, GPKalmanSmoother,
                       LightTrackTracker, OKSNMS, OKSTracker,
                       OneEuroSmoother, PGPTTracker, PredictiveTracker,
                       SmoothNetSmoother)
from .matchers import (BaseAppearanceEmbedder, BasePoseMatcher,
                        NormalizedL2PoseMatcher, PGPTPoseGCNEmbedder,
                        SGCNPoseMatcher, TorchvisionCNNEmbedder,
                        build_appearance_embedder, build_pose_matcher)
from .measurement import (BaseMeasurementModel, PowerScoreMeasurementModel,
                           build_measurement_model)
from .pipeline import PostProcessingPipeline, build_post_processor
from .predictors import BasePredictor, GPKalmanPredictor, build_predictor
from .registry import (POST_PROCESS_APPEARANCE_EMBEDDERS,
                        POST_PROCESS_FILTERS, POST_PROCESS_MEASUREMENT_MODELS,
                        POST_PROCESS_POSE_MATCHERS, POST_PROCESS_PREDICTORS)

__all__ = [
    'BaseFilter',
    'sequence_key_from_path',
    'GPKalmanSmoother',
    'OKSNMS',
    'OKSTracker',
    'OneEuroSmoother',
    'SmoothNetSmoother',
    'PredictiveTracker',
    'DetectAndTrackLinker',
    'LightTrackTracker',
    'PGPTTracker',
    'PostProcessingPipeline',
    'build_post_processor',
    'BasePredictor',
    'GPKalmanPredictor',
    'build_predictor',
    'BaseMeasurementModel',
    'PowerScoreMeasurementModel',
    'build_measurement_model',
    'BasePoseMatcher',
    'BaseAppearanceEmbedder',
    'NormalizedL2PoseMatcher',
    'TorchvisionCNNEmbedder',
    'SGCNPoseMatcher',
    'PGPTPoseGCNEmbedder',
    'build_pose_matcher',
    'build_appearance_embedder',
    'POST_PROCESS_FILTERS',
    'POST_PROCESS_PREDICTORS',
    'POST_PROCESS_MEASUREMENT_MODELS',
    'POST_PROCESS_POSE_MATCHERS',
    'POST_PROCESS_APPEARANCE_EMBEDDERS',
]
