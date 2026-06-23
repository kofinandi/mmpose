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
from .filters import OKSTracker, OneEuroSmoother
from .pipeline import PostProcessingPipeline, build_post_processor
from .registry import POST_PROCESS_FILTERS

__all__ = [
    'BaseFilter',
    'sequence_key_from_path',
    'OKSTracker',
    'OneEuroSmoother',
    'PostProcessingPipeline',
    'build_post_processor',
    'POST_PROCESS_FILTERS',
]
