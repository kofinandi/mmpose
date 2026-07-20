# Copyright (c) OpenMMLab. All rights reserved.
"""Registry for post-processing filters.

Kept separate from mmpose.registry so post-processing components can be
built without triggering the full mmpose model/dataset scope.
"""

from mmengine.registry import Registry

POST_PROCESS_FILTERS = Registry('post_process_filters')
POST_PROCESS_PREDICTORS = Registry('post_process_predictors')
