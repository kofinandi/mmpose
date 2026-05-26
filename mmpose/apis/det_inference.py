# Copyright (c) OpenMMLab. All rights reserved.
"""Detector init/inference helpers with Ultralytics YOLO support."""

from typing import List, Sequence, Union

import numpy as np
import torch.nn as nn
from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmpose.compat.transformers_v5 import install_transformers_v5_shims
from mmpose.utils import adapt_mmdet_pipeline

install_transformers_v5_shims()

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet.registry import MODELS
    from mmdet.structures import DetDataSample
    HAS_MMDET = True
except (ImportError, ModuleNotFoundError):
    HAS_MMDET = False

CUSTOM_DETECTOR_TYPES = frozenset({
    'UltralyticsYOLODetector',
    'RFDETRDetector',
})

_CHECKPOINT_FIELDS = {
    'UltralyticsYOLODetector': 'weights',
    'RFDETRDetector': 'pretrain_weights',
}


def init_det_model(
    config: Union[str, Config],
    checkpoint: str = None,
    device: str = 'cuda:0',
) -> nn.Module:
    """Initialize a detector from a config file.

    For custom detector configs (``UltralyticsYOLODetector``,
    ``RFDETRDetector``), builds the wrapper directly.  Otherwise delegates
    to MMDetection's ``init_detector``.
    """
    if not HAS_MMDET:
        raise ImportError(
            'mmdet is required for detector initialization. '
            'Install it with: pip install mmdet')

    if isinstance(config, str):
        cfg = Config.fromfile(config)
    else:
        cfg = config

    scope = cfg.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(scope)

    # Ensure custom detector modules are registered.
    import mmpose.models.detectors  # noqa: F401

    det_type = cfg.model.get('type')
    if det_type in CUSTOM_DETECTOR_TYPES:
        if checkpoint is not None:
            weights_field = _CHECKPOINT_FIELDS[det_type]
            cfg.model[weights_field] = checkpoint
        cfg.model.device = device
        model = MODELS.build(cfg.model)
        model.cfg = cfg
        return model

    model = init_detector(cfg, checkpoint, device=device)
    model.cfg = adapt_mmdet_pipeline(model.cfg)
    return model


def inference_det_model(
    model: nn.Module,
    imgs: Union[np.ndarray, Sequence[np.ndarray]],
) -> Union[DetDataSample, List[DetDataSample]]:
    """Run detector inference, dispatching to Ultralytics or MMDet as needed."""
    if not HAS_MMDET:
        raise ImportError(
            'mmdet is required for detector inference. '
            'Install it with: pip install mmdet')

    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
        is_batch = False
    else:
        is_batch = True

    if hasattr(model, 'predict_imgs'):
        results = model.predict_imgs(imgs)
    else:
        results = inference_detector(model, imgs)

    if not is_batch:
        return results[0]
    return results
