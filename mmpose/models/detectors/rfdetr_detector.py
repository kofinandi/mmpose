# Copyright (c) OpenMMLab. All rights reserved.
"""RF-DETR detector adapter for top-down pose pipelines."""

import os
import os.path as osp
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
from mmengine.structures import InstanceData

try:
    from mmdet.registry import MODELS
    from mmdet.structures import DetDataSample
except ImportError as exc:
    raise ImportError(
        'mmdet is required for RFDETRDetector. '
        'Install it with: pip install mmdet') from exc

_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))


def _import_rfdetr():
    try:
        import rfdetr_plus.models  # noqa: F401  # enable XL/2XL lazy exports
        import rfdetr
    except ImportError as exc:
        raise ImportError(
            'rfdetr is required for RFDETRDetector. '
            'Install it with: pip install rfdetr') from exc
    return rfdetr


def _configure_model_cache(model_cache_dir: str) -> str:
    """Set RF_HOME to the model cache directory if not already set."""
    cache = model_cache_dir
    if not osp.isabs(cache):
        cache = osp.join(_REPO_ROOT, cache)
    os.makedirs(cache, exist_ok=True)
    os.environ.setdefault('RF_HOME', cache)
    return cache


def _resolve_model_class(rfdetr, model_class: str):
    """Resolve an RF-DETR variant class by name."""
    try:
        return getattr(rfdetr, model_class)
    except (AttributeError, ImportError):
        pass
    try:
        from rfdetr_plus import models as plus_models
        return getattr(plus_models, model_class)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f'Unknown RF-DETR model class {model_class!r}. '
            'Use a name like RFDETRNano or RFDETRXLarge.') from exc


def _is_training_checkpoint(path: str) -> bool:
    """Return True if *path* looks like an RF-DETR training checkpoint."""
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except (OSError, RuntimeError, ValueError):
        return False
    return isinstance(ckpt, dict) and 'args' in ckpt


def _load_rfdetr_model(rfdetr, model_class: str, pretrain_weights: str,
                       device: str):
    """Instantiate an RF-DETR model from weights or a training checkpoint."""
    if _is_training_checkpoint(pretrain_weights):
        return rfdetr.RFDETR.from_checkpoint(
            pretrain_weights, device=device)
    cls = _resolve_model_class(rfdetr, model_class)
    return cls(pretrain_weights=pretrain_weights, device=device)


def _detections_to_data_sample(
    img: np.ndarray,
    detections,
) -> DetDataSample:
    """Convert supervision Detections to an MMDet DetDataSample."""
    h, w = img.shape[:2]
    data_sample = DetDataSample()
    data_sample.set_metainfo(
        dict(ori_shape=(h, w), img_shape=(h, w), pad_shape=(h, w)))

    pred_instances = InstanceData()
    n_dets = len(detections)
    if n_dets > 0:
        pred_instances.bboxes = torch.as_tensor(
            detections.xyxy, dtype=torch.float32)
        pred_instances.scores = torch.as_tensor(
            detections.confidence, dtype=torch.float32)
        pred_instances.labels = torch.as_tensor(
            detections.class_id, dtype=torch.long)
    else:
        pred_instances.bboxes = torch.zeros((0, 4), dtype=torch.float32)
        pred_instances.scores = torch.zeros((0, ), dtype=torch.float32)
        pred_instances.labels = torch.zeros((0, ), dtype=torch.long)

    data_sample.pred_instances = pred_instances
    return data_sample


@MODELS.register_module()
class RFDETRDetector(nn.Module):
    """MMDet-configurable wrapper around an RF-DETR object detector.

    Loads weights in ``__init__`` and exposes ``predict_imgs`` for batched
    BGR numpy inference.  Intended for top-down pose pipelines that expect
    MMDet-style ``DetDataSample.pred_instances`` outputs.
    """

    def __init__(self,
                 model_class: str = 'RFDETRNano',
                 pretrain_weights: str = 'rf-detr-nano.pth',
                 conf_thr: float = 0.05,
                 device: str = 'cuda:0',
                 model_cache_dir: str = 'data/models') -> None:
        super().__init__()
        _configure_model_cache(model_cache_dir)
        rfdetr = _import_rfdetr()
        self.model_class = model_class
        self.pretrain_weights = pretrain_weights
        self.conf_thr = conf_thr
        self._device = device
        self.model_cache_dir = model_cache_dir
        # Keep RF-DETR off the nn.Module tree.  RFDETR.train() starts
        # fine-tuning, so PyTorch's eval()/train() would conflict.
        object.__setattr__(
            self, '_rfdetr',
            _load_rfdetr_model(rfdetr, model_class, pretrain_weights, device))
        self.cfg = None
        self.dataset_meta = {'classes': tuple(self._rfdetr.class_names)}

    def predict_imgs(
        self,
        imgs: Sequence[np.ndarray],
    ) -> List[DetDataSample]:
        """Run batched detection on BGR numpy images."""
        if not imgs:
            return []

        rgb_imgs = [img[..., ::-1].copy() for img in imgs]
        detections = self._rfdetr.predict(
            rgb_imgs,
            threshold=self.conf_thr,
            include_source_image=False,
        )
        if not isinstance(detections, list):
            detections = [detections]

        return [
            _detections_to_data_sample(img, det)
            for img, det in zip(imgs, detections)
        ]

    def forward(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            'RFDETRDetector is inference-only; use predict_imgs().')
