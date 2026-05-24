# Copyright (c) OpenMMLab. All rights reserved.
"""Ultralytics YOLO detector adapter for top-down pose pipelines."""

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
        'mmdet is required for UltralyticsYOLODetector. '
        'Install it with: pip install mmdet') from exc


def _import_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError('ultralytics is required for UltralyticsYOLODetector.') from exc
    return YOLO


@MODELS.register_module()
class UltralyticsYOLODetector(nn.Module):
    """MMDet-configurable wrapper around an Ultralytics YOLO object detector.

    Loads weights in ``__init__`` and exposes ``predict_imgs`` for batched
    BGR numpy inference.  Intended for top-down pose pipelines that expect
    MMDet-style ``DetDataSample.pred_instances`` outputs.
    """

    def __init__(self,
                 weights: str,
                 conf_thr: float = 0.05,
                 iou_thr: float = 0.7,
                 imgsz: int = 640,
                 device: str = 'cuda:0') -> None:
        super().__init__()
        YOLO = _import_ultralytics()
        self.weights = weights
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.imgsz = imgsz
        self._device = device
        # Keep the Ultralytics model off the nn.Module tree.  YOLO overrides
        # train() for fine-tuning, so PyTorch's eval()/train() would start
        # a training run instead of toggling inference mode.
        object.__setattr__(self, '_yolo', YOLO(weights))
        names = self._yolo.names
        if isinstance(names, dict):
            classes = tuple(names[i] for i in sorted(names))
        else:
            classes = tuple(names)
        self.cfg = None
        self.dataset_meta = {'classes': classes}

    def predict_imgs(
        self,
        imgs: Sequence[np.ndarray],
    ) -> List[DetDataSample]:
        """Run batched detection on BGR numpy images."""
        if not imgs:
            return []

        results = self._yolo.predict(
            source=list(imgs),
            conf=self.conf_thr,
            iou=self.iou_thr,
            imgsz=self.imgsz,
            device=self._device,
            verbose=False,
        )

        data_samples: List[DetDataSample] = []
        for img, result in zip(imgs, results):
            h, w = img.shape[:2]
            data_sample = DetDataSample()
            data_sample.set_metainfo(
                dict(ori_shape=(h, w), img_shape=(h, w), pad_shape=(h, w)))

            pred_instances = InstanceData()
            if result.boxes is not None and len(result.boxes) > 0:
                pred_instances.bboxes = result.boxes.xyxy.detach().cpu()
                pred_instances.scores = result.boxes.conf.detach().cpu()
                pred_instances.labels = result.boxes.cls.detach().cpu().long()
            else:
                pred_instances.bboxes = torch.zeros((0, 4), dtype=torch.float32)
                pred_instances.scores = torch.zeros((0, ), dtype=torch.float32)
                pred_instances.labels = torch.zeros((0, ), dtype=torch.long)

            data_sample.pred_instances = pred_instances
            data_samples.append(data_sample)

        return data_samples

    def forward(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            'UltralyticsYOLODetector is inference-only; use predict_imgs().')
