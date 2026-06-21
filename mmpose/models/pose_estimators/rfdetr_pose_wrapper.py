"""Wrapper that integrates RF-DETR Keypoint Preview into MMPose v1.x.

The wrapper is intentionally minimal: all model logic lives in rfdetr.
Only the interface (data format translation and result packing) is adapted
here for bottom-up COCO evaluation.

Usage example::

    model = dict(
        type='RFDETRPoseEstimator',
        pretrain_weights='data/models/rf-detr-keypoint-preview-xlarge.pth',
        conf_thr=0.001,
        data_preprocessor=dict(
            type='PoseDataPreprocessor',
            mean=[0, 0, 0],
            std=[1, 1, 1],
            bgr_to_rgb=False,
            pad_size_divisor=1,
        ),
    )
"""

import os
import os.path as osp
from typing import List, Optional, Sequence

import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmpose.registry import MODELS
from mmpose.utils.typing import SampleList

_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))


def _configure_model_cache(model_cache_dir: str) -> str:
    """Set RF_HOME to the model cache directory if not already set."""
    cache = model_cache_dir
    if not osp.isabs(cache):
        cache = osp.join(_REPO_ROOT, cache)
    os.makedirs(cache, exist_ok=True)
    os.environ.setdefault('RF_HOME', cache)
    return cache


def _import_rfdetr_keypoint():
    try:
        from rfdetr import RFDETRKeypointPreview
    except ImportError as exc:
        raise ImportError(
            'rfdetr>=1.8.0 is required for RFDETRPoseEstimator. '
            'Install it with: pip install "rfdetr>=1.8.0"') from exc
    return RFDETRKeypointPreview


@MODELS.register_module()
class RFDETRPoseEstimator(BaseModel):
    """MMPose v1.x-compatible wrapper for RF-DETR Keypoint Preview.

    Runs end-to-end multi-person pose estimation via
    ``rfdetr.RFDETRKeypointPreview`` and exposes a standard MMPose
    ``predict()`` interface.

    Args:
        pretrain_weights (str): Path to pretrained weights file
            (e.g. ``'data/models/rf-detr-keypoint-preview-xlarge.pth'``).
            Auto-downloads on first use when only a filename is given.
        conf_thr (float): Detection confidence threshold passed to
            ``RFDETRKeypointPreview.predict``. Defaults to ``0.001``.
        device (str): Device string. Updated by :meth:`to`.
            Defaults to ``'cuda:0'``.
        model_cache_dir (str): Directory for auto-downloaded weights.
            Sets ``RF_HOME`` environment variable. Defaults to
            ``'data/models'``.
        num_keypoints (int): Expected number of keypoints per person.
            Used only for empty-result padding. Defaults to ``17``
            (COCO person).
        data_preprocessor (dict | None): Config for
            ``PoseDataPreprocessor``.
        init_cfg (dict | None): Unused; kept for API compatibility.

    Note:
        ``optimize_for_inference()`` is intentionally skipped.
        In rfdetr 1.8.0 the optimized forward path (``forward_export``)
        drops keypoint outputs, silently returning ``Detections`` instead of
        ``KeyPoints``.  The model already runs in eval mode via
        :func:`mmpose.apis.inference.init_model`.
    
    """

    def __init__(
        self,
        pretrain_weights: str = 'rf-detr-keypoint-preview-xlarge.pth',
        conf_thr: float = 0.001,
        device: str = 'cuda:0',
        model_cache_dir: str = 'data/models',
        num_keypoints: int = 17,
        data_preprocessor: Optional[dict] = None,
        init_cfg=None,
    ):
        if data_preprocessor is not None and isinstance(data_preprocessor,
                                                         dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        _configure_model_cache(model_cache_dir)
        RFDETRKeypointPreview = _import_rfdetr_keypoint()

        self.pretrain_weights = pretrain_weights
        self.conf_thr = conf_thr
        self._device = device
        self.model_cache_dir = model_cache_dir
        self.num_keypoints = num_keypoints

        # Keep RF-DETR off the nn.Module tree. RFDETRKeypointPreview.train()
        # starts fine-tuning, so PyTorch's eval()/train() would conflict.
        #
        # Note: optimize_for_inference() is intentionally NOT called here.
        # RFDETRKeypointPreview.forward_export() (used by the optimized model)
        # returns only (outputs_coord, outputs_class) and drops keypoint
        # outputs — a known limitation of the 1.8.0 preview release.
        # The model runs in eval mode already (set by init_model).
        object.__setattr__(
            self, '_rfdetr',
            RFDETRKeypointPreview(
                pretrain_weights=pretrain_weights,
                device=device,
            ))

    def to(self, *args, **kwargs):
        if args:
            self._device = str(args[0])
        elif 'device' in kwargs:
            self._device = str(kwargs['device'])
        return super().to(*args, **kwargs)

    def train(self, mode: bool = True):
        """Keep the wrapper in eval mode; RF-DETR handles its own mode."""
        return super().train(False)

    @staticmethod
    def _tensor_to_rgb_numpy(tensor: torch.Tensor,
                              img_shape: tuple) -> np.ndarray:
        """Convert a preprocessed CHW BGR tensor to an RGB uint8 numpy image.

        RF-DETR expects RGB channel order.
        """
        h, w = int(img_shape[0]), int(img_shape[1])
        img = tensor[:3, :h, :w].detach().cpu().numpy().transpose(1, 2, 0)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        # Reverse channel order BGR → RGB
        return img[..., ::-1].copy()

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'loss':
            raise NotImplementedError(
                'RFDETRPoseEstimator is for inference only.')
        else:
            raise ValueError(f'Unsupported mode "{mode}"')

    def _build_img_metas(self, data_samples: SampleList,
                         inputs: torch.Tensor) -> List[dict]:
        """Extract per-image metadata needed for tensor cropping."""
        _, _, h_pad, w_pad = inputs.shape
        img_metas = []
        for ds in data_samples:
            meta = ds.metainfo
            img_shape = meta.get('img_shape', (h_pad, w_pad, 3))
            if isinstance(img_shape, (tuple, list)) and len(img_shape) == 2:
                img_shape = (img_shape[0], img_shape[1], 3)
            img_metas.append({'img_shape': img_shape})
        return img_metas

    def _inference_forward(
        self,
        inputs: torch.Tensor,
        img_metas: Sequence[dict],
    ) -> list:
        """Run RF-DETR keypoint inference and return raw results.

        This method contains only the ``predict`` call and is patched by
        the :class:`~mmpose.evaluation.metrics.FPS` metric for accurate
        inference-time measurement.
        """
        batch_size = inputs.shape[0]
        imgs = [
            self._tensor_to_rgb_numpy(inputs[i], img_metas[i]['img_shape'])
            for i in range(batch_size)
        ]

        results = self._rfdetr.predict(
            images=imgs,
            threshold=self.conf_thr,
            include_source_image=False,
        )

        # predict() returns a list when given a list of images, but for a
        # single image it may return the KeyPoints directly.
        if not isinstance(results, list):
            results = [results]
        return results

    def predict(self, inputs: torch.Tensor,
                data_samples: SampleList) -> SampleList:
        """Run RF-DETR keypoint inference and populate pred_instances."""
        img_metas = self._build_img_metas(data_samples, inputs)

        with torch.no_grad():
            results = self._inference_forward(inputs, img_metas)

        num_keypoints = self.num_keypoints
        for data_sample, key_points in zip(data_samples, results):
            pred_instances = InstanceData()

            n = len(key_points) if key_points is not None else 0
            if n > 0:
                xy = np.asarray(key_points.xy, dtype=np.float32)
                kp_conf = np.asarray(
                    key_points.keypoint_confidence, dtype=np.float32)
                det_conf = np.asarray(
                    key_points.detection_confidence, dtype=np.float32)
                bboxes = np.asarray(
                    key_points.data['xyxy'], dtype=np.float32)

                num_keypoints = xy.shape[1]
                pred_instances.bboxes = bboxes
                pred_instances.bbox_scores = det_conf
                pred_instances.keypoints = xy
                pred_instances.keypoint_scores = kp_conf
                pred_instances.keypoints_visible = kp_conf
            else:
                pred_instances.bboxes = np.zeros((0, 4), dtype=np.float32)
                pred_instances.bbox_scores = np.zeros((0, ), dtype=np.float32)
                pred_instances.keypoints = np.zeros(
                    (0, num_keypoints, 2), dtype=np.float32)
                pred_instances.keypoint_scores = np.zeros(
                    (0, num_keypoints), dtype=np.float32)
                pred_instances.keypoints_visible = np.zeros(
                    (0, num_keypoints), dtype=np.float32)

            data_sample.pred_instances = pred_instances

        return data_samples
