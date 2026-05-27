"""Wrapper that integrates Ultralytics YOLO26-Pose models into MMPose v1.x.

The wrapper is intentionally minimal: all model logic lives in Ultralytics.
Only the interface (data format translation and result packing) is adapted
here for bottom-up COCO evaluation.

Usage example::

    model = dict(
        type='UltralyticsYOLOPoseEstimator',
        weights='yolo26n-pose.pt',
        conf_thr=0.001,
        iou_thr=0.7,
        imgsz=640,
        data_preprocessor=dict(
            type='PoseDataPreprocessor',
            mean=[0, 0, 0],
            std=[1, 1, 1],
            bgr_to_rgb=False,
            pad_size_divisor=1,
        ),
    )
"""

from typing import List, Optional, Sequence

import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmpose.models.utils.ultralytics_weights import resolve_ultralytics_weights
from mmpose.registry import MODELS
from mmpose.utils.typing import SampleList


def _import_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            'ultralytics is required for UltralyticsYOLOPoseEstimator. '
            'Install it with: pip install ultralytics') from exc
    return YOLO


@MODELS.register_module()
class UltralyticsYOLOPoseEstimator(BaseModel):
    """MMPose v1.x-compatible wrapper for Ultralytics YOLO26-Pose models.

    Runs end-to-end multi-person pose estimation via ``ultralytics.YOLO``
    and exposes a standard MMPose ``predict()`` interface.

    Args:
        weights (str): Path or model name for Ultralytics pose weights
            (e.g. ``yolo26n-pose.pt``). Auto-downloads on first use.
        conf_thr (float): Detection confidence threshold passed to
            ``YOLO.predict``. Defaults to ``0.001``.
        iou_thr (float): NMS IoU threshold passed to ``YOLO.predict``.
            Defaults to ``0.7``.
        imgsz (int): Inference image size passed to ``YOLO.predict``.
            Defaults to ``640``.
        device (str): Device string for ``YOLO.predict``. Updated by
            :meth:`to`. Defaults to ``'cuda:0'``.
        model_cache_dir (str): Directory for auto-downloaded Ultralytics
            weights. Defaults to ``'data/models'``.
        data_preprocessor (dict | None): Config for ``PoseDataPreprocessor``.
        init_cfg (dict | None): Unused; kept for API compatibility.
    """

    def __init__(
        self,
        weights: str,
        conf_thr: float = 0.001,
        iou_thr: float = 0.7,
        imgsz: int = 640,
        device: str = 'cuda:0',
        model_cache_dir: str = 'data/models',
        data_preprocessor: Optional[dict] = None,
        init_cfg=None,
    ):
        if data_preprocessor is not None and isinstance(data_preprocessor,
                                                         dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        YOLO = _import_ultralytics()
        self.weights = resolve_ultralytics_weights(weights, model_cache_dir)
        self.model_cache_dir = model_cache_dir
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.imgsz = imgsz
        self._device = device
        # Keep the Ultralytics model off the nn.Module tree.  YOLO overrides
        # train() for fine-tuning, so PyTorch's eval()/train() would start
        # a training run instead of toggling inference mode.
        object.__setattr__(self, '_yolo', YOLO(self.weights))

    def to(self, *args, **kwargs):
        if args:
            self._device = str(args[0])
        elif 'device' in kwargs:
            self._device = str(kwargs['device'])
        return super().to(*args, **kwargs)

    def train(self, mode: bool = True):
        """Keep the wrapper in eval mode; Ultralytics handles its own mode."""
        return super().train(False)

    @staticmethod
    def _tensor_to_bgr_numpy(tensor: torch.Tensor, img_shape: tuple) -> np.ndarray:
        """Convert a preprocessed CHW tensor to a BGR uint8 numpy image."""
        h, w = int(img_shape[0]), int(img_shape[1])
        img = tensor[:3, :h, :w].detach().cpu().numpy().transpose(1, 2, 0)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'loss':
            raise NotImplementedError(
                'UltralyticsYOLOPoseEstimator is for inference only.')
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
        """Run YOLO pose inference and return raw Ultralytics results.

        This method contains only the Ultralytics ``predict`` call and is
        patched by the :class:`~mmpose.evaluation.metrics.FPS` metric for
        accurate inference-time measurement.
        """
        batch_size = inputs.shape[0]
        imgs = [
            self._tensor_to_bgr_numpy(inputs[i], img_metas[i]['img_shape'])
            for i in range(batch_size)
        ]

        return self._yolo.predict(
            source=imgs,
            conf=self.conf_thr,
            iou=self.iou_thr,
            imgsz=self.imgsz,
            device=self._device,
            verbose=False,
        )

    def predict(self, inputs: torch.Tensor,
                data_samples: SampleList) -> SampleList:
        """Run YOLO26-Pose inference and populate pred_instances."""
        img_metas = self._build_img_metas(data_samples, inputs)

        with torch.no_grad():
            results = self._inference_forward(inputs, img_metas)

        num_keypoints = 17
        for data_sample, result in zip(data_samples, results):
            pred_instances = InstanceData()

            if (result.boxes is not None and len(result.boxes) > 0
                    and result.keypoints is not None):
                bboxes = result.boxes.xyxy.detach().cpu().numpy()
                bbox_scores = result.boxes.conf.detach().cpu().numpy()
                kpts = result.keypoints.data.detach().cpu().numpy()
                num_keypoints = kpts.shape[1]

                pred_instances.bboxes = bboxes
                pred_instances.bbox_scores = bbox_scores
                pred_instances.keypoints = kpts[:, :, :2]
                pred_instances.keypoint_scores = kpts[:, :, 2]
                pred_instances.keypoints_visible = kpts[:, :, 2]
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
