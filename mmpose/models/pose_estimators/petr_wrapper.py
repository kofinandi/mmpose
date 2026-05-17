"""Wrapper that integrates the PETR (Opera) end-to-end pose estimator from
``external/PETR`` into the MMPose v1.x framework for evaluation.

The wrapper is intentionally minimal: all model logic lives in the original
Opera submodule unchanged.  Only the interface (data format translation and
weight loading) is adapted here.

Usage example::

    model = dict(
        type='PETRPoseEstimator',
        petr_root='external/PETR',
        petr_model_cfg=dict(
            type='opera.PETR',
            backbone=dict(type='mmdet.ResNet', depth=50, ...),
            neck=dict(type='mmdet.ChannelMapper', ...),
            bbox_head=dict(type='opera.PETRHead', ...),
            train_cfg=dict(...),
            test_cfg=dict(max_per_img=200),
        ),
        data_preprocessor=dict(
            type='PoseDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=1,
        ),
    )
"""

import os
import sys
from typing import Optional

import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmpose.registry import MODELS
from mmpose.utils.typing import SampleList

_PETR_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'external',
                 'PETR'))


@MODELS.register_module()
class PETRPoseEstimator(BaseModel):
    """MMPose v1.x–compatible wrapper for the PETR end-to-end pose estimator.

    PETR is a bottom-up, end-to-end multi-person pose estimator built on the
    legacy OpenMMLab stack (mmcv v1 / mmdet v2).  This wrapper installs
    compatibility shims and exposes a standard MMPose ``predict()`` interface.

    Args:
        petr_root (str): Path to the root of the external/PETR repository.
            Defaults to ``external/PETR`` relative to the mmpose project root.
        petr_model_cfg (dict): Full model configuration dict for
            ``opera.models.build_model``.  Must include at minimum ``type``,
            ``backbone``, ``neck``, ``bbox_head``, and ``test_cfg``.
        data_preprocessor (dict | None): Config for ``PoseDataPreprocessor``.
        init_cfg (dict | None): Unused; kept for API compatibility.
    """

    def __init__(
        self,
        petr_model_cfg: dict,
        petr_root: str = _PETR_ROOT,
        data_preprocessor: Optional[dict] = None,
        init_cfg=None,
    ):
        if data_preprocessor is not None and isinstance(data_preprocessor,
                                                         dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # ---- Add external/PETR to sys.path --------------------------------
        petr_root = os.path.abspath(petr_root)
        if petr_root not in sys.path:
            sys.path.insert(0, petr_root)

        # ---- Install compatibility shims (idempotent) ---------------------
        from mmpose.models.pose_estimators.petr_compat import (
            install_petr_shims, finalize_petr_shims)
        install_petr_shims()

        # ---- Import opera model builder AFTER shims are installed ---------
        from opera.models import build_model
        finalize_petr_shims()

        # ---- Build the Opera PETR model -----------------------------------
        import copy
        cfg = copy.deepcopy(petr_model_cfg)
        self.petr_model = build_model(cfg)
        self.petr_model.eval()

        self.test_cfg = petr_model_cfg.get('test_cfg', {})
        self._register_load_state_dict_pre_hook(self._remap_checkpoint_keys)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _remap_checkpoint_keys(self, state_dict, prefix, *args, **kwargs):
        """Remap raw Opera PETR checkpoint keys to wrapper layout.

        A raw PETR checkpoint stores weights as ``backbone.*``, ``neck.*``,
        ``bbox_head.*``.  After wrapping they live under
        ``petr_model.backbone.*`` etc.
        """
        petr_prefix = prefix + 'petr_model.'
        skip_prefixes = (petr_prefix, prefix + 'data_preprocessor.')
        for k in list(state_dict.keys()):
            if k.startswith(prefix) and not any(
                    k.startswith(p) for p in skip_prefixes):
                state_dict[petr_prefix + k[len(prefix):]] = state_dict.pop(k)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'loss':
            raise NotImplementedError(
                'PETRPoseEstimator is for inference only.')
        else:
            raise ValueError(f'Unsupported mode "{mode}"')

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _inference_forward(self, inputs: torch.Tensor,
                           img_metas: list) -> list:
        """Run the PETR neural network and return raw detection results.

        This method contains *only* the model forward pass (iterating over
        images one at a time as required by ``PETR.simple_test``) and is the
        target patched by the :class:`~mmpose.evaluation.metrics.FPS` metric
        for accurate inference-time measurement (excluding keypoint packing
        and ``InstanceData`` construction).

        Args:
            inputs (Tensor): Pre-processed image batch ``(N, C, H, W)``.
            img_metas (list[dict]): Per-image metadata dicts in the mmdet v2
                format expected by ``PETR.simple_test``.

        Returns:
            list[tuple]: One ``(bbox_result, kpt_result)`` tuple per image.
        """
        batch_size = inputs.shape[0]
        all_results = []
        for i in range(batch_size):
            single_img = inputs[i:i + 1]
            single_meta = [img_metas[i]]
            res = self.petr_model.simple_test(
                single_img, single_meta, rescale=True)
            all_results.append(res[0])  # (bbox_result, kpt_result)
        return all_results

    def predict(self, inputs: torch.Tensor,
                data_samples: SampleList) -> SampleList:
        """Run PETR inference and populate pred_instances.

        Args:
            inputs (Tensor): Pre-processed image batch (N, C, H, W).
                After ``PoseDataPreprocessor`` padding this may be larger than
                the original images; ``batch_input_shape`` in img_metas tracks
                the padded size.
            data_samples (list[PoseDataSample]): Per-sample metadata.

        Returns:
            list[PoseDataSample]: Updated with ``pred_instances.keypoints``
            (N_persons, K, 2), ``pred_instances.keypoint_scores`` (N_persons,
            K), and ``pred_instances.bboxes`` (N_persons, 4).
        """
        batch_size = inputs.shape[0]
        _, _, H_pad, W_pad = inputs.shape

        img_metas = []
        for ds in data_samples:
            meta = ds.metainfo
            # img_shape: actual (non-padded) image shape (H, W, C)
            img_shape = meta.get('img_shape', (H_pad, W_pad, 3))
            if isinstance(img_shape, (tuple, list)) and len(img_shape) == 2:
                img_shape = (img_shape[0], img_shape[1], 3)

            # scale_factor: ratio of resized/padded image to original image
            scale_factor = meta.get('scale_factor', np.array([1.0, 1.0]))
            if isinstance(scale_factor, (float, int)):
                scale_factor = np.array(
                    [scale_factor, scale_factor], dtype=np.float32)
            else:
                scale_factor = np.array(scale_factor, dtype=np.float32)
            # PETR / mmdet v2 expects (sx, sy, sx, sy)
            if scale_factor.size == 2:
                scale_factor = np.array([
                    scale_factor[0], scale_factor[1],
                    scale_factor[0], scale_factor[1],
                ], dtype=np.float32)

            img_meta_dict = dict(
                img_shape=img_shape,
                scale_factor=scale_factor,
                flip=meta.get('flip', False),
                flip_direction=meta.get('flip_direction', None),
                batch_input_shape=(H_pad, W_pad),
                img_path=meta.get('img_path', ''),
            )
            img_metas.append(img_meta_dict)

        with torch.no_grad():
            all_results = self._inference_forward(inputs, img_metas)

        for i, (data_sample, result) in enumerate(
                zip(data_samples, all_results)):
            bbox_result, kpt_result = result
            # bbox_result: list of len num_classes, each (N_i, 5) with score
            # kpt_result:  list of len num_classes, each (N_i, K, 3) x,y,v
            bboxes = np.concatenate(bbox_result, axis=0)   # (N, 5)
            kpts = np.concatenate(kpt_result, axis=0)      # (N, K, 3)

            pred_instances = InstanceData()
            if bboxes.shape[0] > 0:
                pred_instances.bboxes = bboxes[:, :4]              # (N, 4)
                pred_instances.bbox_scores = bboxes[:, 4]          # (N,)
                pred_instances.keypoints = kpts[:, :, :2]          # (N, K, 2)
                pred_instances.keypoint_scores = kpts[:, :, 2]     # (N, K)
                pred_instances.keypoints_visible = kpts[:, :, 2]   # (N, K)
            else:
                num_kpts = self.petr_model.bbox_head.num_keypoints
                pred_instances.bboxes = np.zeros((0, 4), dtype=np.float32)
                pred_instances.bbox_scores = np.zeros((0,), dtype=np.float32)
                pred_instances.keypoints = np.zeros(
                    (0, num_kpts, 2), dtype=np.float32)
                pred_instances.keypoint_scores = np.zeros(
                    (0, num_kpts), dtype=np.float32)
                pred_instances.keypoints_visible = np.zeros(
                    (0, num_kpts), dtype=np.float32)

            data_sample.pred_instances = pred_instances

        return data_samples
