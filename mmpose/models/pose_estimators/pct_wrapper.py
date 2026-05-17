"""Wrapper that integrates the PCT (Pose Compositional Tokens) model from
``external/PCT`` into the MMPose v1.x framework for inference / evaluation.

The wrapper is intentionally minimal: all model logic lives in the original
PCT submodule unchanged.  Only the interface (data pre/post-processing, weight
loading) is adapted here.

Usage example (in a config file)::

    model = dict(
        type='PCTPoseEstimator',
        pct_root='external/PCT',
        pct_model_cfg=dict(
            backbone=dict(type='SwinV2TransformerRPE2FC', ...),
            keypoint_head=dict(type='PCT_Head', ...),
            test_cfg=dict(flip_test=True, dataset_name='COCO'),
        ),
        data_preprocessor=dict(
            type='PoseDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
        ),
    )
"""

import os
import sys
from typing import List, Optional

import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmpose.registry import MODELS
from mmpose.utils.typing import SampleList


@MODELS.register_module()
class PCTPoseEstimator(BaseModel):
    """MMPose v1.x–compatible wrapper for the PCT pose estimator.

    The wrapper installs compatibility shims for the old mmpose v0.x / mmcv
    v1.x API surface that PCT depends on, then instantiates the original
    ``PCT`` detector class unmodified.

    Args:
        pct_root (str): Path to the root of the external/PCT repository
            (contains the ``models/`` and ``utils/`` sub-packages).
        pct_model_cfg (dict): Model configuration dict passed to the original
            ``PCT`` constructor.  Must contain at minimum ``backbone``,
            ``keypoint_head`` and ``test_cfg``.  The ``test_cfg`` field is
            forwarded as-is to the PCT model so that ``dataset_name``,
            ``flip_test`` etc. are honoured for every target dataset.
        data_preprocessor (dict | None): Config for the MMPose v1.x
            ``PoseDataPreprocessor`` (handles BGR→RGB conversion and
            pixel normalisation).  Defaults to ``None``.
        init_cfg (dict | None): Unused; kept for API compatibility.
    """

    def __init__(
        self,
        pct_root: str,
        pct_model_cfg: dict,
        data_preprocessor: Optional[dict] = None,
        init_cfg=None,
    ):
        # Build data_preprocessor before super().__init__ so that mmengine's
        # BaseModel can attach it via the standard path.
        if data_preprocessor is not None and isinstance(data_preprocessor,
                                                         dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # ---- Install compatibility shims (idempotent) ---------------------
        from mmpose.models.pose_estimators.pct_compat import install_pct_shims
        install_pct_shims()

        # ---- Add external/PCT to sys.path ---------------------------------
        pct_root = os.path.abspath(pct_root)
        if pct_root not in sys.path:
            sys.path.insert(0, pct_root)

        # ---- Import and build the PCT model --------------------------------
        from models.pct_detector import PCT  # noqa: E402
        self.pct_model = PCT(
            backbone=pct_model_cfg['backbone'],
            keypoint_head=pct_model_cfg['keypoint_head'],
            test_cfg=pct_model_cfg.get('test_cfg', {}),
            pretrained=pct_model_cfg.get('pretrained', None),
        )
        self.pct_model.eval()

        # Store test_cfg for easy access; flip_test and dataset_name are
        # already baked into pct_model but we keep a copy for reference.
        self.test_cfg = pct_model_cfg.get('test_cfg', {})

        # Register hook to remap raw PCT checkpoint keys when mmengine
        # calls load_state_dict.  Must be done after pct_model is created.
        self._register_load_state_dict_pre_hook(self._remap_checkpoint_keys)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _remap_checkpoint_keys(self, state_dict, prefix, *args, **kwargs):
        """Pre-hook: remap raw PCT checkpoint keys to our wrapper layout.

        A raw PCT checkpoint stores weights as ``backbone.*`` and
        ``keypoint_head.*``.  After wrapping, those keys live under
        ``pct_model.backbone.*`` / ``pct_model.keypoint_head.*``.  This hook
        is registered via ``_register_load_state_dict_pre_hook`` so that both
        raw PCT checkpoints and any checkpoint saved from this wrapper load
        transparently.
        """
        pct_prefix = prefix + 'pct_model.'
        skip_prefixes = (pct_prefix, prefix + 'data_preprocessor.')
        for k in list(state_dict.keys()):
            if k.startswith(prefix) and not any(
                    k.startswith(p) for p in skip_prefixes):
                state_dict[pct_prefix + k[len(prefix):]] = state_dict.pop(k)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'loss':
            raise NotImplementedError(
                'PCTPoseEstimator is for inference only; '
                'training is not supported via this wrapper.')
        else:
            raise ValueError(f'Unsupported mode "{mode}"')

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _inference_forward(self, inputs: torch.Tensor,
                           img_metas: list) -> dict:
        """Run the PCT neural network and return raw prediction results.

        This method contains *only* the model forward pass and is the target
        patched by the :class:`~mmpose.evaluation.metrics.FPS` metric for
        accurate inference-time measurement (excluding ``InstanceData``
        construction and ``PoseDataSample`` packing).

        Args:
            inputs (Tensor): Pre-processed image batch ``(N, C, H, W)``.
            img_metas (list[dict]): Per-image metadata dicts in the format
                expected by PCT's ``forward_test``.

        Returns:
            dict: Raw prediction dict from PCT (contains ``'preds'`` with
            shape ``(N, K, 3)``).
        """
        return self.pct_model(
            inputs,
            joints_3d=None,
            joints_3d_visible=None,
            img_metas=img_metas,
            return_loss=False,
        )

    def predict(self, inputs: torch.Tensor,
                data_samples: SampleList) -> SampleList:
        """Run inference and return predictions as PoseDataSample objects.

        PCT's ``forward_test`` already applies the inverse affine transform
        (``transform_preds``) to map keypoints from model-input space back to
        the original image coordinate system.  We therefore store the
        predictions directly without any additional spatial remapping.

        Args:
            inputs (Tensor): Pre-processed image batch, shape (N, C, H, W).
            data_samples (list[PoseDataSample]): Per-sample metadata.

        Returns:
            list[PoseDataSample]: Updated data samples with
            ``pred_instances.keypoints`` and
            ``pred_instances.keypoint_scores`` populated.
        """
        batch_size = inputs.shape[0]

        # Build img_metas list expected by PCT's forward_test
        img_metas = []
        for ds in data_samples:
            meta = ds.metainfo
            # input_center / input_scale use the MMPose v1.x convention:
            #   input_scale  = bbox_size (pixels, both axes)
            #   input_center = centre of bbox in original image
            # PCT expects:
            #   center = (cx, cy)           in original image pixels
            #   scale  = bbox_size / 200    (i.e. input_scale / 200)
            center = np.array(meta['input_center'], dtype=np.float32)
            scale = np.array(meta['input_scale'], dtype=np.float32) / 200.0

            img_meta_dict = dict(
                center=center,
                scale=scale,
                image_file=meta.get('img_path', ''),
                bbox_score=float(meta.get('bbox_score', 1.0)),
            )
            if 'bbox_id' in meta:
                img_meta_dict['bbox_id'] = meta['bbox_id']
            img_metas.append(img_meta_dict)

        # PCT forward_test requires img_metas to carry 'bbox_id' for
        # batches > 1 so it can assemble the result dict correctly.
        # Fall back to 0-based index when bbox_id is absent.
        if batch_size > 1 and 'bbox_id' not in img_metas[0]:
            for idx, m in enumerate(img_metas):
                m['bbox_id'] = idx

        with torch.no_grad():
            results = self._inference_forward(inputs, img_metas)

        # results['preds']: (N, K, 3) – (x, y) in original image space + score
        all_preds = results['preds']  # numpy (N, K, 3)

        # Pack predictions into PoseDataSample instances
        for i, data_sample in enumerate(data_samples):
            pred_kpts = all_preds[i, :, :2][np.newaxis]  # (1, K, 2)
            pred_scores = all_preds[i, :, 2][np.newaxis]  # (1, K)

            pred_instances = InstanceData()
            pred_instances.keypoints = pred_kpts
            pred_instances.keypoint_scores = pred_scores
            pred_instances.keypoints_visible = pred_scores

            # Attach ground-truth bbox info (expected by evaluators)
            gt_instances = data_sample.gt_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

        return data_samples
