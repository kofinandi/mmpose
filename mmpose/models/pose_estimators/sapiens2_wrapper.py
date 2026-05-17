"""Wrapper that integrates the sapiens2 pose model from ``external/sapiens2``
into the MMPose v1.x framework for inference / evaluation.

The wrapper is intentionally minimal: all model logic lives in the original
sapiens2 submodule unchanged.  Only the interface (data pre/post-processing,
weight loading, keypoint mapping) is adapted here.

Usage example (in a config file)::

    model = dict(
        type='Sapiens2PoseEstimator',
        sapiens2_root='external/sapiens2',
        sapiens2_model_cfg=dict(
            type='PoseTopdownEstimator',
            backbone=dict(type='Sapiens2', arch='sapiens2_0.4b', ...),
            decode_head=dict(type='PoseHeatmapHead', in_channels=1024, ...),
        ),
        codec=dict(
            type='MSRAHeatmap',
            input_size=(768, 1024),
            heatmap_size=(192, 256),
            sigma=6,
        ),
        keypoint_mapping=dict(
            num_keypoints=17,
            mapping=[
                # (goliath_308_src_channel, target_dst_index)
                (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
                (5, 5), (6, 6), (7, 7), (8, 8),
                (62, 9),   # left_wrist
                (41, 10),  # right_wrist
                (9, 11), (10, 12), (11, 13), (12, 14), (13, 15), (14, 16),
            ],
        ),
        data_preprocessor=dict(
            type='PoseDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
        ),
    )

Checkpoint loading
------------------
The HuggingFace-released sapiens2 pose checkpoints store weights directly
under ``backbone.*`` and ``decode_head.*``.  A ``load_state_dict`` pre-hook
remaps these to ``sapiens2_model.backbone.*`` / ``sapiens2_model.decode_head.*``
so that both raw HF checkpoints and any checkpoint saved from this wrapper
load transparently via MMEngine's ``load_checkpoint`` (invoked by
``tools/test.py`` through ``--checkpoint``).

For ``.safetensors`` files MMEngine's default loader cannot handle them
directly, so we hook into the checkpoint loading via a custom
``load_checkpoint`` call inside ``init_weights``.
"""

import copy
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.typing import SampleList


@MODELS.register_module()
class Sapiens2PoseEstimator(BaseModel):
    """MMPose v1.x-compatible wrapper for the sapiens2 pose estimator.

    Integrates ``PoseTopdownEstimator`` (Sapiens2 backbone + PoseHeatmapHead)
    from ``external/sapiens2`` without modifying the submodule.

    The model outputs 308-channel heatmaps (Goliath keypoint set).  A
    configurable ``keypoint_mapping`` selects and reorders channels to match
    any target dataset that is a subset of Goliath 308 (e.g. COCO-17, MPII).

    Args:
        sapiens2_root (str): Path to the root of the external/sapiens2
            repository (the directory that contains the ``sapiens/`` package).
        sapiens2_model_cfg (dict): Config dict for
            ``sapiens.registry.MODELS.build()``.  Must include ``backbone``
            (type ``Sapiens2``) and ``decode_head`` (type
            ``PoseHeatmapHead``).  The ``init_cfg`` key is automatically
            stripped from the backbone dict so that the full pose checkpoint
            is used rather than a pre-trained backbone-only checkpoint.
        codec (dict): Config for an MMPose ``MSRAHeatmap`` codec used to
            decode the heatmaps to keypoint coordinates in input-image space.
        keypoint_mapping (dict): Mapping from Goliath-308 output channels to
            target dataset keypoints.  Must contain:

            - ``num_keypoints`` (int): Number of target keypoints.
            - ``mapping`` (list of ``(src, dst)`` tuples): Each tuple maps a
              Goliath channel index ``src`` to a target keypoint index ``dst``.

        fp16 (bool): Run the sapiens2 model in float16 for faster inference
            and lower GPU memory usage.  The data preprocessor still outputs
            float32; inputs are cast to fp16 inside ``predict()``.  Defaults
            to ``False``.
        data_preprocessor (dict | None): Config for
            ``PoseDataPreprocessor`` (normalisation, BGR→RGB).  Defaults to
            ``None``.
        init_cfg (dict | None): Unused; kept for API compatibility.
    """

    def __init__(
        self,
        sapiens2_root: str,
        sapiens2_model_cfg: dict,
        codec: dict,
        keypoint_mapping: dict,
        fp16: bool = False,
        data_preprocessor: Optional[dict] = None,
        init_cfg=None,
    ):
        if data_preprocessor is not None and isinstance(data_preprocessor, dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # ---- Shims and sys.path -----------------------------------------
        sapiens2_root = os.path.abspath(sapiens2_root)

        from mmpose.models.pose_estimators.sapiens2_compat import (
            install_sapiens2_shims,
        )
        install_sapiens2_shims(sapiens2_root)

        if sapiens2_root not in sys.path:
            sys.path.insert(0, sapiens2_root)

        # ---- Import sapiens2 components ---------------------------------
        # Importing the package triggers all @MODELS.register_module() calls
        # in backbones, heads, estimator, etc.
        import sapiens  # noqa: F401
        from sapiens.registry import MODELS as SAPIENS_MODELS

        # ---- Build sapiens2 model ---------------------------------------
        model_cfg = copy.deepcopy(sapiens2_model_cfg)

        # Drop backbone init_cfg: the full pose checkpoint contains backbone
        # weights already, loading pretrain-only weights on top would conflict.
        model_cfg.get('backbone', {}).pop('init_cfg', None)

        self.sapiens2_model = SAPIENS_MODELS.build(model_cfg)
        self.sapiens2_model.eval()

        # ---- fp16 -------------------------------------------------------
        self.fp16 = fp16
        if fp16:
            self.sapiens2_model.half()

        # ---- Codec (MSRAHeatmap) ----------------------------------------
        self.codec = KEYPOINT_CODECS.build(codec)

        # ---- Keypoint mapping -------------------------------------------
        self.num_output_keypoints = keypoint_mapping['num_keypoints']
        mapping: List[Tuple[int, int]] = keypoint_mapping['mapping']
        src_indices = [None] * self.num_output_keypoints
        for src, dst in mapping:
            src_indices[dst] = src
        assert all(s is not None for s in src_indices), (
            'keypoint_mapping is incomplete: not all target indices covered. '
            f'Got: {keypoint_mapping}')
        # src_indices[i] = Goliath channel that maps to target keypoint i
        self.src_indices = src_indices

        # ---- Checkpoint remapping hook ----------------------------------
        self._register_load_state_dict_pre_hook(self._remap_checkpoint_keys)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _remap_checkpoint_keys(self, state_dict, prefix, *args, **kwargs):
        """Pre-hook: remap HF-style sapiens2 checkpoint keys.

        Raw HF checkpoints store weights as ``backbone.*`` and
        ``decode_head.*``.  After wrapping, those keys live under
        ``sapiens2_model.backbone.*`` / ``sapiens2_model.decode_head.*``.
        This hook is registered via ``_register_load_state_dict_pre_hook``
        so that both raw HF checkpoints and any checkpoint saved from this
        wrapper load transparently.
        """
        target_prefix = prefix + 'sapiens2_model.'
        skip_prefixes = (target_prefix, prefix + 'data_preprocessor.')
        for k in list(state_dict.keys()):
            if k.startswith(prefix) and not any(
                    k.startswith(p) for p in skip_prefixes):
                new_key = target_prefix + k[len(prefix):]
                state_dict[new_key] = state_dict.pop(k)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'loss':
            raise NotImplementedError(
                'Sapiens2PoseEstimator is for inference only; '
                'training is not supported via this wrapper.')
        else:
            raise ValueError(f'Unsupported mode "{mode}"')

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _inference_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the sapiens2 neural network and return raw heatmaps.

        This method contains *only* the GPU forward pass and is the target
        patched by the :class:`~mmpose.evaluation.metrics.FPS` metric for
        accurate inference-time measurement (excluding heatmap decoding and
        coordinate post-processing).

        Args:
            inputs (Tensor): Pre-processed image batch ``(N, C, H, W)``.
                Float32 or float16 depending on ``self.fp16``.

        Returns:
            Tensor: Raw heatmaps ``(N, 308, H_hm, W_hm)`` in float32.
        """
        model_inputs = inputs.half() if self.fp16 else inputs
        heatmaps = self.sapiens2_model(model_inputs)
        if self.fp16:
            heatmaps = heatmaps.float()
        return heatmaps

    def predict(self, inputs: torch.Tensor,
                data_samples: SampleList) -> SampleList:
        """Run inference and return predictions as ``PoseDataSample`` objects.

        Steps:
          1. Forward through sapiens2 → heatmaps ``(B, 308, H_hm, W_hm)``
          2. Select & reorder channels via ``keypoint_mapping``
          3. Decode each sample's heatmaps with ``MSRAHeatmap.decode()``
             → keypoints in input-image space
          4. Apply the inverse affine transform to get original-image coords
          5. Pack into ``PoseDataSample.pred_instances``

        Args:
            inputs (Tensor): Pre-processed image batch ``(N, C, H, W)``.
            data_samples (list[PoseDataSample]): Per-sample metadata.

        Returns:
            list[PoseDataSample]: Updated samples with
            ``pred_instances.keypoints`` and ``pred_instances.keypoint_scores``
            populated.
        """
        with torch.no_grad():
            # (B, 308, H_hm, W_hm)
            all_heatmaps = self._inference_forward(inputs)

        # Select target keypoint channels: (B, K_target, H_hm, W_hm)
        all_heatmaps = all_heatmaps[:, self.src_indices, :, :]

        # Move to CPU numpy for codec decode
        all_heatmaps_np = all_heatmaps.cpu().numpy()  # (B, K, H, W)

        for i, data_sample in enumerate(data_samples):
            # MSRAHeatmap.decode expects (K, H, W) and returns
            # keypoints (1, K, 2) in input-image space, scores (1, K)
            heatmaps_i = all_heatmaps_np[i]  # (K, H, W)
            keypoints, scores = self.codec.decode(heatmaps_i)

            # Transform keypoints from affine-cropped input space to
            # original image coordinates.  Same formula used in
            # TopdownPoseEstimator.add_pred_to_datasample.
            meta = data_sample.metainfo
            input_center = np.array(meta['input_center'], dtype=np.float32)
            input_scale = np.array(meta['input_scale'], dtype=np.float32)
            input_size = np.array(meta['input_size'], dtype=np.float32)

            keypoints[..., :2] = (
                keypoints[..., :2] / input_size * input_scale
                + input_center - 0.5 * input_scale)

            pred_instances = InstanceData()
            pred_instances.keypoints = keypoints        # (1, K, 2)
            pred_instances.keypoint_scores = scores      # (1, K)
            pred_instances.keypoints_visible = scores    # (1, K)

            gt_instances = data_sample.gt_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

        return data_samples
