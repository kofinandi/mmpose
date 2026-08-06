"""Wrapper that integrates the TAR-ViTPose video pose estimator from
``external/TARViTPose`` into the MMPose v1.x framework for evaluation.

TAR-ViTPose (`Fang et al., CVPR 2026 <https://arxiv.org/abs/2603.05929>`_,
"Beyond Static Frames: Temporal Aggregate-and-Restore Vision Transformer for
Human Pose Estimation") is, like Poseidon, a top-down, multi-frame pose
estimator built directly on the Poseidon codebase (same authors' "DCPose-
style" data/config/eval pipeline; the TAR-ViTPose README credits Poseidon
as its base). Given a temporal window of ``T`` frames (default 5) sharing
the same person crop, a ViTPose backbone (reused verbatim from MMPose via
``mmpose.apis.init_model``) extracts per-frame patch tokens; a
joint-centric temporal aggregation (JTA, learnable per-joint query tokens
cross-attending -- optionally low-confidence-masked -- to every frame's
tokens) and global restoring attention (GRA, folding the aggregated joint
context back into the center frame's tokens) replace Poseidon's pyramid-
pooling/adaptive-weighting fusion, before the same deconvolution head
regresses PoseTrack-17-layout heatmaps for the center frame only.

This wrapper mirrors :mod:`poseidon_wrapper` (same upstream-unmodified-
class, checkpoint-remap, configurable-codec, PoseTrack->COCO conversion
approach) since both models share the same interface contract:

- Sys.path insertion so the upstream package is importable unmodified.
- Checkpoint loading: the released checkpoint is a raw PyTorch training
  checkpoint (``{'epoch', 'model_state_dict', 'optimizer_state_dict',
  'scheduler_state_dict'}``), not an MMEngine-style ``{'state_dict': ...}``
  checkpoint. The same ``load_state_dict`` pre-hook pattern as
  ``PoseidonPoseEstimator`` unwraps ``model_state_dict`` and prefixes keys.
- Heatmap decoding via a configurable MMPose keypoint codec (default
  ``MSRAHeatmap``, matching the paper's own argmax + quarter-pixel-offset
  decode inherited from the Poseidon/DCPose codebase).
- PoseTrack-17 -> COCO-17 layout conversion via
  ``KeypointConverter(src='posetrack18', dst='coco')`` (see
  ``poseidon_wrapper`` for the mapping's coverage / ``synthesize_eyes``
  option -- identical here).

Usage example (in a config file)::

    model = dict(
        type='TARViTPosePoseEstimator',
        tarvitpose_root='external/TARViTPose',
        tarvitpose_cfg=dict(
            MODEL=dict(
                CONFIG_FILE='configs/body_2d_keypoint/topdown_heatmap/'
                             'coco/td-hm_ViTPose-base_8xb64-210e_coco-'
                             '256x192.py',
                EMBED_DIM=768,
                HEATMAP_SIZE=[72, 96],
                NUM_JOINTS=17,
                MASK_THRESHOLD=0.2,
                NUM_LAYERS=6,
            ),
            WINDOWS_SIZE=5,
        ),
        codec=dict(
            type='MSRAHeatmap',
            input_size=(288, 384),
            heatmap_size=(72, 96),
            sigma=3,
        ),
        map_to_coco=True,
        data_preprocessor=dict(
            type='ClipPoseDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
        ),
    )
"""

import copy
import os
import sys
from typing import Optional

import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.typing import SampleList

_TARVITPOSE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'external',
                 'TARViTPose'))


@MODELS.register_module()
class TARViTPosePoseEstimator(BaseModel):
    """MMPose v1.x-compatible wrapper for the TAR-ViTPose video pose
    estimator.

    Args:
        tarvitpose_cfg (dict): Plain-dict version of the upstream
            TAR-ViTPose ``easydict`` training config. Must contain
            ``MODEL.CONFIG_FILE`` (a mmpose ViTPose topdown config
            defining the backbone + heatmap head architecture),
            ``MODEL.EMBED_DIM``, ``MODEL.HEATMAP_SIZE`` ``[W, H]``,
            ``MODEL.NUM_JOINTS``, ``MODEL.MASK_THRESHOLD``,
            ``MODEL.NUM_LAYERS`` (JTA/GRA stack depth), and
            ``WINDOWS_SIZE`` (number of input frames ``T``, must match
            ``num_input_frames`` used to build clips upstream).
        tarvitpose_root (str): Path to the root of the
            external/TARViTPose repository. Defaults to
            ``external/TARViTPose`` relative to the mmpose project root.
        use_mask (bool): Whether the joint-centric cross-attention masks
            out low-confidence spatial locations from the initial
            (single-frame) heatmap estimate, as in the paper. Defaults to
            ``True``.
        codec (dict): Config for an MMPose keypoint codec
            (``MSRAHeatmap``/``UDPHeatmap``) used to decode heatmaps to
            keypoint coordinates in input-crop space.
        map_to_coco (bool): Reproject the native PoseTrack-17 output onto
            COCO-17 via ``KeypointConverter(src='posetrack18', dst='coco')``.
            Defaults to ``True``.
        synthesize_eyes (bool): See ``PoseidonPoseEstimator``. Defaults to
            ``False``.
        eye_offset_ratio (float): See ``PoseidonPoseEstimator``. Defaults
            to 0.3.
        data_preprocessor (dict | None): Config for
            ``ClipPoseDataPreprocessor`` operating on ``(B, T, C, H, W)``
            clip batches. Defaults to ``None``.
        init_cfg (dict | None): Unused; kept for API compatibility.
    """

    def __init__(
        self,
        tarvitpose_cfg: dict,
        tarvitpose_root: str = _TARVITPOSE_ROOT,
        use_mask: bool = True,
        codec: dict = None,
        map_to_coco: bool = True,
        synthesize_eyes: bool = False,
        eye_offset_ratio: float = 0.3,
        data_preprocessor: Optional[dict] = None,
        init_cfg=None,
    ):
        if data_preprocessor is not None and isinstance(data_preprocessor,
                                                         dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # ---- Add external/TARViTPose to sys.path -----------------------
        tarvitpose_root = os.path.abspath(tarvitpose_root)
        if tarvitpose_root not in sys.path:
            sys.path.insert(0, tarvitpose_root)

        # ---- Build the upstream TAR_ViTPose model (random-init backbone,
        # released checkpoint is loaded afterwards by MMEngine via the
        # `_remap_checkpoint_keys` pre-hook below) ----------------------
        from models.best.TAR_ViTPose import TAR_ViTPose

        cfg = _build_tarvitpose_cfg(tarvitpose_cfg, use_mask)
        self.num_input_frames = cfg.WINDOWS_SIZE
        self.tarvitpose_model = TAR_ViTPose(cfg=cfg, device='cpu',
                                           phase='test')
        self.tarvitpose_model.eval()

        # ---- Codec (heatmap -> keypoints), configurable ----------------
        if codec is None:
            heatmap_w, heatmap_h = cfg.MODEL.HEATMAP_SIZE
            image_w, image_h = cfg.MODEL.get('IMAGE_SIZE', [288, 384])
            codec = dict(
                type='MSRAHeatmap',
                input_size=(image_w, image_h),
                heatmap_size=(heatmap_w, heatmap_h),
                sigma=3)
        self.codec = KEYPOINT_CODECS.build(codec)

        # ---- PoseTrack-18 -> COCO-17 layout conversion, configurable ---
        self.map_to_coco = map_to_coco
        self.synthesize_eyes = synthesize_eyes
        self.eye_offset_ratio = eye_offset_ratio
        if map_to_coco:
            from mmpose.datasets.transforms import KeypointConverter
            self.keypoint_converter = KeypointConverter(
                src='posetrack18', dst='coco')
            from mmpose.datasets.transforms.keypoint_registry import (
                get_keypoints)
            self._posetrack_names = get_keypoints('posetrack18')
        else:
            self.keypoint_converter = None

        self._register_load_state_dict_pre_hook(self._remap_checkpoint_keys)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _remap_checkpoint_keys(self, state_dict, prefix, *args, **kwargs):
        """Remap TAR-ViTPose checkpoint keys to the wrapper layout.

        Identical unwrap/prefix logic to
        ``PoseidonPoseEstimator._remap_checkpoint_keys`` -- see that
        docstring for details on the raw training-checkpoint format.
        """
        target_prefix = prefix + 'tarvitpose_model.'
        raw_key = prefix + 'model_state_dict'
        if raw_key in state_dict and isinstance(state_dict[raw_key], dict):
            inner = state_dict.pop(raw_key)
            for k in list(state_dict.keys()):
                if k.startswith(prefix):
                    state_dict.pop(k)
            for k, v in inner.items():
                state_dict[target_prefix + k] = v
            return

        skip_prefixes = (target_prefix, prefix + 'data_preprocessor.')
        for k in list(state_dict.keys()):
            if k.startswith(prefix) and not any(
                    k.startswith(p) for p in skip_prefixes):
                state_dict[target_prefix + k[len(prefix):]] = state_dict.pop(
                    k)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'loss':
            raise NotImplementedError(
                'TARViTPosePoseEstimator is for inference only; training '
                'is not supported via this wrapper.')
        else:
            raise ValueError(f'Unsupported mode "{mode}"')

    def _inference_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the TAR-ViTPose network and return raw center-frame
        heatmaps.

        This method contains *only* the GPU forward pass and is the
        target patched by the :class:`~mmpose.evaluation.metrics.FPS`
        metric for accurate inference-time measurement.

        Args:
            inputs (Tensor): Pre-processed clip batch ``(B, T, C, H, W)``.

        Returns:
            Tensor: Raw heatmaps ``(B, K, H_hm, W_hm)`` for the center
            frame of each clip.
        """
        return self.tarvitpose_model(inputs)

    def predict(self, inputs: torch.Tensor,
               data_samples: SampleList) -> SampleList:
        """Run inference and return predictions as ``PoseDataSample``
        objects. Identical decode/remap/layout-conversion pipeline to
        :meth:`PoseidonPoseEstimator.predict`.

        Args:
            inputs (Tensor): Pre-processed clip batch ``(B, T, C, H, W)``.
            data_samples (list[PoseDataSample]): Per-sample metadata,
                including ``input_center``/``input_scale``/``input_size``
                set by ``GetBBoxCenterScale``/``TopdownAffine``.

        Returns:
            list[PoseDataSample]: Updated samples with
            ``pred_instances.keypoints`` and ``pred_instances.keypoint_scores``
            populated (COCO-17 layout if ``map_to_coco``, else native
            PoseTrack-17 layout).
        """
        with torch.no_grad():
            all_heatmaps = self._inference_forward(inputs)  # (B, K, H, W)

        all_heatmaps_np = all_heatmaps.cpu().numpy()

        for i, data_sample in enumerate(data_samples):
            heatmaps_i = all_heatmaps_np[i]  # (K, H, W)
            keypoints, scores = self.codec.decode(heatmaps_i)  # (1,K,2),(1,K)

            meta = data_sample.metainfo
            input_center = np.array(meta['input_center'], dtype=np.float32)
            input_scale = np.array(meta['input_scale'], dtype=np.float32)
            input_size = np.array(meta['input_size'], dtype=np.float32)

            keypoints[..., :2] = (
                keypoints[..., :2] / input_size * input_scale + input_center
                - 0.5 * input_scale)

            if self.map_to_coco:
                keypoints, scores = self._to_coco(keypoints, scores)

            pred_instances = InstanceData()
            pred_instances.keypoints = keypoints
            pred_instances.keypoint_scores = scores
            pred_instances.keypoints_visible = scores

            gt_instances = data_sample.gt_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

        return data_samples

    def _to_coco(self, keypoints: np.ndarray, scores: np.ndarray):
        """Reproject PoseTrack-17 predictions onto COCO-17. See
        ``PoseidonPoseEstimator._to_coco`` for details."""
        results = dict(
            keypoints=keypoints.astype(np.float64),
            keypoints_visible=scores.astype(np.float64))
        results = self.keypoint_converter.transform(results)
        coco_keypoints = results['keypoints'].astype(np.float32)  # (1,17,2)
        coco_scores = results['keypoints_visible'][..., 0].astype(
            np.float32)  # (1,17)

        if self.synthesize_eyes:
            self._synthesize_eyes(keypoints, scores, coco_keypoints,
                                 coco_scores)

        return coco_keypoints, coco_scores

    def _synthesize_eyes(self, src_keypoints, src_scores, coco_keypoints,
                        coco_scores):
        """Fill in COCO ``left_eye``/``right_eye`` with a coarse
        nose-towards-ear geometric heuristic. See
        ``PoseidonPoseEstimator._synthesize_eyes`` for details."""
        name_to_idx = {n: i for i, n in enumerate(self._posetrack_names)}
        nose_idx = name_to_idx['nose']
        nose_xy = src_keypoints[:, nose_idx]
        nose_score = src_scores[:, nose_idx]
        for ear_name, coco_eye_idx in (('left_ear', 1), ('right_ear', 2)):
            ear_idx = name_to_idx[ear_name]
            ear_xy = src_keypoints[:, ear_idx]
            ear_score = src_scores[:, ear_idx]
            eye_xy = nose_xy + self.eye_offset_ratio * (ear_xy - nose_xy)
            eye_score = 0.5 * np.minimum(nose_score, ear_score)
            coco_keypoints[:, coco_eye_idx] = eye_xy
            coco_scores[:, coco_eye_idx] = eye_score


def _build_tarvitpose_cfg(tarvitpose_cfg: dict, use_mask: bool):
    """Build the upstream ``easydict`` config expected by ``TAR_ViTPose()``
    from a plain dict, filling in defaults for fields the wrapper itself
    always overrides (``CHECKPOINT_FILE=None``: the full TAR-ViTPose
    checkpoint, not the ViTPose init checkpoint, is loaded afterwards by
    MMEngine) or derives from a separate constructor arg (``USE_MASK``)."""
    from easydict import EasyDict

    cfg = copy.deepcopy(tarvitpose_cfg)
    model_cfg = cfg.setdefault('MODEL', {})
    model_cfg['CHECKPOINT_FILE'] = None
    model_cfg.setdefault('USE_MASK', use_mask)
    cfg.setdefault('WINDOWS_SIZE', 5)
    return EasyDict(cfg)
