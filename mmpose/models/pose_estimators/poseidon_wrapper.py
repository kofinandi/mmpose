"""Wrapper that integrates the Poseidon video pose estimator from
``external/poseidon`` into the MMPose v1.x framework for evaluation.

Poseidon (`Tirupati et al., 2024 <https://arxiv.org/abs/2501.08446>`_) is a
top-down, multi-frame pose estimator: given a temporal window of ``T``
frames (default 5) sharing the same person crop, a ViTPose backbone
(reused verbatim from MMPose via ``mmpose.apis.init_model``) extracts
per-frame features, which are fused across frames (pyramid-pooling +
adaptive frame weighting + cross-/self-attention) before a standard
deconvolution head regresses PoseTrack-17-layout heatmaps for the *center*
frame only.

This wrapper is intentionally minimal: the upstream ``Poseidon`` class
(``external/poseidon/models/best/Poseidon.py``) is used unmodified. Only
the interface is adapted here:

- Sys.path insertion so the upstream package (``models``, ``posetimation``,
  ``utils``) is importable without renaming/vendoring.
- Construction of the upstream ``easydict`` config from a plain dict.
- Checkpoint loading: the released Poseidon checkpoints are raw PyTorch
  *training* checkpoints (``{'epoch', 'model_state_dict',
  'optimizer_state_dict'}``), not MMEngine-style ``{'state_dict': ...}``
  checkpoints. A ``load_state_dict`` pre-hook unwraps ``model_state_dict``
  and adds the ``poseidon_model.`` prefix so the released ``.pt`` files
  load transparently via MMEngine's ``load_checkpoint`` (as invoked by
  ``mmpose.apis.init_model`` / ``tools/benchmark_e2e.py``'s
  ``pose_checkpoint`` argument).
- Heatmap decoding: upstream decodes with a DCPose-style argmax + quarter-
  pixel offset. Here decoding uses a **configurable MMPose keypoint codec**
  (default ``MSRAHeatmap`` with no DARK refinement, which is numerically
  the same argmax + quarter-offset scheme); ``UDPHeatmap`` or
  ``MSRAHeatmap(unbiased=True)`` (DARK) can be selected instead via the
  ``codec`` config.
- Keypoint layout: Poseidon outputs the PoseTrack-17 layout (its training
  datasets, PoseTrack18/21, use this layout). ``map_to_coco=True``
  (default) reprojects predictions onto COCO-17 via
  ``KeypointConverter(src='posetrack18', dst='coco')``: 15 of 17 COCO
  keypoints have a same-named PoseTrack counterpart (everything except
  ``left_eye``/``right_eye``), which are left at zero confidence unless
  ``synthesize_eyes=True`` requests a coarse geometric heuristic (see
  ``_synthesize_eyes`` below).

Usage example (in a config file)::

    model = dict(
        type='PoseidonPoseEstimator',
        poseidon_root='external/poseidon',
        poseidon_cfg=dict(
            MODEL=dict(
                CONFIG_FILE='configs/body_2d_keypoint/topdown_heatmap/'
                             'coco/td-hm_ViTPose-small_8xb64-210e_coco-'
                             '256x192.py',
                EMBED_DIM=384,
                HEATMAP_SIZE=[72, 96],
                NUM_JOINTS=17,
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

_POSEIDON_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'external',
                 'poseidon'))


@MODELS.register_module()
class PoseidonPoseEstimator(BaseModel):
    """MMPose v1.x-compatible wrapper for the Poseidon video pose estimator.

    Args:
        poseidon_cfg (dict): Plain-dict version of the upstream Poseidon
            ``easydict`` training config. Must contain ``MODEL.CONFIG_FILE``
            (a mmpose ViTPose topdown config path defining the backbone +
            heatmap head architecture), ``MODEL.EMBED_DIM``,
            ``MODEL.HEATMAP_SIZE`` ``[W, H]``, ``MODEL.NUM_JOINTS``, and
            ``WINDOWS_SIZE`` (number of input frames ``T``, must match
            ``num_input_frames`` used to build clips upstream).
        poseidon_root (str): Path to the root of the external/poseidon
            repository. Defaults to ``external/poseidon`` relative to the
            mmpose project root.
        codec (dict): Config for an MMPose keypoint codec
            (``MSRAHeatmap``/``UDPHeatmap``) used to decode heatmaps to
            keypoint coordinates in input-crop space.
        map_to_coco (bool): Reproject the native PoseTrack-17 output onto
            COCO-17 via ``KeypointConverter(src='posetrack18', dst='coco')``.
            Defaults to ``True``.
        synthesize_eyes (bool): Only used when ``map_to_coco=True``. COCO's
            ``left_eye``/``right_eye`` have no PoseTrack counterpart and are
            zero-confidence by default; if ``True``, approximate them as a
            point offset from the nose towards the corresponding ear
            (``nose + eye_offset_ratio * (ear - nose)``), with confidence
            discounted from ``min(nose_score, ear_score)``. This is a coarse
            geometric heuristic, not a learned prediction. Defaults to
            ``False``.
        eye_offset_ratio (float): See ``synthesize_eyes``. Defaults to 0.3.
        data_preprocessor (dict | None): Config for
            ``ClipPoseDataPreprocessor`` (normalisation, BGR->RGB) operating
            on ``(B, T, C, H, W)`` clip batches. Defaults to ``None``.
        init_cfg (dict | None): Unused; kept for API compatibility.
    """

    def __init__(
        self,
        poseidon_cfg: dict,
        poseidon_root: str = _POSEIDON_ROOT,
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

        # ---- Add external/poseidon to sys.path -----------------------
        poseidon_root = os.path.abspath(poseidon_root)
        if poseidon_root not in sys.path:
            sys.path.insert(0, poseidon_root)

        # ---- Build the upstream Poseidon model (random-init backbone,
        # released checkpoint is loaded afterwards by MMEngine via the
        # `_remap_checkpoint_keys` pre-hook below) ----------------------
        from models.best.Poseidon import Poseidon

        cfg = _build_poseidon_cfg(poseidon_cfg)
        self.num_input_frames = cfg.WINDOWS_SIZE
        self.poseidon_model = Poseidon(
            cfg=cfg, device='cpu', phase='test', inference=True)
        self.poseidon_model.eval()

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
        """Remap Poseidon checkpoint keys to the wrapper layout.

        The released Poseidon checkpoints are raw training checkpoints:
        ``{'epoch': int, 'model_state_dict': {...}, 'optimizer_state_dict':
        {...}}``. This unwraps ``model_state_dict`` (dropping the training
        bookkeeping) and prefixes every key with ``poseidon_model.`` so the
        state dict matches this wrapper's module tree. A checkpoint that
        was instead saved directly from this wrapper (already
        ``poseidon_model.*`` / plain ``backbone.*`` etc.) is also handled,
        matching the ``petr``/``sapiens2`` wrapper convention.
        """
        target_prefix = prefix + 'poseidon_model.'
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
                'PoseidonPoseEstimator is for inference only; training is '
                'not supported via this wrapper.')
        else:
            raise ValueError(f'Unsupported mode "{mode}"')

    def _inference_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the Poseidon network and return raw center-frame heatmaps.

        This method contains *only* the GPU forward pass and is the target
        patched by the :class:`~mmpose.evaluation.metrics.FPS` metric for
        accurate inference-time measurement.

        Args:
            inputs (Tensor): Pre-processed clip batch ``(B, T, C, H, W)``.

        Returns:
            Tensor: Raw heatmaps ``(B, K, H_hm, W_hm)`` for the center
            frame of each clip.
        """
        return self.poseidon_model(inputs)

    def predict(self, inputs: torch.Tensor,
               data_samples: SampleList) -> SampleList:
        """Run inference and return predictions as ``PoseDataSample``
        objects.

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
        """Reproject PoseTrack-17 predictions onto COCO-17.

        Args:
            keypoints (np.ndarray): PoseTrack-17 keypoints, shape (1, 17, 2).
            scores (np.ndarray): PoseTrack-17 keypoint scores, shape (1, 17).

        Returns:
            tuple[np.ndarray, np.ndarray]: COCO-17 ``(keypoints, scores)``,
            shapes ``(1, 17, 2)`` and ``(1, 17)``. ``left_eye``/``right_eye``
            have zero score unless ``synthesize_eyes=True``.
        """
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
        """Fill in COCO ``left_eye``/``right_eye`` (indices 1, 2) with a
        coarse geometric heuristic: offset from the nose towards the
        corresponding (directly-mapped) ear. Modifies ``coco_keypoints`` /
        ``coco_scores`` in place. Not a learned prediction -- see the
        ``synthesize_eyes`` docstring on the class.
        """
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


def _build_poseidon_cfg(poseidon_cfg: dict):
    """Build the upstream ``easydict`` config expected by ``Poseidon()``
    from a plain dict, filling in defaults for fields the wrapper itself
    always overrides (``CHECKPOINT_FILE=None``: the full Poseidon
    checkpoint, not the ViTPose init checkpoint, is loaded afterwards by
    MMEngine)."""
    from easydict import EasyDict

    cfg = copy.deepcopy(poseidon_cfg)
    model_cfg = cfg.setdefault('MODEL', {})
    model_cfg['CHECKPOINT_FILE'] = None
    cfg.setdefault('WINDOWS_SIZE', 5)
    return EasyDict(cfg)
