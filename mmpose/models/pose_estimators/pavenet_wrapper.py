"""Wrapper that integrates the PAVE-Net (Opera) end-to-end video pose
estimator from ``external/PAVENet`` into the MMPose v1.x framework for
evaluation.

PAVE-Net (`Yu et al., AAAI 2026 <https://arxiv.org/abs/2511.13208>`_,
"End-to-End Multi-Person Pose Estimation with Pose-Aware Video Transformer")
is a bottom-up, end-to-end, *multi-frame* pose estimator built directly on
PETR/Opera: given a temporal window of ``T`` frames (the only released
config/checkpoint pair uses ``T=3``), a shared backbone (ResNet-50, with
PAVE-Net's ``input_type='mul_frames'`` flattening ``T`` into the batch
dimension before the usual conv stem -- see ``pavenet_compat.py``) extracts
per-frame multi-scale features; a pose-aware multi-frame deformable
attention/decoder (``opera.TransformerMulFrames``) then jointly detects
people and regresses keypoints for the *center* frame, using the
neighbouring frames only to disambiguate temporal association -- there is no
separate detector and no heatmap: keypoint coordinates and confidences come
directly out of an RLE (normalising-flow) regression head refined by
``forward_refine``.

The wrapper is intentionally minimal, mirroring :mod:`petr_wrapper` (PAVE-Net
reuses almost all of PETR's Opera API surface verbatim -- see
``pavenet_compat.py`` for what is genuinely new): all model logic lives in
the upstream ``opera.PAVE``/``opera.PAVEHeadMulFrames`` classes unmodified.
Only the interface is adapted here:

- ``sys.path`` insertion so the upstream package is importable unmodified.
- ``pavenet_compat.install_pavenet_shims()`` / ``finalize_pavenet_shims()``
  installing the mmcv v1/mmdet v2 compatibility shims (PETR's, plus the
  multi-frame ResNet/attention/decoder pieces PAVE-Net additionally needs).
- Checkpoint loading: the released checkpoint is already a standard
  mmdet-style ``{'meta', 'state_dict', 'optimizer'}`` checkpoint (unlike
  Poseidon/TAR-ViTPose's raw training-checkpoint format), so a
  ``load_state_dict`` pre-hook only needs to add the ``pavenet_model.``
  prefix (identical in spirit to ``PETRPoseEstimator``'s hook).
- Keypoint layout: PAVE-Net's PoseTrack-based training data drops
  ``left_ear``/``right_ear`` from the standard 17-point PoseTrack18 layout
  (see the paper's data-processing note and the upstream
  ``opera.PAVE.show_result``'s ``num_keypoint == 15`` skeleton), leaving 15
  keypoints in PoseTrack18 order with the two ear entries removed:
  ``nose, head_bottom, head_top, left_shoulder, right_shoulder, left_elbow,
  right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee,
  right_knee, left_ankle, right_ankle``. ``map_to_coco=True`` (default)
  reprojects this onto COCO-17 (``left_eye``/``right_eye``/``left_ear``/
  ``right_ear`` all left at zero confidence -- unlike Poseidon/TAR-ViTPose,
  PAVE-Net has no ear keypoints at all to synthesize eyes from).

Usage example (in a config file)::

    model = dict(
        type='PAVENetPoseEstimator',
        pavenet_root='external/PAVENet',
        pavenet_model_cfg=dict(
            type='opera.PAVE',
            backbone=dict(
                type='mmdet.ResNet',
                input_type='mul_frames',
                depth=50,
                ...
            ),
            neck=dict(type='mmdet.ChannelMapper', ...),
            bbox_head=dict(
                type='opera.PAVEHeadMulFrames',
                num_frames=3,
                num_keypoints=15,
                ...
            ),
            test_cfg=dict(max_per_img=20),
        ),
        data_preprocessor=dict(
            type='ClipPoseDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
        ),
    )
"""

import contextlib
import copy
import os
import sys
from typing import Optional

import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmpose.registry import MODELS
from mmpose.utils.typing import SampleList

_PAVENET_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'external',
                 'PAVENet'))

# PoseTrack18 (17 kpts, see configs/_base_/datasets/posetrack18.py) with
# `left_ear`/`right_ear` (indices 3, 4) removed -- see module docstring.
# Maps PAVE-Net's native keypoint index -> COCO-17 index; `head_bottom`/
# `head_top` have no COCO counterpart and are dropped (left at zero score).
_PAVENET_TO_COCO = [
    (0, 0),    # nose -> nose
    (3, 5),    # left_shoulder
    (4, 6),    # right_shoulder
    (5, 7),    # left_elbow
    (6, 8),    # right_elbow
    (7, 9),    # left_wrist
    (8, 10),   # right_wrist
    (9, 11),   # left_hip
    (10, 12),  # right_hip
    (11, 13),  # left_knee
    (12, 14),  # right_knee
    (13, 15),  # left_ankle
    (14, 16),  # right_ankle
]


@MODELS.register_module()
class PAVENetPoseEstimator(BaseModel):
    """MMPose v1.x-compatible wrapper for the PAVE-Net end-to-end video pose
    estimator.

    PAVE-Net is a bottom-up, end-to-end, multi-frame pose estimator (no
    external detector, no heatmap decode -- keypoints come directly out of
    an RLE regression head). This wrapper installs the PAVE-Net-specific
    compatibility shims (see :mod:`pavenet_compat`) and exposes a standard
    MMPose ``predict()`` interface, mirroring :class:`PETRPoseEstimator`.

    Args:
        pavenet_root (str): Path to the root of the external/PAVENet
            repository. Defaults to ``external/PAVENet`` relative to the
            mmpose project root.
        pavenet_model_cfg (dict): Full model configuration dict for
            ``opera.models.build_model``. Must include at minimum ``type``
            (``opera.PAVE``), ``backbone``, ``neck``, ``bbox_head``
            (``opera.PAVEHeadMulFrames``), and ``test_cfg``.
        map_to_coco (bool): Reproject the native 15-keypoint PoseTrack
            layout onto COCO-17 (see module docstring for the mapping).
            Defaults to ``True``.
        fp16 (bool): Run inference under CUDA automatic mixed precision.
            Weights stay in float32; only the forward pass uses half
            precision. Defaults to ``True`` (PAVE-Net's deformable
            attention/transformer stack is memory-heavy, as with PETR).
        torch_compile (bool): Wrap ``pavenet_model`` with ``torch.compile``
            on the first forward pass. Defaults to ``False``.
        data_preprocessor (dict | None): Config for
            ``ClipPoseDataPreprocessor`` operating on the ``(B, T, C, H, W)``
            clip batches produced by the benchmark's bottom-up clip-window
            path. Defaults to ``None``.
        init_cfg (dict | None): Unused; kept for API compatibility.
    """

    def __init__(
        self,
        pavenet_model_cfg: dict,
        pavenet_root: str = _PAVENET_ROOT,
        map_to_coco: bool = True,
        fp16: bool = True,
        torch_compile: bool = False,
        data_preprocessor: Optional[dict] = None,
        init_cfg=None,
    ):
        if data_preprocessor is not None and isinstance(data_preprocessor,
                                                         dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # ---- Add external/PAVENet to sys.path ------------------------
        pavenet_root = os.path.abspath(pavenet_root)
        if pavenet_root not in sys.path:
            sys.path.insert(0, pavenet_root)

        # ---- Install compatibility shims (idempotent) -----------------
        from mmpose.models.pose_estimators.pavenet_compat import (
            finalize_pavenet_shims, install_pavenet_shims)
        install_pavenet_shims()

        # ---- Import opera model builder AFTER shims are installed -----
        from opera.models import build_model
        finalize_pavenet_shims()

        # ---- Build the Opera PAVE model ---------------------------------
        cfg = copy.deepcopy(pavenet_model_cfg)
        self.num_frames = cfg['bbox_head']['num_frames']
        self.num_keypoints = cfg['bbox_head']['num_keypoints']
        self.pavenet_model = build_model(cfg)
        self.pavenet_model.eval()

        self.fp16 = fp16
        self.torch_compile = torch_compile
        self._compiled = False

        self.test_cfg = pavenet_model_cfg.get('test_cfg', {})

        # ---- Native-layout -> COCO-17 conversion, configurable ---------
        self.map_to_coco = map_to_coco

        self._register_load_state_dict_pre_hook(self._remap_checkpoint_keys)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _remap_checkpoint_keys(self, state_dict, prefix, *args, **kwargs):
        """Remap raw Opera PAVE-Net checkpoint keys to wrapper layout.

        The released checkpoint is a standard mmdet-style checkpoint
        (``{'meta', 'state_dict', 'optimizer'}``); MMEngine's
        ``load_checkpoint`` already unwraps the top-level ``state_dict``
        key before this hook runs, so ``state_dict`` here directly holds
        ``backbone.*``, ``neck.*``, ``bbox_head.*`` -- identical in shape
        to a raw PETR checkpoint. After wrapping these live under
        ``pavenet_model.backbone.*`` etc.
        """
        target_prefix = prefix + 'pavenet_model.'
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
                'PAVENetPoseEstimator is for inference only.')
        else:
            raise ValueError(f'Unsupported mode "{mode}"')

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _maybe_compile(self) -> None:
        """Apply ``torch.compile`` once, lazily on first inference."""
        if not self.torch_compile or self._compiled:
            return
        self.pavenet_model = torch.compile(
            self.pavenet_model, mode='reduce-overhead')
        self._compiled = True

    @contextlib.contextmanager
    def _amp_context(self, inputs: torch.Tensor):
        """Enable FP16 autocast for CUDA inference when ``self.fp16`` is
        set."""
        if self.fp16 and inputs.is_cuda:
            with torch.autocast(
                    device_type=inputs.device.type,
                    dtype=torch.float16):
                yield
        else:
            yield

    def _inference_forward(self, inputs: torch.Tensor,
                           img_metas: list) -> list:
        """Run the PAVE-Net network and return raw detection results.

        This method contains *only* the model forward pass (iterating over
        clips one at a time, as required by ``PAVE.simple_test``'s
        ``batch_size == 1`` assertion) and is the target patched by the
        :class:`~mmpose.evaluation.metrics.FPS` metric for accurate
        inference-time measurement (excluding keypoint packing and
        ``InstanceData`` construction).

        Args:
            inputs (Tensor): Pre-processed clip batch
                ``(N, T, C, H, W)``.
            img_metas (list[dict]): Per-clip metadata dicts (length ``N``,
                one per clip -- *not* per frame) in the mmdet v2 format
                expected by ``PAVE.simple_test``.

        Returns:
            list[tuple]: One ``(bbox_result, kpt_result)`` tuple per clip.
        """
        self._maybe_compile()

        batch_size = inputs.shape[0]
        all_results = []
        with self._amp_context(inputs):
            for i in range(batch_size):
                single_clip = inputs[i:i + 1]  # (1, T, C, H, W)
                single_meta = [img_metas[i]]
                # NOTE: `rescale=False`, not the PETR-style `True` --
                # `PAVE._get_bboxes_single`'s own rescale step assumes a
                # simple per-axis `scale_factor` (mmdet v2 convention), but
                # `BottomupResize` (this wrapper's pipeline transform)
                # produces an aspect-ratio-preserving, letterbox-style
                # affine warp instead (like `TopdownAffine`) and does not
                # populate `scale_factor` at all. Keypoints/bboxes are
                # therefore left in padded-canvas pixel space (``img_shape``
                # -- i.e. ``input_size``) here, and mapped back to original-
                # image coordinates via `input_center`/`input_scale` in
                # :meth:`predict` instead (mirroring
                # ``PoseidonPoseEstimator``'s ``TopdownAffine`` inverse-warp
                # convention).
                res = self.pavenet_model.simple_test(
                    single_clip, single_meta, rescale=False)
                all_results.append(res[0])  # (bbox_result, kpt_result)
        return all_results

    def predict(self, inputs: torch.Tensor,
               data_samples: SampleList) -> SampleList:
        """Run PAVE-Net inference and populate ``pred_instances``.

        Args:
            inputs (Tensor): Pre-processed clip batch ``(N, T, C, H, W)``.
                After ``ClipPoseDataPreprocessor`` padding this may be
                larger than the original frames; ``batch_input_shape`` in
                img_metas tracks the padded size.
            data_samples (list[PoseDataSample]): Per-sample metadata.

        Returns:
            list[PoseDataSample]: Updated with ``pred_instances.keypoints``
            (N_persons, K, 2), ``pred_instances.keypoint_scores``
            (N_persons, K), and ``pred_instances.bboxes`` (N_persons, 4).
        """
        _, _, _, H_pad, W_pad = inputs.shape

        img_metas = []
        for ds in data_samples:
            meta = ds.metainfo
            img_shape = meta.get('img_shape', (H_pad, W_pad, 3))
            if isinstance(img_shape, (tuple, list)) and len(img_shape) == 2:
                img_shape = (img_shape[0], img_shape[1], 3)

            scale_factor = meta.get('scale_factor', np.array([1.0, 1.0]))
            if isinstance(scale_factor, (float, int)):
                scale_factor = np.array(
                    [scale_factor, scale_factor], dtype=np.float32)
            else:
                scale_factor = np.array(scale_factor, dtype=np.float32)
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
            bboxes = np.concatenate(bbox_result, axis=0)   # (N, 5)
            kpts = np.concatenate(kpt_result, axis=0)       # (N, K, 3)

            if bboxes.shape[0] > 0:
                bboxes[:, :4], kpts[:, :, :2] = self._unwarp_to_original(
                    data_sample, bboxes[:, :4], kpts[:, :, :2])

            native_keypoints = kpts[:, :, :2]
            native_scores = kpts[:, :, 2]

            if self.map_to_coco:
                keypoints, scores = self._to_coco(native_keypoints,
                                                  native_scores)
            else:
                keypoints, scores = native_keypoints, native_scores

            pred_instances = InstanceData()
            if bboxes.shape[0] > 0:
                pred_instances.bboxes = bboxes[:, :4]              # (N, 4)
                pred_instances.bbox_scores = bboxes[:, 4]          # (N,)
                pred_instances.keypoints = keypoints
                pred_instances.keypoint_scores = scores
                pred_instances.keypoints_visible = scores
            else:
                num_kpts = 17 if self.map_to_coco else self.num_keypoints
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

    @staticmethod
    def _unwarp_to_original(data_sample, bboxes: np.ndarray,
                            keypoints: np.ndarray):
        """Map bboxes/keypoints from ``BottomupResize``'s padded-canvas
        pixel space back to original-image coordinates.

        Inverts the same aspect-ratio-preserving affine warp
        ``BottomupResize`` applied (center ``input_center``, span
        ``input_scale``, output canvas ``input_size``) -- the bottom-up
        equivalent of ``TopdownAffine``'s crop warp, hence the same inverse
        formula as ``PoseidonPoseEstimator``/``TARViTPosePoseEstimator``.

        Args:
            data_sample: The corresponding ``PoseDataSample``, whose
                ``metainfo`` carries ``input_center``/``input_scale``/
                ``input_size`` (set by ``BottomupResize``).
            bboxes (np.ndarray): ``(N, 4)`` bboxes in padded-canvas pixel
                space.
            keypoints (np.ndarray): ``(N, K, 2)`` keypoints in padded-canvas
                pixel space.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(bboxes, keypoints)`` in
            original-image pixel coordinates.
        """
        meta = data_sample.metainfo
        input_center = np.array(meta['input_center'], dtype=np.float32)
        input_scale = np.array(meta['input_scale'], dtype=np.float32)
        input_size = np.array(meta['input_size'], dtype=np.float32)

        def _unwarp(xy):
            return xy / input_size * input_scale + input_center \
                - 0.5 * input_scale

        keypoints = _unwarp(keypoints)
        bboxes = bboxes.reshape(-1, 2, 2)
        bboxes = _unwarp(bboxes).reshape(-1, 4)
        return bboxes, keypoints

    def _to_coco(self, keypoints: np.ndarray, scores: np.ndarray):
        """Reproject PAVE-Net's native 15-keypoint layout onto COCO-17.

        Args:
            keypoints (np.ndarray): Native keypoints, shape (N, 15, 2).
            scores (np.ndarray): Native keypoint scores, shape (N, 15).

        Returns:
            tuple[np.ndarray, np.ndarray]: COCO-17 ``(keypoints, scores)``,
            shapes ``(N, 17, 2)`` and ``(N, 17)``. ``left_eye``/
            ``right_eye``/``left_ear``/``right_ear`` have zero score (no
            counterpart in PAVE-Net's native layout -- see module
            docstring).
        """
        n = keypoints.shape[0]
        coco_keypoints = np.zeros((n, 17, 2), dtype=np.float32)
        coco_scores = np.zeros((n, 17), dtype=np.float32)
        for src_idx, dst_idx in _PAVENET_TO_COCO:
            coco_keypoints[:, dst_idx] = keypoints[:, src_idx]
            coco_scores[:, dst_idx] = scores[:, src_idx]
        return coco_keypoints, coco_scores
