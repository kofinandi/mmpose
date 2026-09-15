"""Inference-only wrapper for DETRPose-MOTIP (``external/detrpose-motip``).

DETRPose-MOTIP is a single-stage multi-person pose estimator (HGNetv2 +
HybridEncoder + transformer queries) with MOTIP's online ID decoder on
top.  Detection and identity assignment run in one ``predict()`` call:
the detector is the inner ``MOTIP.detr``, association is the unmodified
``PoseRuntimeTracker`` (trajectory modeling + ID decoder, called
frame-by-frame).  This is a **different model** from
:class:`~mmpose.models.pose_estimators.detrpose_wrapper.DETRPoseEstimator`,
which has no tracker.

The wrapper loads the training checkout unmodified (``sys.path`` insert +
``LazyConfig.instantiate``) and translates I/O to MMPose
``PoseDataSample`` for ``tools/benchmark_e2e.py``.  Configs that use this
wrapper must set ``emits_track_ids = True`` so the driver calls
:meth:`reset_tracking` at every video boundary.

**Fidelity notes:**

- Detector, ``TrackPostProcess``, and ``PoseRuntimeTracker`` are the
  upstream modules, unmodified.  Sequence reset matches
  ``evaluate_motip``'s ``seq_name`` change.
- Upstream eval batches the *stateless* detector (default 8 frames) then
  steps the tracker one frame at a time.  ``run_tracking()`` uses batch
  size 1; that is a speed difference only.
- ``ck['ema']`` is typically ``None`` (MOTIP-S trains with
  ``use_ema=False``).  Weights are loaded from ``ck['model']`` unless
  ``use_ema=True``.  Do not copy the DETRPose wrapper's EMA-first rule.
- Native layout is PoseTrack-17 with never-annotated ears dropped (15
  joints).  ``map_to_coco=True`` (the default) reprojects onto COCO-17
  for ``tools/benchmark_e2e.py``.  ``left_eye`` / ``right_eye`` /
  ``left_ear`` / ``right_ear`` stay at zero confidence; ``head_bottom`` /
  ``head_top`` have no COCO counterpart and are dropped.
- Bounding boxes stored on ``pred_instances`` are **derived** as
  axis-aligned min/max over the native keypoints (upstream has no box
  head).
- Per-keypoint visibility is hard-coded to ``1.0`` upstream; we mirror
  that.
- Detection filtering is ``PoseRuntimeTracker``'s ``det_thresh`` /
  ``newborn_thresh`` / ``id_thresh`` / ``area_thresh``, not a wrapper
  score cut.  ``score_thr`` is accepted for API compatibility and is
  not applied.
- Resize in the MMPose pipeline is OpenCV bilinear 640x640
  (``BottomupRandomChoiceResize``, ``keep_ratio=False``).  Upstream eval
  uses torchvision/PIL bilinear.  Pixel-level keypoints can differ
  slightly; association is still the published tracker.

Usage::

    emits_track_ids = True

    model = dict(
        type='DETRPoseMOTIPEstimator',
        config='external/detrpose-motip/configs/detrpose/'
               'motip_detrpose_hgnetv2_s_posetrack21.py',
        checkpoint='path/to/checkpoint0039.pth',
        motip_root='external/detrpose-motip',
        map_to_coco=True,
        data_preprocessor=dict(
            type='PoseDataPreprocessor',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            bgr_to_rgb=True,
            pad_size_divisor=1),
    )
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmpose.registry import MODELS
from mmpose.utils.typing import SampleList

_MOTIP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'external',
                 'detrpose-motip'))

_DEFAULT_CONFIG = os.path.join(
    'configs', 'detrpose', 'motip_detrpose_hgnetv2_s_posetrack21.py')

# PoseTrack-17 with left_ear/right_ear (indices 3, 4) removed.  Matches
# DETRPose's PT21_KEYPOINT_NAMES and PAVE-Net's native 15-joint layout.
# Maps native index -> COCO-17 index; head_bottom/head_top have no COCO
# counterpart and are dropped (left at zero score).
_PT21_15_TO_COCO = [
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

_TRACKER_CFG_KEYS = (
    ('tracker_det_thresh', 'det_thresh', 0.3),
    ('tracker_newborn_thresh', 'newborn_thresh', 0.6),
    ('tracker_id_thresh', 'id_thresh', 0.2),
    ('tracker_area_thresh', 'area_thresh', 0.0),
    ('tracker_miss_tolerance', 'miss_tolerance', 20),
    ('tracker_assignment_protocol', 'assignment_protocol', 'object-max'),
)


def _bboxes_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Axis-aligned xyxy from keypoint min/max (derived, not upstream)."""
    n = keypoints.shape[0]
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xy_min = keypoints.min(axis=1)
    xy_max = keypoints.max(axis=1)
    return np.concatenate([xy_min, xy_max], axis=1).astype(np.float32)


def _pt21_15_to_coco(keypoints: np.ndarray, scores: np.ndarray):
    """Reproject PoseTrack21's native 15-keypoint layout onto COCO-17."""
    n = keypoints.shape[0]
    coco_keypoints = np.zeros((n, 17, 2), dtype=np.float32)
    coco_scores = np.zeros((n, 17), dtype=np.float32)
    for src_idx, dst_idx in _PT21_15_TO_COCO:
        coco_keypoints[:, dst_idx] = keypoints[:, src_idx]
        coco_scores[:, dst_idx] = scores[:, src_idx]
    return coco_keypoints, coco_scores


def _ensure_motip_on_path(motip_root: str) -> str:
    """Put ``external/detrpose-motip`` on ``sys.path`` so ``src.*`` imports.

    ``src`` is a generic top-level name.  If a *different* ``src`` package
    is already in ``sys.modules``, importing MOTIP would silently bind
    the wrong tree.  Fail loudly in that case.
    """
    motip_root = os.path.abspath(motip_root)
    if not os.path.isdir(motip_root):
        raise FileNotFoundError(
            f'detrpose-motip root not found: {motip_root}. '
            'Clone or copy the training checkout to external/detrpose-motip.')
    expected_src = os.path.abspath(os.path.join(motip_root, 'src'))
    existing = sys.modules.get('src')
    if existing is not None:
        existing_file = getattr(existing, '__file__', None) or ''
        existing_dir = (
            os.path.abspath(os.path.dirname(existing_file))
            if existing_file else '')
        if not existing_dir.startswith(expected_src):
            raise RuntimeError(
                'DETRPoseMOTIPEstimator needs to import `src` from '
                f'{expected_src}, but `src` is already loaded from '
                f'{existing_file!r}. Run this wrapper in a process that '
                'does not import a different top-level `src` package.')
    if motip_root not in sys.path:
        sys.path.insert(0, motip_root)
    return motip_root


def _resolve_path(path: str, motip_root: str) -> str:
    """Resolve ``path`` against CWD, then the MOTIP checkout."""
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(motip_root, path)),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f'File not found: {path}. Looked in {candidates}.')


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    getter = getattr(cfg, 'get', None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def _tracker_kwargs_from_cfg(cfg: Any,
                             override: Optional[Dict[str, Any]]) -> dict:
    params = _cfg_get(cfg, 'training_params', cfg)
    kwargs: Dict[str, Any] = {}
    for src, dst, default in _TRACKER_CFG_KEYS:
        kwargs[dst] = _cfg_get(params, src, default)
    if override:
        kwargs.update(override)
    kwargs['miss_tolerance'] = int(kwargs['miss_tolerance'])
    kwargs['assignment_protocol'] = str(kwargs['assignment_protocol'])
    for key in ('det_thresh', 'newborn_thresh', 'id_thresh', 'area_thresh'):
        kwargs[key] = float(kwargs[key])
    return kwargs


def _load_motip_state_dict(checkpoint: str, use_ema: bool) -> dict:
    """Load a MOTIP trainer ``.pth`` into a flat state_dict.

    MOTIP-S trains with ``use_ema=False``, so ``ck['ema']`` is ``None``.
    Default is therefore ``ck['model']``.  Requesting EMA when the blob
    is missing is an error, not a silent fallback.
    """
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f'MOTIP checkpoint not found: {checkpoint}.')
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f'Unexpected MOTIP checkpoint type {type(ckpt)} in {checkpoint}')
    if use_ema:
        ema = ckpt.get('ema')
        if ema is None:
            raise RuntimeError(
                f'use_ema=True but {checkpoint} has ema=None. '
                'This MOTIP-S run trained with use_ema=False; leave '
                'use_ema=False so ck["model"] is loaded.')
        state = ema['module'] if isinstance(ema, dict) and 'module' in ema \
            else ema
    elif 'model' in ckpt:
        state = ckpt['model']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise RuntimeError(
            f'Could not extract state_dict from MOTIP checkpoint '
            f'{checkpoint}.')
    keys = list(state.keys())
    if keys and all(k.startswith('module.') for k in keys):
        state = {k[len('module.'):]: v for k, v in state.items()}
    return state


@MODELS.register_module()
class DETRPoseMOTIPEstimator(BaseModel):
    """MMPose inference wrapper for DETRPose-MOTIP (bottom-up + tracking).

    Args:
        config (str): Upstream LazyConfig path
            (``motip_detrpose_hgnetv2_*_posetrack21.py``).  Relative paths
            are resolved against CWD, then ``motip_root``.
        checkpoint (str | None): Local ``.pth`` from ``train_motip.py``.
            Required; ``init_model`` injects the CLI path into this field.
        motip_root (str): Path to the ``external/detrpose-motip`` checkout.
        tracker_cfg (dict | None): Overrides for ``PoseRuntimeTracker``
            (``det_thresh``, ``newborn_thresh``, ``id_thresh``,
            ``area_thresh``, ``miss_tolerance``, ``assignment_protocol``).
            Defaults come from the LazyConfig's ``training_params.tracker_*``.
        use_ema (bool): Load ``ck['ema']['module']``.  Default ``False``
            (MOTIP-S checkpoints store ``ema=None``).
        map_to_coco (bool): Reproject the native 15-keypoint output onto
            COCO-17.  Default ``True``.
        score_thr (float): Unused.  Filtering is ``PoseRuntimeTracker``'s
            ``det_thresh``.  Kept so configs can share a field with
            :class:`DETRPoseEstimator`.
        device (str): Unused at build time; weights stay CPU until
            ``.to(device)`` from ``init_model``.
        data_preprocessor (dict | None): ``PoseDataPreprocessor`` config.
            Must produce ``[0, 1]`` RGB tensors (mean 0, std 255).
        init_cfg: Unused; API compatibility.
    """

    def __init__(
        self,
        config: str = _DEFAULT_CONFIG,
        checkpoint: Optional[str] = None,
        motip_root: str = _MOTIP_ROOT,
        tracker_cfg: Optional[dict] = None,
        use_ema: bool = False,
        map_to_coco: bool = True,
        score_thr: float = 0.0,
        device: str = 'cuda:0',
        data_preprocessor: Optional[dict] = None,
        init_cfg=None,
    ):
        if data_preprocessor is not None and isinstance(data_preprocessor,
                                                         dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.score_thr = float(score_thr)
        self._device_hint = device
        self.use_ema = bool(use_ema)

        motip_root = _ensure_motip_on_path(motip_root)
        self.motip_root = motip_root
        config_path = _resolve_path(config, motip_root)
        self.config_path = config_path

        from src.core import LazyConfig, instantiate
        from src.models.motip.runtime_tracker import PoseRuntimeTracker

        self._PoseRuntimeTracker = PoseRuntimeTracker

        cfg = LazyConfig.load(config_path)
        backbone = _cfg_get(_cfg_get(_cfg_get(cfg, 'model'), 'detr'),
                            'backbone')
        if backbone is not None and hasattr(backbone, 'pretrained'):
            backbone.pretrained = False

        self.motip = instantiate(cfg.model)
        self.postprocessor = instantiate(cfg.postprocessor)
        self.tracker_kwargs = _tracker_kwargs_from_cfg(cfg, tracker_cfg)
        self.num_body_points = int(
            getattr(self.postprocessor, 'num_body_points', 15))

        if checkpoint is None:
            raise FileNotFoundError(
                'DETRPoseMOTIPEstimator requires a local checkpoint path. '
                'Pass checkpoint=... in the config or via init_model().')
        checkpoint = _resolve_path(checkpoint, motip_root)
        self.checkpoint = checkpoint
        state = _load_motip_state_dict(checkpoint, use_ema=self.use_ema)
        incompatible = self.motip.load_state_dict(state, strict=False)
        missing = list(getattr(incompatible, 'missing_keys', []) or [])
        unexpected = list(getattr(incompatible, 'unexpected_keys', []) or [])
        missing = [k for k in missing if 'num_batches_tracked' not in k]
        if missing or unexpected:
            raise RuntimeError(
                f'MOTIP checkpoint appears incompatible with {config_path}: '
                f'{len(missing)} missing / {len(unexpected)} unexpected '
                f'keys. First missing: {missing[:5]}; first unexpected: '
                f'{unexpected[:5]}')

        self.motip.eval()
        self.postprocessor.eval()

        self.map_to_coco = bool(map_to_coco)
        if self.map_to_coco and self.num_body_points != 15:
            raise ValueError(
                'DETRPoseMOTIPEstimator map_to_coco=True is the PoseTrack21 '
                '15-keypoint (ears dropped) -> COCO-17 remap. This model '
                f'reports num_body_points={self.num_body_points}.')

        # Built in reset_tracking() after init_model() moves weights to GPU.
        self.tracker = None

    def reset_tracking(self) -> None:
        """Clear MOTIP trajectory state; called at every sequence boundary.

        Constructs a fresh :class:`PoseRuntimeTracker` on the current
        device.  Must not run in ``__init__``: ``init_model`` only calls
        ``.to(device)`` afterwards, and the tracker caches GPU tensors.
        """
        device = next(self.motip.parameters()).device
        self.tracker = self._PoseRuntimeTracker(
            self.motip,
            num_body_points=self.num_body_points,
            device=device,
            **self.tracker_kwargs,
        )

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        if mode == 'loss':
            raise NotImplementedError(
                'DETRPoseMOTIPEstimator is for inference only.')
        raise ValueError(f'Unsupported mode "{mode}"')

    def _orig_sizes(self, inputs: torch.Tensor,
                    data_samples: SampleList) -> torch.Tensor:
        orig_list = []
        for ds in data_samples:
            ori = ds.metainfo.get('ori_shape', None)
            if ori is None:
                h, w = inputs.shape[-2:]
            else:
                h, w = int(ori[0]), int(ori[1])
            orig_list.append([w, h])
        return torch.tensor(
            orig_list, dtype=torch.float32, device=inputs.device)

    def _pack_frame(self, track_result: dict):
        scores = track_result['scores'].detach().float().cpu().numpy()
        keypoints = track_result['keypoints'].detach().float().cpu().numpy()
        track_ids = track_result['track_ids'].detach().cpu().numpy()

        n = int(scores.shape[0])
        out_k = 17 if self.map_to_coco else self.num_body_points
        if n == 0:
            return (
                np.zeros((0, out_k, 2), dtype=np.float32),
                np.zeros((0, out_k), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, ), dtype=np.float32),
                np.zeros((0, ), dtype=np.int32),
            )

        native = keypoints.reshape(n, self.num_body_points, 3)
        native_kpts = native[:, :, :2].astype(np.float32)
        native_scores = np.ones((n, self.num_body_points), dtype=np.float32)
        bboxes = _bboxes_from_keypoints(native_kpts)
        bbox_scores = scores.astype(np.float32)
        if self.map_to_coco:
            kpts, kpt_scores = _pt21_15_to_coco(native_kpts, native_scores)
        else:
            kpts, kpt_scores = native_kpts, native_scores
        return (
            kpts, kpt_scores, bboxes, bbox_scores,
            np.asarray(track_ids, dtype=np.int32).reshape(-1),
        )

    def predict(self, inputs: torch.Tensor,
                data_samples: SampleList) -> SampleList:
        """Run detector + MOTIP tracker and pack ``pred_instances``."""
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(list(inputs), dim=0)
        if self.tracker is None:
            self.reset_tracking()

        orig_sizes = self._orig_sizes(inputs, data_samples)
        with torch.no_grad():
            outputs = self.motip(part='detr', samples=inputs)
            results: Sequence[dict] = self.postprocessor(outputs, orig_sizes)

        for i, result in enumerate(results):
            track_result = self.tracker.update(
                result['scores'], result['keypoints'],
                result['instance_embeds'])
            kpts, kpt_scores, bboxes, bbox_scores, track_ids = \
                self._pack_frame(track_result)

            pred = InstanceData()
            pred.keypoints = kpts
            pred.keypoint_scores = kpt_scores
            pred.keypoints_visible = kpt_scores
            pred.bboxes = bboxes
            pred.bbox_scores = bbox_scores
            pred.track_ids = track_ids
            data_samples[i].pred_instances = pred

        return data_samples
