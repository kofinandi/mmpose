"""Inference-only wrapper for DETRPose (``external/DETRPose``, inference_only).

DETRPose (`Janampa & Pattichis, arXiv:2506.13027
<https://arxiv.org/abs/2506.13027>`_) is a single-stage end-to-end multi-person
pose estimator (HGNetv2 + HybridEncoder + transformer queries).  This wrapper
loads the upstream ``inference_only`` package unmodified and translates I/O to
MMPose ``PoseDataSample`` for ``tools/benchmark_e2e.py``.

**Fidelity notes (not the paper's box head):**

- Upstream emits class scores and keypoints only (no ``pred_boxes``).  Bounding
  boxes stored on ``pred_instances`` are **derived** as axis-aligned min/max
  over predicted keypoints so ``CocoMetric(score_mode='bbox')`` can run.
- Per-keypoint visibility / scores are hard-coded to ``1.0`` upstream; we
  mirror that.
- Official COCO eval keeps all top-``num_select`` queries (no score threshold).
  ``score_thr`` defaults to ``0.0`` for the same behaviour.

Usage::

    model = dict(
        type='DETRPoseEstimator',
        model_name='detrpose_hgnetv2_n',
        checkpoint='data/models/detrpose_hgnetv2_n.pth',
        detrpose_root='external/DETRPose',
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
from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData
from omegaconf import OmegaConf

from mmpose.registry import MODELS
from mmpose.utils.typing import SampleList

_DETRpose_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'external',
                 'DETRPose'))


def _bboxes_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Axis-aligned xyxy from keypoint min/max (derived, not upstream)."""
    n = keypoints.shape[0]
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xy_min = keypoints.min(axis=1)
    xy_max = keypoints.max(axis=1)
    return np.concatenate([xy_min, xy_max], axis=1).astype(np.float32)


def _load_detrpose_state_dict(checkpoint: str) -> dict:
    """Load a DETRPose ``.pth`` / HF-style dict into a flat state_dict."""
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f'DETRPose checkpoint not found: {checkpoint}. '
            'Download from '
            'https://github.com/SebastianJanampa/DETRPose/releases '
            'or Hugging Face SebasJanampa/DETRPose_* and place under '
            'data/models/.')
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f'Unexpected DETRPose checkpoint type {type(ckpt)} in {checkpoint}')
    if 'ema' in ckpt:
        state = ckpt['ema']
        if isinstance(state, dict) and 'module' in state:
            state = state['module']
    elif 'model' in ckpt:
        state = ckpt['model']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise RuntimeError(
            f'Could not extract state_dict from DETRPose checkpoint '
            f'{checkpoint} (ema/model keys present but not a dict).')
    # Strip optional wrapping prefixes from DDP / HF exports.
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ('module.', 'model.'):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


@MODELS.register_module()
class DETRPoseEstimator(BaseModel):
    """MMPose inference wrapper for DETRPose (bottom-up, end-to-end).

    Args:
        model_name (str): Upstream config stem, e.g. ``detrpose_hgnetv2_n``
            or ``detrpose_hgnetv2_n_crowdpose``.
        checkpoint (str | None): Local ``.pth`` path.  Required for offline
            eval; when ``None``, upstream would pull HF weights (not used by
            ``init_model`` CSV path).
        detrpose_root (str): Path to the ``external/DETRPose`` checkout.
        img_size (int | tuple): Square (or HxW) inference size; default 640.
        score_thr (float): Filter instance scores after PostProcess.  Use
            ``0.0`` to keep all top-k queries (official COCO eval).
        device (str): Unused at build time; weights stay CPU until
            ``.to(device)`` from ``init_model``.  Accepted for
            ``CUSTOM_POSE_WRAPPER_TYPES`` compatibility.
        data_preprocessor (dict | None): ``PoseDataPreprocessor`` config.
            Must produce ``[0, 1]`` RGB tensors (mean 0, std 255).
        init_cfg: Unused; API compatibility.
    """

    def __init__(
        self,
        model_name: str = 'detrpose_hgnetv2_n',
        checkpoint: Optional[str] = None,
        detrpose_root: str = _DETRpose_ROOT,
        img_size: Union[int, Tuple[int, int]] = 640,
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

        self.model_name = model_name
        self.checkpoint = checkpoint
        self.score_thr = float(score_thr)
        self.img_size = img_size
        self._device_hint = device

        detrpose_root = os.path.abspath(detrpose_root)
        if detrpose_root not in sys.path:
            sys.path.insert(0, detrpose_root)

        from detrpose.core import LazyConfig
        from detrpose.core.lazy import _visit_dict_config
        from detrpose.core.utils import _convert_target_to_string
        from detrpose.engine.hf_model import HFModel
        from detrpose.engine.model import _resolve_config_path

        config_path = _resolve_config_path(model_name)
        cfg = LazyConfig.load(config_path)

        def _replace_type_by_name(x):
            if '_target_' in x and callable(x._target_):
                try:
                    x._target_ = _convert_target_to_string(x._target_)
                except AttributeError:
                    pass

        cfg_copy = deepcopy(cfg)
        _visit_dict_config(cfg_copy, _replace_type_by_name)
        hf_config = {
            'model': OmegaConf.to_container(cfg_copy.model, resolve=True),
            'postprocessor':
            OmegaConf.to_container(cfg_copy.postprocessor, resolve=True),
        }

        hf_model = HFModel(hf_config)
        if checkpoint is None:
            raise FileNotFoundError(
                'DETRPoseEstimator requires a local checkpoint path. '
                'Pass checkpoint=... in the config or via init_model().')
        state = _load_detrpose_state_dict(checkpoint)
        incompatible = hf_model.model.load_state_dict(state, strict=False)
        missing = list(getattr(incompatible, 'missing_keys', []) or [])
        unexpected = list(getattr(incompatible, 'unexpected_keys', []) or [])
        # Backbone ImageNet init keys may appear as missing when resume
        # disables pretrained; warn only if the bulk of the net is empty.
        if len(missing) > 50:
            raise RuntimeError(
                f'DETRPose checkpoint appears incompatible with '
                f'{model_name}: {len(missing)} missing / '
                f'{len(unexpected)} unexpected keys. First missing: '
                f'{missing[:5]}')

        self.detrpose_model = hf_model.deploy()
        self.detrpose_model.eval()

        try:
            self.num_keypoints = int(
                self.detrpose_model.model.transformer.num_keypoints)
        except AttributeError:
            self.num_keypoints = 17

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        if mode == 'loss':
            raise NotImplementedError(
                'DETRPoseEstimator is for inference only.')
        raise ValueError(f'Unsupported mode "{mode}"')

    def _inference_forward(self, inputs: torch.Tensor,
                           orig_sizes: torch.Tensor):
        """NN + PostProcess only (FPS metric target).

        Args:
            inputs: ``(N, 3, H, W)`` in ``[0, 1]``.
            orig_sizes: ``(N, 2)`` as ``[W, H]`` original image sizes.
        """
        outputs = self.detrpose_model.model(inputs)
        return self.detrpose_model.postprocessor(outputs, orig_sizes)

    def predict(self, inputs: torch.Tensor,
                data_samples: SampleList) -> SampleList:
        """Run DETRPose and pack ``pred_instances`` (derived boxes)."""
        device = inputs.device
        batch = inputs.shape[0]

        orig_list = []
        for ds in data_samples:
            # ori_shape is (H, W); postprocessor expects [W, H].
            ori = ds.metainfo.get('ori_shape', None)
            if ori is None:
                h, w = inputs.shape[-2:]
            else:
                h, w = int(ori[0]), int(ori[1])
            orig_list.append([w, h])
        orig_sizes = torch.tensor(
            orig_list, dtype=torch.float32, device=device)

        with torch.no_grad():
            results = self._inference_forward(inputs, orig_sizes)

        # deploy PostProcess: (scores, labels, keypoints) — no boxes.
        if not isinstance(results, (tuple, list)):
            raise RuntimeError(
                f'Unexpected DETRPose postprocessor output type: '
                f'{type(results)}')
        scores_b, labels_b, keypoints_b = results[0], results[1], results[-1]

        for i in range(batch):
            scores = scores_b[i].detach().float().cpu().numpy()
            keypoints = keypoints_b[i].detach().float().cpu().numpy()
            # keypoints: (Q, K, 2)
            keep = scores > self.score_thr
            scores = scores[keep]
            keypoints = keypoints[keep]

            n = int(scores.shape[0])
            k = self.num_keypoints
            if n == 0:
                kpts = np.zeros((0, k, 2), dtype=np.float32)
                kpt_scores = np.zeros((0, k), dtype=np.float32)
                bboxes = np.zeros((0, 4), dtype=np.float32)
                bbox_scores = np.zeros((0, ), dtype=np.float32)
            else:
                kpts = keypoints[:, :, :2].astype(np.float32)
                # Upstream hard-codes visibility to 1 in non-deploy eval.
                kpt_scores = np.ones((n, k), dtype=np.float32)
                bboxes = _bboxes_from_keypoints(kpts)
                bbox_scores = scores.astype(np.float32)

            pred = InstanceData()
            pred.keypoints = kpts
            pred.keypoint_scores = kpt_scores
            pred.keypoints_visible = kpt_scores
            pred.bboxes = bboxes
            pred.bbox_scores = bbox_scores
            data_samples[i].pred_instances = pred

        return data_samples
