"""Inference-only wrapper for GroupPose (``external/GroupPose``).

GroupPose (`Liu et al., ICCV 2023 <https://arxiv.org/abs/2308.07313>`_) is a
DETR-style end-to-end multi-person pose estimator.  This wrapper loads the
upstream code unmodified (after compiling
``MultiScaleDeformableAttention``) and translates I/O to MMPose
``PoseDataSample`` for ``tools/benchmark_e2e.py``.

**Fidelity notes:**

- Upstream PostProcess returns scores, labels, and COCO ``x,y,v`` keypoints
  only (no boxes).  ``pred_instances.bboxes`` are **derived** from keypoint
  min/max so ``CocoMetric(score_mode='bbox')`` can run; ``bbox_scores`` are
  the instance class scores.
- Swin backbones call ``torch.load`` on ImageNet weights at build time.  When
  those files are absent we skip that load (strict=False empty dict) because
  the released pose checkpoint already contains the backbone.

Compile the CUDA op once before first use::

    cd external/GroupPose/models/grouppose/ops
    python setup.py build install
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from typing import Optional

import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmpose.registry import MODELS
from mmpose.utils.typing import SampleList

_GROUPPOSE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'external',
                 'GroupPose'))


def _bboxes_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Axis-aligned xyxy from keypoint min/max (derived, not upstream)."""
    n = keypoints.shape[0]
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xy_min = keypoints.min(axis=1)
    xy_max = keypoints.max(axis=1)
    return np.concatenate([xy_min, xy_max], axis=1).astype(np.float32)


def _ensure_msda_importable() -> None:
    try:
        import MultiScaleDeformableAttention  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            'GroupPose requires the compiled MultiScaleDeformableAttention '
            'CUDA extension. Build it with:\n'
            '  cd external/GroupPose/models/grouppose/ops\n'
            '  python setup.py build install\n'
            f'Original import error: {exc}') from exc


def _install_torchvision_pretrained_shim() -> None:
    """Map legacy ``pretrained=`` to torchvision >=0.13 ``weights=``."""
    import torchvision.models as tvm

    for name in ('resnet50', 'resnet101'):
        orig = getattr(tvm, name, None)
        if orig is None or getattr(orig, '_mmpose_pretrained_shim', False):
            continue

        def _make(orig_fn):

            def wrapped(*args, pretrained=None, weights=None, **kwargs):
                if pretrained is not None and weights is None:
                    weights = 'DEFAULT' if pretrained else None
                return orig_fn(*args, weights=weights, **kwargs)

            wrapped._mmpose_pretrained_shim = True  # type: ignore[attr-defined]
            return wrapped

        setattr(tvm, name, _make(orig))


def _load_grouppose_state_dict(checkpoint: str) -> dict:
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f'GroupPose checkpoint not found: {checkpoint}. '
            'Download from the official Google Drive folder '
            'https://drive.google.com/drive/folders/'
            '1exJMkr7j_HbItRM-u7DWT7scx1n4htiF and place under data/models/.')
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        return ckpt['model']
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(
        f'Unexpected GroupPose checkpoint type {type(ckpt)} in {checkpoint}')


@MODELS.register_module()
class GroupPosePoseEstimator(BaseModel):
    """MMPose inference wrapper for GroupPose (bottom-up, end-to-end).

    Args:
        config_file (str): Upstream SLConfig path relative to
            ``grouppose_root``, default ``config/grouppose.py``.
        checkpoint (str | None): Local pose ``.pth`` path.
        backbone (str): ``resnet50``, ``swin_T_224_1k``, or
            ``swin_L_384_22k``.
        num_body_points (int): 17 for COCO, 14 for CrowdPose.
        grouppose_root (str): Path to ``external/GroupPose``.
        swin_pretrain_path (str): Directory for optional ImageNet Swin
            weights (skipped if files are missing).
        score_thr (float): Filter instance scores; ``0.0`` keeps all
            ``num_select`` queries (official eval).
        device (str): Accepted for custom-wrapper ``init_model`` API.
        data_preprocessor (dict | None): ``PoseDataPreprocessor`` config
            (ImageNet mean/std on uint8).
        init_cfg: Unused; API compatibility.
    """

    def __init__(
        self,
        config_file: str = 'config/grouppose.py',
        checkpoint: Optional[str] = None,
        backbone: str = 'resnet50',
        num_body_points: int = 17,
        grouppose_root: str = _GROUPPOSE_ROOT,
        swin_pretrain_path: str = '.',
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
        self.num_keypoints = int(num_body_points)
        self._device_hint = device

        grouppose_root = os.path.abspath(grouppose_root)
        if grouppose_root not in sys.path:
            sys.path.insert(0, grouppose_root)

        _ensure_msda_importable()
        _install_torchvision_pretrained_shim()

        from models.registry import MODULE_BUILD_FUNCS
        from util.slconfig import SLConfig
        # Import side-effect: registers build_grouppose.
        import models.grouppose  # noqa: F401

        cfg_path = config_file
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(grouppose_root, cfg_path)
        cfg = SLConfig.fromfile(cfg_path)

        args = Namespace()
        for k, v in cfg._cfg_dict.to_dict().items():
            setattr(args, k, v)
        args.backbone = backbone
        args.swin_pretrain_path = swin_pretrain_path
        args.num_body_points = num_body_points
        args.device = 'cpu'  # build on CPU; moved later by init_model
        args.dataset_file = 'coco' if num_body_points == 17 else 'crowdpose'
        if not hasattr(args, 'use_ema'):
            args.use_ema = False
        if not hasattr(args, 'debug'):
            args.debug = False

        # Skip missing ImageNet Swin files; pose ckpt overwrites backbone.
        _real_torch_load = torch.load

        def _torch_load_skip_missing_swin(path, *a, **kw):
            if (isinstance(path, (str, os.PathLike)) and not os.path.isfile(path)
                    and 'swin' in os.path.basename(str(path)).lower()):
                return {'model': {}}
            return _real_torch_load(path, *a, **kw)

        torch.load = _torch_load_skip_missing_swin  # type: ignore[assignment]
        try:
            assert args.modelname in MODULE_BUILD_FUNCS._module_dict
            build_func = MODULE_BUILD_FUNCS.get(args.modelname)
            model, _criterion, postprocessors = build_func(args)
        finally:
            torch.load = _real_torch_load  # type: ignore[assignment]

        if checkpoint is None:
            raise FileNotFoundError(
                'GroupPosePoseEstimator requires a local checkpoint path.')
        state = _load_grouppose_state_dict(checkpoint)
        incompatible = model.load_state_dict(state, strict=False)
        missing = list(getattr(incompatible, 'missing_keys', []) or [])
        if len(missing) > 50:
            raise RuntimeError(
                f'GroupPose checkpoint appears incompatible: '
                f'{len(missing)} missing keys. First: {missing[:5]}')

        model.eval()
        self.grouppose_model = model
        self.postprocessor = postprocessors['keypoints']

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        if mode == 'loss':
            raise NotImplementedError(
                'GroupPosePoseEstimator is for inference only.')
        raise ValueError(f'Unsupported mode "{mode}"')

    def _inference_forward(self, samples, orig_sizes: torch.Tensor):
        """NN + PostProcess (FPS metric target)."""
        outputs = self.grouppose_model(samples)
        return self.postprocessor(outputs, orig_sizes)

    def predict(self, inputs: torch.Tensor,
                data_samples: SampleList) -> SampleList:
        from util.misc import nested_tensor_from_tensor_list

        device = inputs.device
        # Split batch into a list so NestedTensor masks match each image.
        # benchmark_e2e uses batch_size=1, but keep the general path.
        tensor_list = [inputs[i] for i in range(inputs.shape[0])]
        samples = nested_tensor_from_tensor_list(tensor_list).to(device)

        orig_list = []
        for ds in data_samples:
            ori = ds.metainfo.get('ori_shape', None)
            if ori is None:
                h, w = int(inputs.shape[-2]), int(inputs.shape[-1])
            else:
                h, w = int(ori[0]), int(ori[1])
            # PostProcess expects (H, W).
            orig_list.append([h, w])
        orig_sizes = torch.tensor(
            orig_list, dtype=torch.float32, device=device)

        with torch.no_grad():
            results = self._inference_forward(samples, orig_sizes)

        k = self.num_keypoints
        for i, res in enumerate(results):
            scores = res['scores'].detach().float().cpu().numpy()
            kpts_flat = res['keypoints'].detach().float().cpu().numpy()
            keep = scores > self.score_thr
            scores = scores[keep]
            kpts_flat = kpts_flat[keep]

            n = int(scores.shape[0])
            if n == 0:
                kpts = np.zeros((0, k, 2), dtype=np.float32)
                kpt_scores = np.zeros((0, k), dtype=np.float32)
                bboxes = np.zeros((0, 4), dtype=np.float32)
                bbox_scores = np.zeros((0, ), dtype=np.float32)
            else:
                kpts_xyz = kpts_flat.reshape(n, k, 3)
                kpts = kpts_xyz[:, :, :2].astype(np.float32)
                kpt_scores = kpts_xyz[:, :, 2].astype(np.float32)
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
