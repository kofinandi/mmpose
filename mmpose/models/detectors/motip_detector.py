# Copyright (c) OpenMMLab. All rights reserved.
"""Official MOTIP detector adapter for top-down pose pipelines.

This wraps the published MOTIP box tracker (MCG-NJU/MOTIP, CVPR 2025),
not :class:`~mmpose.models.pose_estimators.detrpose_motip_wrapper.DETRPoseMOTIPEstimator`.
The network, ``RuntimeTracker``, resize, and ID assignment are the
upstream modules. The wrapper translates BGR numpy images into the
upstream PIL / torchvision input and xywh boxes into xyxy.

**Fidelity notes:**

- ``predict_imgs`` accepts one image. ``reset_tracking`` drops the
  tracker; the next call builds a new ``RuntimeTracker`` with that
  frame's height and width, matching ``submit_and_evaluate``'s
  per-sequence tracker. A later frame in the same sequence whose size
  differs raises, because upstream unnormalises every box with the
  sequence size captured at construction.
- Shorter-side resize is 800, the default of ``submit_and_evaluate``'s
  ``image_max_shorter`` argument (it is not a yaml key). The longer-side
  cap is the yaml's ``INFERENCE_MAX_LONGER``. Score thresholds, miss
  tolerance, and the assignment protocol are read from the yaml after
  ``load_super_config``, so SportsMOT's overrides (miss tolerance 60,
  relative position length 60) are kept.
- Track ids are MOTIP's ``next_id`` values and start at 0. That is a
  real identity, not an empty-track sentinel.
- TF32 is left at the process default. Upstream turns it off in
  ``submit_and_evaluate.py``'s ``__main__``, but this process also runs
  the pose model, and the flag is global.
- A failed image decode must not be passed in. A blank frame would be
  tracked as a real observation.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from PIL import Image

try:
    from mmdet.registry import MODELS
    from mmdet.structures import DetDataSample
except ImportError as exc:
    raise ImportError(
        'mmdet is required for MOTIPDetector. '
        'Install it with: pip install mmdet') from exc

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Top-level names MOTIP imports. A different package already bound to
# one of these would be used silently after the sys.path insert.
_GENERIC_PACKAGES = ('models', 'utils', 'structures', 'configs')

# Upstream submit_and_evaluate passes this as image_max_shorter. It is
# not stored in the yaml.
_DEFAULT_MAX_SHORTER = 800

_MOTIP_DEVICE = {'device': torch.device('cpu')}
_UPSTREAM = None


def _patched_distributed_device():
    """Device for ``RuntimeTracker`` tensors.

    ``RuntimeTracker`` binds ``distributed_device`` at import. Accelerate's
    ``PartialState`` would otherwise pin that state to its own default GPU.
    """
    return _MOTIP_DEVICE['device']


def _package_dir(module) -> str:
    module_file = getattr(module, '__file__', None) or ''
    if not module_file:
        paths = getattr(module, '__path__', None)
        if paths:
            return os.path.abspath(next(iter(paths)))
        return ''
    return os.path.abspath(os.path.dirname(module_file))


def _ensure_motip_on_path(motip_root: str) -> str:
    motip_root = os.path.abspath(motip_root)
    if not os.path.isdir(motip_root):
        raise FileNotFoundError(
            f'MOTIP root not found: {motip_root}. '
            'Add the official checkout as external/MOTIP.')
    for name in _GENERIC_PACKAGES:
        existing = sys.modules.get(name)
        if existing is None:
            continue
        existing_dir = _package_dir(existing)
        if not existing_dir.startswith(motip_root + os.sep) and (
                existing_dir != motip_root):
            raise RuntimeError(
                f'MOTIPDetector needs to import `{name}` from {motip_root}, '
                f'but `{name}` is already loaded from {existing_dir!r}. '
                'Run this wrapper in a process that has not imported a '
                f'different top-level `{name}` package.')
    if motip_root not in sys.path:
        sys.path.insert(0, motip_root)
    return motip_root


def _import_upstream(motip_root: str):
    """Import MOTIP after the device hook is installed."""
    global _UPSTREAM
    motip_root = _ensure_motip_on_path(motip_root)
    import utils.misc as motip_misc
    motip_misc.distributed_device = _patched_distributed_device
    import models.runtime_tracker as runtime_tracker
    runtime_tracker.distributed_device = _patched_distributed_device
    from configs.util import load_super_config
    from models.misc import load_checkpoint
    from models.motip import build as build_motip
    from models.runtime_tracker import RuntimeTracker
    from utils.misc import yaml_to_dict
    from utils.nested_tensor import nested_tensor_from_tensor_list
    _UPSTREAM = SimpleNamespace(
        motip_root=motip_root,
        load_super_config=load_super_config,
        load_checkpoint=load_checkpoint,
        build_motip=build_motip,
        RuntimeTracker=RuntimeTracker,
        yaml_to_dict=yaml_to_dict,
        nested_tensor_from_tensor_list=nested_tensor_from_tensor_list,
    )
    return _UPSTREAM


def _resolve_file(path: str, motip_root: str) -> str:
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(motip_root, path)),
        os.path.abspath(os.path.join(_REPO_ROOT, path)),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f'{path} was not found relative to the working directory, '
        f'{motip_root}, or {_REPO_ROOT}.')


def _load_motip_config(config_path: str, motip_root: str) -> dict:
    """Load a MOTIP yaml, resolving ``SUPER_CONFIG_PATH`` inside the checkout.

    SportsMOT's super path is ``./configs/...``. Opening that from the
    mmpose working directory would miss the file. ``load_super_config``
    then overlays the child, so SportsMOT's miss tolerance and relative
    position length replace the DanceTrack values.
    """
    upstream = _import_upstream(motip_root)
    config_path = _resolve_file(config_path, motip_root)
    config = upstream.yaml_to_dict(config_path)
    super_path = config.get('SUPER_CONFIG_PATH')
    if super_path:
        resolved = _resolve_file(super_path, motip_root)
        config = upstream.load_super_config(config, resolved)
    return config


def _inference_dtype(name: str) -> torch.dtype:
    key = str(name).upper()
    if key == 'FP32':
        return torch.float32
    if key == 'FP16':
        return torch.float16
    raise ValueError(
        f'INFERENCE_DTYPE {name!r} is not supported. Use FP32 or FP16.')


def _empty_instances() -> InstanceData:
    pred = InstanceData()
    pred.bboxes = torch.zeros((0, 4), dtype=torch.float32)
    pred.scores = torch.zeros((0, ), dtype=torch.float32)
    pred.labels = torch.zeros((0, ), dtype=torch.long)
    pred.track_ids = torch.zeros((0, ), dtype=torch.int32)
    return pred


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1).to(dtype=torch.float32)


@MODELS.register_module()
class MOTIPDetector(nn.Module):
    """MMDet-style wrapper around official MOTIP.

    ``emits_track_ids`` is true: ``tools/benchmark_e2e.py`` steps this
    detector one frame at a time, calls :meth:`reset_tracking` on each
    sequence change, and copies ``pred_instances.track_ids`` onto the
    pose instances. A second NMS is not applied.
    """

    emits_track_ids = True

    def __init__(self,
                 config: str,
                 checkpoint: str,
                 motip_root: str = 'external/MOTIP',
                 device: str = 'cuda:0',
                 max_shorter: int = _DEFAULT_MAX_SHORTER) -> None:
        super().__init__()
        from torchvision.transforms import v2

        if not os.path.isabs(motip_root):
            motip_root = os.path.join(_REPO_ROOT, motip_root)
        self.motip_root = os.path.abspath(motip_root)
        self.config_path = config
        self.checkpoint = checkpoint
        self._device = torch.device(device)
        self.max_shorter = int(max_shorter)
        _MOTIP_DEVICE['device'] = self._device

        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(
                f'MOTIP checkpoint not found: {self.checkpoint}. '
                'Download a published weight, for example '
                'https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/'
                'r50_deformable_detr_motip_dancetrack.pth')

        motip_cfg = _load_motip_config(config, self.motip_root)
        only_detr = motip_cfg['INFERENCE_ONLY_DETR']
        if only_detr is None:
            only_detr = motip_cfg['ONLY_DETR']
        if only_detr:
            raise ValueError(
                f'{config} sets ONLY_DETR / INFERENCE_ONLY_DETR. '
                'MOTIPDetector is the tracking model and requires the '
                'ID decoder.')

        self.max_longer = int(motip_cfg['INFERENCE_MAX_LONGER'])
        self.det_thresh = float(motip_cfg['DET_THRESH'])
        self.newborn_thresh = float(motip_cfg['NEWBORN_THRESH'])
        self.id_thresh = float(motip_cfg['ID_THRESH'])
        self.area_thresh = motip_cfg.get('AREA_THRESH', 0)
        self.miss_tolerance = int(motip_cfg['MISS_TOLERANCE'])
        self.assignment_protocol = str(
            motip_cfg.get('ASSIGNMENT_PROTOCOL', 'hungarian'))
        self.use_sigmoid = bool(motip_cfg.get('USE_FOCAL_LOSS', False))
        self._dtype = _inference_dtype(
            motip_cfg.get('INFERENCE_DTYPE', 'FP32'))
        self._size_divisibility = int(motip_cfg.get('SIZE_DIVISIBILITY') or 0)
        # build() places the unused criterion on this device.
        motip_cfg['DEVICE'] = str(self._device)

        upstream = _import_upstream(self.motip_root)
        model, _criterion = upstream.build_motip(config=motip_cfg)
        upstream.load_checkpoint(model, path=self.checkpoint)
        model.to(device=self._device, dtype=self._dtype)
        model.eval()
        # Keep MOTIP off this module's parameter tree. init_det_model does
        # not call .to() itself, and the tracker holds the same module.
        object.__setattr__(self, '_motip', model)
        object.__setattr__(self, '_tracker', None)
        object.__setattr__(self, '_seq_hw', None)
        object.__setattr__(self, '_upstream', upstream)

        self._transform = v2.Compose([
            v2.Resize(size=self.max_shorter, max_size=self.max_longer),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.cfg = None
        self.dataset_meta = {'classes': ('person',)}

    def reset_tracking(self) -> None:
        """Drop trajectory state. The next frame starts a new sequence."""
        object.__setattr__(self, '_tracker', None)
        object.__setattr__(self, '_seq_hw', None)

    def _ensure_tracker(self, height: int, width: int):
        _MOTIP_DEVICE['device'] = self._device
        if self._tracker is None:
            tracker = self._upstream.RuntimeTracker(
                model=self._motip,
                sequence_hw=(height, width),
                use_sigmoid=self.use_sigmoid,
                assignment_protocol=self.assignment_protocol,
                miss_tolerance=self.miss_tolerance,
                det_thresh=self.det_thresh,
                newborn_thresh=self.newborn_thresh,
                id_thresh=self.id_thresh,
                area_thresh=self.area_thresh,
                only_detr=False,
                dtype=self._dtype,
            )
            object.__setattr__(self, '_tracker', tracker)
            object.__setattr__(self, '_seq_hw', (height, width))
            return tracker
        if self._seq_hw != (height, width):
            raise ValueError(
                'MOTIP unnormalises boxes with the sequence size captured '
                f'on the first frame {self._seq_hw}, but this frame is '
                f'{(height, width)}. Call reset_tracking() when the '
                'sequence changes, and keep a constant frame size inside '
                'a sequence, as submit_and_evaluate does.')
        return self._tracker

    def _preprocess(self, bgr: np.ndarray):
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(
                f'MOTIPDetector expected a BGR image of shape (H, W, 3), '
                f'got {getattr(bgr, "shape", None)}.')
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        image = Image.fromarray(rgb)
        transformed = self._transform(image)
        if self._dtype != torch.float32:
            transformed = transformed.to(self._dtype)
        nested = self._upstream.nested_tensor_from_tensor_list(
            [transformed], self._size_divisibility)
        nested.tensors = nested.tensors.to(self._device)
        nested.mask = nested.mask.to(self._device)
        return nested

    def _to_data_sample(self, bgr: np.ndarray, results: dict) -> DetDataSample:
        height, width = bgr.shape[:2]
        data_sample = DetDataSample()
        data_sample.set_metainfo(dict(
            ori_shape=(height, width),
            img_shape=(height, width),
            pad_shape=(height, width)))
        boxes = results.get('bbox')
        if boxes is None or boxes.numel() == 0:
            data_sample.pred_instances = _empty_instances()
            return data_sample

        pred = InstanceData()
        pred.bboxes = _xywh_to_xyxy(boxes.detach().cpu())
        pred.scores = results['score'].detach().cpu().to(dtype=torch.float32)
        pred.labels = results['category'].detach().cpu().to(dtype=torch.long)
        pred.track_ids = results['id'].detach().cpu().to(dtype=torch.int32)
        data_sample.pred_instances = pred
        return data_sample

    def predict_imgs(
        self,
        imgs: Sequence[np.ndarray],
    ) -> List[DetDataSample]:
        """Track one BGR image and return its boxes and track ids.

        The caller steps frames in order and calls :meth:`reset_tracking`
        at each sequence boundary. Passing more than one image is refused
        so a batch cannot cross that boundary unnoticed.
        """
        if not imgs:
            return []
        if len(imgs) != 1:
            raise ValueError(
                'MOTIPDetector tracks one frame at a time so IDs stay in '
                f'sequence order. Got a batch of {len(imgs)}. Call '
                'reset_tracking() at each sequence boundary, then '
                'predict_imgs with a single image.')
        image = imgs[0]
        height, width = int(image.shape[0]), int(image.shape[1])
        tracker = self._ensure_tracker(height, width)
        nested = self._preprocess(image)
        with torch.no_grad():
            tracker.update(image=nested)
        return [self._to_data_sample(image, tracker.get_track_results())]

    def forward(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            'MOTIPDetector is inference-only; use predict_imgs().')
