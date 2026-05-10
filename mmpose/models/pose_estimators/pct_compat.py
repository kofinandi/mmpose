"""Compatibility shims to allow importing external/PCT (mmpose v0.x based)
in an mmpose v1.x + mmcv 2.x environment without modifying the submodule.

Call ``install_pct_shims()`` once before importing any PCT model code.
"""

import functools
import logging
import sys
import types

import cv2
import numpy as np
import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# mmcv.runner shims
# ---------------------------------------------------------------------------

def _get_dist_info():
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def _auto_fp16(fn=None, apply_to=None, out_fp32=False):
    """No-op replacement for mmcv.runner.auto_fp16."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    if fn is not None:
        return decorator(fn)
    return decorator


class _DefaultOptimizerConstructor:
    """Stub for mmcv.runner.DefaultOptimizerConstructor (inference-only)."""
    def __init__(self, model_cfg, paramwise_cfg=None):
        self.model_cfg = model_cfg
        self.paramwise_cfg = paramwise_cfg or {}
        self.base_lr = model_cfg.get('lr', 1e-3)
        self.base_wd = model_cfg.get('weight_decay', 1e-4)

    def __call__(self, model):
        return torch.optim.AdamW(model.parameters())

    def add_params(self, params, module, **kwargs):
        for p in module.parameters():
            if p.requires_grad:
                params.append({'params': [p]})


class _FakeRegistry:
    """Minimal stub for mmcv.runner.OPTIMIZER_BUILDERS."""
    def register_module(self, *args, **kwargs):
        def decorator(cls):
            return cls
        if args and callable(args[0]):
            return args[0]
        return decorator


# ---------------------------------------------------------------------------
# mmcv.parallel shims
# ---------------------------------------------------------------------------

def _is_module_wrapper(module):
    from mmengine.model import is_model_wrapper
    return is_model_wrapper(module)


# ---------------------------------------------------------------------------
# mmcv.cnn shims
# ---------------------------------------------------------------------------

def _constant_init(module, val, bias=0):
    from mmengine.model import constant_init
    constant_init(module, val, bias=bias)


def _normal_init(module, mean=0, std=1, bias=0):
    from mmengine.model import normal_init
    normal_init(module, mean=mean, std=std, bias=bias)


# ---------------------------------------------------------------------------
# mmcv.utils shims
# ---------------------------------------------------------------------------

def _build_from_cfg(cfg, registry, default_args=None):
    from mmengine.registry import build_from_cfg
    return build_from_cfg(cfg, registry, default_args)


def _get_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


# ---------------------------------------------------------------------------
# mmpose.core.post_processing shims
# ---------------------------------------------------------------------------

def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return [
        src_point[0] * cs - src_point[1] * sn,
        src_point[0] * sn + src_point[1] * cs,
    ]


def _get_affine_transform(center, scale, rot, output_size,
                          shift=np.array([0, 0], dtype=np.float32),
                          inv=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def _affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float64)
    return t.dot(new_pt)[:2]


def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Map keypoint coordinates from model-input space to original image space.

    Reimplements ``mmpose.core.post_processing.transform_preds`` from
    mmpose v0.x so that external/PCT can run unchanged under mmpose v1.x.

    Args:
        coords (np.ndarray): Coordinates of shape (K, 2) or (K, 3).
        center (np.ndarray): Crop centre in original image, shape (2,).
        scale (np.ndarray): Crop scale (bbox_size / 200), shape (2,).
        output_size (list[int]): [width, height] of the model input.
        use_udp (bool): Whether to use UDP-style transform.

    Returns:
        np.ndarray: Coordinates in original image space, same shape as input.
    """
    assert coords.shape[1] in (2, 3)
    target_coords = np.ones_like(coords)
    if use_udp:
        scale_x = scale[0] * 200.0 / output_size[0]
        scale_y = scale[1] * 200.0 / output_size[1]
        target_coords[:, 0] = (
            coords[:, 0] * scale_x + center[0] - scale[0] * 100)
        target_coords[:, 1] = (
            coords[:, 1] * scale_y + center[1] - scale[1] * 100)
    else:
        trans = _get_affine_transform(center, scale, 0, output_size, inv=True)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = _affine_transform(coords[p, 0:2], trans)
    return target_coords


# ---------------------------------------------------------------------------
# mmpose.models.detectors.base shims
# ---------------------------------------------------------------------------

class BasePose(torch.nn.Module):
    """Minimal stub for mmpose v0.x BasePose used by external/PCT."""

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def show_result(self, *args, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# mmpose.models.heads.topdown_heatmap_base_head shims
# ---------------------------------------------------------------------------

class TopdownHeatmapBaseHead(torch.nn.Module):
    """Minimal stub for mmpose v0.x TopdownHeatmapBaseHead."""

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Master installer
# ---------------------------------------------------------------------------

_shims_installed = False


def install_pct_shims():
    """Patch sys.modules so that external/PCT can be imported under mmpose v1.x
    and mmcv 2.x without any modifications to the submodule.

    Safe to call multiple times; installation only happens once.
    """
    global _shims_installed
    if _shims_installed:
        return
    _shims_installed = True

    # ---- mmcv.runner -------------------------------------------------------
    runner_mod = types.ModuleType('mmcv.runner')
    runner_mod.auto_fp16 = _auto_fp16
    runner_mod.get_dist_info = _get_dist_info
    runner_mod.DefaultOptimizerConstructor = _DefaultOptimizerConstructor
    runner_mod.OPTIMIZER_BUILDERS = _FakeRegistry()
    sys.modules['mmcv.runner'] = runner_mod
    # make mmcv.runner accessible as an attribute of the mmcv package
    import mmcv
    mmcv.runner = runner_mod

    # ---- mmcv.parallel -----------------------------------------------------
    parallel_mod = types.ModuleType('mmcv.parallel')
    parallel_mod.is_module_wrapper = _is_module_wrapper
    sys.modules['mmcv.parallel'] = parallel_mod
    mmcv.parallel = parallel_mod

    # ---- mmcv.cnn (augment existing module) --------------------------------
    import mmcv.cnn as mmcv_cnn
    if not hasattr(mmcv_cnn, 'constant_init'):
        mmcv_cnn.constant_init = _constant_init
    if not hasattr(mmcv_cnn, 'normal_init'):
        mmcv_cnn.normal_init = _normal_init

    # ---- mmcv.utils (augment existing module) ------------------------------
    import mmcv.utils as mmcv_utils
    if not hasattr(mmcv_utils, 'build_from_cfg'):
        mmcv_utils.build_from_cfg = _build_from_cfg
    if not hasattr(mmcv_utils, 'get_logger'):
        mmcv_utils.get_logger = _get_logger
    # also expose on top-level mmcv for direct `mmcv.utils.X` access
    mmcv.utils = mmcv_utils

    # ---- mmpose.models.builder (add POSENETS) ------------------------------
    from mmpose.models import builder as mmpose_builder
    from mmpose.registry import MODELS
    if not hasattr(mmpose_builder, 'POSENETS'):
        mmpose_builder.POSENETS = MODELS

    # ---- mmpose.models.detectors.base (BasePose stub) ----------------------
    _ensure_module('mmpose.models.detectors')
    _ensure_module('mmpose.models.detectors.base', {
        'BasePose': BasePose,
    })

    # ---- mmpose.models.heads.topdown_heatmap_base_head ---------------------
    _ensure_module('mmpose.models.heads.topdown_heatmap_base_head', {
        'TopdownHeatmapBaseHead': TopdownHeatmapBaseHead,
    })

    # ---- mmpose.core.post_processing ---------------------------------------
    _ensure_module('mmpose.core')
    _ensure_module('mmpose.core.post_processing', {
        'transform_preds': transform_preds,
    })


def _ensure_module(name, attrs=None):
    """Create a fake module in sys.modules if it doesn't already exist,
    optionally populating it with attributes from a dict.
    """
    if name not in sys.modules:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        # attach as attribute on parent package
        parent_name, _, child_name = name.rpartition('.')
        if parent_name and parent_name in sys.modules:
            setattr(sys.modules[parent_name], child_name, mod)
    if attrs:
        for k, v in attrs.items():
            setattr(sys.modules[name], k, v)
