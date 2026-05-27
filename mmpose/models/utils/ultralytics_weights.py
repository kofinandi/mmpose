# Copyright (c) OpenMMLab. All rights reserved.
"""Helpers for resolving Ultralytics model weight paths."""

import os
import os.path as osp

_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
DEFAULT_ULTRALYTICS_MODEL_DIR = 'data/models'


def _cache_dir(model_cache_dir: str) -> str:
    if osp.isabs(model_cache_dir):
        return model_cache_dir
    return osp.join(_REPO_ROOT, model_cache_dir)


def resolve_ultralytics_weights(
    weights: str,
    model_cache_dir: str = DEFAULT_ULTRALYTICS_MODEL_DIR,
) -> str:
    """Resolve Ultralytics weights to a path under the project model cache.

    Bare filenames such as ``yolo26n-pose.pt`` are mapped to
    ``<repo>/data/models/yolo26n-pose.pt`` so Ultralytics auto-downloads
    land in the shared model directory instead of the current working
    directory.
    """
    if not weights or weights.startswith(('http://', 'https://')):
        return weights

    cache_dir = _cache_dir(model_cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    if osp.isabs(weights):
        if osp.isfile(weights):
            return weights
        return osp.join(cache_dir, osp.basename(weights))

    if osp.dirname(weights):
        path = osp.join(_REPO_ROOT, weights)
        os.makedirs(osp.dirname(path), exist_ok=True)
        return path

    return osp.join(cache_dir, weights)
