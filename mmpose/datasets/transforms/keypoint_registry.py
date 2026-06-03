# Copyright (c) OpenMMLab. All rights reserved.
"""Keypoint convention registry and auto-mapping utilities.

Keypoint name lists are loaded on demand directly from the canonical dataset
metainfo files in ``configs/_base_/datasets/``, so there is no duplication.
``get_mapping`` auto-generates the ``(src_idx, dst_idx)`` pairs that
:class:`KeypointConverter` needs, by matching keypoint names between two
conventions.
"""
from typing import Dict, List, Optional, Tuple, Union

# Maps convention name → path of the corresponding metainfo config file,
# relative to the mmpose repository / installation root.
_METAINFO_FILES: Dict[str, str] = {
    'coco':         'configs/_base_/datasets/coco.py',
    'mpii':         'configs/_base_/datasets/mpii.py',
    'aic':          'configs/_base_/datasets/aic.py',
    'crowdpose':    'configs/_base_/datasets/crowdpose.py',
    'jhmdb':        'configs/_base_/datasets/jhmdb.py',
    'halpe26':      'configs/_base_/datasets/halpe26.py',
    'ochuman':      'configs/_base_/datasets/ochuman.py',
    'posetrack18':  'configs/_base_/datasets/posetrack18.py',
    'coco_wholebody': 'configs/_base_/datasets/coco_wholebody.py',
    'threedpw': 'configs/_base_/datasets/threedpw.py',
    'emdb': 'configs/_base_/datasets/emdb.py',
    'ubody': 'configs/_base_/datasets/ubody2d.py',
}

# Lazy-loaded cache: convention name → list of keypoint name strings
_cache: Dict[str, List[str]] = {}


def get_keypoints(convention: str) -> List[str]:
    """Return the ordered list of keypoint names for *convention*.

    Names are loaded from the metainfo file registered in
    :data:`_METAINFO_FILES` and cached after the first call.

    Args:
        convention (str): A key in :data:`_METAINFO_FILES`.

    Returns:
        list[str]: Keypoint names in dataset index order.

    Raises:
        KeyError: If *convention* is not registered.
    """
    if convention in _cache:
        return _cache[convention]

    if convention not in _METAINFO_FILES:
        raise KeyError(
            f'Unknown keypoint convention {convention!r}. '
            f'Registered conventions: {sorted(_METAINFO_FILES)}. '
            'To add a new one, update _METAINFO_FILES in '
            'mmpose/datasets/transforms/keypoint_registry.py.')

    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    meta = parse_pose_metainfo(
        dict(from_file=_METAINFO_FILES[convention]))
    names = [meta['keypoint_id2name'][i]
             for i in range(meta['num_keypoints'])]
    _cache[convention] = names
    return names


def get_flip_indices(convention: str) -> Optional[List[int]]:
    """Return the flip index list for *convention*, or ``None`` if unavailable.

    Args:
        convention (str): A key in :data:`_METAINFO_FILES`.

    Returns:
        list[int] or None: ``flip_indices[i]`` is the index of the keypoint
        that is symmetric to keypoint *i*.
    """
    if convention not in _METAINFO_FILES:
        return None

    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    meta = parse_pose_metainfo(dict(from_file=_METAINFO_FILES[convention]))
    return meta.get('flip_indices', None)


def get_mapping(
    src: Union[str, List[str]],
    dst: Union[str, List[str]],
) -> List[Tuple[int, int]]:
    """Auto-generate a keypoint index mapping from *src* to *dst*.

    Mapping is built by matching keypoint names: for each keypoint in *dst*,
    if a keypoint with the same name exists in *src*, the pair
    ``(src_idx, dst_idx)`` is added to the result.  This produces the same
    ``mapping`` list expected by :class:`KeypointConverter`.

    Args:
        src (str or list[str]): Source convention.  Either a key in
            :data:`_METAINFO_FILES` or an explicit list of keypoint name
            strings.
        dst (str or list[str]): Destination convention.  Same format as
            *src*.

    Returns:
        list[tuple[int, int]]: Mapping pairs ``(src_idx, dst_idx)`` for every
        keypoint name that appears in both conventions.

    Example:
        >>> from mmpose.datasets.transforms.keypoint_registry import (
        ...     get_mapping)
        >>> get_mapping('mpii', 'coco')
        [(13, 5), (12, 6), (14, 7), (11, 8), (15, 9), (10, 10),
         (3, 11), (2, 12), (4, 13), (1, 14), (5, 15), (0, 16)]
    """
    src_names: List[str] = get_keypoints(src) if isinstance(src, str) else src
    dst_names: List[str] = get_keypoints(dst) if isinstance(dst, str) else dst

    src_name_to_idx = {name: idx for idx, name in enumerate(src_names)}

    mapping: List[Tuple[int, int]] = []
    for dst_idx, name in enumerate(dst_names):
        if name in src_name_to_idx:
            mapping.append((src_name_to_idx[name], dst_idx))
    return mapping
