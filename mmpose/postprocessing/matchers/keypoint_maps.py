# Copyright (c) OpenMMLab. All rights reserved.
"""Keypoint layout conversions for trackers trained on PoseTrack.

The prediction bundles this package post-processes carry COCO-17 keypoints,
but LightTrack's Siamese GCN and PGPT's OKS stage were both built around
15-joint PoseTrack layouts - and, importantly, *two different ones*.  Each
map below is stated in terms of the upstream code that pins it down.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

#: One entry per target joint: an ``int`` copies that source joint, a
#: 2-tuple averages the two named source joints.
IndexMap = Sequence[Union[int, Tuple[int, int]]]

# COCO-17 joint indices, for readability below.
_NOSE = 0
_L_EYE, _R_EYE = 1, 2
_L_SHO, _R_SHO = 5, 6
_L_ELB, _R_ELB = 7, 8
_L_WRI, _R_WRI = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNE, _R_KNE = 13, 14
_L_ANK, _R_ANK = 15, 16

#: COCO-17 -> the 15-joint layout of LightTrack's ``Graph('PoseTrack')``.
#:
#: The graph's edge list (``external/lighttrack/graph/gcn_utils/graph.py``:
#: ``[(0,1),(1,2),(3,4),(4,5),(2,8),(8,7),(7,6),(8,12),(12,9),(9,10),
#: (10,11),(9,3),(12,13),(13,14)]`` with ``center=12``) pins the ordering to
#: the MPII/PoseTrack convention::
#:
#:     0 r_ankle   1 r_knee   2 r_hip     3 l_hip      4 l_knee
#:     5 l_ankle   6 r_wrist  7 r_elbow   8 r_shoulder 9 l_shoulder
#:    10 l_elbow  11 l_wrist 12 head_bottom (neck, the graph centre)
#:    13 nose     14 head_top
#:
#: COCO-17 has neither head_bottom nor head_top, so both are synthesised:
#: the neck as the shoulder midpoint and head_top as the eye midpoint.  This
#: is an approximation of joints the network was trained on - see the
#: fidelity notes on :class:`SGCNPoseMatcher`.
COCO17_TO_POSETRACK15_LIGHTTRACK: IndexMap = (
    _R_ANK, _R_KNE, _R_HIP, _L_HIP, _L_KNE, _L_ANK,
    _R_WRI, _R_ELB, _R_SHO, _L_SHO, _L_ELB, _L_WRI,
    (_L_SHO, _R_SHO),   # 12 head_bottom / neck
    _NOSE,              # 13
    (_L_EYE, _R_EYE),   # 14 head_top
)

#: COCO-17 -> PGPT's 15-joint layout, which is simply COCO-17 with the two
#: ear joints dropped.
#:
#: ``PoseNet.detect_pose`` does ``np.delete(preds, [3, 4], axis=0)`` on its
#: 17-channel output (``external/PGPT/inference/pose_estimation_graph.py``),
#: and ``Matcher.oks_iou`` uses sigmas
#: ``[.26,.25,.25,.79,.79,.72,.72,.62,.62,1.07,1.07,.87,.87,.89,.89]`` -
#: exactly the COCO sigmas with the two ``.35`` ear entries removed, which
#: confirms the ordering is otherwise unchanged.  No joint is synthesised.
COCO17_TO_PGPT15: IndexMap = (
    _NOSE, _L_EYE, _R_EYE,
    _L_SHO, _R_SHO, _L_ELB, _R_ELB, _L_WRI, _R_WRI,
    _L_HIP, _R_HIP, _L_KNE, _R_KNE, _L_ANK, _R_ANK,
)

#: PGPT's OKS sigmas, in :data:`COCO17_TO_PGPT15` order
#: (``external/PGPT/inference/match.py``).
PGPT15_SIGMAS = np.array(
    [.26, .25, .25, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87,
     .89, .89], dtype=np.float32) / 10.0


def convert_keypoints(kpts: np.ndarray, index_map: IndexMap) -> np.ndarray:
    """Re-order/synthesise keypoints according to ``index_map``.

    Args:
        kpts: Source keypoints, shape ``(..., K_src, C)``; ``C`` is 2 for
            coordinates or absent-per-joint scalars work too when passed as
            ``(..., K_src)``.
        index_map: One entry per target joint; an ``int`` copies that
            source joint, a 2-tuple averages the two named source joints.

    Returns:
        Keypoints in the target layout, shape ``(..., len(index_map), C)``.
    """
    kpts = np.asarray(kpts)
    out = np.empty(
        kpts.shape[:-2] + (len(index_map), ) + kpts.shape[-1:],
        dtype=kpts.dtype)
    for dst, src in enumerate(index_map):
        if isinstance(src, (tuple, list)):
            a, b = src
            out[..., dst, :] = 0.5 * (kpts[..., a, :] + kpts[..., b, :])
        else:
            out[..., dst, :] = kpts[..., src, :]
    return out


def convert_scores(scores: np.ndarray, index_map: IndexMap) -> np.ndarray:
    """Re-order/synthesise per-keypoint scores according to ``index_map``.

    Synthesised joints take the **minimum** of their two source scores: a
    joint averaged from an unreliable source is itself unreliable.

    Args:
        scores: Source scores, shape ``(..., K_src)``.
        index_map: As in :func:`convert_keypoints`.

    Returns:
        Scores in the target layout, shape ``(..., len(index_map))``.
    """
    scores = np.asarray(scores)
    out = np.empty(scores.shape[:-1] + (len(index_map), ), dtype=scores.dtype)
    for dst, src in enumerate(index_map):
        if isinstance(src, (tuple, list)):
            a, b = src
            out[..., dst] = np.minimum(scores[..., a], scores[..., b])
        else:
            out[..., dst] = scores[..., src]
    return out
