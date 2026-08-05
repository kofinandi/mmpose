# Copyright (c) OpenMMLab. All rights reserved.
"""Helpers for identifying evaluable ("good") frames in video benchmarks.

EMDB, 3DPW, and PoseTrack21 annotate frames that lack reliable GT with an
image-level ``good_frame=False`` flag. Inference and post-processing often
still run over those frames (for temporal continuity), but detection/pose/
tracking metrics should exclude them. This module centralises the
good/bad decision used by ``tools/benchmark_e2e.py`` and
``tools/postprocess_predictions.py``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

# Datasets whose preprocessors write an explicit ``good_frame`` image field.
# The zero-GT heuristic fallback is only safe on these datasets: COCO
# val2017 legitimately contains images with no person annotations that must
# stay in the evaluation set.
BAD_FRAME_DATASETS = frozenset({'emdb', 'emdb-mini', '3dpw', 'posetrack21'})


def _has_non_crowd_gt(frame: dict) -> bool:
    """Return True when *frame* has at least one non-crowd GT instance.

    PoseTrack21 unlabeled frames can still carry ``iscrowd=1`` ignore
    regions, so those alone must not count as "good".
    """
    instances = frame.get('ground_truth', {}).get('instances', [])
    for inst in instances:
        if int(inst.get('iscrowd', 0)) == 0:
            return True
    return False


def frame_record_is_good(
    frame: dict,
    *,
    allow_heuristic: bool = False,
) -> bool:
    """Decide whether a saved frame record should be evaluated.

    Prefer the explicit ``good_frame`` field written by
    :func:`~mmpose.evaluation.functional.frame_metrics.build_frame_record`.
    When the key is missing (legacy bundles produced before that field
    existed):

    * if ``allow_heuristic`` is True, fall back to "has at least one
      non-crowd GT instance";
    * otherwise treat the frame as good (do not drop).

    Args:
        frame: One entry from a prediction-bundle ``frames.json``.
        allow_heuristic: Enable the zero-GT fallback for legacy bundles.
            Callers should only set this when
            ``test_dataset in BAD_FRAME_DATASETS``.

    Returns:
        ``True`` when the frame should be passed to metric ``process()``.
    """
    if 'good_frame' in frame:
        return bool(frame['good_frame'])
    if allow_heuristic:
        return _has_non_crowd_gt(frame)
    return True


def partition_frame_records(
    frames: Sequence[dict],
    *,
    allow_heuristic: bool = False,
) -> Tuple[List[int], bool]:
    """Return indices of good frames and whether the heuristic was used.

    Args:
        frames: Ordered frame records from a prediction bundle.
        allow_heuristic: See :func:`frame_record_is_good`.

    Returns:
        ``(good_idxs, used_heuristic)`` where ``good_idxs`` are the
        indices into *frames* that should be evaluated, and
        ``used_heuristic`` is True iff at least one frame lacked an
        explicit ``good_frame`` key and the heuristic was consulted.
    """
    good_idxs: List[int] = []
    used_heuristic = False
    for i, frame in enumerate(frames):
        if 'good_frame' not in frame and allow_heuristic:
            used_heuristic = True
        if frame_record_is_good(frame, allow_heuristic=allow_heuristic):
            good_idxs.append(i)
    return good_idxs, used_heuristic
