# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import numpy as np


def keypoint_mpjve(
        pred: np.ndarray,
        gt: np.ndarray,
        mask: np.ndarray,
        norm_factor: Optional[np.ndarray] = None) -> Tuple[float, int]:
    """Calculate the Mean Per-Joint Velocity Error (MPJVE) for a single track.

    Velocity is defined as the first-order finite difference of joint
    positions: ``v_t = P_t - P_{t-1}``.  The error is the sum of Euclidean
    distances between predicted and ground-truth velocity vectors over all
    visible joints and all valid consecutive frame pairs.

    An optional per-frame-pair normalisation factor (e.g. bbox diagonal or
    torso diameter) can be supplied to make the error scale-invariant.

    Note:
        - number of frames: T
        - num_keypoints: K
        - keypoint_dims: D

    Args:
        pred (np.ndarray): Predicted keypoint positions with shape ``[T, K, D]``.
        gt (np.ndarray): Ground-truth keypoint positions with shape ``[T, K, D]``.
        mask (np.ndarray): Boolean visibility mask with shape ``[T, K]``.
            ``True`` indicates a visible (valid) joint.
        norm_factor (np.ndarray, optional): Per-frame-pair normalisation
            scale with shape ``[T-1]``.  Each velocity error at pair
            ``(t-1, t)`` is divided by ``norm_factor[t-1]`` before
            accumulation.  Defaults to ``None`` (no normalisation).

    Returns:
        Tuple[float, int]: ``(error_sum, count)`` where ``error_sum`` is the
        sum of (optionally normalised) per-joint velocity errors over all
        visible joints and valid frame pairs, and ``count`` is the number of
        such joint-frame-pair entries.  Both are ``0`` when no valid pairs
        exist.  To obtain the mean, divide ``error_sum`` by ``count``.
    """
    assert pred.shape == gt.shape, (
        f'pred and gt must have the same shape, got {pred.shape} vs {gt.shape}')
    assert mask.shape == pred.shape[:2], (
        f'mask shape {mask.shape} must match pred shape[:2] {pred.shape[:2]}')

    if pred.shape[0] < 2:
        return 0.0, 0

    # Velocity: [T-1, K, D]
    v_pred = pred[1:] - pred[:-1]
    v_gt = gt[1:] - gt[:-1]

    # A pair is valid only when the joint is visible in both frames: [T-1, K]
    vel_mask = mask[1:] & mask[:-1]

    if not vel_mask.any():
        return 0.0, 0

    # Euclidean velocity error per joint: [T-1, K]
    vel_error = np.linalg.norm(v_pred - v_gt, axis=-1)

    if norm_factor is not None:
        # norm_factor: [T-1] → broadcast over keypoints
        vel_error = vel_error / norm_factor[:, np.newaxis]

    return float(vel_error[vel_mask].sum()), int(vel_mask.sum())


def keypoint_mpjae(
        pred: np.ndarray,
        gt: np.ndarray,
        mask: np.ndarray,
        norm_factor: Optional[np.ndarray] = None) -> Tuple[float, int]:
    """Calculate the Mean Per-Joint Acceleration Error (MPJAE) for a single track.

    Acceleration is defined as the second-order finite difference of joint
    positions: ``a_t = P_t - 2*P_{t-1} + P_{t-2}``.  The error is the sum
    of Euclidean distances between predicted and ground-truth acceleration
    vectors over all visible joints and all valid frame triplets.

    An optional per-triplet normalisation factor can be supplied to make the
    error scale-invariant.

    Note:
        - number of frames: T
        - num_keypoints: K
        - keypoint_dims: D

    Args:
        pred (np.ndarray): Predicted keypoint positions with shape ``[T, K, D]``.
        gt (np.ndarray): Ground-truth keypoint positions with shape ``[T, K, D]``.
        mask (np.ndarray): Boolean visibility mask with shape ``[T, K]``.
            ``True`` indicates a visible (valid) joint.
        norm_factor (np.ndarray, optional): Per-triplet normalisation scale
            with shape ``[T-2]``.  Each acceleration error at triplet
            ``(t-2, t-1, t)`` is divided by ``norm_factor[t-2]`` before
            accumulation.  Defaults to ``None`` (no normalisation).

    Returns:
        Tuple[float, int]: ``(error_sum, count)`` where ``error_sum`` is the
        sum of (optionally normalised) per-joint acceleration errors over all
        visible joints and valid frame triplets, and ``count`` is the number
        of such joint-triplet entries.  Both are ``0`` when no valid triplets
        exist.  To obtain the mean, divide ``error_sum`` by ``count``.
    """
    assert pred.shape == gt.shape, (
        f'pred and gt must have the same shape, got {pred.shape} vs {gt.shape}')
    assert mask.shape == pred.shape[:2], (
        f'mask shape {mask.shape} must match pred shape[:2] {pred.shape[:2]}')

    if pred.shape[0] < 3:
        return 0.0, 0

    # Acceleration: [T-2, K, D]
    a_pred = pred[2:] - 2 * pred[1:-1] + pred[:-2]
    a_gt = gt[2:] - 2 * gt[1:-1] + gt[:-2]

    # A triplet is valid when the joint is visible in all three frames: [T-2, K]
    acc_mask = mask[2:] & mask[1:-1] & mask[:-2]

    if not acc_mask.any():
        return 0.0, 0

    # Euclidean acceleration error per joint: [T-2, K]
    acc_error = np.linalg.norm(a_pred - a_gt, axis=-1)

    if norm_factor is not None:
        # norm_factor: [T-2] → broadcast over keypoints
        acc_error = acc_error / norm_factor[:, np.newaxis]

    return float(acc_error[acc_mask].sum()), int(acc_mask.sum())
