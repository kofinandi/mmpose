# Copyright (c) OpenMMLab. All rights reserved.
"""SmoothNet offline smoother for tracked pose sequences.

SmoothNet architecture ported from the legacy MMPose v0.x implementation
(commit 6be61ce0, mmpose/core/legacy/post_processing/temporal_filters/
smoothnet_filter.py).

Reference: "SmoothNet: A Plug-and-Play Network for Refining Human Poses in
Videos", arXiv 2021, https://arxiv.org/abs/2112.13715

Official MMPose checkpoints (Human3.6M, window sizes 8/16/32/64):
  https://download.openmmlab.com/mmpose/plugin/smoothnet/smoothnet_ws<W>_h36m.pth
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from mmpose.structures import PoseDataSample

from ..base import BaseFilter, sequence_key_from_path
from ..registry import POST_PROCESS_FILTERS


# ---------------------------------------------------------------------------
# SmoothNet neural network (architecture verbatim from legacy source)
# ---------------------------------------------------------------------------

class SmoothNetResBlock(nn.Module):
    """Residual block used inside SmoothNet.

    Args:
        in_channels (int): Input channel number.
        hidden_channels (int): Hidden feature channel number.
        dropout (float): Dropout probability. Default: 0.5.

    Shape:
        Input / Output: ``(*, in_channels)``
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        return x + identity


class SmoothNet(nn.Module):
    """Temporal-only MLP that refines human pose trajectories.

    Operates on sliding windows over a pose sequence.  The number of channels
    ``C = K * coord_dim`` is treated as a batch dimension so the same
    checkpoint can run on any number of keypoints / coordinate dimensions.

    Args:
        window_size (int): Sliding-window width (= number of input frames).
        output_size (int): Number of refined frames per window.
        hidden_size (int): Encoder/decoder hidden dimension. Default: 512.
        res_hidden_size (int): Residual-block hidden dimension. Default: 256.
        num_blocks (int): Number of residual blocks. Default: 3.
        dropout (float): Dropout probability. Default: 0.5.

    Shape:
        Input:  ``(N, C, T)`` – original pose sequence.
        Output: ``(N, C, T)`` – smoothed pose sequence (overlap-averaged).
    """

    def __init__(
        self,
        window_size: int,
        output_size: int,
        hidden_size: int = 512,
        res_hidden_size: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        assert output_size <= window_size, (
            f'output_size ({output_size}) must be <= window_size ({window_size})')

        self.window_size = window_size
        self.output_size = output_size

        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.res_blocks = nn.Sequential(*[
            SmoothNetResBlock(hidden_size, res_hidden_size, dropout)
            for _ in range(num_blocks)
        ])
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        N, C, T = x.shape
        num_windows = T - self.window_size + 1

        assert T >= self.window_size, (
            f'Sequence length T={T} must be >= window_size={self.window_size}')

        # [N, C, num_windows, window_size]
        x = x.unfold(2, self.window_size, 1)
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)  # [N, C, num_windows, output_size]

        # Overlap-add accumulation
        out = x.new_zeros(N, C, T)
        count = x.new_zeros(T)
        for t in range(num_windows):
            out[..., t:t + self.output_size] += x[:, :, t]
            count[t:t + self.output_size] += 1.0

        return out.div(count)


# ---------------------------------------------------------------------------
# Filter wrapper
# ---------------------------------------------------------------------------

@POST_PROCESS_FILTERS.register_module()
class SmoothNetSmoother(BaseFilter):
    """Offline per-track SmoothNet smoother.

    Groups predicted instances by ``(sequence_key, track_id)`` into temporal
    trajectories and refines each trajectory with a pretrained
    :class:`SmoothNet` model.  Instances in tracks shorter than
    ``window_size`` are returned unchanged.

    This filter is **offline** (``online = False``): it overrides
    :meth:`process_sequence` and must be driven by
    :class:`~mmpose.postprocessing.pipeline.PostProcessingPipeline` in its
    buffered/evaluate mode.

    Requires ``pred_instances.track_ids`` to be set by a preceding tracker
    (e.g. :class:`~mmpose.postprocessing.filters.OKSTracker`).

    Only ``keypoints[:, :, :2]`` (coordinates) are modified.  Scores,
    bboxes, and ``track_ids`` are preserved.

    Args:
        window_size (int): Sliding-window length fed to SmoothNet.
        output_size (int | None): Output window size.  Defaults to
            ``window_size`` (same as all official checkpoints).
        checkpoint (str | None): URL or local path of the pretrained
            SmoothNet weights.  Supports MMPose HTTP URLs; loaded via
            :func:`mmengine.runner.load_checkpoint`.  When ``None`` the
            model runs with random weights.
        hidden_size (int): SmoothNet encoder/decoder width. Default: 512.
        res_hidden_size (int): Residual-block hidden width. Default: 256.
        num_blocks (int): Number of residual blocks. Default: 3.
        dropout (float): Dropout probability (zero-ed at inference anyway).
            Default: 0.5.
        device (str): Device for the SmoothNet forward pass. Default: ``'cpu'``.
        root_index (int | None): If set, keypoints are centered around this
            joint before smoothing and de-centered afterwards.  The official
            H36M checkpoints use ``root_index=0`` (pelvis); for COCO-17 2D
            where no canonical root exists, leave as ``None``. Default: ``None``.
    """

    online = False

    def __init__(
        self,
        window_size: int,
        output_size: Optional[int] = None,
        checkpoint: Optional[str] = None,
        hidden_size: int = 512,
        res_hidden_size: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.5,
        device: str = 'cpu',
        root_index: Optional[int] = None,
    ) -> None:
        self.window_size = window_size
        self.output_size = output_size if output_size is not None else window_size
        self.device = device
        self.root_index = root_index

        self.smoothnet = SmoothNet(
            window_size=self.window_size,
            output_size=self.output_size,
            hidden_size=hidden_size,
            res_hidden_size=res_hidden_size,
            num_blocks=num_blocks,
            dropout=dropout,
        )

        if checkpoint is not None:
            from mmengine.runner import load_checkpoint
            load_checkpoint(self.smoothnet, checkpoint, map_location='cpu')

        self.smoothnet.to(device)
        self.smoothnet.eval()
        for p in self.smoothnet.parameters():
            p.requires_grad_(False)

    def reset(self) -> None:
        pass  # stateless; all state lives in the per-call trajectory dicts

    def process_frame(
        self,
        ds: PoseDataSample,
        seq_key: str,
    ) -> PoseDataSample:
        # SmoothNet is offline; process_frame should never be called by the
        # pipeline (offline filters go through process_sequence).
        raise NotImplementedError(
            'SmoothNetSmoother is an offline filter; use process_sequence().')

    def _smooth(self, traj: np.ndarray) -> np.ndarray:
        """Apply SmoothNet to a single-track trajectory.

        Args:
            traj: ``(T, K, 2)`` float32 keypoint coordinates.

        Returns:
            ``(T, K, 2)`` float32 smoothed coordinates (unchanged if
            ``T < window_size``).
        """
        T, K, C = traj.shape
        if T < self.window_size:
            return traj

        root_index = self.root_index
        if root_index is not None:
            root = traj[:, root_index:root_index + 1, :]   # (T,1,C)
            traj = traj.copy()
            traj = np.delete(traj, root_index, axis=1)
            traj = traj - root

        dtype = traj.dtype

        with torch.no_grad():
            x = torch.tensor(traj, dtype=torch.float32, device=self.device)
            # [T, K', C] -> [1, K'*C, T]
            Kp = traj.shape[1]
            x = x.view(T, Kp * C).t().unsqueeze(0)  # [1, Kp*C, T]
            out = self.smoothnet(x)                   # [1, Kp*C, T]
            # [1, Kp*C, T] -> [T, Kp, C]
            out = out.squeeze(0).t().view(T, Kp, C)
            smoothed = out.cpu().numpy().astype(dtype)

        if root_index is not None:
            smoothed = smoothed + root
            smoothed = np.concatenate([
                smoothed[:, :root_index],
                root,
                smoothed[:, root_index:],
            ], axis=1)

        return smoothed

    def process_sequence(
        self,
        frames: List[PoseDataSample],
    ) -> List[PoseDataSample]:
        """Smooth all tracks in the frame sequence.

        Args:
            frames: Ordered list of :class:`PoseDataSample`s carrying
                ``pred_instances.track_ids`` (set by a preceding tracker).

        Returns:
            Same-length, same-order list with smoothed keypoints written back.

        Raises:
            RuntimeError: If any non-empty frame lacks ``track_ids``.
        """

        # ── Build per-frame keypoint arrays (mutable copies) ────────────────
        # kpts_per_frame[i]: (N_i, K, 2) float32 – will be mutated in-place
        kpts_per_frame: List[Optional[np.ndarray]] = []

        # trajectories[(seq_key, track_id)] = [(frame_idx, inst_idx), ...]
        trajectories: Dict[Tuple[str, int], List[Tuple[int, int]]] = \
            defaultdict(list)

        for fi, ds in enumerate(frames):
            inst = ds.pred_instances
            if inst is None or len(inst) == 0:
                kpts_per_frame.append(None)
                continue

            if not hasattr(inst, 'track_ids') or inst.track_ids is None:
                raise RuntimeError(
                    'SmoothNetSmoother requires pred_instances.track_ids. '
                    'Make sure OKSTracker (or another tracker) runs before it.')

            kpts = np.asarray(inst.keypoints, dtype=np.float32).copy()
            kpts_per_frame.append(kpts)

            img_path = ds.metainfo.get('img_path', '')
            seq_key = sequence_key_from_path(img_path)
            track_ids = np.asarray(inst.track_ids, dtype=np.int32)

            for ii, tid in enumerate(track_ids):
                trajectories[(seq_key, int(tid))].append((fi, ii))

        # ── Run SmoothNet per trajectory ─────────────────────────────────────
        for (seq_key, tid), occurrences in trajectories.items():
            if not occurrences:
                continue

            traj = np.stack(
                [kpts_per_frame[fi][ii] for fi, ii in occurrences],
                axis=0,
            )  # (T, K, 2)

            smoothed = self._smooth(traj)  # (T, K, 2)

            for t, (fi, ii) in enumerate(occurrences):
                kpts_per_frame[fi][ii] = smoothed[t]

        # ── Assemble output PoseDataSamples ───────────────────────────────────
        out_frames: List[PoseDataSample] = []
        for fi, ds in enumerate(frames):
            kpts = kpts_per_frame[fi]
            if kpts is None:
                out_frames.append(ds)
                continue

            new_ds = ds.new()
            new_ds.set_metainfo(ds.metainfo)
            if hasattr(ds, 'gt_instances'):
                new_ds.gt_instances = ds.gt_instances

            new_inst = deepcopy(ds.pred_instances)
            new_inst.keypoints = kpts
            new_ds.pred_instances = new_inst
            out_frames.append(new_ds)

        return out_frames
