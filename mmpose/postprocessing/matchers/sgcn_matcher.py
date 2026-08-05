# Copyright (c) OpenMMLab. All rights reserved.
"""Wrapper around LightTrack's Siamese Graph Convolutional Network (SGCN).

    Ning et al., "LightTrack: A Generic Framework for Online Top-Down Human
    Pose Tracking", CVPRW 2020.  https://github.com/Guanghan/lighttrack

LightTrack's re-identification module reads a pose as a graph over 15
PoseTrack joints, with box-relative pixel coordinates as node features,
embeds it into a 128-D space with a two-layer ST-GCN, and calls two poses
the same person when their embeddings are close.

Like :class:`~mmpose.models.pose_estimators.PETRPoseEstimator`, this is a
thin wrapper: the network itself is imported unmodified from the submodule
(``external/lighttrack/graph/gcn_utils/gcn_model.py``) and only the
interface is translated.  What is *not* imported is
``graph/visualize_pose_matching.py``, which owns upstream's matching
helpers but instantiates a ``Pose_Matcher`` at module scope - parsing argv,
allocating a GPU and loading a checkpoint on import - and upstream's
``torchlight`` IO layer, which mutates ``CUDA_VISIBLE_DEVICES``, writes a
work_dir, and calls ``yaml.load`` in a way PyYAML >= 6 rejects.  The few
lines those files contribute (``keypoints_to_graph``, ``graph_pair_to_data``
and the embedding distance) are ported here instead.

Checkpoint
----------
**The published weights are unavailable.**  ``weights/GCN/epoch210_model.pt``
ships in ``GCN.zip``, which 404s on ``guanghan.info`` with no Wayback
snapshot; the training-pair tarball is gone too, and upstream issue #21
("Download link expired") is still open.  ``tools/train_lighttrack_sgcn.py``
retrains the same architecture with the same recipe on PoseTrack21, and
that is what the shipped configs point at.  Weights produced that way are
**not** the paper's, so numbers obtained with them are not a reproduction
of the paper's numbers.
"""

from __future__ import annotations

import os.path as osp
import sys
from typing import Optional

import numpy as np
import torch

from ..registry import POST_PROCESS_POSE_MATCHERS
from .base import BasePoseMatcher
from .keypoint_maps import to_lighttrack15

_LIGHTTRACK_ROOT = osp.abspath(
    osp.join(osp.dirname(__file__), '..', '..', '..', 'external',
             'lighttrack'))


@POST_PROCESS_POSE_MATCHERS.register_module()
class SGCNPoseMatcher(BasePoseMatcher):
    """Embedding distance between two poses under LightTrack's SGCN.

    Poses are converted to the network's 15-joint graph layout by
    :func:`~mmpose.postprocessing.matchers.to_lighttrack15`, shifted to
    their own box origin and truncated to integer pixels (upstream does
    ``int(x - x0)``), which is what the network is trained on.  Keypoint
    confidences are ignored, as upstream: ``keypoints_to_graph`` reads each
    joint's score and then discards it.

    Args:
        checkpoint: Path to SGCN weights.  See the module docstring - the
            published file is unavailable, so this normally points at the
            output of ``tools/train_lighttrack_sgcn.py``.
        lighttrack_root: Path to the ``external/lighttrack`` submodule.
        device: Torch device for the network.
        num_class: Embedding dimensionality, upstream ``128``.
        batch_size: Maximum poses embedded per forward pass.
    """

    def __init__(
        self,
        checkpoint: str = 'data/models/lighttrack_sgcn_posetrack21.pt',
        lighttrack_root: str = _LIGHTTRACK_ROOT,
        device: str = 'cpu',
        num_class: int = 128,
        batch_size: int = 256,
    ) -> None:
        model_cls = self._import_model(lighttrack_root)

        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)
        self.batch_size = int(batch_size)

        # Architecture from external/lighttrack/graph/config/inference.yaml.
        self.model = model_cls(
            in_channels=2,
            num_class=num_class,
            edge_importance_weighting=True,
            graph_args=dict(layout='PoseTrack', strategy='spatial'))
        self._load_weights(checkpoint)
        self.model.eval().to(self.device)

    @staticmethod
    def _import_model(lighttrack_root: str):
        """Import the unmodified upstream ``Model`` from the submodule."""
        graph_dir = osp.join(osp.abspath(lighttrack_root), 'graph')
        if not osp.isdir(graph_dir):
            raise FileNotFoundError(
                f'LightTrack submodule not found at {lighttrack_root!r}. '
                f'Run `git submodule update --init external/lighttrack`.')
        if graph_dir not in sys.path:
            sys.path.insert(0, graph_dir)
        from gcn_utils.gcn_model import Model
        return Model

    def _load_weights(self, checkpoint: str) -> None:
        """Load weights, mirroring ``torchlight.IO.load_weights``."""
        if not osp.isfile(checkpoint):
            raise FileNotFoundError(
                f'SGCN checkpoint not found: {checkpoint!r}.\n'
                f"LightTrack's published weights (weights/GCN/"
                f'epoch210_model.pt, from GCN.zip) are no longer '
                f'downloadable - guanghan.info returns 404 and there is no '
                f'Wayback snapshot (upstream issue #21).\n'
                f'Either train a replacement:\n'
                f'    python tools/train_lighttrack_sgcn.py --out '
                f'{checkpoint}\n'
                f'or run the checkpoint-free variant instead:\n'
                f'    configs/post_processing/lighttrack_l2.py')

        weights = torch.load(checkpoint, map_location='cpu')
        if isinstance(weights, dict) and 'state_dict' in weights:
            weights = weights['state_dict']
        # Upstream strips DataParallel's 'module.' prefix the same way.
        weights = {k.split('module.')[-1]: v for k, v in weights.items()}
        self.model.load_state_dict(weights)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _to_graph_tensor(
        self,
        kpts: np.ndarray,
        bboxes: np.ndarray,
    ) -> Optional[torch.Tensor]:
        """Pack poses into the ``(N, C=2, T=1, V=15, M=1)`` input tensor.

        Port of ``keypoints_to_graph`` + ``graph_pair_to_data``: convert to
        the 15-joint layout, shift each pose to its own box origin and
        truncate to integer pixels.

        Returns ``None`` when there is nothing to embed.
        """
        kpts = np.asarray(kpts, dtype=np.float32)
        if kpts.ndim == 2:
            kpts = kpts[None]
        if len(kpts) == 0:
            return None

        bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        graphs = to_lighttrack15(kpts)                     # (N, 15, 2)
        graphs = graphs - bboxes[:, None, :2]              # box-origin shift
        graphs = graphs.astype(np.int32).astype(np.float32)  # int truncation

        # (N, 15, 2) -> (N, 2, 1, 15, 1)
        packed = graphs.transpose(0, 2, 1)[:, :, None, :, None]
        return torch.from_numpy(np.ascontiguousarray(packed))

    @torch.no_grad()
    def _embed(self, kpts: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """Embed a batch of poses into the SGCN feature space."""
        tensor = self._to_graph_tensor(kpts, bboxes)
        if tensor is None:
            return np.zeros((0, 1), dtype=np.float32)

        feats = []
        for start in range(0, len(tensor), self.batch_size):
            chunk = tensor[start:start + self.batch_size].float().to(
                self.device)
            # The Siamese model embeds both inputs with shared weights;
            # feeding the same batch twice yields that shared embedding.
            out, _ = self.model(chunk, chunk)
            feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32)

    @staticmethod
    def _degenerate(bboxes: np.ndarray) -> np.ndarray:
        """Boxes upstream would reject via ``bbox_invalid``."""
        bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        return (w <= 0) | (h <= 0) | (w > 2000) | (h > 2000)

    # ------------------------------------------------------------------
    # BasePoseMatcher interface
    # ------------------------------------------------------------------

    def distance(
        self,
        kpts_a: np.ndarray,
        bbox_a: np.ndarray,
        kpts_b: np.ndarray,
        bbox_b: np.ndarray,
        scores_a: Optional[np.ndarray] = None,
        scores_b: Optional[np.ndarray] = None,
    ) -> float:
        return float(self.distance_matrix(
            np.asarray(kpts_a)[None], np.asarray(bbox_a)[None],
            np.asarray(kpts_b)[None], np.asarray(bbox_b)[None])[0, 0])

    def distance_matrix(
        self,
        kpts_a: np.ndarray,
        bboxes_a: np.ndarray,
        kpts_b: np.ndarray,
        bboxes_b: np.ndarray,
        scores_a: Optional[np.ndarray] = None,
        scores_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Pairwise embedding distances.

        Keypoint confidences are ignored, as upstream: ``keypoints_to_graph``
        reads each joint's score and then discards it.
        """
        m, n = len(kpts_a), len(kpts_b)
        out = np.full((m, n), self.INVALID_DISTANCE, dtype=np.float32)
        if m == 0 or n == 0:
            return out

        feats_a = self._embed(kpts_a, bboxes_a)
        feats_b = self._embed(kpts_b, bboxes_b)

        # Upstream: euclidean distance between the two 128-D embeddings.
        dists = np.linalg.norm(
            feats_a[:, None, :] - feats_b[None, :, :], axis=-1)

        bad_a = self._degenerate(bboxes_a)
        bad_b = self._degenerate(bboxes_b)
        valid = ~bad_a[:, None] & ~bad_b[None, :]
        out[valid] = dists[valid]
        return out
