# Copyright (c) OpenMMLab. All rights reserved.
"""PGPT's PoseGCN appearance embedding.

    Zhang et al., "Pose-Guided Tracking-by-Detection: Robust Multi-Person
    Pose Tracking", IEEE TMM 2020.  https://github.com/JDAI-CV/PGPT

PGPT's re-identification feature is a 2048-D descriptor from a ResNet-152
in which the pose branch conditions the appearance branch.  The network is
imported unmodified from the submodule
(``external/PGPT/lib/pose/models/pose_gcn.py``) and the published
``pose_gcn.pth.tar`` weights are loaded into it; only the forward pass and
the pre/post-processing are re-implemented here.

Which embedding variant actually ships
--------------------------------------
``PoseResNet.forward`` offers two appearance paths, and **the released
checkpoint only covers one of them**:

``'pose_gated'`` (upstream ``flag=2``) - default, fully weighted.
    ``embedding_layer(backbone)`` is multiplied by ``pose_conv(heatmaps)``,
    so the pose branch gates the appearance map, and ``fc_feature`` pools
    the result to 2048-D.  Every parameter it needs is in
    ``pose_gcn.pth.tar``.

``'graph'`` (upstream ``flag=1``) - **weights were never released**.
    The heatmaps pick a 3x3 patch of the embedding map around each of 15
    joints and two graph-convolution layers propagate those part features
    over the skeleton, fused as ``0.1 * global + 0.9 * part``.  This is the
    path ``PoseNet.embedding`` nominally calls, but ``graph_layer1``,
    ``graph_layer2`` and ``fc_feature_align`` are absent from the published
    checkpoint.  Upstream's ``_load_model`` merges the checkpoint into
    ``model.state_dict()`` and so leaves them **randomly initialised**,
    silently - running the demo as shipped embeds crops with an untrained
    graph head.  The path is implemented and verified here so that real
    weights drop straight in, but selecting it without supplying them
    raises rather than producing noise.

Why the forward pass is re-implemented
--------------------------------------
Upstream's ``forward`` cannot run on this stack: it builds its joint-patch
mask as ``torch.cuda.ByteTensor`` and calls ``masked_select`` with it - a
``uint8`` mask, which torch >= 1.2 rejects outright (verified: torch 2.4
raises ``RuntimeError: masked_select: expected BoolTensor``) - and it
hard-codes ``torch.cuda.LongTensor`` / ``torch.cuda.FloatTensor``, as does
``GraphConvolution.forward``, so nothing can run on CPU.  The patch
extraction is also a Python loop over every (sample, joint) pair.

The functions below reproduce the same computation with bool masks,
device-agnostic tensors and a vectorised patch gather.  **Weights and
architecture are untouched** - this is an execution shim in the spirit of
``mmpose/models/pose_estimators/petr_compat.py``, not a redesign.
:func:`_align_features_reference` is the literal transcription of the
upstream loop, and :func:`_align_features` is checked against it bit-exactly
(including the clamped-border cases) in the tests.

Deviation: inference-time dropout
---------------------------------
Upstream's graph branch runs ``F.dropout(F.relu(self.graph_layer1(...)))``.
``torch.nn.functional.dropout`` defaults to ``training=True``, so that
dropout stays active even though the caller has just invoked
``model.eval()`` - the embedding is randomly masked and rescaled at
inference, and successive calls on the same crop disagree.  This wrapper
disables it by default (``upstream_dropout=False``) so tracking is
deterministic; pass ``upstream_dropout=True`` to reproduce the literal
behaviour.  Only the ``'graph'`` variant is affected.
"""

from __future__ import annotations

import os.path as osp
import sys
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..registry import POST_PROCESS_APPEARANCE_EMBEDDERS
from .base import BaseAppearanceEmbedder

_PGPT_ROOT = osp.abspath(
    osp.join(osp.dirname(__file__), '..', '..', '..', 'external', 'PGPT'))

#: Upstream ``PoseNet`` constants (``inference/pose_estimation_graph.py``).
_INPUT_W, _INPUT_H = 288, 384
_PIXEL_STD = 200.0
_PADDING = 1.25
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def pgpt_area(
    bbox: np.ndarray,
    aspect_ratio: float = _INPUT_W / _INPUT_H,
    pixel_std: float = _PIXEL_STD,
    padding: float = _PADDING,
) -> float:
    """OKS area normaliser as PGPT computes it.

    Port of ``PoseNet.x1y1x2y2_to_cs`` followed by
    ``area = np.prod(scale * 200, 1)`` in ``Track_And_Detect.oks_filter``:
    the box is first stretched to the pose network's input aspect ratio,
    then padded by 1.25, and its area in ``pixel_std`` units is the OKS
    normaliser.

    Args:
        bbox: ``xyxy`` box, shape ``(4,)``.
        aspect_ratio: Width/height of the pose network's input, upstream
            ``288/384``.
        pixel_std: Upstream ``self.pixel_std``.
        padding: Upstream's fixed ``scale * 1.25``.

    Returns:
        Area used to normalise OKS.
    """
    _, scale = x1y1x2y2_to_cs(bbox, aspect_ratio, pixel_std, padding)
    return float(np.prod(scale * pixel_std))


def x1y1x2y2_to_cs(
    bbox: np.ndarray,
    aspect_ratio: float = _INPUT_W / _INPUT_H,
    pixel_std: float = _PIXEL_STD,
    padding: float = _PADDING,
) -> Tuple[np.ndarray, np.ndarray]:
    """Port of ``PoseNet.x1y1x2y2_to_cs``: box -> (centre, scale)."""
    x, y, xmax, ymax = (float(v) for v in np.asarray(bbox).reshape(-1)[:4])
    w, h = xmax - x, ymax - y
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w / pixel_std, h / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * padding
    return center, scale


def pgpt_adjacency() -> torch.Tensor:
    """Port of ``PoseNet.reset_adj_no1``: the 15x15 skeleton matrix.

    Self-connections are ``0.9``, skeleton edges ``1.0``, everything else
    ``0.2``.
    """
    adj = torch.zeros(15, 15)
    edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (5, 7), (4, 6),
             (3, 9), (4, 10), (9, 10), (9, 11), (10, 12), (12, 14)]
    for i, j in edges:
        adj[i][j] = 1
        adj[j][i] = 1
    # Upstream sets these one-way only; reproduced exactly.
    adj[6][8] = 1
    adj[11][13] = 1

    for i in range(15):
        for j in range(15):
            if i == j:
                adj[i][j] = 1 - 0.1
            elif adj[i][j] != 1:
                adj[i][j] = 0.2
    return adj


# ---------------------------------------------------------------------------
# Joint-patch feature alignment
# ---------------------------------------------------------------------------

def _peak_indices(hp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-joint heatmap peak, with upstream's two-step reduction order.

    Args:
        hp: ``(M, H, W)`` heatmaps.

    Returns:
        ``(y, x)`` index tensors of shape ``(M,)``.
    """
    col_max, col_arg = hp.max(dim=1)      # over y, for each x
    _, x_idx = col_max.max(dim=1)         # over x
    y_idx = col_arg.gather(1, x_idx[:, None]).squeeze(1)
    return y_idx, x_idx


def _align_features_reference(
    hp: torch.Tensor,
    emb: torch.Tensor,
) -> torch.Tensor:
    """Literal transcription of upstream's ``align_feature`` loop.

    Kept as the correctness reference for :func:`_align_features`; the only
    changes are a bool mask instead of ``torch.cuda.ByteTensor`` and
    device-agnostic allocation.

    Args:
        hp: ``(B, 15, H, W)`` per-joint heatmaps.
        emb: ``(B, 256, H, W)`` embedding map.

    Returns:
        ``(B, 15, 9 * 256)`` aligned part features.
    """
    b_n, j_n, h, w = hp.shape
    feature = hp.unsqueeze(dim=2) * emb.unsqueeze(dim=1)  # (B,15,256,H,W)
    out = torch.zeros(b_n, j_n, 9 * 256, device=hp.device, dtype=hp.dtype)

    for b in range(b_n):
        for i in range(j_n):
            mask = torch.zeros(h, w, dtype=torch.bool, device=hp.device)
            temp = hp[b][i]
            y_values, ys = temp.max(dim=0)
            _, xc = y_values.max(dim=0)
            y = ys[xc]
            for n in range(0, 2):
                y_l = min(int(y) + n, h - 1)
                y_s = max(int(y) - n, 0)
                xc_l = min(int(xc) + n, w - 1)
                xc_s = max(int(xc) - n, 0)
                for yy in (y_l, y_s, int(y)):
                    for xx in (xc_l, xc_s, int(xc)):
                        mask[yy][xx] = True
            temp_feature = torch.masked_select(feature[b][i], mask)
            cnt = int(mask.sum())
            if cnt < 9:
                mean = torch.mean(temp_feature).repeat(256)
                while cnt < 9:
                    temp_feature = torch.cat((temp_feature, mean), dim=0)
                    cnt += 1
            out[b][i] = temp_feature
    return out


def _align_features(hp: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    """Vectorised equivalent of :func:`_align_features_reference`.

    The mask upstream builds is the 3x3 neighbourhood of each joint's
    heatmap peak, clamped to the map and deduplicated (it is a mask, so
    clamped-together positions collapse).  ``masked_select`` then reads the
    ``(256, H, W)`` product channel-major, taking the kept positions in
    row-major order, and short results are padded at the end with the mean
    of what was selected.  All of that is reproduced here without the
    per-(sample, joint) Python loop.

    Args:
        hp: ``(B, 15, H, W)`` per-joint heatmaps.
        emb: ``(B, 256, H, W)`` embedding map.

    Returns:
        ``(B, 15, 9 * 256)`` aligned part features.
    """
    b_n, j_n, h, w = hp.shape
    c_n = emb.shape[1]
    m = b_n * j_n

    hp_flat = hp.reshape(m, h, w)
    y_idx, x_idx = _peak_indices(hp_flat)

    offs = torch.tensor([-1, 0, 1], device=hp.device)
    rows = (y_idx[:, None] + offs[None, :]).clamp(0, h - 1)   # (M,3)
    cols = (x_idx[:, None] + offs[None, :]).clamp(0, w - 1)   # (M,3)

    # A clamped duplicate collapses in the mask; keep only first occurrences.
    keep_r = torch.ones_like(rows, dtype=torch.bool)
    keep_r[:, 1:] = rows[:, 1:] != rows[:, :-1]
    keep_c = torch.ones_like(cols, dtype=torch.bool)
    keep_c[:, 1:] = cols[:, 1:] != cols[:, :-1]

    pos = (rows[:, :, None] * w + cols[:, None, :]).reshape(m, 9)
    keep = (keep_r[:, :, None] & keep_c[:, None, :]).reshape(m, 9)
    counts = keep.sum(dim=1)                                   # in {4, 6, 9}

    # feature[b, i, c, p] = hp[b, i, p] * emb[b, c, p]
    hp_p = hp_flat.reshape(m, h * w)
    emb_p = emb.reshape(b_n, c_n, h * w)
    emb_rep = emb_p.repeat_interleave(j_n, dim=0)              # (M, C, HW)

    out = torch.zeros(m, 9 * c_n, device=hp.device, dtype=hp.dtype)
    for n in counts.unique():
        sel = (counts == n).nonzero(as_tuple=True)[0]
        n_i = int(n)
        # Positions kept, compacted and still in ascending (row-major) order.
        idx = pos[sel][keep[sel]].reshape(len(sel), n_i)
        gathered = (emb_rep[sel].gather(
            2, idx[:, None, :].expand(-1, c_n, -1))
            * hp_p[sel].gather(1, idx)[:, None, :])           # (m_i, C, n)
        flat = gathered.reshape(len(sel), c_n * n_i)
        if n_i < 9:
            mean = flat.mean(dim=1, keepdim=True)
            pad = mean.expand(-1, c_n * (9 - n_i))
            flat = torch.cat((flat, pad), dim=1)
        out[sel] = flat

    return out.reshape(b_n, j_n, 9 * c_n)


def _backbone(model, x: torch.Tensor) -> torch.Tensor:
    """Shared ResNet trunk, as at the top of ``PoseResNet.forward``."""
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    return x


def _embed_forward_pose_gated(model, x: torch.Tensor) -> torch.Tensor:
    """Device-agnostic re-implementation of ``PoseResNet.forward(flag=2)``.

    The appearance map is gated by the pose branch
    (``embedding_layer(trunk) * pose_conv(heatmaps)``) and pooled to
    2048-D.  This is the variant the released checkpoint covers.

    Args:
        model: The upstream ``PoseResNet``.
        x: ``(B, 3, 384, 288)`` normalised input.

    Returns:
        ``(B, 2048)`` embeddings.
    """
    trunk = _backbone(model, x)
    heatmaps = model.final_layer(model.deconv_layers(trunk))
    gated = model.embedding_layer(trunk) * model.pose_conv(heatmaps)
    out = model.fc_feature(gated.reshape(gated.size(0), -1))
    if not model.feature_flag:
        out = model.classification(out)
    return out


def _embed_forward_graph(
    model,
    x: torch.Tensor,
    adj: torch.Tensor,
    upstream_dropout: bool = False,
) -> torch.Tensor:
    """Device-agnostic re-implementation of ``PoseResNet.forward(flag=1)``.

    Requires ``graph_layer1``, ``graph_layer2`` and ``fc_feature_align``,
    none of which are in the released checkpoint - see the module
    docstring.

    Args:
        model: The upstream ``PoseResNet``.
        x: ``(B, 3, 384, 288)`` normalised input.
        adj: ``(B, 15, 15)`` adjacency.
        upstream_dropout: Keep upstream's inference-time dropout (see the
            module docstring).

    Returns:
        ``(B, 2048)`` embeddings.
    """
    trunk = _backbone(model, x)

    hp = model.final_layer(model.deconv_layers(trunk))
    # Upstream: F.upsample(..., mode='bilinear'), i.e. align_corners=False
    # for the torch 0.4 it targets.
    hp = F.interpolate(hp, size=[12, 9], mode='bilinear', align_corners=False)
    hp = torch.index_select(hp, 1, model.indices.to(hp.device))

    emb = model.embedding_layer(trunk)
    align_feature = _align_features(hp, emb)

    # GraphConvolution.forward hard-codes torch.cuda.FloatTensor, so apply
    # the same maths directly on the layer's parameters instead.
    def graph_conv(layer, inp):
        support = torch.matmul(inp, layer.weight)
        out = torch.bmm(adj, support)
        if layer.bias is not None:
            out = out + layer.bias
        return out

    x_part = F.relu(graph_conv(model.graph_layer1, align_feature))
    x_part = F.dropout(x_part, training=upstream_dropout)
    x_part = graph_conv(model.graph_layer2, x_part)
    x_part = model.fc_feature_align(x_part.reshape(x_part.size(0), -1))

    x_glob = model.fc_feature(emb.reshape(emb.size(0), -1))
    out = 0.1 * x_glob + 0.9 * x_part
    if not model.feature_flag:
        out = model.classification(out)
    return out


@POST_PROCESS_APPEARANCE_EMBEDDERS.register_module()
class PGPTPoseGCNEmbedder(BaseAppearanceEmbedder):
    """2048-D PoseGCN appearance descriptors for detection crops.

    Each box is affine-warped to 288x384 the way upstream's
    ``PoseNet.embedding`` does, ImageNet-normalised, and embedded; with
    ``flip_test`` the horizontally flipped crop is embedded too and the two
    descriptors are averaged, as upstream.  Distances are cosine on the
    2048-D vector (``Matcher.distance``).

    Args:
        checkpoint: Path to PGPT's ``pose_gcn.pth.tar``.  Obtain with
            ``gdown 1emHrW4OFFOndmR5OIUfHDq4xf8yhdPjR``.
        pgpt_root: Path to the ``external/PGPT`` submodule.
        device: Torch device for the network.
        variant: ``'pose_gated'`` (default, upstream ``flag=2``) or
            ``'graph'`` (upstream ``flag=1``).  See the module docstring -
            the released checkpoint only carries weights for the first.
        flip_test: Average with the horizontally flipped crop, upstream
            ``cfg.TEST.FLIP_TEST = True``.
        batch_size: Maximum crops per forward pass.  Upstream embeds one
            box at a time; batching is equivalent in eval mode.
        upstream_dropout: Reproduce upstream's active-at-inference dropout,
            which makes embeddings stochastic.  ``'graph'`` variant only.
    """

    def __init__(
        self,
        checkpoint: str = 'data/models/pgpt_pose_gcn.pth.tar',
        pgpt_root: str = _PGPT_ROOT,
        device: str = 'cuda:0',
        variant: str = 'pose_gated',
        flip_test: bool = True,
        batch_size: int = 16,
        upstream_dropout: bool = False,
    ) -> None:
        if variant not in ('pose_gated', 'graph'):
            raise ValueError(
                f"variant must be 'pose_gated' or 'graph', got {variant!r}")
        get_pose_net, cfg = self._import_upstream(pgpt_root)

        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)
        self.variant = variant
        self.flip_test = bool(flip_test)
        self.batch_size = int(batch_size)
        self.upstream_dropout = bool(upstream_dropout)

        self.model = get_pose_net(cfg, is_train=False, flag=1)
        self._load_weights(checkpoint)
        self.model.eval().to(self.device)

        self._adj = pgpt_adjacency().to(self.device)
        self._mean = np.array(_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array(_STD, dtype=np.float32).reshape(3, 1, 1)

    @staticmethod
    def _import_upstream(pgpt_root: str):
        """Import the unmodified upstream network and build its config.

        ``pose.core.config`` is deliberately not used: its ``update_config``
        calls ``yaml.load`` without a loader, which PyYAML >= 6 rejects, and
        it carries a literal ``'${PGPT_ROOT}'`` path.  The few fields the
        model constructor reads are loaded from the shipped YAML directly.
        """
        import yaml
        from mmengine.config import ConfigDict

        pgpt_root = osp.abspath(pgpt_root)
        lib_dir = osp.join(pgpt_root, 'lib')
        if not osp.isdir(lib_dir):
            raise FileNotFoundError(
                f'PGPT submodule not found at {pgpt_root!r}. Run '
                f'`git submodule update --init external/PGPT`.')
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)

        from pose.models.pose_gcn import get_pose_net

        cfg_path = osp.join(pgpt_root, 'cfgs', 'pose_res152.yaml')
        with open(cfg_path, 'r') as f:
            raw = yaml.safe_load(f)
        return get_pose_net, ConfigDict(raw)

    def _load_weights(self, checkpoint: str) -> None:
        """Load ``pose_gcn.pth.tar``, mirroring ``PoseNet._load_model``."""
        if not osp.isfile(checkpoint):
            raise FileNotFoundError(
                f'PGPT PoseGCN checkpoint not found: {checkpoint!r}.\n'
                f'Download it with:\n'
                f'    python -m gdown 1emHrW4OFFOndmR5OIUfHDq4xf8yhdPjR '
                f'-O {checkpoint}\n'
                f'(link from external/PGPT/README.md), or run the '
                f'appearance-free variant instead:\n'
                f'    configs/post_processing/pgpt_geom.py')

        state = torch.load(checkpoint, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        state = {(k[7:] if k.startswith('module.') else k): v
                 for k, v in state.items()}
        missing, _ = self.model.load_state_dict(state, strict=False)

        # The released checkpoint has no weights for the graph branch;
        # upstream silently leaves them randomly initialised, which would
        # embed crops with an untrained head.  Refuse instead.
        graph_params = {'graph_layer1.weight', 'graph_layer1.bias',
                        'graph_layer2.weight', 'graph_layer2.bias',
                        'fc_feature_align.weight', 'fc_feature_align.bias'}
        missing_graph = sorted(graph_params & set(missing))
        if self.variant == 'graph' and missing_graph:
            raise RuntimeError(
                f"variant='graph' needs the graph-branch parameters "
                f'{missing_graph}, which are absent from {checkpoint!r}. '
                f"PGPT never released them - the published pose_gcn.pth.tar "
                f"covers only the pose-gated embedding. Use "
                f"variant='pose_gated' (the default), or supply a checkpoint "
                f'that contains the graph head.')

        needed_missing = [
            k for k in missing if k not in graph_params
        ]
        if self.variant == 'pose_gated':
            # pose_gated uses trunk + deconv/final + embedding_layer +
            # pose_conv + fc_feature; nothing else may be missing.
            if needed_missing:
                raise RuntimeError(
                    f'PGPT checkpoint {checkpoint!r} is missing '
                    f'{len(needed_missing)} parameters the pose-gated '
                    f'embedding needs, e.g. {needed_missing[:5]}. Is this '
                    f'the pose_gcn.pth.tar from the PGPT README?')

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _crop(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Affine-warp a box to 288x384 and normalise, as upstream.

        Upstream feeds the BGR frame straight into a ``ToTensor`` +
        ``Normalize`` chain without swapping channels, so the network sees
        BGR.  That is reproduced here rather than "fixed": the published
        weights were used that way.
        """
        from pose.utils.transforms import get_affine_transform

        center, scale = x1y1x2y2_to_cs(bbox)
        trans = get_affine_transform(center, scale, 0, (_INPUT_W, _INPUT_H))
        warped = cv2.warpAffine(
            image, trans, (_INPUT_W, _INPUT_H), flags=cv2.INTER_LINEAR)
        chw = warped.transpose(2, 0, 1).astype(np.float32) / 255.0
        return (chw - self._mean) / self._std

    @torch.no_grad()
    def embed(self, image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        if len(bboxes) == 0:
            return np.zeros((0, 2048), dtype=np.float32)
        if image is None:
            raise ValueError(
                'PGPTPoseGCNEmbedder.embed() needs frame pixels; the caller '
                'passed image=None.')

        crops = np.stack([self._crop(image, b) for b in bboxes])

        def forward(batch: torch.Tensor) -> torch.Tensor:
            if self.variant == 'pose_gated':
                return _embed_forward_pose_gated(self.model, batch)
            adj = self._adj[None].expand(len(batch), -1, -1)
            return _embed_forward_graph(
                self.model, batch, adj, self.upstream_dropout)

        feats = []
        for start in range(0, len(crops), self.batch_size):
            chunk = torch.from_numpy(
                crops[start:start + self.batch_size]).to(self.device)
            out = forward(chunk)
            if self.flip_test:
                out = (out + forward(torch.flip(chunk, dims=[3]))) * 0.5
            feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32)
