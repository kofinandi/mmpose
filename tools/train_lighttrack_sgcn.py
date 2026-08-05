# Copyright (c) OpenMMLab. All rights reserved.
"""Train LightTrack's Siamese GCN pose matcher on PoseTrack21.

    Ning et al., "LightTrack: A Generic Framework for Online Top-Down Human
    Pose Tracking", CVPRW 2020.  https://github.com/Guanghan/lighttrack

Why this script exists
----------------------
LightTrack's published SGCN checkpoint (``weights/GCN/epoch210_model.pt``,
from ``GCN.zip``) is no longer downloadable: ``guanghan.info`` returns 404
for it and for the ``gcn_data_train_val.tar.gz`` training pairs, and the
Wayback Machine has no snapshot of either (upstream issue #21, "Download
link expired", is still open).  The network definition and training recipe
*are* in the repository, so the matcher is retrained here from local
PoseTrack21 annotations.

**The resulting weights are not the published ones.**  The architecture,
loss, optimiser and schedule follow ``graph/config/train.yaml``, and the
model, contrastive loss and feeder are imported unmodified from the
submodule, but the training data is regenerated (see below), so results
obtained with this checkpoint are not a reproduction of the paper's
numbers.

Pair generation
---------------
Mirrors ``graph/gcn_utils/keypoints_to_graph*.py``:

* **Positives** - the same ``track_id`` in two frames of one sequence, at
  most ``max_frame_gap`` frames apart (upstream requires consecutive
  annotated frames).
* **Negatives** - two different ``track_id``\\ s.  Half are sampled within a
  frame or its neighbour (*hard*: same scene, similar scale, which is what
  the tracker actually has to separate), half uniformly at random across
  the split, matching upstream's split between
  ``keypoints_to_graph_negative`` and ``keypoints_to_graph_hard_neg``.
* Boxes with width or height <= 100 px are dropped
  (``validate_bbox``), and each pose is expressed as box-origin-shifted,
  integer-truncated pixel coordinates (``keypoints_to_graph``).

Keypoint layout
---------------
Poses are converted with
:func:`~mmpose.postprocessing.matchers.to_lighttrack15`, the *same*
function :class:`SGCNPoseMatcher` applies at inference.  It reads only the
nose and limb joints, which COCO-17 and PoseTrack-17 index identically, and
synthesises the graph's head_bottom/head_top nodes from nose and shoulders.
Training therefore sees exactly the representation inference produces -
including the synthesised head joints - rather than PoseTrack's annotated
head joints, which COCO predictions could never supply.

Usage::

    python tools/train_lighttrack_sgcn.py \\
        --out data/models/lighttrack_sgcn_posetrack21.pt

    # quick smoke run
    python tools/train_lighttrack_sgcn.py --epochs 2 --max-pairs 5000 \\
        --out /tmp/sgcn_smoke.pt
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mmpose.postprocessing.matchers import to_lighttrack15

_LIGHTTRACK_ROOT = osp.abspath(
    osp.join(osp.dirname(__file__), '..', 'external', 'lighttrack'))

#: Upstream ``validate_bbox``: boxes narrower/shorter than this are skipped.
_MIN_BOX_SIDE = 100


def _import_upstream(lighttrack_root: str):
    """Import the unmodified SGCN model, loss and feeder from the submodule.

    Only ``graph/`` is put on ``sys.path``; ``graph.visualize_pose_matching``
    is deliberately *not* imported (it instantiates a ``Pose_Matcher`` at
    module scope, which parses argv, allocates a GPU and loads a checkpoint
    on import), and neither is ``torchlight``'s IO layer (it mutates
    ``CUDA_VISIBLE_DEVICES`` and writes a work_dir, and its ``yaml.load``
    call is broken under PyYAML >= 6).
    """
    graph_dir = osp.join(lighttrack_root, 'graph')
    if not osp.isdir(graph_dir):
        raise FileNotFoundError(
            f'LightTrack submodule not found at {lighttrack_root!r}. Run '
            f'`git submodule update --init external/lighttrack`.')
    if graph_dir not in sys.path:
        sys.path.insert(0, graph_dir)

    from gcn_utils.contrastive import ContrastiveLoss
    from gcn_utils.gcn_model import Model
    return Model, ContrastiveLoss


# ── Pair generation ────────────────────────────────────────────────────────

def _load_tracks(ann_file: str) -> Dict[str, Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]]]:
    """Group annotations into ``{sequence: {track_id: [(frame, kpts, bbox)]}}``.

    Keypoints come back in LightTrack's 15-joint graph layout; boxes are
    ``xywh`` as stored in the COCO-style annotation file.
    """
    with open(ann_file, 'r') as f:
        data = json.load(f)

    img_meta = {
        int(im['id']): (im.get('seq_name', ''), int(im.get('frame_id', 0)))
        for im in data['images']
    }

    tracks: Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for ann in data['annotations']:
        if ann.get('iscrowd', 0):
            continue
        track_id = ann.get('track_id')
        if track_id is None:
            continue
        bbox = np.asarray(ann['bbox'], dtype=np.float32)  # xywh
        if bbox[2] <= _MIN_BOX_SIDE or bbox[3] <= _MIN_BOX_SIDE:
            continue

        kp = np.asarray(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
        if kp.shape[0] != 17:
            continue
        # The synthesised head joints need nose and both shoulders; the
        # limbs are used as-is (upstream likewise keeps unlabelled joints,
        # which arrive as zeros).
        if not np.all(kp[[0, 5, 6], 2] > 0):
            continue

        seq, frame = img_meta.get(int(ann['image_id']), ('', 0))
        graph = to_lighttrack15(kp[:, :2])
        tracks[seq][int(track_id)].append((frame, graph, bbox))

    return tracks


def _to_graph(kpts15: np.ndarray, bbox_xywh: np.ndarray) -> np.ndarray:
    """Port of ``keypoints_to_graph``: box-origin shift, integer truncation."""
    shifted = kpts15 - bbox_xywh[:2][None, :]
    return shifted.astype(np.int32).astype(np.float32)


def build_pairs(
    ann_file: str,
    max_frame_gap: int = 1,
    max_pairs: int = 0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build positive and negative graph pairs from a PoseTrack21 split.

    Returns:
        ``(a, b, labels)`` where ``a``/``b`` are ``(N, 15, 2)`` graphs and
        ``labels`` is ``(N,)`` with 1 for "same person".
    """
    rng = random.Random(seed)
    tracks = _load_tracks(ann_file)

    positives: List[Tuple[np.ndarray, np.ndarray]] = []
    flat: List[Tuple[str, int, np.ndarray, np.ndarray, int]] = []
    by_frame: Dict[Tuple[str, int], list] = defaultdict(list)

    for seq, seq_tracks in tracks.items():
        for track_id, entries in seq_tracks.items():
            entries.sort(key=lambda e: e[0])
            for frame, kpts, bbox in entries:
                g = _to_graph(kpts, bbox)
                flat.append((seq, track_id, g, bbox, frame))
                by_frame[(seq, frame)].append(len(flat) - 1)
            for (f0, k0, b0), (f1, k1, b1) in zip(entries, entries[1:]):
                if 0 < f1 - f0 <= max_frame_gap:
                    positives.append((_to_graph(k0, b0), _to_graph(k1, b1)))

    rng.shuffle(positives)
    if max_pairs:
        positives = positives[:max_pairs // 2]

    # Negatives: half hard (same or adjacent frame), half random.
    n_neg = len(positives)
    n_hard = n_neg // 2
    negatives: List[Tuple[np.ndarray, np.ndarray]] = []

    frame_keys = [k for k, v in by_frame.items() if len(v) >= 2]
    rng.shuffle(frame_keys)
    for key in frame_keys:
        if len(negatives) >= n_hard:
            break
        idxs = by_frame[key]
        for _ in range(min(3, len(idxs))):
            i, j = rng.sample(idxs, 2)
            if flat[i][1] != flat[j][1]:
                negatives.append((flat[i][2], flat[j][2]))
            if len(negatives) >= n_hard:
                break

    guard = 0
    while len(negatives) < n_neg and guard < 50 * n_neg:
        guard += 1
        i, j = rng.randrange(len(flat)), rng.randrange(len(flat))
        # Different person = different (sequence, track_id).
        if (flat[i][0], flat[i][1]) != (flat[j][0], flat[j][1]):
            negatives.append((flat[i][2], flat[j][2]))

    pairs = ([(a, b, 1) for a, b in positives]
             + [(a, b, 0) for a, b in negatives])
    rng.shuffle(pairs)

    a = np.stack([p[0] for p in pairs]).astype(np.float32)
    b = np.stack([p[1] for p in pairs]).astype(np.float32)
    y = np.array([p[2] for p in pairs], dtype=np.float32)
    return a, b, y


def _as_tensors(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> TensorDataset:
    """Pack graphs into the ``(N, C=2, T=1, V=15, M=1)`` network input."""

    def pack(arr: np.ndarray) -> torch.Tensor:
        # (N, 15, 2) -> (N, 2, 1, 15, 1)
        return torch.from_numpy(
            arr.transpose(0, 2, 1)[:, :, None, :, None].copy())

    return TensorDataset(pack(a), pack(b), torch.from_numpy(y))


# ── Training ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, margin: float) -> Tuple[float, float, float]:
    """Return ``(accuracy, mean same-person dist, mean diff-person dist)``."""
    model.eval()
    correct = total = 0
    same_d: List[float] = []
    diff_d: List[float] = []
    for x1, x2, y in loader:
        f1, f2 = model(x1.float().to(device), x2.float().to(device))
        dist = torch.sqrt(torch.sum((f1 - f2)**2, dim=1) + 1e-12).cpu()
        pred = (dist < margin).float()
        correct += int((pred == y).sum())
        total += len(y)
        same_d += dist[y > 0.5].tolist()
        diff_d += dist[y < 0.5].tolist()
    return (correct / max(total, 1),
            float(np.mean(same_d)) if same_d else 0.0,
            float(np.mean(diff_d)) if diff_d else 0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--train-ann',
                   default='data/posetrack21/annotations/posetrack21_train.json')
    p.add_argument('--val-ann',
                   default='data/posetrack21/annotations/posetrack21_val.json')
    p.add_argument('--out', default='data/models/lighttrack_sgcn_posetrack21.pt')
    p.add_argument('--lighttrack-root', default=_LIGHTTRACK_ROOT)
    # Defaults below reproduce external/lighttrack/graph/config/train.yaml.
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--base-lr', type=float, default=0.01)
    p.add_argument('--steps', type=int, nargs='+', default=[40, 60, 100, 150])
    p.add_argument('--margin', type=float, default=1.0,
                   help="ContrastiveLoss margin (upstream 'margin=1')")
    p.add_argument('--match-margin', type=float, default=0.2,
                   help='Distance below which a pair counts as a match when '
                        'reporting accuracy (upstream inference margin)')
    p.add_argument('--max-frame-gap', type=int, default=1)
    p.add_argument('--max-pairs', type=int, default=0,
                   help='Cap on training pairs (0 = all); for smoke runs')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=4)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    Model, ContrastiveLoss = _import_upstream(args.lighttrack_root)

    print(f'Building training pairs from {args.train_ann} ...')
    tr = build_pairs(args.train_ann, args.max_frame_gap, args.max_pairs,
                     args.seed)
    print(f'  train pairs: {len(tr[2])} '
          f'({int(tr[2].sum())} positive, {int((1 - tr[2]).sum())} negative)')

    print(f'Building validation pairs from {args.val_ann} ...')
    va = build_pairs(args.val_ann, args.max_frame_gap,
                     max(args.max_pairs // 4, 0), args.seed + 1)
    print(f'  val pairs  : {len(va[2])}')

    train_loader = DataLoader(
        _as_tensors(*tr), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(
        _as_tensors(*va), batch_size=256, shuffle=False,
        num_workers=args.num_workers)

    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')
    model = Model(
        in_channels=2, num_class=128, edge_importance_weighting=True,
        graph_args=dict(layout='PoseTrack', strategy='spatial')).to(device)
    loss_fn = ContrastiveLoss(margin=args.margin)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.base_lr, momentum=0.9, nesterov=True,
        weight_decay=0.0001)

    print(f'Training on {device} for {args.epochs} epochs '
          f'({sum(p.numel() for p in model.parameters())} params) ...')
    best_acc = -1.0
    os.makedirs(osp.dirname(osp.abspath(args.out)) or '.', exist_ok=True)

    for epoch in range(args.epochs):
        # Upstream adjust_lr: 0.1x at each passed milestone.
        lr = args.base_lr * (0.1**int(np.sum(epoch >= np.array(args.steps))))
        for group in optimizer.param_groups:
            group['lr'] = lr

        model.train()
        losses = []
        for x1, x2, y in train_loader:
            f1, f2 = model(x1.float().to(device), x2.float().to(device))
            loss = loss_fn(f1, f2, y.float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            acc, d_same, d_diff = evaluate(
                model, val_loader, device, args.match_margin)
            print(f'  epoch {epoch:3d}  lr {lr:.5f}  '
                  f'loss {np.mean(losses):.4f}  val acc {acc:.4f}  '
                  f'd(same) {d_same:.3f}  d(diff) {d_diff:.3f}')
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), args.out)

    print(f'\nBest validation accuracy: {best_acc:.4f}')
    print(f'Checkpoint written to {osp.abspath(args.out)}')
    print('NOTE: these are retrained weights, not the published '
          'epoch210_model.pt (which is no longer downloadable).')


if __name__ == '__main__':
    main()
