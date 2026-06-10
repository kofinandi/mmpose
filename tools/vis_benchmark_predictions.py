# Copyright (c) OpenMMLab. All rights reserved.
"""Interactive viewer for benchmark prediction exports.

Server (automatic when ``--model-name`` is set on benchmark_e2e.py)::

    python tools/benchmark_e2e.py CONFIG CKPT \\
        --det-config ... --det-checkpoint ... \\
        --model-name RTMPose --model-variant m-rfdetr

Local::

    rsync -av server:.../benchmark/predictions/DATE_coco_topdown/RTMPose-m-rfdetr/ ./preds/
    python tools/vis_benchmark_predictions.py preds/ --data-root data/ \\
        --badcase --badcase-thr 0.5

Controls: Left/Right arrows navigate; ``p`` pred keypoints; ``g`` GT keypoints;
``b`` bboxes; ``v`` visibility markers; ``m`` match OKS/IoU labels;
``c`` iscrowd coloring; ``q`` quit.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mmcv
import numpy as np

from mmpose.evaluation.functional.frame_metrics import (
    normalize_keypoint_visibility,
)

# ---- Color constants (RGB 0–1 floats) ----------------------------------------
_GT_COLOR = (0.2, 0.4, 1.0)      # blue  – GT keypoints / matched bboxes
_PRED_COLOR = (1.0, 0.25, 0.25)  # red   – pred keypoints / matched bboxes
_CROWD_COLOR = (1.0, 0.6, 0.0)   # orange – iscrowd GT bboxes


# ---- Argument parsing --------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Browse benchmark prediction exports interactively')
    parser.add_argument(
        'pred_dir',
        help='Directory containing manifest.json and frames.json')
    parser.add_argument(
        '--data-root',
        default=None,
        help='Root for resolving relative img_path values '
        '(default: manifest data_root)')
    parser.add_argument(
        '--badcase',
        action='store_true',
        help='Only show frames where metrics[metric_key] <= badcase_thr')
    parser.add_argument(
        '--badcase-thr',
        type=float,
        default=None,
        help='Badcase threshold (default: manifest badcase_defaults.thr '
        'or 0.5)')
    parser.add_argument(
        '--metric-key',
        default=None,
        help='Metric key for badcase filter (default: mean_oks)')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Keypoint score threshold for drawing predictions')
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Initial frame index within the (filtered) list')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        choices=['mmpose', 'openpose'],
        help='Skeleton style (unused; kept for CLI compatibility)')
    parser.add_argument(
        '--render-scale',
        type=float,
        default=2.0,
        help='Upscale factor for rendering. Use 1.0 for native resolution. '
        'Default: 2.0')
    return parser.parse_args()


# ---- Data loading ------------------------------------------------------------

def load_bundle(pred_dir: str):
    manifest_path = osp.join(pred_dir, 'manifest.json')
    frames_path = osp.join(pred_dir, 'frames.json')
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    with open(frames_path, 'r', encoding='utf-8') as f:
        frames = json.load(f)
    return manifest, frames


def _resolve_image_path(frame: dict, data_root: str) -> str:
    img_path = frame['img_path']
    if osp.isabs(img_path):
        return img_path
    return osp.join(data_root, img_path)


# ---- Instance parsing --------------------------------------------------------

def _parse_instances(instances_json: List[dict]) -> List[dict]:
    """Parse raw JSON instance list to a list of dicts with numpy arrays.

    Each returned dict may contain:
        keypoints (K, 2), keypoint_scores (K,), keypoints_visible (K,),
        bbox (4,) [x1,y1,x2,y2], bbox_score (float), iscrowd (int 0/1).

    When ``keypoints_visible_coco`` is present in the JSON (raw COCO 0/1/2
    flags), it is stored as ``keypoints_visible`` and ``_coco_vis=True`` is
    set so the drawing code can distinguish occluded (v=1) from visible (v=2).
    For older bundles that only have the binary ``keypoints_visible`` field,
    ``_coco_vis=False`` is set and all labeled keypoints are drawn filled.
    """
    result = []
    for item in instances_json:
        inst: dict = {}
        if 'keypoints' in item:
            inst['keypoints'] = np.asarray(
                item['keypoints'], dtype=np.float32).reshape(-1, 2)
        n_kpts = len(inst['keypoints']) if 'keypoints' in inst else None
        if 'keypoint_scores' in item:
            inst['keypoint_scores'] = np.asarray(
                item['keypoint_scores'], dtype=np.float32).reshape(-1)
        if 'keypoints_visible_coco' in item:
            # Prefer raw COCO 0/1/2 visibility when available
            inst['keypoints_visible'] = normalize_keypoint_visibility(
                item['keypoints_visible_coco'], n_kpts)
            inst['_coco_vis'] = True
        elif 'keypoints_visible' in item:
            # Binary 0/1 from older bundles or non-COCO datasets
            inst['keypoints_visible'] = normalize_keypoint_visibility(
                item['keypoints_visible'], n_kpts)
            inst['_coco_vis'] = False
        else:
            inst['_coco_vis'] = False
        if 'bbox' in item:
            # split_instances stores bbox as [[x1,y1,x2,y2]] (trailing comma
            # artefact), so we flatten before slicing.
            inst['bbox'] = np.asarray(
                item['bbox'], dtype=np.float32).reshape(-1)[:4]
        if 'bbox_score' in item:
            inst['bbox_score'] = float(item['bbox_score'])
        inst['iscrowd'] = int(item.get('iscrowd', 0))
        result.append(inst)
    return result


def _count_valid_gt(gt_insts: List[dict]) -> int:
    """Count GT instances that are valid matching targets.

    Replicates the :func:`is_valid_instance` criteria from
    ``benchmark_data.py``: non-crowd, has at least one labeled keypoint,
    positive bbox area, keypoints not all zero.
    """
    count = 0
    for inst in gt_insts:
        if inst.get('iscrowd', 0):
            continue
        kpts = inst.get('keypoints')
        if kpts is None or len(kpts) == 0:
            continue
        vis = inst.get('keypoints_visible')
        if vis is not None and int(np.sum(vis > 0)) == 0:
            continue
        bbox = inst.get('bbox')
        if bbox is not None:
            if (bbox[2] - bbox[0]) <= 0 or (bbox[3] - bbox[1]) <= 0:
                continue
        if np.max(kpts) <= 0:
            continue
        count += 1
    return count


# ---- Drawing helpers ---------------------------------------------------------

def _upscale_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Upscale image for sharper overlay rendering."""
    if scale <= 1.0:
        return img
    h, w = img.shape[:2]
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return mmcv.imresize(img, new_size, interpolation='bicubic')


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Axis-aligned IoU of two [x1, y1, x2, y2] boxes."""
    ix1 = max(float(box_a[0]), float(box_b[0]))
    iy1 = max(float(box_a[1]), float(box_b[1]))
    ix2 = min(float(box_a[2]), float(box_b[2]))
    iy2 = min(float(box_a[3]), float(box_b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = (float(box_a[2]) - float(box_a[0])) * (
        float(box_a[3]) - float(box_a[1]))
    area_b = (float(box_b[2]) - float(box_b[0])) * (
        float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _build_match_lookups(
    matches: List[dict],
) -> Tuple[Set[int], Set[int], Dict[int, dict]]:
    """Return (gt_matched, pred_matched, gt_idx → match) from match list."""
    gt_matched: Set[int] = set()
    pred_matched: Set[int] = set()
    gt_to_match: Dict[int, dict] = {}
    for m in matches:
        gt_matched.add(int(m['gt_idx']))
        pred_matched.add(int(m['pred_idx']))
        gt_to_match[int(m['gt_idx'])] = m
    return gt_matched, pred_matched, gt_to_match


def _draw_kpts_and_skeleton(
    ax,
    instances: List[dict],
    color,
    is_gt: bool,
    kpt_thr: float,
    skeleton_links: List,
    scale: float,
    show_visibility: bool,
) -> None:
    """Draw skeleton links and keypoint circles on *ax*.

    For GT instances visibility is taken from ``keypoints_visible``.
    When the instance carries ``_coco_vis=True`` (raw COCO 0/1/2 values) and
    *show_visibility* is True, v==1 (labeled but occluded) keypoints are drawn
    as empty circles; v==2 (visible) are filled; v==0 are skipped.
    For older bundles with binary visibility (``_coco_vis=False``), v==1 is
    treated as visible (filled) since the occluded/visible distinction is lost.
    For pred instances visibility is determined by ``keypoint_scores``.
    """
    radius = max(3.0, 3.0 * scale)
    lw = max(1.0, scale)

    for inst in instances:
        kpts = inst.get('keypoints')
        if kpts is None:
            continue
        kpts_s = kpts * scale  # scaled coordinates

        if is_gt:
            vis = inst.get('keypoints_visible')
            # When no visibility info, assume all keypoints are visible
            if vis is None:
                vis = np.full(len(kpts), 2.0, dtype=np.float32)
            coco_vis = inst.get('_coco_vis', False)

            # Skeleton links – both endpoints must be labeled (v > 0)
            for sk in skeleton_links:
                i0, i1 = int(sk[0]), int(sk[1])
                if i0 >= len(kpts) or i1 >= len(kpts):
                    continue
                v0 = float(vis[i0]) if i0 < len(vis) else 2.0
                v1 = float(vis[i1]) if i1 < len(vis) else 2.0
                if v0 < 0.5 or v1 < 0.5:
                    continue
                ax.plot(
                    [kpts_s[i0, 0], kpts_s[i1, 0]],
                    [kpts_s[i0, 1], kpts_s[i1, 1]],
                    color=color, linewidth=lw, alpha=0.7, zorder=3,
                    solid_capstyle='round',
                )

            # Keypoint circles
            for k_idx in range(len(kpts)):
                kx, ky = kpts_s[k_idx]
                v = float(vis[k_idx]) if k_idx < len(vis) else 2.0
                if v < 0.5:
                    # v == 0: not labeled – skip
                    continue
                # Empty circle only when raw COCO vis is available and v==1
                # (occluded).  For binary-only bundles v==1 means "visible".
                if show_visibility and coco_vis and v < 1.5:
                    # COCO v == 1: labeled but occluded – empty circle
                    circ = mpatches.Circle(
                        (kx, ky), radius=radius,
                        facecolor='none', edgecolor=color,
                        linewidth=lw, zorder=4,
                    )
                else:
                    # COCO v == 2 or binary v == 1 or visibility markers off
                    circ = mpatches.Circle(
                        (kx, ky), radius=radius,
                        facecolor=color, edgecolor=color,
                        linewidth=lw, alpha=0.85, zorder=4,
                    )
                ax.add_patch(circ)

        else:
            scores = inst.get('keypoint_scores')
            if scores is None:
                scores = np.ones(len(kpts), dtype=np.float32)

            # Skeleton links – both endpoints above threshold
            for sk in skeleton_links:
                i0, i1 = int(sk[0]), int(sk[1])
                if i0 >= len(kpts) or i1 >= len(kpts):
                    continue
                s0 = float(scores[i0]) if i0 < len(scores) else 1.0
                s1 = float(scores[i1]) if i1 < len(scores) else 1.0
                if s0 < kpt_thr or s1 < kpt_thr:
                    continue
                ax.plot(
                    [kpts_s[i0, 0], kpts_s[i1, 0]],
                    [kpts_s[i0, 1], kpts_s[i1, 1]],
                    color=color, linewidth=lw, alpha=0.7, zorder=3,
                    solid_capstyle='round',
                )

            # Keypoint circles
            for k_idx in range(len(kpts)):
                kx, ky = kpts_s[k_idx]
                sc = float(scores[k_idx]) if k_idx < len(scores) else 1.0
                if sc < kpt_thr:
                    continue
                circ = mpatches.Circle(
                    (kx, ky), radius=radius,
                    facecolor=color, edgecolor=color,
                    linewidth=lw, alpha=0.85, zorder=4,
                )
                ax.add_patch(circ)


def _draw_bboxes(
    ax,
    instances: List[dict],
    base_color,
    matched_set: Set[int],
    show_iscrowd: bool,
    scale: float,
) -> None:
    """Draw bounding boxes for all instances.

    - Matched bboxes: solid outline.
    - Non-matched bboxes: dashed outline.
    - GT iscrowd==1 (when *show_iscrowd* is True): orange outline, overrides
      *base_color* regardless of match status.
    """
    lw = max(1.5, 1.5 * scale)
    for idx, inst in enumerate(instances):
        bbox = inst.get('bbox')
        if bbox is None:
            continue
        x1, y1, x2, y2 = (float(v) * scale for v in bbox)
        w, h = x2 - x1, y2 - y1
        is_matched = idx in matched_set
        is_crowd = show_iscrowd and inst.get('iscrowd', 0) == 1
        color = _CROWD_COLOR if is_crowd else base_color
        linestyle = '-' if is_matched else '--'
        rect = mpatches.Rectangle(
            (x1, y1), w, h,
            fill=False,
            edgecolor=color,
            linewidth=lw,
            linestyle=linestyle,
            zorder=5,
        )
        ax.add_patch(rect)


def _draw_match_labels(
    ax,
    gt_insts: List[dict],
    pred_insts: List[dict],
    gt_to_match: Dict[int, dict],
    scale: float,
) -> None:
    """Draw OKS + bbox-IoU text labels above each matched GT-pred pair."""
    for gt_idx, match in gt_to_match.items():
        pred_idx = int(match['pred_idx'])
        oks_val = float(match['oks'])
        if gt_idx >= len(gt_insts) or pred_idx >= len(pred_insts):
            continue

        gt_bbox = gt_insts[gt_idx].get('bbox')
        pred_bbox = pred_insts[pred_idx].get('bbox')

        if gt_bbox is not None and pred_bbox is not None:
            iou_val = _bbox_iou(gt_bbox, pred_bbox)
            label = f'OKS={oks_val:.2f} IoU={iou_val:.2f}'
        else:
            label = f'OKS={oks_val:.2f}'

        # Anchor at GT bbox top-left, fall back to pred bbox
        ref_bbox = gt_bbox if gt_bbox is not None else pred_bbox
        if ref_bbox is None:
            continue
        tx = float(ref_bbox[0]) * scale
        ty = float(ref_bbox[1]) * scale

        ax.text(
            tx, ty,
            label,
            color='white',
            fontsize=max(6, int(7 * scale)),
            ha='left', va='bottom',
            zorder=6,
            bbox=dict(
                facecolor='black', alpha=0.55,
                boxstyle='round,pad=0.2',
                edgecolor='none',
            ),
        )


# ---- Frame filtering / metrics -----------------------------------------------

def _filter_frames(frames: List[dict], manifest: dict, args) -> List[dict]:
    if not args.badcase:
        return frames
    defaults = manifest.get('badcase_defaults', {})
    metric_key = args.metric_key or defaults.get('metric_key', 'mean_oks')
    thr = (args.badcase_thr if args.badcase_thr is not None
           else defaults.get('thr', 0.5))
    filtered = []
    for frame in frames:
        metrics = frame.get('metrics', {})
        if metric_key not in metrics:
            continue
        if metrics[metric_key] <= thr:
            filtered.append(frame)
    return filtered


def _format_metrics(metrics: dict) -> str:
    parts = []
    for key in sorted(metrics.keys()):
        if key == 'matches':
            continue
        val = metrics[key]
        if isinstance(val, float):
            parts.append(f'{key}={val:.4f}')
        else:
            parts.append(f'{key}={val}')
    return '  '.join(parts)


# ---- Browser -----------------------------------------------------------------

class PredictionBrowser:
    """Matplotlib browser for benchmark prediction frames."""

    def __init__(self, frames: List[dict], manifest: dict, args):
        self.frames = frames
        self.manifest = manifest
        self.args = args
        self.data_root = args.data_root or manifest.get('data_root', 'data/')
        self.index = min(max(args.start_index, 0), max(len(frames) - 1, 0))

        self.show_pred = True
        self.show_gt = True
        self.show_bbox = True
        self.show_visibility = True   # v: empty circles for occluded GT kpts
        self.show_match_labels = True  # m: OKS + IoU text labels
        self.show_iscrowd = True      # c: orange bboxes for iscrowd GT

        self.render_scale = max(float(args.render_scale), 1.0)

        # Extract skeleton connectivity from dataset_meta (robust fallback)
        dmeta = manifest.get('dataset_meta', {})
        raw_links = dmeta.get('skeleton_links', [])
        self.skeleton_links: List = [list(lk) for lk in raw_links]

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._render()

    def _load_image(self, frame: dict) -> np.ndarray:
        path = _resolve_image_path(frame, self.data_root)
        if not osp.isfile(path):
            raise FileNotFoundError(
                f'Image not found: {path} (data_root={self.data_root})')
        return mmcv.imread(path, channel_order='rgb')

    def _render(self) -> None:
        if not self.frames:
            self.ax.clear()
            self.ax.set_title('No frames to display')
            self.fig.canvas.draw_idle()
            return

        frame = self.frames[self.index]
        img = self._load_image(frame)
        img = _upscale_image(img, self.render_scale)

        pred_json = frame['predictions']['instances']
        gt_json = frame['ground_truth']['instances']
        pred_insts = _parse_instances(pred_json)
        gt_insts = _parse_instances(gt_json)

        # Build match lookups from stored metrics
        matches: List[dict] = frame.get('metrics', {}).get('matches', [])
        gt_matched, pred_matched, gt_to_match = _build_match_lookups(matches)

        # Render base image
        self.ax.clear()
        self.ax.imshow(img, interpolation='nearest')
        self.ax.axis('off')
        self.ax.autoscale(False)

        scale = self.render_scale

        # GT overlays
        if self.show_gt:
            if self.show_bbox:
                _draw_bboxes(
                    self.ax, gt_insts, _GT_COLOR, gt_matched,
                    show_iscrowd=self.show_iscrowd, scale=scale,
                )
            _draw_kpts_and_skeleton(
                self.ax, gt_insts, _GT_COLOR,
                is_gt=True,
                kpt_thr=self.args.kpt_thr,
                skeleton_links=self.skeleton_links,
                scale=scale,
                show_visibility=self.show_visibility,
            )

        # Pred overlays
        if self.show_pred:
            if self.show_bbox:
                _draw_bboxes(
                    self.ax, pred_insts, _PRED_COLOR, pred_matched,
                    show_iscrowd=False, scale=scale,
                )
            _draw_kpts_and_skeleton(
                self.ax, pred_insts, _PRED_COLOR,
                is_gt=False,
                kpt_thr=self.args.kpt_thr,
                skeleton_links=self.skeleton_links,
                scale=scale,
                show_visibility=False,
            )

        # Match labels (OKS + IoU)
        if self.show_match_labels and matches:
            _draw_match_labels(
                self.ax, gt_insts, pred_insts, gt_to_match, scale=scale)

        # num_valid_gt: GT instances that count as matching targets (non-crowd,
        # have keypoints, positive bbox area, not all-zero coords).
        num_valid_gt = _count_valid_gt(gt_insts)

        # Title
        toggles = [
            f'pred={"ON" if self.show_pred else "off"}',
            f'gt={"ON" if self.show_gt else "off"}',
            f'bbox={"ON" if self.show_bbox else "off"}',
            f'vis={"ON" if self.show_visibility else "off"}',
            f'labels={"ON" if self.show_match_labels else "off"}',
            f'crowd={"ON" if self.show_iscrowd else "off"}',
        ]
        metrics_str = _format_metrics(frame.get('metrics', {}))
        # Append live num_valid_gt (may differ from stored num_gt which counts
        # all GT annotations including crowd / zero-keypoint instances).
        metrics_str += f'  num_valid_gt={num_valid_gt}'
        title = (
            f'[{self.index + 1}/{len(self.frames)}] '
            f'img_id={frame["img_id"]}  {frame["img_path"]}\n'
            f'{metrics_str}\n'
            f'{"  ".join(toggles)}  |  '
            f'arrows:prev/next  p/g/b/v/m/c:toggle  q:quit'
        )
        self.ax.set_title(title, fontsize=9)
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key in ('right', 'down'):
            if self.frames:
                self.index = (self.index + 1) % len(self.frames)
                self._render()
        elif event.key in ('left', 'up'):
            if self.frames:
                self.index = (self.index - 1) % len(self.frames)
                self._render()
        elif event.key == 'p':
            self.show_pred = not self.show_pred
            self._render()
        elif event.key == 'g':
            self.show_gt = not self.show_gt
            self._render()
        elif event.key == 'b':
            self.show_bbox = not self.show_bbox
            self._render()
        elif event.key == 'v':
            self.show_visibility = not self.show_visibility
            self._render()
        elif event.key == 'm':
            self.show_match_labels = not self.show_match_labels
            self._render()
        elif event.key == 'c':
            self.show_iscrowd = not self.show_iscrowd
            self._render()
        elif event.key == 'q':
            plt.close(self.fig)

    def show(self) -> None:
        plt.show()


def main():
    args = parse_args()
    pred_dir = osp.abspath(args.pred_dir)
    manifest, frames = load_bundle(pred_dir)
    frames = _filter_frames(frames, manifest, args)

    if args.badcase and not frames:
        print('No badcase frames matched the filter criteria.')
        return

    print(f'Loaded {len(frames)} frames from {pred_dir}')
    browser = PredictionBrowser(frames, manifest, args)
    browser.show()


if __name__ == '__main__':
    main()
