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
``b`` bboxes; ``q`` quit.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import List, Optional

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from mmengine.structures import InstanceData

from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer


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
        help='Skeleton style for visualization')
    parser.add_argument(
        '--render-scale',
        type=float,
        default=2.0,
        help='Upscale factor for rendering (image + overlays). '
        'Use 1.0 for native resolution. Default: 2.0')
    return parser.parse_args()


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


def _instances_from_json(instances: List[dict]) -> InstanceData:
    inst = InstanceData()
    if not instances:
        inst.keypoints = np.zeros((0, 17, 2), dtype=np.float32)
        inst.keypoint_scores = np.zeros((0, 17), dtype=np.float32)
        inst.bboxes = np.zeros((0, 4), dtype=np.float32)
        return inst

    keypoints = []
    scores = []
    bboxes = []
    visibility = []
    for item in instances:
        if 'bbox' in item:
            bbox = np.asarray(item['bbox'], dtype=np.float32).reshape(-1)[:4]
            bboxes.append(bbox)
        if 'keypoints' not in item:
            continue
        kpts = np.asarray(item['keypoints'], dtype=np.float32).reshape(-1, 2)
        keypoints.append(kpts)
        if 'keypoint_scores' in item:
            sc = np.asarray(item['keypoint_scores'], dtype=np.float32).reshape(-1)
        else:
            sc = np.ones(kpts.shape[0], dtype=np.float32)
        scores.append(sc)
        if 'keypoints_visible' in item:
            vis = np.asarray(item['keypoints_visible'], dtype=np.float32).reshape(-1)
            visibility.append(vis)

    if keypoints:
        inst.keypoints = np.stack(keypoints, axis=0)
        inst.keypoint_scores = np.stack(scores, axis=0)
        if visibility:
            inst.keypoints_visible = np.stack(visibility, axis=0)
    if bboxes:
        inst.bboxes = np.stack(bboxes, axis=0)
    return inst


def _build_data_sample(frame: dict, pred_inst: InstanceData,
                       gt_inst: InstanceData) -> PoseDataSample:
    ds = PoseDataSample()
    ds.set_metainfo({
        'img_id': frame['img_id'],
        'ori_shape': tuple(frame['ori_shape']),
    })
    ds.pred_instances = pred_inst
    ds.gt_instances = gt_inst
    return ds


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


def _upscale_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Upscale image for sharper overlay rendering."""
    if scale <= 1.0:
        return img
    h, w = img.shape[:2]
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return mmcv.imresize(img, new_size, interpolation='bicubic')


def _scale_instances(inst: InstanceData, scale: float) -> InstanceData:
    """Scale keypoints and bboxes to match an upscaled canvas."""
    if scale <= 1.0:
        return inst

    scaled = InstanceData()
    if not hasattr(inst, 'keypoints') or len(inst.keypoints) == 0:
        scaled.keypoints = inst.keypoints
        scaled.keypoint_scores = inst.keypoint_scores
        if hasattr(inst, 'bboxes'):
            scaled.bboxes = inst.bboxes
        if hasattr(inst, 'keypoints_visible'):
            scaled.keypoints_visible = inst.keypoints_visible
        return scaled

    scaled.keypoints = inst.keypoints * scale
    scaled.keypoint_scores = inst.keypoint_scores
    if hasattr(inst, 'keypoints_visible'):
        scaled.keypoints_visible = inst.keypoints_visible
    if hasattr(inst, 'bboxes') and inst.bboxes is not None:
        scaled.bboxes = inst.bboxes * scale
    return scaled


def _apply_render_scale(
    img: np.ndarray,
    pred_inst: InstanceData,
    gt_inst: InstanceData,
    scale: float,
) -> tuple:
    """Return upscaled image and coordinate-scaled instances."""
    if scale <= 1.0:
        return img, pred_inst, gt_inst
    return (
        _upscale_image(img, scale),
        _scale_instances(pred_inst, scale),
        _scale_instances(gt_inst, scale),
    )


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
        self.render_scale = max(float(args.render_scale), 1.0)

        self.visualizer = PoseLocalVisualizer(name='benchmark_vis')
        self.visualizer.radius = 3 * self.render_scale
        self.visualizer.line_width = max(1, int(round(self.render_scale)))
        self.visualizer.set_dataset_meta(
            manifest['dataset_meta'],
            skeleton_style=args.skeleton_style)

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.im_artist = None
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

        pred_json = frame['predictions']['instances']
        gt_json = frame['ground_truth']['instances']
        pred_inst = _instances_from_json(pred_json)
        gt_inst = _instances_from_json(gt_json)
        img, pred_inst, gt_inst = _apply_render_scale(
            img, pred_inst, gt_inst, self.render_scale)
        data_sample = _build_data_sample(frame, pred_inst, gt_inst)

        canvas = img.copy()

        if self.show_gt:
            self.visualizer.kpt_color = 'blue'
            self.visualizer.link_color = 'blue'
            self.visualizer.add_datasample(
                'gt',
                canvas,
                data_sample=data_sample,
                draw_gt=True,
                draw_pred=False,
                draw_bbox=self.show_bbox,
                draw_heatmap=False,
                show=False,
                kpt_thr=self.args.kpt_thr,
            )
            canvas = self.visualizer.get_image()

        if self.show_pred:
            self.visualizer.kpt_color = 'red'
            self.visualizer.link_color = 'red'
            self.visualizer.add_datasample(
                'pred',
                canvas,
                data_sample=data_sample,
                draw_gt=False,
                draw_pred=True,
                draw_bbox=self.show_bbox,
                draw_heatmap=False,
                show=False,
                kpt_thr=self.args.kpt_thr,
            )
            canvas = self.visualizer.get_image()

        self.ax.clear()
        self.ax.imshow(canvas, interpolation='nearest')
        self.ax.axis('off')

        toggles = []
        toggles.append(f'pred={"ON" if self.show_pred else "off"}')
        toggles.append(f'gt={"ON" if self.show_gt else "off"}')
        toggles.append(f'bbox={"ON" if self.show_bbox else "off"}')
        metrics_str = _format_metrics(frame.get('metrics', {}))
        title = (
            f'[{self.index + 1}/{len(self.frames)}] '
            f'img_id={frame["img_id"]}  {frame["img_path"]}\n'
            f'{metrics_str}\n'
            f'{"  ".join(toggles)}  |  arrows: prev/next  p/g/b: toggle  q: quit'
        )
        self.ax.set_title(title, fontsize=10)
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
