# Copyright (c) OpenMMLab. All rights reserved.
"""Pre-render benchmark prediction overlays for one sequence into a video.

This is the "movie" counterpart to ``tools/vis_benchmark_predictions.py``:
instead of interactively stepping frame-by-frame, it renders *every* frame of
a chosen EMDB/3DPW sequence with the same overlays (pred/GT keypoints,
bboxes, visibility markers, match labels, iscrowd coloring, track IDs) baked
in up front, then encodes them into an mp4 that plays back smoothly.

The interactive viewer's toggles (``p g b v m c t o``) become CLI flags here
since there is no interactivity once the video is encoded.

Usage::

    # List the sequences available in a prediction bundle
    python tools/vis_benchmark_video.py preds/ --list-sequences

    # Render one EMDB sequence (post-processed source, all overlays on)
    python tools/vis_benchmark_video.py preds/ --sequence 14_outdoor_climb \\
        --source postproc --out vis_results/14_outdoor_climb.mp4

    # Render a 3DPW sequence without track-id / match-label overlays
    python tools/vis_benchmark_video.py preds/ --sequence downtown_arguing_00 \\
        --no-track-ids --no-match-labels

Flag <-> interactive key mapping:
``--no-pred``/``p``, ``--no-gt``/``g``, ``--no-bbox``/``b``,
``--no-visibility``/``v``, ``--no-match-labels``/``m``, ``--no-iscrowd``/``c``,
``--no-track-ids``/``t``, ``--source``/``o``.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
import textwrap
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use('Agg')  # headless rendering – must precede pyplot import

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mmcv  # noqa: E402
import numpy as np  # noqa: E402

# tools/ has no __init__.py, so when this script is invoked directly
# (``python tools/vis_benchmark_video.py``) Python already puts its own
# directory on sys.path[0]; the explicit insert below is a safety net for
# other invocation styles (e.g. via a wrapper that changes cwd first).
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from mmpose.evaluation.functional.benchmark_data import resize_to_ori_shape

from vis_benchmark_predictions import (  # noqa: E402
    _GT_COLOR,
    _PRED_COLOR,
    _build_match_lookups,
    _count_valid_gt,
    _draw_bboxes,
    _draw_kpts_and_skeleton,
    _draw_match_labels,
    _draw_track_ids,
    _parse_instances,
    _resolve_image_path,
    _upscale_image,
    load_bundle,
)


# ---- Argument parsing --------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pre-render benchmark prediction overlays for one '
        'sequence into a video file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'pred_dir',
        help='Directory containing manifest.json and frames.json')
    parser.add_argument(
        '--postproc-dir',
        default=None,
        help='Directory with post-processed predictions (manifest.json + '
        'frames.json). When omitted, the sibling directory '
        '"<pred_dir>__postproc" is used automatically if it exists.')
    parser.add_argument(
        '--data-root',
        default=None,
        help='Root for resolving relative img_path values '
        '(default: manifest data_root)')

    parser.add_argument(
        '--sequence',
        default=None,
        help='Sequence to render: the directory portion of img_path '
        '(e.g. "14_outdoor_climb" or "downtown_arguing_00"). May be an '
        'exact match or a unique substring. Required unless '
        '--list-sequences is given.')
    parser.add_argument(
        '--list-sequences',
        action='store_true',
        help='List all sequences (with frame counts) found in the bundle '
        'and exit without rendering')

    # Toggles – default to the interactive viewer's ON defaults.
    parser.add_argument(
        '--no-pred', dest='show_pred', action='store_false',
        help='Hide predicted keypoints/bboxes (interactive key: p)')
    parser.add_argument(
        '--no-gt', dest='show_gt', action='store_false',
        help='Hide GT keypoints/bboxes (interactive key: g)')
    parser.add_argument(
        '--no-bbox', dest='show_bbox', action='store_false',
        help='Hide bounding boxes (interactive key: b)')
    parser.add_argument(
        '--no-visibility', dest='show_visibility', action='store_false',
        help='Hide occluded/visible GT keypoint distinction '
        '(interactive key: v)')
    parser.add_argument(
        '--no-match-labels', dest='show_match_labels', action='store_false',
        help='Hide OKS/IoU match labels (interactive key: m)')
    parser.add_argument(
        '--no-iscrowd', dest='show_iscrowd', action='store_false',
        help='Hide orange iscrowd GT bbox coloring (interactive key: c)')
    parser.add_argument(
        '--no-track-ids', dest='show_track_ids', action='store_false',
        help='Hide track ID labels on predictions (interactive key: t)')
    parser.set_defaults(
        show_pred=True, show_gt=True, show_bbox=True, show_visibility=True,
        show_match_labels=True, show_iscrowd=True, show_track_ids=True)

    parser.add_argument(
        '--source',
        default='orig',
        choices=['orig', 'postproc'],
        help='Prediction source to render (interactive key: o)')

    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Keypoint score threshold for drawing predictions')
    parser.add_argument(
        '--render-scale',
        type=float,
        default=2.0,
        help='Upscale factor for rendering. Use 1.0 for native resolution.')
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Output video frame rate')
    parser.add_argument(
        '--overlay-info',
        action='store_true',
        help='Burn a HUD (frame index, metrics, num_valid_gt) into the '
        'top-left corner of each frame, mirroring the interactive '
        "viewer's title")
    parser.add_argument(
        '--out',
        default=None,
        help='Output video path (default: '
        'vis_results/<pred_dir_name>/<sequence>.mp4)')
    return parser.parse_args()


# ---- Sequence grouping --------------------------------------------------------

def _sequence_key(img_path: str) -> str:
    """Directory portion of img_path, used to group frames into sequences."""
    return osp.dirname(img_path)


def _group_by_sequence(frames: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for frame in frames:
        key = _sequence_key(frame['img_path'])
        groups.setdefault(key, []).append(frame)
    return groups


def _resolve_sequence(
    groups: Dict[str, List[dict]],
    query: str,
) -> Tuple[str, List[dict]]:
    """Resolve *query* to a single sequence key (exact or unique substring)."""
    if query in groups:
        return query, groups[query]
    matches = [key for key in groups if query in key]
    if len(matches) == 1:
        return matches[0], groups[matches[0]]
    if not matches:
        available = '\n  '.join(sorted(groups.keys()))
        raise SystemExit(
            f'No sequence matches "{query}". Available sequences:\n  '
            f'{available}')
    matches_str = '\n  '.join(sorted(matches))
    raise SystemExit(
        f'Ambiguous sequence "{query}" matches multiple sequences:\n  '
        f'{matches_str}\nPlease be more specific.')


def _print_sequence_list(groups: Dict[str, List[dict]]) -> None:
    print(f'{len(groups)} sequence(s) found:')
    for key in sorted(groups.keys()):
        print(f'  {key}  ({len(groups[key])} frames)')


# ---- Metrics formatting (mirrors vis_benchmark_predictions._format_metrics) --

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


# ---- Renderer ------------------------------------------------------------

class VideoRenderer:
    """Renders one sequence's frames with baked-in overlays to an mp4."""

    def __init__(
        self,
        seq_frames: List[dict],
        manifest: dict,
        args,
        postproc_by_img_id: Optional[Dict[int, dict]] = None,
    ):
        self.seq_frames = seq_frames
        self.manifest = manifest
        self.args = args
        self.data_root = args.data_root or manifest.get('data_root', 'data/')
        self.postproc_by_img_id = postproc_by_img_id
        self.active_source = args.source
        self.render_scale = max(float(args.render_scale), 1.0)

        dmeta = manifest.get('dataset_meta', {})
        raw_links = dmeta.get('skeleton_links', [])
        self.skeleton_links: List = [list(lk) for lk in raw_links]

        # Fixed canvas size for the whole sequence, derived from the first
        # frame after upscaling, so every encoded frame has identical shape.
        first_img = self._load_image(self.seq_frames[0])
        first_img = _upscale_image(first_img, self.render_scale)
        h, w = first_img.shape[:2]
        self.frame_size = (w, h)  # (width, height), OpenCV convention

        dpi = 100.0
        self.fig = plt.figure(
            figsize=(w / dpi, h / dpi), dpi=dpi, frameon=False)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_axis_off()
        self.fig.canvas.draw()

    # ------------------------------------------------------------------

    def _active_pred_frame(self, orig_frame: dict) -> dict:
        if self.active_source == 'postproc' and self.postproc_by_img_id:
            img_id = orig_frame.get('img_id')
            pp_frame = self.postproc_by_img_id.get(img_id)
            if pp_frame is not None:
                return pp_frame
        return orig_frame

    def _load_image(self, frame: dict) -> np.ndarray:
        path = _resolve_image_path(frame, self.data_root)
        if not osp.isfile(path):
            raise FileNotFoundError(
                f'Image not found: {path} (data_root={self.data_root})')
        img = mmcv.imread(path, channel_order='rgb')
        return resize_to_ori_shape(img, frame.get('ori_shape'))

    # ------------------------------------------------------------------

    def render_frame(self, seq_idx: int, orig_frame: dict) -> np.ndarray:
        """Render one frame and return it as an (H, W, 3) BGR uint8 array."""
        pred_frame = self._active_pred_frame(orig_frame)

        img = self._load_image(orig_frame)
        img = _upscale_image(img, self.render_scale)

        pred_json = pred_frame['predictions']['instances']
        gt_json = orig_frame['ground_truth']['instances']
        pred_insts = _parse_instances(pred_json)
        gt_insts = _parse_instances(gt_json)

        matches: List[dict] = pred_frame.get('metrics', {}).get('matches', [])
        gt_matched, pred_matched, gt_to_match = _build_match_lookups(matches)

        ax = self.ax
        ax.clear()
        ax.set_axis_off()
        ax.imshow(img, interpolation='nearest')
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.autoscale(False)

        scale = self.render_scale
        args = self.args

        if args.show_gt:
            if args.show_bbox:
                _draw_bboxes(
                    ax, gt_insts, _GT_COLOR, gt_matched,
                    show_iscrowd=args.show_iscrowd, scale=scale,
                )
            _draw_kpts_and_skeleton(
                ax, gt_insts, _GT_COLOR,
                is_gt=True,
                kpt_thr=args.kpt_thr,
                skeleton_links=self.skeleton_links,
                scale=scale,
                show_visibility=args.show_visibility,
            )

        if args.show_pred:
            if args.show_bbox:
                _draw_bboxes(
                    ax, pred_insts, _PRED_COLOR, pred_matched,
                    show_iscrowd=False, scale=scale,
                )
            _draw_kpts_and_skeleton(
                ax, pred_insts, _PRED_COLOR,
                is_gt=False,
                kpt_thr=args.kpt_thr,
                skeleton_links=self.skeleton_links,
                scale=scale,
                show_visibility=False,
            )
            if args.show_track_ids:
                _draw_track_ids(ax, pred_insts, scale=scale)

        if args.show_match_labels and matches:
            _draw_match_labels(
                ax, gt_insts, pred_insts, gt_to_match, scale=scale)

        if args.overlay_info:
            num_valid_gt = _count_valid_gt(gt_insts)
            metrics_str = _format_metrics(pred_frame.get('metrics', {}))
            metrics_str += f'  num_valid_gt={num_valid_gt}'
            fontsize = max(6, int(7 * scale))
            # Rough monospace-ish char width estimate (points -> pixels via
            # dpi, with a fudge factor) so long metric strings wrap instead
            # of overflowing past the frame edge.
            out_w = self.frame_size[0]
            char_px = fontsize * (self.fig.dpi / 72.0) * 0.6
            wrap_width = max(20, int(out_w * 0.95 / char_px))
            header = (
                f'[{seq_idx + 1}/{len(self.seq_frames)}] '
                f'img_id={orig_frame["img_id"]}  {orig_frame["img_path"]}'
            )
            lines = textwrap.wrap(header, width=wrap_width) + textwrap.wrap(
                metrics_str, width=wrap_width)
            hud = '\n'.join(lines)
            ax.text(
                0.01, 0.99, hud,
                transform=ax.transAxes,
                color='white',
                fontsize=fontsize,
                ha='left', va='top',
                zorder=10,
                bbox=dict(
                    facecolor='black', alpha=0.55,
                    boxstyle='round,pad=0.3',
                    edgecolor='none',
                ),
            )

        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        rgb = buf[:, :, :3]
        bgr = rgb[:, :, ::-1].copy()
        # Guard against per-frame ori_shape variance / dpi rounding so every
        # frame handed to the video writer has an identical, fixed shape.
        out_w, out_h = self.frame_size
        if (bgr.shape[1], bgr.shape[0]) != (out_w, out_h):
            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return bgr

    def close(self) -> None:
        plt.close(self.fig)


# ---- Main ---------------------------------------------------------------

def main():
    args = parse_args()
    pred_dir = osp.abspath(args.pred_dir)
    manifest, frames = load_bundle(pred_dir)
    print(f'Loaded {len(frames)} frames from {pred_dir}')

    postproc_dir = args.postproc_dir
    if postproc_dir is None:
        candidate = pred_dir + '__postproc'
        if osp.isdir(candidate) and osp.isfile(
                osp.join(candidate, 'manifest.json')):
            postproc_dir = candidate
            print(f'Auto-detected post-processed bundle: {postproc_dir}')

    postproc_by_img_id: Optional[Dict[int, dict]] = None
    if postproc_dir is not None:
        postproc_dir = osp.abspath(postproc_dir)
        if not osp.isdir(postproc_dir):
            print(f'Warning: --postproc-dir does not exist: {postproc_dir}')
        else:
            try:
                _, pp_frames = load_bundle(postproc_dir)
                postproc_by_img_id = {f['img_id']: f for f in pp_frames}
                print(
                    f'Loaded {len(pp_frames)} post-processed frames '
                    f'from {postproc_dir}')
            except Exception as exc:
                print(f'Warning: could not load post-processed bundle: {exc}')

    if args.source == 'postproc' and postproc_by_img_id is None:
        raise SystemExit(
            '--source postproc was requested but no post-processed bundle '
            'could be loaded (checked --postproc-dir / '
            f'"{pred_dir}__postproc").')

    groups = _group_by_sequence(frames)

    if args.list_sequences:
        _print_sequence_list(groups)
        return

    if not args.sequence:
        _print_sequence_list(groups)
        raise SystemExit('\nPlease specify --sequence (see list above).')

    seq_key, seq_frames = _resolve_sequence(groups, args.sequence)
    seq_frames = sorted(seq_frames, key=lambda f: f.get('frame_id', 0))
    print(f'Rendering sequence "{seq_key}" ({len(seq_frames)} frames)')

    out_path = args.out
    if out_path is None:
        pred_dir_name = osp.basename(pred_dir.rstrip('/'))
        seq_name = seq_key.replace('/', '_')
        out_path = osp.join('vis_results', pred_dir_name, f'{seq_name}.mp4')
    out_path = osp.abspath(out_path)
    os.makedirs(osp.dirname(out_path), exist_ok=True)

    renderer = VideoRenderer(
        seq_frames, manifest, args, postproc_by_img_id=postproc_by_img_id)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        out_path, fourcc, args.fps, renderer.frame_size)
    if not writer.isOpened():
        renderer.close()
        raise SystemExit(f'Failed to open video writer for: {out_path}')

    try:
        n_frames = len(seq_frames)
        for idx, frame in enumerate(seq_frames):
            bgr = renderer.render_frame(idx, frame)
            writer.write(bgr)
            if (idx + 1) % 50 == 0 or idx + 1 == n_frames:
                print(f'  [{idx + 1}/{n_frames}] frames rendered', end='\r')
        print()
    finally:
        writer.release()
        renderer.close()

    print(f'Wrote {len(seq_frames)} frames to {out_path}')


if __name__ == '__main__':
    main()
