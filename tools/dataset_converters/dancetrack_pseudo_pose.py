#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Pre-annotate DanceTrack with PCT-huge keypoints in PoseTrack21 JSON style.

Uses ground-truth MOT boxes (``gt/gt.txt``) and a top-down pose model to write
merged COCO-style JSON with:

* original DanceTrack bboxes
* 1:1 pose tracks (``orig_track_id``) remapped to globally unique ``track_id``
* 19-keypoint union: COCO-17 plus synthesized ``head_bottom`` / ``head_top``

Train/val only (test has no boxes).  Per-sequence shards under ``--work-dir``
make the run resumable.

Example::

    python tools/dataset_converters/dancetrack_pseudo_pose.py \\
        --device cuda:0 --resume

    python tools/dataset_converters/dancetrack_pseudo_pose.py \\
        --seq dancetrack0004 --max-frames 8 --render-seq dancetrack0004
"""

from __future__ import annotations

import argparse
import colorsys
import configparser
import json
import os
import os.path as osp
import re
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Joint indices in the 19-keypoint union (COCO-17 + PoseTrack head).
_NOSE = 0
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_HEAD_BOTTOM = 17
_HEAD_TOP = 18
_NUM_COCO = 17
_NUM_UNION = 19
# head_top = head_bottom + scale * (nose - head_bottom).  1.5 sits
# beyond the nose without the 2.0x overshoot of a pure reflection.
_HEAD_TOP_EXTEND = 1.5

_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
    'right_knee', 'left_ankle', 'right_ankle', 'head_bottom', 'head_top',
]

# 1-based COCO skeleton plus PoseTrack head links.
_SKELETON_1BASED = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
    [1, 18], [1, 19], [18, 19], [18, 6], [18, 7],
]

_CATEGORIES = [{
    'supercategory': 'person',
    'id': 1,
    'name': 'person',
    'keypoints': _KEYPOINT_NAMES,
    'skeleton': _SKELETON_1BASED,
}]

# Physical folders on disk for each logical split.  The HF zip layout uses
# train1/train2; the download script may flatten those into train/.
_SPLIT_FOLDERS = {
    'train': ('train1', 'train2', 'train'),
    'val': ('val',),
}

_SEQ_NUM_RE = re.compile(r'(\d+)$')

_DEFAULT_CONFIG = (
    'configs/body_2d_keypoint/topdown_regression/coco/'
    'td-reg_pct-huge_coco-256x256.py')
_DEFAULT_CHECKPOINT = 'external/PCT/weights/pct/swin_huge.pth'


def _now() -> str:
    return time.strftime('%H:%M:%S')


def _log(msg: str) -> None:
    print(f'[{_now()}] {msg}', flush=True)


def parse_seqinfo(seq_dir: str) -> dict:
    """Read ``seqinfo.ini`` (frame rate, length, resolution)."""
    path = osp.join(seq_dir, 'seqinfo.ini')
    cfg = configparser.ConfigParser()
    if not cfg.read(path):
        raise FileNotFoundError(f'Missing seqinfo.ini in {seq_dir}')
    sec = cfg['Sequence']
    return dict(
        name=sec.get('name', osp.basename(seq_dir)),
        im_dir=sec.get('imDir', 'img1'),
        frame_rate=int(sec.get('frameRate', '20')),
        seq_length=int(sec.get('seqLength', '0')),
        im_width=int(sec.get('imWidth', '0')),
        im_height=int(sec.get('imHeight', '0')),
        im_ext=sec.get('imExt', '.jpg'),
    )


def parse_gt_txt(gt_path: str) -> Dict[int, List[dict]]:
    """Parse MOT ``gt.txt`` into ``frame -> list of {orig_track_id, bbox}``."""
    by_frame: Dict[int, List[dict]] = defaultdict(list)
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.replace(' ', '').split(',')
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            orig_id = int(float(parts[1]))
            x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]),
                          float(parts[5]))
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            if conf == 0:
                continue
            by_frame[frame].append(
                dict(orig_track_id=orig_id, bbox=[x, y, w, h]))
    return by_frame


def discover_sequences(
    data_root: str,
    split: str,
) -> List[Tuple[str, str, str]]:
    """Return ``[(seq_name, rel_dir, abs_dir), ...]`` for one split."""
    seen = set()
    seqs: List[Tuple[str, str, str]] = []
    for folder in _SPLIT_FOLDERS[split]:
        parent = osp.join(data_root, folder)
        if not osp.isdir(parent):
            continue
        for name in sorted(os.listdir(parent)):
            if name in seen:
                continue
            seq_dir = osp.join(parent, name)
            if not osp.isdir(seq_dir):
                continue
            if not osp.isdir(osp.join(seq_dir, 'img1')):
                continue
            if not osp.isfile(osp.join(seq_dir, 'gt', 'gt.txt')):
                continue
            seen.add(name)
            rel = osp.relpath(seq_dir, data_root).replace('\\', '/')
            seqs.append((name, rel, seq_dir))
    return seqs


def seq_numeric_id(seq_name: str) -> int:
    match = _SEQ_NUM_RE.search(seq_name)
    if not match:
        raise ValueError(f'Cannot parse sequence number from {seq_name!r}')
    return int(match.group(1))


def image_id_for(seq_name: str, frame_id: int) -> int:
    return seq_numeric_id(seq_name) * 1_000_000 + int(frame_id)


def frame_image_path(seq_dir: str, im_dir: str, frame_id: int,
                     im_ext: str) -> str:
    return osp.join(seq_dir, im_dir, f'{frame_id:08d}{im_ext}')


def clip_bbox_xywh(bbox: Sequence[float], width: int,
                   height: int) -> Optional[List[float]]:
    """Clip an xywh box to the image; return None if it has no area."""
    x, y, w, h = bbox
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(width), float(x) + float(w))
    y2 = min(float(height), float(y) + float(h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def _extend_head_top(
    nose_x: float,
    nose_y: float,
    head_bottom_x: float,
    head_bottom_y: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float]:
    """Place head_top along head_bottom→nose, scaled by ``_HEAD_TOP_EXTEND``."""
    hx = head_bottom_x + _HEAD_TOP_EXTEND * (nose_x - head_bottom_x)
    hy = head_bottom_y + _HEAD_TOP_EXTEND * (nose_y - head_bottom_y)
    hx = float(np.clip(hx, 0, img_w - 1))
    hy = float(np.clip(hy, 0, img_h - 1))
    return hx, hy


def recompute_head_tops(data: dict) -> int:
    """Rewrite labeled ``head_top`` joints in a COCO-style dict in place.

    Leaves unlabeled head_top joints (visibility 0) untouched.  Returns the
    number of annotations whose head_top coordinates changed.
    """
    img_size = {
        int(img['id']): (int(img['width']), int(img['height']))
        for img in data.get('images', [])
    }
    n_updated = 0
    nose_x, nose_y, nose_v = 0, 1, 2
    hb_x = _HEAD_BOTTOM * 3
    hb_y = hb_x + 1
    hb_v = hb_x + 2
    ht_x = _HEAD_TOP * 3
    ht_y = ht_x + 1
    ht_v = ht_x + 2
    for ann in data.get('annotations', []):
        kpts = ann.get('keypoints')
        if not kpts or len(kpts) < _NUM_UNION * 3:
            continue
        if kpts[ht_v] < 1.0 or kpts[nose_v] < 1.0 or kpts[hb_v] < 1.0:
            continue
        width, height = img_size.get(int(ann['image_id']), (0, 0))
        if width <= 0 or height <= 0:
            continue
        hx, hy = _extend_head_top(
            float(kpts[nose_x]), float(kpts[nose_y]),
            float(kpts[hb_x]), float(kpts[hb_y]),
            width, height)
        hx, hy = round(hx, 4), round(hy, 4)
        if hx != kpts[ht_x] or hy != kpts[ht_y]:
            kpts[ht_x] = hx
            kpts[ht_y] = hy
            n_updated += 1
    return n_updated


def coco17_to_union19(
    kpts_xy: np.ndarray,
    scores: np.ndarray,
    kpt_thr: float,
    img_w: int,
    img_h: int,
) -> Tuple[List[float], int]:
    """Map PCT COCO-17 ``(x,y)`` + scores onto the 19-joint union.

    Visibility is COCO-style: 2 if score >= ``kpt_thr``, else 0 with xy
    zeroed.  ``head_bottom`` is the shoulder midpoint; ``head_top`` extends
    the head_bottom→nose line by ``_HEAD_TOP_EXTEND``.
    """
    out = np.zeros((_NUM_UNION, 3), dtype=np.float64)
    xy = np.asarray(kpts_xy, dtype=np.float64).reshape(-1, 2)
    sc = np.asarray(scores, dtype=np.float64).reshape(-1)

    for i in range(min(_NUM_COCO, len(xy))):
        if sc[i] >= kpt_thr:
            out[i, 0] = xy[i, 0]
            out[i, 1] = xy[i, 1]
            out[i, 2] = 2.0

    if (sc[_LEFT_SHOULDER] >= kpt_thr and sc[_RIGHT_SHOULDER] >= kpt_thr):
        out[_HEAD_BOTTOM, 0] = 0.5 * (xy[_LEFT_SHOULDER, 0] +
                                      xy[_RIGHT_SHOULDER, 0])
        out[_HEAD_BOTTOM, 1] = 0.5 * (xy[_LEFT_SHOULDER, 1] +
                                      xy[_RIGHT_SHOULDER, 1])
        out[_HEAD_BOTTOM, 2] = 2.0

    if out[_HEAD_BOTTOM, 2] >= 2.0 and sc[_NOSE] >= kpt_thr:
        hx, hy = _extend_head_top(
            xy[_NOSE, 0], xy[_NOSE, 1],
            out[_HEAD_BOTTOM, 0], out[_HEAD_BOTTOM, 1],
            img_w, img_h)
        out[_HEAD_TOP, 0] = hx
        out[_HEAD_TOP, 1] = hy
        out[_HEAD_TOP, 2] = 2.0

    num_kpts = int((out[:, 2] >= 1.0).sum())
    flat = [round(float(v), 4) for v in out.reshape(-1)]
    return flat, num_kpts


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pose_pipeline(model):
    """Build (and cache) the model's test pipeline once per process."""
    pipeline = getattr(model, '_dancetrack_pipeline', None)
    if pipeline is not None:
        return pipeline
    from mmengine.dataset import Compose
    from mmengine.registry import init_default_scope
    scope = model.cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    model._dancetrack_pipeline = pipeline
    return pipeline


def infer_frame(
    model,
    image: np.ndarray,
    boxes_xywh: List[List[float]],
    kpt_thr: float,
    img_w: int,
    img_h: int,
    chunk_size: int = 32,
) -> List[Tuple[List[float], int]]:
    """Run top-down pose on one frame; return union keypoints per box."""
    import torch
    from mmengine.dataset import pseudo_collate
    from mmpose.structures.bbox import bbox_xywh2xyxy

    if not boxes_xywh:
        return []

    pipeline = _pose_pipeline(model)
    results_union: List[Tuple[List[float], int]] = []
    all_xyxy = bbox_xywh2xyxy(
        np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4))

    for start in range(0, len(all_xyxy), chunk_size):
        chunk = all_xyxy[start:start + chunk_size]
        data_list = []
        for bbox in chunk:
            data_info = dict(img=image.copy())
            data_info['bbox'] = bbox[None]
            data_info['bbox_score'] = np.ones(1, dtype=np.float32)
            data_info.update(model.dataset_meta)
            data_list.append(pipeline(data_info))
        batch = pseudo_collate(data_list)
        with torch.no_grad():
            pose_results = model.test_step(batch)
        for ds in pose_results:
            kpts = _to_numpy(ds.pred_instances.keypoints)[0]
            scores = _to_numpy(ds.pred_instances.keypoint_scores)[0]
            results_union.append(
                coco17_to_union19(kpts, scores, kpt_thr, img_w, img_h))
    return results_union


def shard_path(work_dir: str, split: str, seq_name: str) -> str:
    return osp.join(work_dir, 'shards', f'{split}_{seq_name}.json')


def dump_json(obj: dict, path: str) -> None:
    """Write JSON atomically. Falls back to a direct write if replace fails
    (observed on some networked filesystems for large files)."""
    import shutil

    path = osp.abspath(path)
    os.makedirs(osp.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(tmp, path)
    except OSError:
        shutil.copyfile(tmp, path)
        os.remove(tmp)


def process_sequence(
    model,
    data_root: str,
    split: str,
    seq_name: str,
    rel_dir: str,
    seq_dir: str,
    work_dir: str,
    kpt_thr: float,
    max_frames: Optional[int],
    resume: bool,
) -> str:
    """Run PCT on one sequence and write a shard JSON.  Returns shard path."""
    out_path = shard_path(work_dir, split, seq_name)
    if resume and osp.isfile(out_path):
        _log(f'skip {split}/{seq_name} (shard exists)')
        return out_path

    info = parse_seqinfo(seq_dir)
    gt = parse_gt_txt(osp.join(seq_dir, 'gt', 'gt.txt'))
    img_w, img_h = info['im_width'], info['im_height']
    nframes = info['seq_length']
    im_dir, im_ext = info['im_dir'], info['im_ext']
    vid_id = f'{seq_numeric_id(seq_name):04d}'

    if max_frames is not None:
        nframes = min(nframes, int(max_frames))

    images: List[dict] = []
    annotations: List[dict] = []
    t0 = time.time()
    n_boxes = 0

    for frame_id in range(1, nframes + 1):
        img_file = frame_image_path(seq_dir, im_dir, frame_id, im_ext)
        if not osp.isfile(img_file):
            continue
        rel_file = osp.relpath(img_file, data_root).replace('\\', '/')
        persons = gt.get(frame_id, [])
        img_id = image_id_for(seq_name, frame_id)
        images.append({
            'id': img_id,
            'file_name': rel_file,
            'width': img_w,
            'height': img_h,
            'nframes': info['seq_length'],
            'frame_id': frame_id,
            'seq_name': seq_name,
            'vid_id': vid_id,
            'is_labeled': True,
            'good_frame': True,
        })

        if not persons:
            continue

        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        if image is None:
            _log(f'warning: failed to read {img_file}')
            continue

        infer_boxes: List[List[float]] = []
        kept_persons: List[dict] = []
        for person in persons:
            clipped = clip_bbox_xywh(person['bbox'], img_w, img_h)
            if clipped is None:
                annotations.append({
                    'image_id': img_id,
                    'category_id': 1,
                    'iscrowd': 0,
                    'orig_track_id': int(person['orig_track_id']),
                    'bbox': [round(float(v), 4) for v in person['bbox']],
                    'area': round(float(person['bbox'][2] * person['bbox'][3]),
                                  4),
                    'num_keypoints': 0,
                    'keypoints': [0.0] * (_NUM_UNION * 3),
                })
                continue
            infer_boxes.append(clipped)
            kept_persons.append(person)

        unions = infer_frame(model, image, infer_boxes, kpt_thr, img_w, img_h)
        for person, (kpts, num_kpts) in zip(kept_persons, unions):
            bbox = [round(float(v), 4) for v in person['bbox']]
            annotations.append({
                'image_id': img_id,
                'category_id': 1,
                'iscrowd': 0,
                'orig_track_id': int(person['orig_track_id']),
                'bbox': bbox,
                'area': round(float(bbox[2] * bbox[3]), 4),
                'num_keypoints': num_kpts,
                'keypoints': kpts,
            })
            n_boxes += 1

        if frame_id % 50 == 0 or frame_id == nframes:
            elapsed = max(time.time() - t0, 1e-6)
            _log(f'{split}/{seq_name} frame {frame_id}/{nframes} '
                 f'({frame_id / elapsed:.1f} fps, {n_boxes} boxes)')

    dump_json(
        {
            'split': split,
            'seq_name': seq_name,
            'rel_dir': rel_dir,
            'frame_rate': info['frame_rate'],
            'images': images,
            'annotations': annotations,
        },
        out_path,
    )
    _log(f'wrote {out_path} ({len(images)} images, {len(annotations)} anns)')
    return out_path


def merge_shards(
    work_dir: str,
    split: str,
    seq_names: Sequence[str],
    out_path: str,
) -> str:
    """Merge per-sequence shards into one PoseTrack21-style JSON."""
    images: List[dict] = []
    annotations: List[dict] = []
    track_key_to_id: Dict[Tuple[str, int], int] = {}
    next_track = 1
    next_ann = 1

    for seq_name in seq_names:
        path = shard_path(work_dir, split, seq_name)
        if not osp.isfile(path):
            raise FileNotFoundError(f'Missing shard for {split}/{seq_name}: '
                                    f'{path}')
        with open(path, 'r') as f:
            shard = json.load(f)
        images.extend(shard['images'])
        for ann in shard['annotations']:
            key = (seq_name, int(ann['orig_track_id']))
            if key not in track_key_to_id:
                track_key_to_id[key] = next_track
                next_track += 1
            out_ann = dict(ann)
            out_ann['id'] = next_ann
            out_ann['track_id'] = track_key_to_id[key]
            next_ann += 1
            annotations.append(out_ann)

    coco_json = {
        'info': {
            'description': (
                f'DanceTrack {split} split with PCT-huge pose pre-annotations '
                '(COCO-17 + PoseTrack head_bottom/head_top)'),
            'url': 'https://github.com/DanceTrack/DanceTrack',
            'version': '1.0',
            'year': 2022,
            'contributor': 'PCT-huge pseudo labels on DanceTrack GT boxes',
        },
        'categories': _CATEGORIES,
        'images': images,
        'annotations': annotations,
    }
    dump_json(coco_json, out_path)
    _log(f'{split}: {len(images)} images, {len(annotations)} anns, '
         f'{len(track_key_to_id)} tracks -> {out_path}')
    return out_path


def track_color_bgr(orig_track_id: int) -> Tuple[int, int, int]:
    """Stable saturated colour from a track id (OpenCV BGR)."""
    hue = (int(orig_track_id) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.98)
    return int(b * 255), int(g * 255), int(r * 255)


def _draw_label(image: np.ndarray, text: str, x: int, y: int,
                color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.8, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    x1 = max(0, int(x))
    y2 = max(th + 2 * pad, int(y))
    y1 = y2 - th - 2 * pad
    if y1 < 0:
        y1 = 0
        y2 = th + 2 * pad
    x2 = min(image.shape[1] - 1, x1 + tw + 2 * pad)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
    cv2.putText(
        image,
        text,
        (x1 + pad, y2 - pad - baseline // 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _kpts_array(flat: Sequence[float]) -> np.ndarray:
    arr = np.asarray(flat, dtype=np.float32).reshape(-1, 3)
    if arr.shape[0] < _NUM_UNION:
        pad = np.zeros((_NUM_UNION - arr.shape[0], 3), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr


def draw_instances(image: np.ndarray, anns: Iterable[dict],
                   kpt_thr_v: float = 1.0) -> np.ndarray:
    """Draw bbox, skeleton, and a readable track-id label per person."""
    vis = image.copy()
    for ann in anns:
        orig_id = int(ann.get('orig_track_id', ann.get('track_id', 0)))
        track_id = ann.get('track_id')
        color = track_color_bgr(orig_id)
        x, y, w, h = ann['bbox']
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        kpts = _kpts_array(ann.get('keypoints', []))
        for a, b in _SKELETON_1BASED:
            ka, kb = kpts[a - 1], kpts[b - 1]
            if ka[2] < kpt_thr_v or kb[2] < kpt_thr_v:
                continue
            pa = (int(round(ka[0])), int(round(ka[1])))
            pb = (int(round(kb[0])), int(round(kb[1])))
            cv2.line(vis, pa, pb, color, 2, cv2.LINE_AA)
        for kp in kpts:
            if kp[2] < kpt_thr_v:
                continue
            cv2.circle(
                vis, (int(round(kp[0])), int(round(kp[1]))), 3, color, -1,
                cv2.LINE_AA)

        if track_id is not None and int(track_id) != orig_id:
            label = f'ID {orig_id} (#{int(track_id)})'
        else:
            label = f'ID {orig_id}'
        _draw_label(vis, label, x1, y1, color)
    return vis


def load_seq_from_json(json_path: str, seq_name: str) -> Tuple[List[dict],
                                                              List[dict]]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    images = [img for img in data['images'] if img.get('seq_name') == seq_name]
    img_ids = {img['id'] for img in images}
    anns = [ann for ann in data['annotations'] if ann['image_id'] in img_ids]
    images.sort(key=lambda img: img['frame_id'])
    return images, anns


def find_seq_annotations(
    seq_name: str,
    work_dir: str,
    out_dir: str,
    splits: Sequence[str],
) -> Tuple[List[dict], List[dict], int]:
    """Load images/anns for ``seq_name`` from merged JSON or a shard."""
    for split in splits:
        merged = osp.join(out_dir, f'dancetrack_pose_{split}.json')
        if osp.isfile(merged):
            images, anns = load_seq_from_json(merged, seq_name)
            if images:
                with open(merged, 'r') as f:
                    # frame_rate is not in merged file; fall back later
                    pass
                return images, anns, 20
        shard = shard_path(work_dir, split, seq_name)
        if osp.isfile(shard):
            with open(shard, 'r') as f:
                shard_data = json.load(f)
            return (shard_data['images'], shard_data['annotations'],
                    int(shard_data.get('frame_rate', 20)))
    raise FileNotFoundError(
        f'No annotations found for {seq_name} in {out_dir} or {work_dir}/shards'
    )


def render_sequence(
    data_root: str,
    seq_name: str,
    images: List[dict],
    annotations: List[dict],
    out_mp4: str,
    frame_rate: int,
    max_frames: Optional[int] = None,
) -> str:
    """Write an mp4 with skeleton, bbox, and track-id labels."""
    if not images:
        raise ValueError(f'No frames to render for {seq_name}')
    images = sorted(images, key=lambda img: img['frame_id'])
    if max_frames is not None:
        images = images[:int(max_frames)]

    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann['image_id']].append(ann)

    first = cv2.imread(osp.join(data_root, images[0]['file_name']))
    if first is None:
        raise FileNotFoundError(images[0]['file_name'])
    height, width = first.shape[:2]
    os.makedirs(osp.dirname(out_mp4) or '.', exist_ok=True)
    writer = cv2.VideoWriter(
        out_mp4,
        cv2.VideoWriter_fourcc(*'mp4v'),
        float(frame_rate),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open VideoWriter for {out_mp4}')

    for i, img_info in enumerate(images):
        path = osp.join(data_root, img_info['file_name'])
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            _log(f'warning: missing frame {path}')
            continue
        vis = draw_instances(frame, anns_by_image.get(img_info['id'], []))
        header = f'{seq_name}  frame {img_info["frame_id"]}'
        cv2.putText(
            vis,
            header,
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(vis)
        if (i + 1) % 100 == 0:
            _log(f'render {seq_name} {i + 1}/{len(images)}')
    writer.release()
    _log(f'wrote video {out_mp4}')
    return out_mp4


def build_model(config: str, checkpoint: str, device: str):
    from mmpose.apis import init_model
    _log(f'loading pose model {checkpoint} on {device}')
    return init_model(config, checkpoint, device=device)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Pre-annotate DanceTrack with PCT-huge in PoseTrack21 '
                    'JSON format.')
    parser.add_argument('--data-root', default='data/dancetrack')
    parser.add_argument(
        '--out-dir', default='data/dancetrack/annotations',
        help='Directory for merged dancetrack_pose_{split}.json files.')
    parser.add_argument(
        '--work-dir', default='work_dirs/dancetrack_pct_huge')
    parser.add_argument(
        '--splits', nargs='+', default=['train', 'val'],
        choices=['train', 'val'])
    parser.add_argument('--config', default=_DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default=_DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument(
        '--seq', nargs='+', default=None,
        help='Process only these sequence names (e.g. dancetrack0004).')
    parser.add_argument(
        '--max-frames', type=int, default=None,
        help='Cap frames per sequence (smoke tests).')
    parser.add_argument(
        '--resume', action='store_true',
        help='Skip sequences whose shard JSON already exists.')
    parser.add_argument(
        '--merge-only', action='store_true',
        help='Skip inference; merge existing shards.')
    parser.add_argument(
        '--render-seq', default=None,
        help='After inference/merge, render this sequence to mp4.')
    parser.add_argument(
        '--render-only', action='store_true',
        help='Only render --render-seq from existing JSON/shards.')
    parser.add_argument(
        '--render-out', default=None,
        help='Output mp4 path (default: work_dir/<seq>_pose.mp4).')
    parser.add_argument(
        '--recompute-head-top', action='store_true',
        help='Rewrite head_top in existing shards and merged JSON using '
             'head_bottom + 1.5*(nose-head_bottom). Does not run the model.')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    data_root = osp.abspath(args.data_root)
    out_dir = args.out_dir
    work_dir = args.work_dir
    os.makedirs(osp.join(work_dir, 'shards'), exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if args.recompute_head_top:
        shard_dir = osp.join(work_dir, 'shards')
        if osp.isdir(shard_dir):
            for name in sorted(os.listdir(shard_dir)):
                if not name.endswith('.json'):
                    continue
                path = osp.join(shard_dir, name)
                with open(path, 'r') as f:
                    shard = json.load(f)
                n = recompute_head_tops(shard)
                dump_json(shard, path)
                _log(f'recomputed head_top in {path} ({n} anns)')
        for split in args.splits:
            merged = osp.join(out_dir, f'dancetrack_pose_{split}.json')
            if not osp.isfile(merged):
                continue
            with open(merged, 'r') as f:
                data = json.load(f)
            n = recompute_head_tops(data)
            dump_json(data, merged)
            _log(f'recomputed head_top in {merged} ({n} anns)')
        if args.render_seq:
            images, anns, fps = find_seq_annotations(
                args.render_seq, work_dir, out_dir, args.splits)
            for split in args.splits:
                for _, _, seq_dir in discover_sequences(data_root, split):
                    if osp.basename(seq_dir) == args.render_seq:
                        fps = parse_seqinfo(seq_dir)['frame_rate']
                        break
            out_mp4 = args.render_out or osp.join(
                work_dir, f'{args.render_seq}_pose.mp4')
            render_sequence(
                data_root, args.render_seq, images, anns, out_mp4, fps,
                max_frames=args.max_frames)
        return

    if args.render_only:
        if not args.render_seq:
            raise SystemExit('--render-only requires --render-seq')
        images, anns, fps = find_seq_annotations(
            args.render_seq, work_dir, out_dir, args.splits)
        # Prefer seqinfo frame rate when the sequence is still on disk.
        for split in args.splits:
            for _, _, seq_dir in discover_sequences(data_root, split):
                if osp.basename(seq_dir) == args.render_seq:
                    fps = parse_seqinfo(seq_dir)['frame_rate']
                    break
        out_mp4 = args.render_out or osp.join(
            work_dir, f'{args.render_seq}_pose.mp4')
        render_sequence(
            data_root, args.render_seq, images, anns, out_mp4, fps,
            max_frames=args.max_frames)
        return

    seqs_by_split: Dict[str, List[Tuple[str, str, str]]] = {}
    for split in args.splits:
        seqs = discover_sequences(data_root, split)
        if args.seq:
            wanted = set(args.seq)
            seqs = [s for s in seqs if s[0] in wanted]
        if not seqs and not args.seq:
            _log(f'warning: no sequences found for split {split} under '
                 f'{data_root}')
        seqs_by_split[split] = seqs
        _log(f'{split}: {len(seqs)} sequences')

    if not args.merge_only:
        any_seq = any(seqs_by_split.values())
        if not any_seq:
            raise SystemExit(f'No matching sequences under {data_root}')
        model = build_model(args.config, args.checkpoint, args.device)
        for split, seqs in seqs_by_split.items():
            for i, (seq_name, rel_dir, seq_dir) in enumerate(seqs, 1):
                _log(f'[{i}/{len(seqs)}] {split}/{seq_name}')
                process_sequence(
                    model,
                    data_root=data_root,
                    split=split,
                    seq_name=seq_name,
                    rel_dir=rel_dir,
                    seq_dir=seq_dir,
                    work_dir=work_dir,
                    kpt_thr=args.kpt_thr,
                    max_frames=args.max_frames,
                    resume=args.resume,
                )
        del model

    if args.max_frames is None and not args.seq:
        for split, seqs in seqs_by_split.items():
            if not seqs:
                continue
            merge_shards(
                work_dir,
                split,
                [s[0] for s in seqs],
                osp.join(out_dir, f'dancetrack_pose_{split}.json'),
            )
    elif args.merge_only:
        for split, seqs in seqs_by_split.items():
            if not seqs:
                continue
            merge_shards(
                work_dir,
                split,
                [s[0] for s in seqs],
                osp.join(out_dir, f'dancetrack_pose_{split}.json'),
            )

    if args.render_seq:
        images, anns, fps = find_seq_annotations(
            args.render_seq, work_dir, out_dir, args.splits)
        for split in args.splits:
            for name, _, seq_dir in seqs_by_split.get(split, []):
                if name == args.render_seq:
                    fps = parse_seqinfo(seq_dir)['frame_rate']
        out_mp4 = args.render_out or osp.join(
            work_dir, f'{args.render_seq}_pose.mp4')
        render_sequence(
            data_root, args.render_seq, images, anns, out_mp4, fps,
            max_frames=args.max_frames)


if __name__ == '__main__':
    sys.exit(main())
