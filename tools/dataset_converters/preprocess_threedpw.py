# Copyright (c) OpenMMLab. All rights reserved.
"""Convert 3DPW sequence PKL files to COCO-format JSON annotations.

The 3DPW ``poses2d`` field stores 2D keypoints in OpenPose-18 order:
  0:nose  1:neck  2:r_shoulder  3:r_elbow  4:r_wrist
  5:l_shoulder  6:l_elbow  7:l_wrist  8:r_hip  9:r_knee  10:r_ankle
  11:l_hip  12:l_knee  13:l_ankle  14:r_eye  15:l_eye  16:r_ear  17:l_ear

These are remapped to COCO-17 order (OpenPose neck has no COCO equivalent
and is dropped):
  0:nose  1:l_eye  2:r_eye  3:l_ear  4:r_ear
  5:l_shoulder  6:r_shoulder  7:l_elbow  8:r_elbow  9:l_wrist  10:r_wrist
  11:l_hip  12:r_hip  13:l_knee  14:r_knee  15:l_ankle  16:r_ankle

Usage::

    python tools/dataset_converters/preprocess_threedpw.py \\
        --data-root data/3dpw \\
        --out-dir data/3dpw/annotations
"""

import argparse
import os
import os.path as osp
import pickle
from collections import defaultdict

import cv2
import numpy as np

try:
    import mmengine
    USE_MMENGINE = True
except ImportError:
    import json
    USE_MMENGINE = False

# Maps (openpose_18_idx, coco_17_idx); OpenPose joint 1 (neck) is dropped.
_OPENPOSE18_TO_COCO17 = [
    (0, 0),    # nose
    (15, 1),   # left_eye
    (14, 2),   # right_eye
    (17, 3),   # left_ear
    (16, 4),   # right_ear
    (5, 5),    # left_shoulder
    (2, 6),    # right_shoulder
    (6, 7),    # left_elbow
    (3, 8),    # right_elbow
    (7, 9),    # left_wrist
    (4, 10),   # right_wrist
    (11, 11),  # left_hip
    (8, 12),   # right_hip
    (12, 13),  # left_knee
    (9, 14),   # right_knee
    (13, 15),  # left_ankle
    (10, 16),  # right_ankle
]

_COCO_CATEGORIES = [{
    'supercategory': 'person',
    'id': 1,
    'name': 'person',
    'keypoints': [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
        'right_knee', 'left_ankle', 'right_ankle'
    ],
    'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                 [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                 [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
}]

_IMG_DIM_CACHE: dict = {}


def _get_img_dims(img_path):
    """Read (height, width) of an image, with caching."""
    if img_path not in _IMG_DIM_CACHE:
        img = cv2.imread(img_path)
        if img is None:
            return None, None
        _IMG_DIM_CACHE[img_path] = (img.shape[0], img.shape[1])
    return _IMG_DIM_CACHE[img_path]


def _openpose18_to_coco17(frame_kpts):
    """Convert one frame's OpenPose-18 keypoints to COCO-17.

    Args:
        frame_kpts (np.ndarray): shape (3, 18) -- rows are x, y, confidence.

    Returns:
        np.ndarray: shape (17, 3) -- columns are x, y, visibility (0 or 2).
    """
    coco = np.zeros((17, 3), dtype=np.float32)
    for op_idx, coco_idx in _OPENPOSE18_TO_COCO17:
        x = float(frame_kpts[0, op_idx])
        y = float(frame_kpts[1, op_idx])
        conf = float(frame_kpts[2, op_idx])
        vis = 2 if conf > 0.0 else 0
        coco[coco_idx] = [x, y, vis]
    return coco


def _compute_bbox(kpts_coco17, img_h, img_w, pad=0.2):
    """Compute padded bounding box from visible COCO-17 keypoints.

    Args:
        kpts_coco17 (np.ndarray): shape (17, 3).
        img_h (int): image height for clamping.
        img_w (int): image width for clamping.
        pad (float): fractional padding added on each side.

    Returns:
        list | None: [x, y, w, h] in COCO format, or None if no visible joint.
    """
    visible = kpts_coco17[kpts_coco17[:, 2] > 0]
    if len(visible) == 0:
        return None
    x_min, y_min = visible[:, 0].min(), visible[:, 1].min()
    x_max, y_max = visible[:, 0].max(), visible[:, 1].max()
    bw = x_max - x_min
    bh = y_max - y_min
    x_min -= bw * pad
    y_min -= bh * pad
    x_max += bw * pad
    y_max += bh * pad
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    x_max = min(float(img_w), x_max)
    y_max = min(float(img_h), y_max)
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None
    return [x_min, y_min, w, h]


def process_split(data_root, split, out_dir):
    """Convert one data split to a COCO JSON file."""
    seq_dir = osp.join(data_root, 'sequenceFiles', split)
    img_root = osp.join(data_root, 'imageFiles')

    pkl_files = sorted(
        f for f in os.listdir(seq_dir) if f.endswith('.pkl'))

    images = []
    annotations = []
    img_id_counter = 0
    ann_id_counter = 0
    track_id_counter = 0

    # Map (seq_name, frame_idx) -> image_id to deduplicate image entries when
    # multiple actors share the same frame.
    img_key_to_id = {}
    # Map (seq_name, actor_idx) -> track_id; each actor in a sequence is a
    # single persistent track across all frames.
    track_key_to_id = {}

    for pkl_name in pkl_files:
        pkl_path = osp.join(seq_dir, pkl_name)
        with open(pkl_path, 'rb') as f:
            seq = pickle.load(f, encoding='latin1')

        seq_name = seq['sequence']
        poses2d_list = seq['poses2d']   # list of (n_frames, 3, 18) per actor
        n_actors = len(poses2d_list)

        # campose_valid: (n_actors, n_frames) or similar
        campose_valid = np.array(seq['campose_valid'])  # (n_actors, n_frames)

        n_frames = poses2d_list[0].shape[0]

        for actor_idx in range(n_actors):
            actor_kpts = poses2d_list[actor_idx]  # (n_frames, 3, 18)

            # Assign a globally unique track ID for this (sequence, actor).
            track_key = (seq_name, actor_idx)
            if track_key not in track_key_to_id:
                track_id_counter += 1
                track_key_to_id[track_key] = track_id_counter
            track_id = track_key_to_id[track_key]

            for frame_idx in range(n_frames):

                # Only use frames with valid camera alignment
                if not campose_valid[actor_idx, frame_idx]:
                    continue

                frame_kpts_op18 = actor_kpts[frame_idx]  # (3, 18)
                kpts_coco17 = _openpose18_to_coco17(frame_kpts_op18)

                # Skip if no visible keypoints
                if kpts_coco17[:, 2].sum() == 0:
                    continue

                # Construct image file path
                img_filename = osp.join(
                    seq_name, f'image_{frame_idx:05d}.jpg')
                abs_img_path = osp.join(img_root, img_filename)

                if not osp.exists(abs_img_path):
                    continue

                img_h, img_w = _get_img_dims(abs_img_path)
                if img_h is None:
                    continue

                bbox = _compute_bbox(kpts_coco17, img_h, img_w)
                if bbox is None:
                    continue

                # Register image entry (shared across actors in same frame)
                img_key = (seq_name, frame_idx)
                if img_key not in img_key_to_id:
                    img_id_counter += 1
                    img_entry = {
                        'id': img_id_counter,
                        'file_name': img_filename,
                        'width': img_w,
                        'height': img_h,
                        'nframes': n_frames,
                        'frame_id': frame_idx,
                        'seq_name': seq_name,
                    }
                    images.append(img_entry)
                    img_key_to_id[img_key] = img_id_counter

                image_id = img_key_to_id[img_key]

                # Flatten keypoints: [x, y, v, x, y, v, ...]
                kpts_flat = kpts_coco17.reshape(-1).tolist()
                num_visible = int((kpts_coco17[:, 2] > 0).sum())

                ann_id_counter += 1
                ann_entry = {
                    'id': ann_id_counter,
                    'image_id': image_id,
                    'category_id': 1,
                    'iscrowd': 0,
                    'track_id': track_id,
                    'bbox': [round(v, 2) for v in bbox],
                    'area': round(bbox[2] * bbox[3], 2),
                    'num_keypoints': num_visible,
                    'keypoints': [round(v, 4) for v in kpts_flat],
                }
                annotations.append(ann_entry)

    coco_json = {
        'info': {
            'description': f'3DPW {split} split in COCO format',
            'url': 'https://virtualhumans.mpi-inf.mpg.de/3DPW/',
            'version': '1.0',
            'year': 2018,
            'contributor': 'MPI-INF',
        },
        'categories': _COCO_CATEGORIES,
        'images': images,
        'annotations': annotations,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f'threedpw_{split}.json')
    if USE_MMENGINE:
        mmengine.dump(coco_json, out_path)
    else:
        with open(out_path, 'w') as f:
            import json
            json.dump(coco_json, f)

    n_tracks = len(track_key_to_id)
    print(f'[{split}] {len(images):6d} images, '
          f'{len(annotations):7d} annotations, '
          f'{n_tracks:4d} tracks -> {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert 3DPW PKL sequences to COCO-format JSON.')
    parser.add_argument(
        '--data-root',
        default='data/3dpw',
        help='Root directory of the 3DPW dataset.')
    parser.add_argument(
        '--out-dir',
        default='data/3dpw/annotations',
        help='Output directory for JSON annotation files.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'validation', 'test'],
        help='Dataset splits to process.')
    args = parser.parse_args()

    for split in args.splits:
        process_split(args.data_root, split, args.out_dir)


if __name__ == '__main__':
    main()
