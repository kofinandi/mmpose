# Copyright (c) OpenMMLab. All rights reserved.
"""Convert EMDB sequence PKL files to COCO-format JSON annotations.

EMDB ``kp2d`` stores 2D keypoints as SMPL-24 joints projected to the image.
SMPL-24 order (body model):
  0:pelvis  1:left_hip  2:right_hip  3:spine1  4:left_knee  5:right_knee
  6:spine2  7:left_ankle  8:right_ankle  9:spine3  10:left_foot  11:right_foot
  12:neck  13:left_collar  14:right_collar  15:head
  16:left_shoulder  17:right_shoulder  18:left_elbow  19:right_elbow
  20:left_wrist  21:right_wrist  22:left_hand  23:right_hand

Exported as COCO-17 layout for MMPose compatibility.  SMPL has no nose,
eyes, or ears; indices 0-4 are left at (0, 0) with visibility 0 so they
are not trained or evaluated.  Body joints map as:
  5:left_shoulder  6:right_shoulder  7:left_elbow  8:right_elbow
  9:left_wrist  10:right_wrist  11:left_hip  12:right_hip
  13:left_knee  14:right_knee  15:left_ankle  16:right_ankle

Usage::

    python tools/dataset_converters/preprocess_emdb.py \\
        --data-root data/emdb \\
        --out-dir data/emdb/annotations
"""

import argparse
import glob
import os
import os.path as osp
import pickle

import numpy as np

try:
    import mmengine
    USE_MMENGINE = True
except ImportError:
    USE_MMENGINE = False

# COCO-17 indices 0-4 (face) are not present in SMPL-24; kept invisible.
# COCO indices 5-16 -> SMPL-24 body joints.
_SMPL24_TO_COCO17_BODY = [
    16, 17, 18, 19, 20, 21,  # shoulders, elbows, wrists
    1, 2, 4, 5, 7, 8,  # hips, knees, ankles
]
_COCO_BODY_START = 5

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


def _smpl24_to_coco17(frame_kp2d):
    """Convert one frame's SMPL-24 2D joints to COCO-17 layout.

    Face keypoints (COCO 0-4) are marked invisible because SMPL-24 does not
    define facial landmarks.  Only body joints 5-16 are populated.

    Args:
        frame_kp2d (np.ndarray): shape (24, 2).

    Returns:
        np.ndarray: shape (17, 3) -- x, y, visibility (2 if visible else 0).
    """
    coco = np.zeros((17, 3), dtype=np.float32)
    for offset, smpl_idx in enumerate(_SMPL24_TO_COCO17_BODY):
        coco_idx = _COCO_BODY_START + offset
        x = float(frame_kp2d[smpl_idx, 0])
        y = float(frame_kp2d[smpl_idx, 1])
        vis = 2 if np.isfinite(x) and np.isfinite(y) else 0
        coco[coco_idx] = [x, y, vis]
    return coco


def _xyxy_to_xywh(bbox_xyxy):
    """Convert xyxy bbox to COCO xywh."""
    x_min, y_min, x_max, y_max = bbox_xyxy
    w = x_max - x_min
    h = y_max - y_min
    return [float(x_min), float(y_min), float(w), float(h)]


def process_emdb(data_root, out_dir):
    """Convert all EMDB sequences to a single COCO JSON file."""
    pkl_pattern = osp.join(data_root, 'P*', '*', '*_data.pkl')
    pkl_files = sorted(glob.glob(pkl_pattern))

    images = []
    annotations = []
    img_id_counter = 0
    ann_id_counter = 0
    track_id_counter = 0
    track_key_to_id = {}

    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            seq = pickle.load(f)

        seq_name = seq['name']
        n_frames = int(seq['n_frames'])
        kp2d = np.asarray(seq['kp2d'])  # (n_frames, 24, 2)
        bboxes_xyxy = np.asarray(seq['bboxes']['bboxes'])
        invalid_idxs = set(int(i) for i in seq['bboxes']['invalid_idxs'])
        good_frames_mask = np.asarray(seq['good_frames_mask'], dtype=bool)
        emdb1 = bool(seq['emdb1'])
        emdb2 = bool(seq['emdb2'])
        img_w = int(seq['camera']['width'])
        img_h = int(seq['camera']['height'])

        # One person per sequence -> one track
        if seq_name not in track_key_to_id:
            track_id_counter += 1
            track_key_to_id[seq_name] = track_id_counter
        track_id = track_key_to_id[seq_name]

        # file_name prefix: P0/00_mvs_a/images/00000.jpg
        parts = pkl_path.replace('\\', '/').split('/')
        # .../P0/00_mvs_a/P0_00_mvs_a_data.pkl
        subject_dir = parts[-3]
        sequence_dir = parts[-2]
        rel_prefix = osp.join(subject_dir, sequence_dir, 'images')

        for frame_idx in range(n_frames):
            if frame_idx in invalid_idxs:
                continue

            bbox_xyxy = bboxes_xyxy[frame_idx]
            bw = bbox_xyxy[2] - bbox_xyxy[0]
            bh = bbox_xyxy[3] - bbox_xyxy[1]
            if bw <= 0 or bh <= 0:
                continue

            img_filename = osp.join(rel_prefix, f'{frame_idx:05d}.jpg')
            abs_img_path = osp.join(data_root, img_filename)
            if not osp.exists(abs_img_path):
                continue

            kpts_coco17 = _smpl24_to_coco17(kp2d[frame_idx])
            if kpts_coco17[:, 2].sum() == 0:
                continue

            bbox = _xyxy_to_xywh(bbox_xyxy)
            good_frame = bool(good_frames_mask[frame_idx])

            img_id_counter += 1
            img_entry = {
                'id': img_id_counter,
                'file_name': img_filename,
                'width': img_w,
                'height': img_h,
                'nframes': n_frames,
                'frame_id': frame_idx,
                'seq_name': seq_name,
                'emdb1': emdb1,
                'emdb2': emdb2,
                'good_frame': good_frame,
            }
            images.append(img_entry)

            kpts_flat = kpts_coco17.reshape(-1).tolist()
            num_visible = int((kpts_coco17[:, 2] > 0).sum())

            ann_id_counter += 1
            ann_entry = {
                'id': ann_id_counter,
                'image_id': img_id_counter,
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
            'description': 'EMDB dataset in COCO format',
            'url': 'https://ait.ethz.ch/emdb',
            'version': '1.0',
            'year': 2023,
            'contributor': 'ETH Zurich AIT',
        },
        'categories': _COCO_CATEGORIES,
        'images': images,
        'annotations': annotations,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, 'emdb_all.json')
    if USE_MMENGINE:
        mmengine.dump(coco_json, out_path)
    else:
        import json
        with open(out_path, 'w') as f:
            json.dump(coco_json, f)

    n_tracks = len(track_key_to_id)
    print(f'{len(images):6d} images, {len(annotations):7d} annotations, '
          f'{n_tracks:4d} tracks -> {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert EMDB PKL sequences to COCO-format JSON.')
    parser.add_argument(
        '--data-root',
        default='data/emdb',
        help='Root directory of the EMDB dataset.')
    parser.add_argument(
        '--out-dir',
        default='data/emdb/annotations',
        help='Output directory for JSON annotation files.')
    args = parser.parse_args()
    process_emdb(args.data_root, args.out_dir)


if __name__ == '__main__':
    main()
