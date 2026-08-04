# Copyright (c) OpenMMLab. All rights reserved.
"""Merge PoseTrack21 per-video annotations into a single COCO-format JSON.

PoseTrack21 ships one COCO-style JSON per video under
``PoseTrack21/posetrack_data/{train,val}/``.  This script merges each split
into a single annotation file that :class:`~mmpose.datasets.PoseTrack21Dataset`
(and therefore ``tools/benchmark_e2e.py``) can load, adding the fields the
raw release omits.

What the raw release does *not* provide, and this script adds:

* ``width`` / ``height`` -- read once per video from the first frame on disk.
* ``area`` / ``iscrowd`` / ``num_keypoints`` on each annotation.
* **Globally unique** ``track_id``.  PoseTrack restarts ``track_id`` at 0 in
  every video; the temporal metrics (``MPJVE``/``MPJAE``) group tracks by
  ``track_id`` alone, so without a global remap every "person 0" in the
  dataset would be merged into a single track.
* ``frame_id`` -- the frame's index inside its video, matching the numeric
  image filename that ``MPJVE``/``MPJAE``/``IDSwitch`` parse.
* One ``iscrowd=1``, keypoint-free annotation per **ignore region** (people
  the dataset deliberately leaves unannotated, present on ~40% of labeled
  frames).  These make COCOeval discount detections that land on them
  instead of scoring them as false positives -- the same mechanism COCO and
  CrowdPose use for crowd regions.  Worth +2.5 AP (+6.4 AP on medium
  objects) on the val split; recall is unaffected, since ignored GT does not
  enter the recall denominator.
* ``good_frame`` -- ``True`` for frames carrying pose annotations.  Only
  roughly 44% of PoseTrack frames are labeled (every 4th frame, plus a dense
  ~30-frame consecutive block per video).  ``PoseTrack21Dataset`` loads only
  good frames by default; ``good_frame_mask=False`` (``--include-bad-frames``)
  loads all of them, giving temporal post-processing a continuous sequence.

Keypoints keep the native PoseTrack-17 ordering (identical to
``configs/_base_/datasets/posetrack18.py``); the benchmark loader maps them
onto COCO-17 via ``KeypointConverter(src='posetrack18', dst='coco')``.

Only the visibility column is rewritten.  PoseTrack's third value means
"visible" (1) versus "labeled but occluded" (0) -- it is *not* COCO's
"unlabeled" flag.  Unlabeled joints are instead marked with the sentinel
coordinates ``(0, 0)`` or ``(-1, -1)``.  Deciding on the flag alone would
discard ~12% of the annotated joints, so labeled-ness is decided by the
coordinates and re-encoded in COCO's convention (2 = visible,
1 = occluded, 0 = unlabeled).

Known limitations:

* PoseTrack annotates no eyes, and its ears are always unlabeled, so only 13
  of the 17 COCO joints carry ground truth.
* Ignore regions are reduced to their axis-aligned bounding box, and
  ``COCOeval`` then widens that box 3x in each dimension before testing
  containment.  The suppression is therefore more generous than the source
  polygon.  Measured against exact polygon containment the difference is
  worth ~0.3 AP, because the extra discounted detections are low-scoring
  ones that barely move the precision-recall curve.

Usage::

    python tools/dataset_converters/preprocess_posetrack21.py \\
        --data-root data/posetrack21 \\
        --out-dir data/posetrack21/annotations
"""

import argparse
import glob
import json
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mmengine
    USE_MMENGINE = True
except ImportError:
    USE_MMENGINE = False

# PoseTrack-17 layout, identical to configs/_base_/datasets/posetrack18.py.
_POSETRACK_CATEGORIES = [{
    'supercategory': 'person',
    'id': 1,
    'name': 'person',
    'keypoints': [
        'nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
        'right_knee', 'left_ankle', 'right_ankle'
    ],
    'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                 [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                 [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
}]

_NUM_KEYPOINTS = 17


def _is_unlabeled(x: float, y: float) -> bool:
    """Whether a PoseTrack keypoint uses the "not annotated" sentinel.

    PoseTrack marks missing joints with ``(0, 0)`` or ``(-1, -1)`` rather
    than with the visibility flag (which distinguishes visible from
    occluded).
    """
    return (x == 0.0 and y == 0.0) or (x == -1.0 and y == -1.0)


def _to_coco_visibility(keypoints: List[float]) -> Tuple[List[float], int]:
    """Re-encode a flat PoseTrack keypoint list in COCO visibility terms.

    Args:
        keypoints (list[float]): Flat ``[x, y, v] * 17`` list as stored in
            the PoseTrack release, where ``v`` is 1 for visible and 0 for
            occluded, and unlabeled joints carry sentinel coordinates.

    Returns:
        tuple:
        - list[float]: Flat ``[x, y, v] * 17`` list with COCO visibility
          (2 = visible, 1 = labeled but occluded, 0 = unlabeled).  Unlabeled
          joints are zeroed so they cannot leak sentinel coordinates into a
          metric.
        - int: Number of labeled joints (COCO ``num_keypoints``).
    """
    kpts = np.asarray(keypoints, dtype=np.float64).reshape(-1, 3)
    out = np.zeros((_NUM_KEYPOINTS, 3), dtype=np.float64)

    num_labeled = 0
    for i in range(min(len(kpts), _NUM_KEYPOINTS)):
        x, y, v = kpts[i]
        if _is_unlabeled(x, y):
            continue
        out[i] = [x, y, 2.0 if v >= 1.0 else 1.0]
        num_labeled += 1

    return [round(float(v), 4) for v in out.reshape(-1)], num_labeled


def _ignore_region_bbox(
    xs: List[float],
    ys: List[float],
) -> Optional[List[float]]:
    """Axis-aligned COCO ``[x, y, w, h]`` bounds of an ignore-region polygon.

    Returns ``None`` for degenerate polygons (fewer than two distinct points,
    or zero extent along either axis).
    """
    if not xs or not ys:
        return None
    x0, x1 = float(min(xs)), float(max(xs))
    y0, y1 = float(min(ys)), float(max(ys))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def _video_image_size(
    data_root: str,
    image_entries: List[dict],
) -> Optional[Tuple[int, int]]:
    """Return ``(width, height)`` for a video, probing frames on disk.

    Every frame of a PoseTrack video shares a resolution, so the first
    readable frame settles it.  Later frames are probed only if earlier ones
    are missing.
    """
    from PIL import Image

    for img in image_entries:
        path = osp.join(data_root, img['file_name'])
        if not osp.exists(path):
            continue
        try:
            with Image.open(path) as handle:
                width, height = handle.size
            return int(width), int(height)
        except OSError:
            continue
    return None


def process_split(
    data_root: str,
    split: str,
    out_dir: str,
) -> Optional[str]:
    """Merge one PoseTrack21 split into a single COCO-format JSON file."""
    split_dir = osp.join(
        data_root, 'PoseTrack21', 'posetrack_data', split)
    video_files = sorted(glob.glob(osp.join(split_dir, '*.json')))
    if not video_files:
        print(f'No per-video annotations found in {split_dir}, skipping '
              f'split {split!r}.')
        return None

    images: List[dict] = []
    annotations: List[dict] = []
    ann_id_counter = 0
    track_key_to_id: Dict[Tuple[str, int], int] = {}
    n_missing_files = 0
    n_unsized_videos = 0
    n_ignore_anns = 0

    for video_path in video_files:
        with open(video_path, 'r') as f:
            video = json.load(f)

        image_entries = video.get('images', [])
        if not image_entries:
            continue

        size = _video_image_size(data_root, image_entries)
        if size is None:
            n_unsized_videos += 1
            print(f'Warning: no readable frame for {video_path}, skipping.')
            continue
        img_w, img_h = size

        seq_name = osp.splitext(osp.basename(video_path))[0]

        # Frames are emitted in ascending frame order so that consumers
        # iterating the annotation file in order (e.g. the benchmark's
        # post-processing pipeline) see each video as one contiguous run.
        kept_img_ids = set()
        for img in sorted(
                image_entries,
                key=lambda i: osp.basename(i['file_name'])):
            file_name = img['file_name']
            if not osp.exists(osp.join(data_root, file_name)):
                n_missing_files += 1
                continue

            frame_id = int(osp.splitext(osp.basename(file_name))[0])
            img_id = int(img['image_id'])
            is_labeled = bool(img.get('is_labeled', False))

            images.append({
                'id': img_id,
                'file_name': file_name,
                'width': img_w,
                'height': img_h,
                'nframes': int(img.get('nframes', len(image_entries))),
                'frame_id': frame_id,
                'seq_name': seq_name,
                'vid_id': str(img.get('vid_id', seq_name)),
                'is_labeled': is_labeled,
                'good_frame': (
                    is_labeled and bool(img.get('has_labeled_person', True))),
            })
            kept_img_ids.add(img_id)

            # Ignore regions mark people the dataset deliberately leaves
            # unannotated.  Emitted as keypoint-free ``iscrowd=1``
            # annotations, which is how COCO/CrowdPose crowd regions
            # suppress false positives: ``COCOeval._prepare`` flags any GT
            # with zero visible keypoints as ignored, and ``computeOks``
            # falls back to a containment test against the (3x expanded)
            # bbox, so a detection landing inside scores OKS 1.0 and is
            # dropped from both the TP and the FP count.  ``iscrowd=1``
            # additionally lets one region absorb several detections.
            #
            # ``compute_oks_pairs`` (mmpose/evaluation/functional/
            # frame_metrics.py) skips crowd and keypoint-free GT when
            # *matching*, so these never become a GT track and
            # MPJVE/MPJAE/IDSwitch, ``gt_recall`` and ``mean_oks`` are
            # unaffected.  MOTA/IDF1/HOTA (mmpose/evaluation/metrics/
            # mot_metrics.py) additionally read them back out via
            # ``iscrowd``/``bboxes`` on ``gt_instances`` to suppress
            # unmatched predictions landing inside one, so a correct
            # detection here is not counted as a false positive; see that
            # module's docstring for the two overlap conventions.
            # ``track_id`` is 0, a sentinel that no real track uses (those
            # start at 1).
            #
            # Emitted for unlabeled frames too, so --include-bad-frames
            # gets the same false-positive suppression.
            for xs, ys in zip(img.get('ignore_regions_x') or [],
                              img.get('ignore_regions_y') or []):
                ig_bbox = _ignore_region_bbox(xs, ys)
                if ig_bbox is None:
                    continue
                ann_id_counter += 1
                n_ignore_anns += 1
                annotations.append({
                    'id': ann_id_counter,
                    'image_id': img_id,
                    'category_id': 1,
                    'iscrowd': 1,
                    'track_id': 0,
                    'bbox': [round(v, 2) for v in ig_bbox],
                    'area': round(ig_bbox[2] * ig_bbox[3], 2),
                    'num_keypoints': 0,
                    'keypoints': [0.0] * (_NUM_KEYPOINTS * 3),
                })

        for ann in video.get('annotations', []):
            img_id = int(ann['image_id'])
            if img_id not in kept_img_ids:
                continue
            if 'bbox' not in ann or 'keypoints' not in ann:
                continue

            bbox = [float(v) for v in ann['bbox'][:4]]
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue

            kpts, num_labeled = _to_coco_visibility(ann['keypoints'])

            # PoseTrack track_ids restart at 0 in every video; make them
            # unique across the split so temporal metrics keep tracks apart.
            # Keyed on the video rather than on `person_id`, which is a
            # re-identification label and is shared across videos.
            track_key = (seq_name, int(ann.get('track_id', 0)))
            if track_key not in track_key_to_id:
                track_key_to_id[track_key] = len(track_key_to_id) + 1

            ann_id_counter += 1
            annotations.append({
                'id': ann_id_counter,
                'image_id': img_id,
                'category_id': 1,
                'iscrowd': 0,
                'track_id': track_key_to_id[track_key],
                'bbox': [round(v, 2) for v in bbox],
                'area': round(bbox[2] * bbox[3], 2),
                'num_keypoints': num_labeled,
                'keypoints': kpts,
            })

    coco_json = {
        'info': {
            'description': f'PoseTrack21 {split} split in COCO format',
            'url': 'https://github.com/anDoer/PoseTrack21',
            'version': '1.0',
            'year': 2022,
            'contributor': 'PoseTrack21 authors',
        },
        'categories': _POSETRACK_CATEGORIES,
        'images': images,
        'annotations': annotations,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f'posetrack21_{split}.json')
    if USE_MMENGINE:
        mmengine.dump(coco_json, out_path)
    else:
        with open(out_path, 'w') as f:
            json.dump(coco_json, f)

    n_good = sum(1 for img in images if img['good_frame'])
    n_pose = len(annotations) - n_ignore_anns
    print(f'{split:5s}: {len(images):6d} images ({n_good} labeled), '
          f'{n_pose:7d} pose annotations, '
          f'{n_ignore_anns:6d} ignore regions, '
          f'{len(track_key_to_id):5d} tracks -> {out_path}')
    if n_missing_files:
        print(f'       {n_missing_files} annotated frames had no image file '
              f'and were dropped.')
    if n_unsized_videos:
        print(f'       {n_unsized_videos} videos had no readable frame and '
              f'were skipped entirely.')
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Merge PoseTrack21 per-video annotations into COCO-format '
                    'JSON files.')
    parser.add_argument(
        '--data-root',
        default='data/posetrack21',
        help='Root directory of the PoseTrack21 dataset (the directory '
             'containing `images/` and `PoseTrack21/`).')
    parser.add_argument(
        '--out-dir',
        default='data/posetrack21/annotations',
        help='Output directory for the merged JSON annotation files.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['val', 'train'],
        help='Splits to convert. Default: val train')
    args = parser.parse_args()

    for split in args.splits:
        process_split(args.data_root, split, args.out_dir)


if __name__ == '__main__':
    main()
