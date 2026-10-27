# Copyright (c) OpenMMLab. All rights reserved.
"""Translation from benchmark frames to the official PoseTrack21 eval kit.

Keypoint HOTA, keypoint mAP and keypoint MOTA call the kit unchanged. The
keypoint kit drops ignore-region joints with a point-in-polygon test, which
is not the bug CoMotion and DETRAM report.

Box MOTA/IDF1 and box HOTA use the kit's own ``motmetrics`` and
``PoseTrackMOT``. Only the ignore-region test is replaced:

* ``official`` — IoU with the region > 0.1, and every frame is scored.
  A box that sits entirely inside a large region has a tiny IoU and is
  still a false positive. Frames with no ground-truth boxes count every
  prediction as a false positive.
* ``fixed`` — ``area(box ∩ region) / area(box) > 0.1``, and frames with no
  ground-truth boxes are skipped. This is the threshold in the released
  CoMotion and DETRAM scripts (the paper text says 0.3).

A prediction already matched to a ground-truth box is never removed.
Invalid polygons are repaired with ``buffer(0)``; a MultiPolygon contributes
the sum of its parts, which is what the kit computed under Shapely 1
(``for poly in multi_poly`` does not run under Shapely 2).
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from mmpose.evaluation.functional.keypoint_space import convert_keypoint_arrays

_EPS = float(np.finfo(float).eps)

# PoseTrack raw-17, the order the kit reads before it deletes the ears.
POSETRACK_RAW_NAMES = (
    'nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
)

HOTA_FIELDS = (
    'HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA')
POSE_PARTS = (
    'Head', 'Shou', 'Elb', 'Wri', 'Hip', 'Knee', 'Ankl', 'Total')

_FALLBACK_EVAL_ROOT = Path(
    '/local/home/nkoefarago/benchmark_ext/PoseTrack21_full/eval')
_FALLBACK_POSEVAL = Path('/local/home/nkoefarago/benchmark_ext/poseval')


@dataclass(frozen=True)
class IgnoreRule:
    """How an unmatched box is excused by an ignore region."""

    name: str
    overlap: str  # 'iou' or 'box_fraction'
    threshold: float
    skip_frames_without_gt: bool


VARIANTS = {
    'official': IgnoreRule('official', 'iou', 0.1, False),
    'fixed': IgnoreRule('fixed', 'box_fraction', 0.1, True),
}


def ensure_numpy_aliases() -> None:
    """The vendored kit still refers to ``np.float`` / ``np.int`` / ``np.bool``.

    NumPy warns on the lookup itself, so the check is wrapped.
    """
    import warnings
    for name, ty in (('float', float), ('int', int), ('bool', bool)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            try:
                getattr(np, name)
            except AttributeError:
                setattr(np, name, ty)


def resolve_eval_root(explicit: Optional[str] = None) -> Path:
    """Directory that contains both ``posetrack21/`` and ``mot/motmetrics``.

    Search order: ``explicit``, ``POSETRACK21_EVAL_ROOT``, an already
    importable ``posetrack21``, then the known checkout. The error names
    ``--official-posetrack21`` so a normal benchmark run, which does not
    request the kit, never hits this.
    """
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get('POSETRACK21_EVAL_ROOT')
    if env:
        candidates.append(Path(env))
    try:
        import posetrack21
        candidates.append(Path(posetrack21.__file__).resolve().parents[1])
    except ImportError:
        pass
    candidates.append(_FALLBACK_EVAL_ROOT)
    candidates.append(Path('external/PoseTrack21/eval'))

    seen = set()
    for cand in candidates:
        cand = cand.expanduser()
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if _is_eval_root(cand):
            return cand.resolve()
        # Caller pointed at the inner package directory.
        if _is_eval_root(cand.parent):
            return cand.parent.resolve()
    raise ImportError(
        'PoseTrack21 eval kit not found. Pass --pt21-eval-root or set '
        'POSETRACK21_EVAL_ROOT to the kit eval/ directory (it contains '
        'posetrack21/ and mot/). Required by --official-posetrack21.')


def _is_eval_root(path: Path) -> bool:
    return ((path / 'posetrack21' / 'posetrack21' / '__init__.py').is_file()
            and (path / 'mot' / 'motmetrics' / '__init__.py').is_file())


def _import_posetrack21(eval_root: Path):
    ensure_numpy_aliases()
    pkg_parent = str(eval_root / 'posetrack21')
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)
    import posetrack21.api as api
    return api


def _allow_motmetrics_import() -> None:
    """The kit's ``motmetrics.io`` imports ``xmltodict`` at import time.

    Box scoring never reads those XML loaders. A missing ``xmltodict``
    would otherwise block ``MOTAccumulator``, so the name is bound to an
    empty module when the package is not installed.
    """
    import types
    if 'xmltodict' in sys.modules:
        return
    try:
        import xmltodict  # noqa: F401
    except ImportError:
        sys.modules['xmltodict'] = types.ModuleType('xmltodict')


def import_kit_motmetrics(eval_root: Path):
    """The kit's modified py-motmetrics (it accepts ``ignore_candidates``).

    The package imports itself as top-level ``motmetrics``. If another
    ``motmetrics`` is already loaded and lacks that argument, it is
    unloaded so the kit copy can take the name.
    """
    import importlib
    import inspect

    _allow_motmetrics_import()
    mot_root = str(eval_root / 'mot')
    if mot_root not in sys.path:
        sys.path.insert(0, mot_root)
    loaded = sys.modules.get('motmetrics')
    if loaded is not None:
        acc = getattr(loaded, 'MOTAccumulator', None)
        params = (inspect.signature(acc.update).parameters
                  if acc is not None else {})
        if 'ignore_candidates' not in params:
            for key in list(sys.modules):
                if key == 'motmetrics' or key.startswith('motmetrics.'):
                    del sys.modules[key]
    mm = importlib.import_module('motmetrics')
    _patch_ignore_candidates(mm)
    return mm


def _patch_ignore_candidates(mm) -> None:
    """poseval calls ``MOTAccumulator.update`` without ignore candidates.

    The kit's copy defaults that argument to ``None`` and then iterates
    it. An empty list is the same as "nothing to ignore".
    """
    acc = mm.MOTAccumulator
    if getattr(acc.update, '_pt21_ignore_patch', False):
        return
    orig = acc.update

    def update(self, oids, hids, dists, frameid=None, vf='',
               ignore_candidates=None):
        if ignore_candidates is None:
            ignore_candidates = []
        return orig(self, oids, hids, dists, frameid=frameid, vf=vf,
                    ignore_candidates=ignore_candidates)

    update._pt21_ignore_patch = True
    acc.update = update


def _shapely():
    from shapely.geometry import MultiPolygon, Polygon
    return Polygon, MultiPolygon


def _polygon_parts(region):
    """Valid polygon, or the pieces of ``buffer(0)`` for an invalid one."""
    _, multi = _shapely()
    if region.is_valid:
        return [region]
    repaired = region.buffer(0)
    if isinstance(repaired, multi):
        return list(repaired.geoms)
    return [repaired]


def region_overlap(region, box, overlap: str) -> float:
    """Overlap of a predicted box polygon with one ignore-region polygon.

    ``overlap='iou'`` is the official kit. ``overlap='box_fraction'`` is
    the CoMotion/DETRAM fix: the fraction of the *box* that lies inside
    the region, which is 1 for a box fully contained in a huge region.
    """
    parts = _polygon_parts(region)
    inter = sum(part.intersection(box).area for part in parts)
    if overlap == 'iou':
        union = sum(part.union(box).area for part in parts)
        return inter / union if union > 0 else 0.0
    if overlap != 'box_fraction':
        raise ValueError(
            f"overlap must be 'iou' or 'box_fraction', got {overlap!r}")
    return inter / box.area if box.area > 0 else 0.0


def _box_polygon(xyxy: Sequence[float]):
    polygon, _ = _shapely()
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def _xyxy_iou(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Pairwise IoU of ``(M, 4)`` and ``(N, 4)`` xyxy boxes."""
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((len(gt), len(pred)), dtype=np.float64)
    inter_w = (np.minimum(gt[:, None, 2], pred[None, :, 2])
               - np.maximum(gt[:, None, 0], pred[None, :, 0])).clip(min=0)
    inter_h = (np.minimum(gt[:, None, 3], pred[None, :, 3])
               - np.maximum(gt[:, None, 1], pred[None, :, 1])).clip(min=0)
    inter = inter_w * inter_h
    area_g = ((gt[:, 2] - gt[:, 0]).clip(min=0)
              * (gt[:, 3] - gt[:, 1]).clip(min=0))
    area_p = ((pred[:, 2] - pred[:, 0]).clip(min=0)
              * (pred[:, 3] - pred[:, 1]).clip(min=0))
    union = area_g[:, None] + area_p[None, :] - inter
    return inter / np.maximum(union, _EPS)


def ignore_candidate_indices(
    pred_xyxy: np.ndarray,
    regions,
    rule: IgnoreRule,
) -> List[int]:
    """Every prediction whose overlap with some region exceeds the rule.

    This is the MOTA kit's candidate list. ``motmetrics`` then refrains
    from counting an *unmatched* candidate as a false positive; a
    candidate that is matched to ground truth stays a match.
    """
    if len(pred_xyxy) == 0 or not regions:
        return []
    kept = []
    for j, box in enumerate(np.asarray(pred_xyxy, dtype=np.float64)):
        poly = _box_polygon(box)
        if max(region_overlap(region, poly, rule.overlap)
               for region in regions) > rule.threshold:
            kept.append(j)
    return kept


def hota_remove_indices(
    pred_xyxy: np.ndarray,
    gt_xyxy: np.ndarray,
    regions,
    rule: IgnoreRule,
) -> np.ndarray:
    """Tracker rows the box-HOTA preprocessor deletes before scoring.

    Frames with no ground truth are emptied entirely under ``fixed`` and
    left untouched under ``official`` (those predictions are false
    positives). Otherwise only unmatched predictions that overlap an
    ignore region are removed. A box matched at IoU >= 0.5 is kept even
    when it also lies inside a region.
    """
    pred_xyxy = np.asarray(pred_xyxy, dtype=np.float64).reshape(-1, 4)
    gt_xyxy = np.asarray(gt_xyxy, dtype=np.float64).reshape(-1, 4)
    n_pred = len(pred_xyxy)
    if rule.skip_frames_without_gt and len(gt_xyxy) == 0:
        return np.arange(n_pred)
    if n_pred == 0 or len(gt_xyxy) == 0 or not regions:
        return np.array([], dtype=int)
    scores = _xyxy_iou(gt_xyxy, pred_xyxy)
    scores[scores < 0.5 - _EPS] = 0
    rows, cols = linear_sum_assignment(-scores)
    matched = set(int(c) for r, c in zip(rows, cols)
                  if scores[r, c] > _EPS)
    remove = []
    for j in range(n_pred):
        if j in matched:
            continue
        poly = _box_polygon(pred_xyxy[j])
        if max(region_overlap(region, poly, rule.overlap)
               for region in regions) > rule.threshold:
            remove.append(j)
    return np.asarray(remove, dtype=int)


def prediction_counts_as_fp(
    pred_xyxy: Sequence[float],
    gt_xyxy: np.ndarray,
    regions,
    rule: IgnoreRule,
) -> bool:
    """Whether one unmatched box is a false positive under ``rule``.

    ``fixed`` skips a frame that has no ground-truth boxes, so the
    prediction does not count. ``official`` scores it, and only drops it
    when the (usually tiny) IoU with an ignore region clears 0.1.
    """
    gt_xyxy = np.asarray(gt_xyxy, dtype=np.float64).reshape(-1, 4)
    if rule.skip_frames_without_gt and len(gt_xyxy) == 0:
        return False
    if not regions:
        return True
    poly = _box_polygon(pred_xyxy)
    overlap = max(region_overlap(region, poly, rule.overlap)
                  for region in regions)
    return overlap <= rule.threshold


def instances_to_raw17(
    keypoints: np.ndarray,
    scores: np.ndarray,
    src_names: Sequence[str],
) -> np.ndarray:
    """``(N, K, 2)`` eval-space joints -> ``(N, 17, 3)`` raw PoseTrack.

    The third channel is the score. Joints the source layout does not
    have (``head_bottom`` / ``head_top`` for a COCO model, ears for the
    native 15) stay at score 0. The kit treats a joint at (0, 0) as not
    predicted, so those coordinates are cleared too.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if keypoints.ndim == 2:
        keypoints = keypoints[None]
    if scores.ndim == 1:
        scores = scores[None]
    coords, mapped_scores = convert_keypoint_arrays(
        keypoints[..., :2], scores, src_names, POSETRACK_RAW_NAMES)
    raw = np.zeros((coords.shape[0], 17, 3), dtype=np.float32)
    raw[:, :, :2] = coords
    raw[:, :, 2] = mapped_scores
    missing = raw[:, :, 2] <= 0
    raw[:, :, 0][missing] = 0
    raw[:, :, 1][missing] = 0
    return raw


def _compact_track_ids(per_frame_ids: List[np.ndarray]
                       ) -> Tuple[List[np.ndarray], bool]:
    """Renumber track ids to ``1..N`` inside one sequence.

    Negative ids and ids that repeat inside a single frame each become
    their own track. The returned flag is true when that happened, so
    association numbers are not compared to a tracker. The same positive
    id on later frames keeps the number it was given the first time.
    """
    synthetic = False
    mapping: Dict[int, int] = {}
    next_id = 1
    out: List[np.ndarray] = []
    for ids in per_frame_ids:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        frame = np.empty(len(ids), dtype=np.int64)
        seen: set = set()
        for i, tid in enumerate(int(v) for v in ids.tolist()):
            duplicate = tid in seen
            if tid >= 0:
                seen.add(tid)
            if tid < 0 or duplicate:
                synthetic = True
                frame[i] = next_id
                next_id += 1
                continue
            if tid not in mapping:
                mapping[tid] = next_id
                next_id += 1
            frame[i] = mapping[tid]
        out.append(frame)
    return out, synthetic


def _load_json(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def labeled_image_ids(gt_pose_dir: Path) -> Dict[str, set]:
    """``{seq: labeled image ids}`` for every official sequence JSON."""
    out = {}
    for path in sorted(Path(gt_pose_dir).glob('*.json')):
        data = _load_json(path)
        out[path.stem] = {
            int(img['id']) for img in data.get('images', [])
            if img.get('is_labeled', True)
        }
    return out


def unlabeled_image_ids(gt_pose_dir: Path) -> Dict[str, set]:
    """``{seq: unlabeled image ids}`` for every official sequence JSON."""
    out = {}
    for path in sorted(Path(gt_pose_dir).glob('*.json')):
        data = _load_json(path)
        out[path.stem] = {
            int(img['id']) for img in data.get('images', [])
            if not img.get('is_labeled', True)
        }
    return out


def write_sequence_files(
    records: List[dict],
    src_names: Sequence[str],
    gt_pose_dir: Path,
    gt_mot_dir: Path,
    out_dir: Path,
) -> Tuple[Path, Path, Path, Path, bool]:
    """Write kit prediction files and a GT subset of the frames we ran.

    Returns ``(pose_dir, mot_dir, gt_pose_subset, gt_mot_subset)`` and
    whether any track id had to be invented. Sequences and frames that
    were not evaluated are left out of the subset, so they are not scored
    as misses. Image order follows the official list.
    """
    by_seq: Dict[str, List[dict]] = {}
    for rec in records:
        by_seq.setdefault(rec['seq_id'], []).append(rec)

    pose_dir = out_dir / 'pose'
    mot_dir = out_dir / 'mot'
    gt_pose_out = out_dir / 'gt_pose'
    gt_mot_out = out_dir / 'gt_mot'
    for folder in (pose_dir, mot_dir, gt_pose_out, gt_mot_out):
        folder.mkdir(parents=True, exist_ok=True)

    synthetic = False
    for seq, frames in by_seq.items():
        gt_pose_path = Path(gt_pose_dir) / f'{seq}.json'
        if not gt_pose_path.is_file():
            raise FileNotFoundError(
                f'Official PoseTrack21 sequence JSON not found: '
                f'{gt_pose_path}. Required by --official-posetrack21.')
        gt = _load_json(gt_pose_path)
        images = list(gt.get('images', []))
        by_id = {int(img['id']): img for img in images}
        evaluated = []
        seen = set()
        for img in images:
            iid = int(img['id'])
            if iid in seen:
                continue
            if any(int(fr['img_id']) == iid for fr in frames):
                evaluated.append(img)
                seen.add(iid)
        unknown = [
            int(fr['img_id']) for fr in frames if int(fr['img_id']) not in by_id
        ]
        if unknown:
            raise KeyError(
                f'{len(unknown)} predictions in {seq} use image ids that '
                f'are not in {gt_pose_path.name} (first: {unknown[0]}).')

        id_to_frame = {int(fr['img_id']): fr for fr in frames}
        ordered_ids = [int(img['id']) for img in evaluated]
        id_rows = []
        for iid in ordered_ids:
            fr = id_to_frame[iid]
            ids = np.asarray(fr['track_ids'], dtype=np.int64).reshape(-1)
            id_rows.append(ids)
        compact, seq_synthetic = _compact_track_ids(id_rows)
        synthetic = synthetic or seq_synthetic

        annotations = []
        mot_lines = []
        for frame_i, iid in enumerate(ordered_ids):
            fr = id_to_frame[iid]
            kpts = np.asarray(fr['keypoints'], dtype=np.float32)
            scores = np.asarray(fr['scores'], dtype=np.float32)
            if kpts.size == 0:
                raw = np.zeros((0, 17, 3), dtype=np.float32)
            else:
                raw = instances_to_raw17(kpts, scores, src_names)
            boxes = np.asarray(fr['boxes'], dtype=np.float64).reshape(-1, 4)
            box_scores = np.asarray(fr['box_scores'], dtype=np.float64).reshape(-1)
            for inst_i, tid in enumerate(compact[frame_i].tolist()):
                present = raw[inst_i, :, 2] > 0
                xyv = np.zeros((17, 3), dtype=np.float64)
                xyv[present, :2] = raw[inst_i, present, :2]
                xyv[:, 2] = present.astype(np.float64)
                annotations.append({
                    'image_id': int(iid),
                    'track_id': int(tid),
                    'keypoints': [round(float(v), 2) for v in xyv.reshape(-1)],
                    'scores': [round(float(v), 4) for v in raw[inst_i, :, 2]],
                })
                if inst_i >= len(boxes):
                    continue
                x1, y1, x2, y2 = boxes[inst_i]
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                # Kit MOT text is 1-based in frame, id and the box origin,
                # and stores width/height with a +1. ``frame_i`` is the
                # 0-based index into the subset image list.
                vals = [x1 + 1, y1 + 1, (x2 - x1) + 1, (y2 - y1) + 1]
                score = float(box_scores[inst_i]) if inst_i < len(box_scores) else 0.0
                mot_lines.append(
                    f'{frame_i + 1},{int(tid) + 1},'
                    + ','.join(f'{v:.2f}' for v in vals)
                    + f',{score:.4f},-1,-1,-1')

        pose_payload = {
            'images': evaluated,
            'annotations': annotations,
            'categories': gt.get('categories', []),
        }
        (pose_dir / f'{seq}.json').write_text(json.dumps(pose_payload))
        (mot_dir / f'{seq}.txt').write_text(
            '\n'.join(mot_lines) + ('\n' if mot_lines else ''))

        gt_pose_payload = {
            'images': evaluated,
            'annotations': [
                ann for ann in gt.get('annotations', [])
                if int(ann['image_id']) in seen
            ],
            'categories': gt.get('categories', []),
        }
        (gt_pose_out / f'{seq}.json').write_text(json.dumps(gt_pose_payload))
        _write_mot_subset(
            seq, images, evaluated, Path(gt_mot_dir), gt_mot_out)

    return pose_dir, mot_dir, gt_pose_out, gt_mot_out, synthetic


def _write_mot_subset(
    seq: str,
    official_images: List[dict],
    evaluated_images: List[dict],
    gt_mot_dir: Path,
    out_dir: Path,
) -> None:
    """Copy ``image_info.json`` and ``gt/gt.txt``, renumbering frames.

    ``frame_index`` in the official file is 1-based and matches the first
    column of ``gt.txt``. The subset numbers frames from 1 in official
    image order so predictions, which use that same order, line up.
    """
    src = Path(gt_mot_dir) / seq
    info_path = src / 'image_info.json'
    gt_path = src / 'gt' / 'gt.txt'
    if not info_path.is_file() or not gt_path.is_file():
        raise FileNotFoundError(
            f'Official PoseTrack21 MOT files missing for {seq} under '
            f'{src}. Required by --official-posetrack21.')
    info = _load_json(info_path)
    by_id = {int(row['id']): row for row in info}
    # Official pose image id -> original 1-based frame_index.
    old_index = {}
    for img in official_images:
        row = by_id.get(int(img['id']))
        if row is None:
            continue
        old_index[int(img['id'])] = int(row['frame_index'])
    new_of_old = {}
    new_info = []
    for new_i, img in enumerate(evaluated_images, start=1):
        row = dict(by_id[int(img['id'])])
        new_of_old[old_index[int(img['id'])]] = new_i
        row['frame_index'] = new_i
        new_info.append(row)
    seq_out = out_dir / seq
    (seq_out / 'gt').mkdir(parents=True, exist_ok=True)
    (seq_out / 'image_info.json').write_text(json.dumps(new_info))
    kept = []
    with open(gt_path, 'r') as f:
        for row in csv.reader(f):
            if not row:
                continue
            old = int(float(row[0]))
            if old not in new_of_old:
                continue
            row[0] = str(new_of_old[old])
            kept.append(','.join(row))
    (seq_out / 'gt' / 'gt.txt').write_text(
        '\n'.join(kept) + ('\n' if kept else ''))


def _tracker_without_offsets(mot_dir: Path, seqs: Sequence[str],
                             dest: Path) -> Path:
    """Box HOTA reads boxes on the GT pixel grid. Undo the +1 offsets."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for seq in seqs:
        src = mot_dir / f'{seq}.txt'
        lines = []
        if src.is_file():
            for line in src.read_text().splitlines():
                if not line.strip():
                    continue
                row = line.split(',')
                vals = [float(v) - 1 for v in row[2:6]]
                lines.append(','.join(
                    [row[0], row[1]] + [f'{v:.2f}' for v in vals]
                    + [row[6], '1', '-1', '-1']))
        (dest / f'{seq}.txt').write_text(
            '\n'.join(lines) + ('\n' if lines else ''))
    return dest


def keypoint_hota(pose_dir: Path, gt_dir: Path, eval_root: Path,
                  num_cores: int) -> dict:
    api = _import_posetrack21(eval_root)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        evaluator = api.get_api(
            trackers_folder=str(pose_dir), gt_folder=str(gt_dir),
            eval_type='tracking', use_parallel=num_cores > 1,
            num_parallel_cores=max(num_cores, 1),
            OUTPUT_PAPER_SUMMARY=False, PRINT_CONFIG=False)
        res = evaluator.eval()
    out = {}
    for key in HOTA_FIELDS:
        if key not in res:
            continue
        arr = np.asarray(res[key], dtype=np.float64)
        value = float(np.mean(arr[:, -1] if arr.ndim == 2 else arr))
        out[f'kp_{key}'] = 100.0 * value
    return out


def keypoint_map(pose_dir: Path, gt_dir: Path, eval_root: Path,
                 num_cores: int) -> dict:
    api = _import_posetrack21(eval_root)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        evaluator = api.get_api(
            trackers_folder=str(pose_dir), gt_folder=str(gt_dir),
            eval_type='pose_estim', use_parallel=num_cores > 1,
            num_parallel_cores=max(num_cores, 1),
            OUTPUT_PAPER_SUMMARY=False, PRINT_CONFIG=False)
        res = evaluator.eval()
    pose_map = res['map']
    return {f'kp_mAP_{part}': float(pose_map[part]) for part in POSE_PARTS}


def _import_poseval():
    """``poseval.evaluateTracking``. Its MOTA accumulator imports ``motmetrics``.

    The mmpose env often has neither package. The fallback is the known
    poseval checkout, and the kit's own ``motmetrics`` (it still provides
    ``MOTAccumulator``) is placed on the path so that import succeeds.
    """
    ensure_numpy_aliases()

    def _load():
        from poseval import eval_helpers
        from poseval.evaluateTracking import evaluateTracking
        return eval_helpers, evaluateTracking

    try:
        return _load()
    except ImportError:
        pass
    try:
        import_kit_motmetrics(resolve_eval_root(None))
    except ImportError:
        pass
    if _FALLBACK_POSEVAL.is_dir() and str(_FALLBACK_POSEVAL) not in sys.path:
        sys.path.insert(0, str(_FALLBACK_POSEVAL))
    try:
        return _load()
    except ImportError as exc:
        raise ImportError(
            'poseval.evaluateTracking is required for official PoseTrack21 '
            'keypoint MOTA (--official-posetrack21). Install the PoseTrack '
            'poseval package (it imports motmetrics).') from exc


def keypoint_mota(pose_dir: Path, gt_dir: Path) -> dict:
    eval_helpers, evaluateTracking = _import_poseval()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gt_frames, pr_frames = eval_helpers.load_data_dir(
            ['', f'{gt_dir}/', f'{pose_dir}/'])
        with tempfile.TemporaryDirectory(prefix='pt21_poseval_') as tmp:
            metrics = evaluateTracking(
                gt_frames, pr_frames, tmp, False, False)
    n = eval_helpers.Joint().count
    return {
        'kp_MOTA': float(metrics['mota'][0, n]),
        'kp_MOTP': float(metrics['motp'][0, n]),
    }


def _load_gt_mot_sequence(seq_dir: Path):
    """Per-frame ``{gt_id: xyxy}`` and ignore polygons, kit frame order."""
    polygon, _ = _shapely()
    info = _load_json(seq_dir / 'image_info.json')
    boxes = {int(row['frame_index']): {} for row in info}
    gt_file = seq_dir / 'gt' / 'gt.txt'
    with open(gt_file, 'r') as f:
        for row in csv.reader(f):
            if not row or int(float(row[6])) != 1:
                continue
            x1 = float(row[2])
            y1 = float(row[3])
            x2 = x1 + float(row[4])
            y2 = y1 + float(row[5])
            boxes[int(float(row[0]))][int(float(row[1]))] = np.array(
                [x1, y1, x2, y2], dtype=np.float64)
    frames, ignores = [], []
    for row in info:
        frames.append(boxes[int(row['frame_index'])])
        ignores.append(_ignore_polygons(
            row.get('ignore_regions_x') or [],
            row.get('ignore_regions_y') or [], polygon))
    return frames, ignores


def _ignore_polygons(xs, ys, polygon_cls) -> list:
    if len(xs) > 0 and not isinstance(xs[0], list):
        xs, ys = [xs], [ys]
    polys = []
    for rx, ry in zip(xs, ys):
        if len(rx) == 0 or len(ry) == 0:
            continue
        polys.append(polygon_cls(list(zip(rx, ry))))
    return polys


def _read_mot_file(path: Path) -> Dict[int, Dict[int, List[float]]]:
    """``{track_id: {0-based frame: xyxy}}``, undoing the +1 MOT text."""
    results: Dict[int, Dict[int, List[float]]] = {}
    if not path.is_file():
        return results
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = line.split(',')
        frame = int(float(row[0])) - 1
        tid = int(float(row[1])) - 1
        x1 = float(row[2]) - 1
        y1 = float(row[3]) - 1
        w = float(row[4]) - 1
        h = float(row[5]) - 1
        results.setdefault(tid, {})[frame] = [x1, y1, x1 + w, y1 + h]
    return results


def _sequence_accumulator(mm, results, frames, ignores, rule: IgnoreRule):
    acc = mm.MOTAccumulator(auto_id=True)
    for i, gt in enumerate(frames):
        if rule.skip_frames_without_gt and len(gt) == 0:
            continue
        gt_ids = list(gt.keys())
        if gt_ids:
            gt_boxes = np.stack([gt[k] for k in gt_ids])
            gt_xywh = np.concatenate(
                [gt_boxes[:, :2], gt_boxes[:, 2:] - gt_boxes[:, :2]], axis=1)
        else:
            gt_xywh = np.zeros((0, 4))
        track_ids, track_boxes = [], []
        for tid, per_frame in results.items():
            if i in per_frame:
                track_ids.append(tid)
                track_boxes.append(per_frame[i][:4])
        ignore = []
        if ignores[i] and track_boxes:
            ignore = ignore_candidate_indices(
                np.asarray(track_boxes, dtype=np.float64), ignores[i], rule)
        if track_boxes:
            tb = np.asarray(track_boxes, dtype=np.float64)
            tr_xywh = np.concatenate(
                [tb[:, :2], tb[:, 2:] - tb[:, :2]], axis=1)
        else:
            tr_xywh = np.zeros((0, 4))
        dist = mm.distances.iou_matrix(gt_xywh, tr_xywh, max_iou=0.5)
        acc.update(gt_ids, track_ids, dist, ignore_candidates=ignore)
    return acc


def box_clear(mot_dir: Path, gt_mot_dir: Path, eval_root: Path,
              variants: Iterable[str] = ('official', 'fixed')) -> dict:
    """Box MOTA / IDF1. Values are percentages except the raw counts."""
    mm = import_kit_motmetrics(eval_root)
    seqs = sorted(
        p.name for p in Path(gt_mot_dir).iterdir()
        if (p / 'gt' / 'gt.txt').is_file())
    gts = {seq: _load_gt_mot_sequence(Path(gt_mot_dir) / seq) for seq in seqs}
    preds = {seq: _read_mot_file(Path(mot_dir) / f'{seq}.txt') for seq in seqs}
    out = {}
    for name in variants:
        rule = VARIANTS[name]
        accs = [
            _sequence_accumulator(mm, preds[seq], *gts[seq], rule)
            for seq in seqs
        ]
        summary = mm.metrics.create().compute_many(
            accs,
            metrics=[
                'mota', 'motp', 'idf1', 'num_false_positives',
                'num_misses', 'num_switches',
            ],
            names=seqs, generate_overall=True)
        row = summary.loc['OVERALL']
        out[f'box_MOTA_{name}'] = 100.0 * float(row['mota'])
        out[f'box_IDF1_{name}'] = 100.0 * float(row['idf1'])
        out[f'box_FP_{name}'] = int(row['num_false_positives'])
        out[f'box_FN_{name}'] = int(row['num_misses'])
        out[f'box_IDSW_{name}'] = int(row['num_switches'])
    return out


def _fixed_preprocessed(self, raw_data, cls):
    """``PoseTrackMOT.get_preprocessed_seq_data`` with a swappable ignore rule.

    The match (IoU >= 0.5) and the "matched boxes are kept" rule are the
    kit's. ``self.rule`` changes only which unmatched boxes are dropped
    and whether a frame with no ground truth is scored.
    """
    from posetrack21.trackeval.utils import TrackEvalException

    self._check_unique_ids(raw_data)
    data_keys = [
        'gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets',
        'tracker_confidences', 'similarity_scores',
    ]
    data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
    unique_gt_ids: List[int] = []
    unique_tracker_ids: List[int] = []
    num_gt_dets = 0
    num_tracker_dets = 0
    ignore_regions = self.seq_ignore_regions[raw_data['seq']]
    frame_keys = list(ignore_regions.keys())
    rule: IgnoreRule = self.rule

    for t in range(raw_data['num_timesteps']):
        gt_ids = raw_data['gt_ids'][t]
        gt_dets = raw_data['gt_dets'][t]
        tracker_ids = raw_data['tracker_ids'][t]
        tracker_dets = raw_data['tracker_dets'][t]
        tracker_classes = raw_data['tracker_classes'][t]
        tracker_confidences = raw_data['tracker_confidences'][t]
        similarity_scores = raw_data['similarity_scores'][t]

        if len(tracker_classes) > 0 and np.max(tracker_classes) > 1:
            raise TrackEvalException(
                'Evaluation is only valid for the pedestrian class. '
                f'Class {int(np.max(tracker_classes))} found in '
                f'{raw_data["seq"]} at timestep {t}.')

        to_remove = np.array([], dtype=int)
        if rule.skip_frames_without_gt and len(gt_ids) == 0:
            to_remove = np.arange(len(tracker_ids))
        elif len(gt_ids) > 0 and len(tracker_ids) > 0:
            scores = similarity_scores.copy()
            scores[scores < 0.5 - _EPS] = 0
            rows, cols = linear_sum_assignment(-scores)
            matched_cols = cols[scores[rows, cols] > _EPS]
            regions = ignore_regions[frame_keys[t]]
            unmatched = [
                j for j in range(len(tracker_ids)) if j not in set(matched_cols)
            ]
            if regions and unmatched:
                remove = []
                for j in unmatched:
                    det = np.atleast_2d(tracker_dets[j])[0]
                    x1, y1, w, h = [float(v) for v in det[:4]]
                    poly = _box_polygon((x1, y1, x1 + w, y1 + h))
                    if max(region_overlap(region, poly, rule.overlap)
                           for region in regions) > rule.threshold:
                        remove.append(j)
                to_remove = np.asarray(remove, dtype=int)

        data['tracker_ids'][t] = np.delete(tracker_ids, to_remove, axis=0)
        data['tracker_dets'][t] = np.delete(tracker_dets, to_remove, axis=0)
        data['tracker_confidences'][t] = np.delete(
            tracker_confidences, to_remove, axis=0)
        data['similarity_scores'][t] = np.delete(
            similarity_scores, to_remove, axis=1)
        data['gt_ids'][t] = gt_ids
        data['gt_dets'][t] = gt_dets
        unique_gt_ids += list(np.unique(gt_ids))
        unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
        num_tracker_dets += len(data['tracker_ids'][t])
        num_gt_dets += len(gt_ids)

    for key, uniq in (('gt_ids', unique_gt_ids),
                      ('tracker_ids', unique_tracker_ids)):
        if len(uniq) > 0:
            uniq_arr = np.unique(uniq)
            id_map = np.nan * np.ones(int(np.max(uniq_arr)) + 1)
            id_map[uniq_arr] = np.arange(len(uniq_arr))
            for t in range(raw_data['num_timesteps']):
                if len(data[key][t]) > 0:
                    data[key][t] = id_map[data[key][t]].astype(int)
            if key == 'gt_ids':
                unique_gt_ids = list(uniq_arr)
            else:
                unique_tracker_ids = list(uniq_arr)
    data['num_tracker_dets'] = num_tracker_dets
    data['num_gt_dets'] = num_gt_dets
    data['num_tracker_ids'] = len(unique_tracker_ids)
    data['num_gt_ids'] = len(unique_gt_ids)
    data['num_timesteps'] = raw_data['num_timesteps']
    data['seq'] = raw_data['seq']
    self._check_unique_ids(data, after_preproc=True)
    return data


def fixed_posetrack_mot_class(eval_root: Path):
    """``PoseTrackMOT`` subclass whose ignore step follows ``self.rule``.

    Stored on this module so a forked worker can unpickle it.
    """
    cached = getattr(fixed_posetrack_mot_class, 'cls', None)
    if cached is not None:
        return cached
    ensure_numpy_aliases()
    pkg_parent = str(Path(eval_root) / 'posetrack21')
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)
    from posetrack21.trackeval.datasets.posetrack_mot import PoseTrackMOT

    class FixedPoseTrackMOT(PoseTrackMOT):
        rule = VARIANTS['official']

        def get_preprocessed_seq_data(self, raw_data, cls):
            return _fixed_preprocessed(self, raw_data, cls)

    FixedPoseTrackMOT.__module__ = __name__
    FixedPoseTrackMOT.__name__ = 'FixedPoseTrackMOT'
    FixedPoseTrackMOT.__qualname__ = 'FixedPoseTrackMOT'
    globals()['FixedPoseTrackMOT'] = FixedPoseTrackMOT
    fixed_posetrack_mot_class.cls = FixedPoseTrackMOT
    return FixedPoseTrackMOT


def box_hota(mot_dir: Path, gt_mot_dir: Path, eval_root: Path,
             num_cores: int,
             variants: Iterable[str] = ('official', 'fixed')) -> dict:
    """Box HOTA. Sub-metrics are percentages, averaged over alpha."""
    ensure_numpy_aliases()
    try:
        import multiprocessing as mp
        mp.set_start_method('fork')
    except RuntimeError:
        pass
    pkg_parent = str(Path(eval_root) / 'posetrack21')
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)
    import posetrack21.trackeval as trackeval

    seqs = sorted(
        p.name for p in Path(gt_mot_dir).iterdir()
        if (p / 'gt' / 'gt.txt').is_file())
    trackers = _tracker_without_offsets(
        Path(mot_dir), seqs, Path(mot_dir).parent / 'mot_hota')
    eval_config = trackeval.EvaluatorMOT.get_default_eval_config()
    # Parallel workers must unpickle FixedPoseTrackMOT. Fork inherits the
    # class we just installed on this module; spawn would not.
    use_parallel = num_cores > 1 and mp.get_start_method(allow_none=True) == 'fork'
    eval_config.update({
        'USE_PARALLEL': use_parallel,
        'NUM_PARALLEL_CORES': max(num_cores, 1),
        'PRINT_RESULTS': False,
        'PRINT_ONLY_COMBINED': True,
        'OUTPUT_DETAILED': False,
        'PRINT_PAPER_SUMMARY': False,
        'OUTPUT_PAPER_SUMMARY': False,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': False,
        'OUTPUT_SUMMARY': False,
        'PLOT_CURVES': False,
    })
    defaults = trackeval.EvaluatorMOT.get_default_eval_config()
    eval_config = {k: v for k, v in eval_config.items() if k in defaults}
    ds_config = {
        'GT_FOLDER': str(gt_mot_dir),
        'TRACKERS_FOLDER': str(trackers),
        'PRINT_CONFIG': False,
        'ASSURE_SINGLE_TRACKER': True,
    }
    cls = fixed_posetrack_mot_class(eval_root)
    out = {}
    for name in variants:
        dataset = cls(ds_config)
        dataset.rule = VARIANTS[name]
        summary_dir = Path(dataset.get_output_fol(dataset.tracker_list[0]))
        summary_dir.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                res, _ = trackeval.EvaluatorMOT(eval_config).evaluate(
                    [dataset], [trackeval.metrics.HOTA()])
        finally:
            if summary_dir.resolve() != trackers.resolve():
                shutil.rmtree(summary_dir, ignore_errors=True)
        tracker = list(res[dataset.get_name()].keys())[0]
        hota = res[dataset.get_name()][tracker]['COMBINED_SEQ']['pedestrian']['HOTA']
        for key in HOTA_FIELDS:
            arr = np.asarray(hota[key], dtype=np.float64)
            value = float(np.mean(arr if arr.ndim == 1 else arr))
            out[f'box_{key}_{name}'] = 100.0 * value
    return out
