# Copyright (c) OpenMMLab. All rights reserved.
"""Post-processor that tracks pose-estimator boxes with BoxMOT.

    Broström, "BoxMOT: Pluggable SOTA tracking modules"
    https://github.com/mikel-brostrom/boxmot

This is a thin wrapper, not a port.  Association, Kalman filters, ReID
crops and camera-motion compensation all run inside the installed
``boxmot`` package; this filter only translates ``PoseDataSample`` boxes
into BoxMOT's ``(N, 6)`` detection layout, calls ``tracker.update``, and
writes the returned ids back onto the original pose instances via the
``det_ind`` column.

Fidelity notes
--------------
* **OK - association.**  Unmodified BoxMOT Python backend, selected by
  the ``tracker`` config key (``bytetrack``, ``ocsort``, ``botsort``,
  ...).  Defaults come from BoxMOT's packaged YAML; ``tracker_kwargs``
  overrides individual parameters.
* **Integration point, not a substitution - detections.**  Boxes and
  scores come from the prediction bundle (whatever detector + pose
  estimator produced them), not from a BoxMOT-bundled YOLO.  Keypoints
  are carried through and never re-estimated.
* **Dropped by default - Kalman-only tracks.**  BoxMOT may emit a row
  with ``det_ind < 0`` for a predicted box that has no matching
  detection this frame.  There is no pose to attach, so those rows are
  discarded unless ``emit_predicted_tracks=True`` (which emits the
  predicted box with zeroed keypoints).
* **Dummy image for motion-only trackers.**  ``bytetrack`` / ``ocsort``
  / ``sfsort`` only read ``img.shape``.  When ``requires_images`` is
  false a broadcast dummy of ``ori_shape`` is passed and no image I/O
  is performed.  Trackers that enable ReID or CMC (``use_cmc=True``)
  set ``requires_images=True`` and need ``ds.img``.
* **frame_rate.**  BoxMOT defaults to 30 fps for buffer scaling.
  PoseTrack21 clips are ~25 fps; that is left at the packaged default
  rather than silently retuned.  Override with
  ``tracker_kwargs=dict(frame_rate=25)``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mmengine.structures import InstanceData

from mmpose.structures import PoseDataSample

from ..base import BaseFilter
from ..registry import POST_PROCESS_FILTERS

_INSTALL_HINT = (
    'BoxMOT is not installed. Install it into this environment with:\n'
    '  pip install -r requirements/boxmot.txt\n'
    '  pip install --no-deps boxmot==22.0.0\n'
    'A plain `pip install boxmot` upgrades numpy to 2.x and breaks mmcv.'
)

# Trackers whose published pipeline always reads pixels, even when the
# YAML has no use_cmc / with_reid flags.
_ALWAYS_NEEDS_IMAGES = frozenset({'sam2mot'})


def _import_boxmot():
    """Import BoxMOT lazily so the rest of post-processing still loads."""
    try:
        import logging

        # BoxMOT logs a full parameter banner on every tracker construct.
        # We rebuild at each sequence boundary, so leave that at WARNING.
        logging.getLogger('boxmot').setLevel(logging.WARNING)
        from boxmot.trackers.registry import (
            create_tracker,
            get_tracker_definition,
        )
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return create_tracker, get_tracker_definition


def _as_xyxy(bboxes: np.ndarray, n: int) -> np.ndarray:
    """Flatten stored boxes to ``(N, 4)`` xyxy."""
    arr = np.asarray(bboxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((n, 4), dtype=np.float32)
    arr = arr.reshape(n, -1)
    if arr.shape[1] < 4:
        raise ValueError(
            f'Expected xyxy boxes with 4 columns, got shape {arr.shape}')
    return arr[:, :4]


def _boxes_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    kpts = np.asarray(keypoints, dtype=np.float32)
    if kpts.ndim != 3 or kpts.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x1 = np.min(kpts[..., 0], axis=1)
    y1 = np.min(kpts[..., 1], axis=1)
    x2 = np.max(kpts[..., 0], axis=1)
    y2 = np.max(kpts[..., 1], axis=1)
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


@POST_PROCESS_FILTERS.register_module()
class BoxMOTTracker(BaseFilter):
    """Online bbox tracker wrapping a swappable BoxMOT implementation.

    Args:
        tracker: BoxMOT tracker name (``bytetrack``, ``ocsort``,
            ``botsort``, ``strongsort``, ``deepocsort``, ``boosttrack``,
            ``occluboost``, ``hybridsort``, ``sfsort``, ``sam2mot``).
        tracker_config: Optional path to a BoxMOT YAML.  ``None`` uses
            the packaged default for ``tracker``.
        tracker_kwargs: Parameter overrides applied after YAML defaults
            (e.g. ``track_thresh``, ``track_buffer``, ``frame_rate``,
            ``with_reid``, ``use_cmc``).
        reid_weights: Path or filename of a BoxMOT ReID checkpoint.
            Unused for motion-only trackers.  If the file is missing
            and the name is in BoxMOT's zoo, BoxMOT downloads it.
        device: Device for the ReID backend (``'cuda:0'``, ``'cpu'``).
        half: Run the ReID backend in FP16.
        min_bbox_score: Detections below this score are not passed to
            BoxMOT.  They are also omitted from the output (even with
            ``keep_untracked=True``).
        score_source: ``'bbox'`` uses ``bbox_scores``; ``'keypoint_mean'``
            uses the per-instance mean of ``keypoint_scores``.
        keep_untracked: If True, detections that BoxMOT did not return
            (unconfirmed / unmatched) are emitted with freshly allocated
            track ids rather than dropped.
        emit_predicted_tracks: If True, BoxMOT rows with ``det_ind < 0``
            (Kalman-only, no matching detection) are emitted as extra
            instances with zeroed keypoints.  Default False: drop them.
        requires_images: ``None`` infers from the tracker (ReID, CMC, or
            ``sam2mot``).  Set True/False to override.
    """

    online = True

    def __init__(
        self,
        tracker: str = 'bytetrack',
        tracker_config: Optional[str] = None,
        tracker_kwargs: Optional[Dict[str, Any]] = None,
        reid_weights: Optional[str] = None,
        device: str = 'cuda:0',
        half: bool = False,
        min_bbox_score: float = 0.0,
        score_source: str = 'bbox',
        keep_untracked: bool = False,
        emit_predicted_tracks: bool = False,
        requires_images: Optional[bool] = None,
    ) -> None:
        if score_source not in ('bbox', 'keypoint_mean'):
            raise ValueError(
                f"score_source must be 'bbox' or 'keypoint_mean', "
                f'got {score_source!r}')

        self.tracker_name = str(tracker).strip().lower()
        self.tracker_config = tracker_config
        self.tracker_kwargs = dict(tracker_kwargs) if tracker_kwargs else {}
        self.reid_weights = reid_weights
        self.device = device
        self.half = bool(half)
        self.min_bbox_score = float(min_bbox_score)
        self.score_source = score_source
        self.keep_untracked = bool(keep_untracked)
        self.emit_predicted_tracks = bool(emit_predicted_tracks)

        self._reid_model = None
        self._id_map: Dict[int, int] = {}
        self._next_id = 1
        self._warned_missing_image = False
        self._dummy_key: Optional[Tuple[int, int]] = None
        self._dummy_img: Optional[np.ndarray] = None

        self.tracker = self._build_tracker()
        if requires_images is None:
            self.requires_images = self._infer_requires_images()
        else:
            self.requires_images = bool(requires_images)

    # ------------------------------------------------------------------
    # BoxMOT construction
    # ------------------------------------------------------------------

    def _build_tracker(self):
        create_tracker, get_tracker_definition = _import_boxmot()
        definition = get_tracker_definition(self.tracker_name)
        kwargs: Dict[str, Any] = dict(
            tracker_type=self.tracker_name,
            tracker_config=self.tracker_config,
            device=self.device,
            half=self.half,
            per_class=False,
            tracker_backend='python',
        )
        if self.tracker_kwargs:
            kwargs['tracker_kwargs'] = dict(self.tracker_kwargs)
        if definition.needs_reid:
            if self._reid_model is not None:
                kwargs['reid_model'] = self._reid_model
            elif self.reid_weights:
                kwargs['reid_weights'] = self.reid_weights
        tracker = create_tracker(**kwargs)
        if definition.needs_reid and self._reid_model is None:
            model = getattr(tracker, 'model', None)
            if model is not None:
                self._reid_model = model
        return tracker

    def _infer_requires_images(self) -> bool:
        if self.tracker_name in _ALWAYS_NEEDS_IMAGES:
            return True
        use_cmc = bool(getattr(self.tracker, 'use_cmc', False))
        cmc = getattr(self.tracker, 'cmc', None)
        has_cmc = use_cmc or (cmc is not None and getattr(cmc, 'enabled', True))
        with_reid = bool(getattr(self.tracker, 'with_reid', False))
        has_model = getattr(self.tracker, 'model', None) is not None
        return bool(has_cmc or with_reid or has_model)

    def reset(self) -> None:
        """Rebuild the BoxMOT tracker and clear per-sequence id maps.

        BoxMOT's own ``reset()`` does not clear every tracker-specific
        list (e.g. ByteTrack ``lost_stracks``), so a fresh instance is
        safer at sequence boundaries.  The ReID backend is reused.
        """
        self.tracker = self._build_tracker()
        self._id_map.clear()
        self._next_id = 1
        self._warned_missing_image = False

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        ds: PoseDataSample,
        seq_key: str,
    ) -> PoseDataSample:
        img = self._resolve_image(ds)
        instances = ds.pred_instances
        if instances is None or len(instances) == 0:
            self.tracker.update(
                np.zeros((0, 6), dtype=np.float32), img)
            return self._empty_output(ds)

        n = len(instances)
        bboxes, scores = self._read_boxes_scores(instances, n)
        passed = scores >= self.min_bbox_score
        src_idx = np.flatnonzero(passed).astype(np.int64)

        if src_idx.size == 0:
            dets = np.zeros((0, 6), dtype=np.float32)
        else:
            dets = np.concatenate(
                [
                    bboxes[src_idx],
                    scores[src_idx, None],
                    np.zeros((src_idx.size, 1), dtype=np.float32),
                ],
                axis=1,
            ).astype(np.float32, copy=False)

        res = self.tracker.update(dets, img)
        emitted, pred_rows = self._match_results(res, src_idx, n)
        return self._build_output(ds, emitted, pred_rows)

    def _match_results(
        self,
        res,
        src_idx: np.ndarray,
        n_orig: int,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[np.ndarray, int, float]]]:
        emitted: List[Tuple[int, int]] = []
        pred_rows: List[Tuple[np.ndarray, int, float]] = []

        if res is None or len(res) == 0:
            if self.keep_untracked:
                for i in src_idx.tolist():
                    emitted.append((int(i), self._alloc_local_id()))
            return emitted, pred_rows

        det_ind = np.asarray(res.det_ind, dtype=np.int32)
        track_ids = np.asarray(res.id, dtype=np.int32)
        xyxy = np.asarray(res.xyxy, dtype=np.float32)
        confs = np.asarray(res.conf, dtype=np.float32)

        claimed: set = set()
        n_in = int(src_idx.size)
        for row in range(len(res)):
            di = int(det_ind[row])
            local_id = self._map_id(int(track_ids[row]))
            if di < 0 or di >= n_in:
                if self.emit_predicted_tracks:
                    pred_rows.append((xyxy[row].copy(), local_id, float(confs[row])))
                continue
            orig = int(src_idx[di])
            if orig in claimed:
                continue
            claimed.add(orig)
            emitted.append((orig, local_id))

        if self.keep_untracked:
            passed = set(int(i) for i in src_idx.tolist())
            for i in range(n_orig):
                if i in claimed:
                    continue
                if i not in passed:
                    continue
                emitted.append((i, self._alloc_local_id()))

        return emitted, pred_rows

    def _map_id(self, boxmot_id: int) -> int:
        mapped = self._id_map.get(boxmot_id)
        if mapped is None:
            mapped = self._next_id
            self._id_map[boxmot_id] = mapped
            self._next_id += 1
        return mapped

    def _alloc_local_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    # ------------------------------------------------------------------
    # Inputs / outputs
    # ------------------------------------------------------------------

    def _read_boxes_scores(
        self,
        instances,
        n: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(instances, 'bboxes') and instances.bboxes is not None:
            bboxes = _as_xyxy(instances.bboxes, n)
        elif hasattr(instances, 'keypoints') and instances.keypoints is not None:
            bboxes = _boxes_from_keypoints(instances.keypoints)
        else:
            bboxes = np.zeros((n, 4), dtype=np.float32)

        if self.score_source == 'keypoint_mean':
            if (hasattr(instances, 'keypoint_scores')
                    and instances.keypoint_scores is not None):
                ks = np.asarray(instances.keypoint_scores, dtype=np.float32)
                scores = np.nanmean(ks.reshape(n, -1), axis=1)
            else:
                scores = np.ones(n, dtype=np.float32)
        elif hasattr(instances, 'bbox_scores') and instances.bbox_scores is not None:
            scores = np.asarray(instances.bbox_scores, dtype=np.float32).reshape(-1)
            if scores.shape[0] != n:
                scores = np.resize(scores, n)
        elif (hasattr(instances, 'keypoint_scores')
              and instances.keypoint_scores is not None):
            ks = np.asarray(instances.keypoint_scores, dtype=np.float32)
            scores = np.nanmean(ks.reshape(n, -1), axis=1)
        else:
            scores = np.ones(n, dtype=np.float32)
        return bboxes, scores.astype(np.float32, copy=False)

    def _resolve_image(self, ds: PoseDataSample) -> np.ndarray:
        img = ds.get('img', None)
        if img is not None:
            return np.asarray(img)
        ori = ds.metainfo.get('ori_shape', (1, 1))
        dummy = self._dummy_image(ori)
        if self.requires_images:
            if not self._warned_missing_image:
                print(
                    'Warning: BoxMOTTracker requires frame images but a '
                    'frame arrived without pixels; using a blank dummy. '
                    'Appearance matching and camera-motion compensation '
                    'are inactive for such frames.')
                self._warned_missing_image = True
            h, w = dummy.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)
        return dummy

    def _dummy_image(self, ori_shape: Sequence[int]) -> np.ndarray:
        h = int(ori_shape[0]) if len(ori_shape) >= 1 else 1
        w = int(ori_shape[1]) if len(ori_shape) >= 2 else 1
        h = max(h, 1)
        w = max(w, 1)
        key = (h, w)
        if self._dummy_key != key or self._dummy_img is None:
            self._dummy_key = key
            self._dummy_img = np.broadcast_to(
                np.zeros((1, 1, 3), dtype=np.uint8), (h, w, 3))
        return self._dummy_img

    @staticmethod
    def _empty_output(ds: PoseDataSample) -> PoseDataSample:
        new_ds = ds.new()
        new_ds.set_metainfo(ds.metainfo)
        if hasattr(ds, 'gt_instances'):
            new_ds.gt_instances = ds.gt_instances
        if ds.pred_instances is None:
            return new_ds
        empty = deepcopy(ds.pred_instances[np.zeros(0, dtype=np.int64)])
        empty.track_ids = np.zeros(0, dtype=np.int32)
        new_ds.pred_instances = empty
        return new_ds

    def _build_output(
        self,
        ds: PoseDataSample,
        emitted: List[Tuple[int, int]],
        pred_rows: List[Tuple[np.ndarray, int, float]],
    ) -> PoseDataSample:
        new_ds = ds.new()
        new_ds.set_metainfo(ds.metainfo)
        if hasattr(ds, 'gt_instances'):
            new_ds.gt_instances = ds.gt_instances

        if ds.pred_instances is None:
            return new_ds

        parts: List[InstanceData] = []
        if emitted:
            emitted_sorted = sorted(emitted, key=lambda e: e[1])
            keep = np.array([e[0] for e in emitted_sorted], dtype=np.int64)
            ids = np.array([e[1] for e in emitted_sorted], dtype=np.int32)
            inst = deepcopy(ds.pred_instances[keep])
            inst.track_ids = ids
            parts.append(inst)

        if pred_rows:
            k = 17
            if len(ds.pred_instances) > 0:
                k = int(np.asarray(ds.pred_instances.keypoints).shape[1])
            pred_sorted = sorted(pred_rows, key=lambda r: r[1])
            extra = InstanceData()
            extra.keypoints = np.zeros(
                (len(pred_sorted), k, 2), dtype=np.float32)
            extra.keypoint_scores = np.zeros(
                (len(pred_sorted), k), dtype=np.float32)
            extra.bboxes = np.stack(
                [r[0] for r in pred_sorted]).astype(np.float32)
            extra.bbox_scores = np.array(
                [r[2] for r in pred_sorted], dtype=np.float32)
            extra.track_ids = np.array(
                [r[1] for r in pred_sorted], dtype=np.int32)
            parts.append(extra)

        if not parts:
            empty = deepcopy(ds.pred_instances[np.zeros(0, dtype=np.int64)])
            empty.track_ids = np.zeros(0, dtype=np.int32)
            new_ds.pred_instances = empty
            return new_ds

        if len(parts) == 1:
            new_ds.pred_instances = parts[0]
        else:
            new_ds.pred_instances = InstanceData.cat(parts)
        return new_ds
