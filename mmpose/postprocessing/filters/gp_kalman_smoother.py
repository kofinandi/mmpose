# Copyright (c) OpenMMLab. All rights reserved.
"""GP-Kalman offline smoother for pose predictions.

Wraps the Local Gaussian Process Regression Filter with Bayesian Fusion
implemented in ``projects/gp_kalman_filter/filter.py``.

Prototype simplifications
--------------------------
* No tracker — always uses prediction instance 0.
* Runs the filter independently on every joint and on x / y separately.
* Offline (needs the full sequence).
"""

from __future__ import annotations

import copy
import os
import sys
from typing import List

import numpy as np

from mmpose.structures import PoseDataSample

from ..base import BaseFilter, sequence_key_from_path
from ..registry import POST_PROCESS_FILTERS

# ---------------------------------------------------------------------------
# Locate and import the GP-Kalman filter module from the projects tree.
# File layout:
#   <repo>/mmpose/postprocessing/filters/  ← this file
#   <repo>/projects/gp_kalman_filter/filter.py
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_GP_FILTER_DIR = os.path.join(_REPO_ROOT, 'projects', 'gp_kalman_filter')
if _GP_FILTER_DIR not in sys.path:
    sys.path.insert(0, _GP_FILTER_DIR)

from filter import run_filter  # noqa: E402  (path added above)


@POST_PROCESS_FILTERS.register_module()
class GPKalmanSmoother(BaseFilter):
    """Offline GP-Kalman smoother applied per-joint per-coordinate.

    Implements the Local Gaussian Process Regression Filter with Bayesian
    Fusion from ``GP_Kalman_Filter_Steps.md``.  Each (joint, coordinate)
    pair is filtered independently.  The filter is re-initialised at every
    sequence boundary.

    Args:
        window_size     : Sliding-window length N (number of past frames
                          the GP is conditioned on).
        pixel_scale     : Scale factor for the measurement-variance model
                          ``R = ((1 - score)² × pixel_scale)``.
        gp_length_scale : RBF kernel length-scale in frames.
        gp_signal_var   : GP prior signal variance used during bootstrap
                          (no history yet).
    """

    online: bool = False

    # process_frame is not used for offline filters, but the ABC requires it.
    def process_frame(
        self, ds: PoseDataSample, seq_key: str
    ) -> PoseDataSample:
        return ds

    # ------------------------------------------------------------------

    def process_sequence(
        self, frames: List[PoseDataSample]
    ) -> List[PoseDataSample]:
        """Apply the GP-Kalman filter to an ordered list of frames.

        Frames spanning multiple sequences are handled automatically: the
        filter state is reset at every sequence boundary.

        Args:
            frames: Prediction-only :class:`PoseDataSample`s in order.

        Returns:
            Deep-copied list with ``pred_instances.keypoints[0]`` replaced
            by the GP-Kalman posterior for each frame that has an instance.
        """
        results = [copy.deepcopy(ds) for ds in frames]

        # ── Identify contiguous sequence spans ────────────────────────────
        # frames are globally ordered; group consecutive frames that share
        # the same sequence key.
        spans: list = []   # list of (seq_key, global_start, global_end)
        current_key: str | None = None
        span_start = 0
        for i, ds in enumerate(frames):
            key = sequence_key_from_path(ds.metainfo.get('img_path', ''))
            if key != current_key:
                if current_key is not None:
                    spans.append((current_key, span_start, i))
                current_key = key
                span_start = i
        if current_key is not None:
            spans.append((current_key, span_start, len(frames)))

        total_seqs = len(spans)
        for seq_i, (seq_key, s, e) in enumerate(spans):
            seq_frames = frames[s:e]
            n_frames = e - s
            print(
                f'  GP-Kalman [{seq_i + 1}/{total_seqs}] '
                f'{seq_key!r}  ({n_frames} frames) …',
                flush=True,
            )

            # Detect number of keypoints from the first frame with instances
            num_kpts = 17
            for ds in seq_frames:
                pi = ds.pred_instances
                if pi.keypoints.shape[0] > 0:
                    num_kpts = pi.keypoints.shape[1]
                    break

            frame_ids = list(range(n_frames))

            # ── Extract measurements and scores per joint ─────────────────
            # meas[j][c]  : list[float | None]  – coordinate c of joint j
            # scores_j[j] : list[float]         – confidence for joint j
            meas   = [[[] for _ in range(2)] for _ in range(num_kpts)]
            sc_buf = [[]                         for _ in range(num_kpts)]
            has_instance: list[bool] = []

            for ds in seq_frames:
                pi = ds.pred_instances
                if pi.keypoints.shape[0] > 0:
                    kpts   = pi.keypoints[0]        # (K, 2)
                    scores = pi.keypoint_scores[0]  # (K,)
                    has_instance.append(True)
                    for j in range(num_kpts):
                        meas[j][0].append(float(kpts[j, 0]))
                        meas[j][1].append(float(kpts[j, 1]))
                        sc_buf[j].append(float(scores[j]))
                else:
                    has_instance.append(False)
                    for j in range(num_kpts):
                        meas[j][0].append(None)
                        meas[j][1].append(None)
                        sc_buf[j].append(0.0)

            # ── Run filter per joint per coordinate ───────────────────────
            filtered = np.zeros((n_frames, num_kpts, 2), dtype=np.float32)

            for j in range(num_kpts):
                for c in range(2):
                    records = run_filter(
                        frame_ids, meas[j][c], sc_buf[j], precompute_forecast=False
                    )
                    for t, rec in enumerate(records):
                        filtered[t, j, c] = float(rec['mu_post'])

            # ── Write filtered keypoints back ─────────────────────────────
            for t in range(n_frames):
                if not has_instance[t]:
                    continue   # no instance to update
                results[s + t].pred_instances.keypoints[0] = filtered[t]

        return results
