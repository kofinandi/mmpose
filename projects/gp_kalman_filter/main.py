#!/usr/bin/env python3
"""
GP-Kalman Filter — entry point
================================
Edit the configuration block below, then run:

    python main.py

Navigation: ← / → arrow keys  or  Prev / Next buttons.
"""

import matplotlib.pyplot as plt

from filter import KEYPOINT_NAMES, load_sequence, run_filter
from viewer import StepViewer

# ── Configuration ──────────────────────────────────────────────────────────────

FRAMES_JSON = (
    "/Users/kofinandi/Documents/ETH/Thesis/mmpose/benchmark/predictions"
    "/20260622_emdb_topdown/ViTPose-small-rfdetr/frames.json"
)
SEQUENCE  = "P1/14_outdoor_climb"

JOINT_IDX = 5       # 0-based COCO joint index  (5 = left_shoulder)
COORD     = "x"     # "x"  or  "y"

WINDOW_SIZE = 60    # sliding-window length N
N_FORECAST  = 20    # frames to project forward after current step

# Heatmap-to-pixel scale for the measurement-noise model:
#   R = (PIXEL_SCALE / (score · √(2π)))²
PIXEL_SCALE = 1.0

# GP kernel hyper-parameters (fixed – not optimised for speed)
GP_LENGTH_SCALE = 15.0    # RBF length-scale in frames
GP_SIGNAL_VAR   = 2000.0  # GP prior signal variance  [px²]

# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    frame_ids, meas, scores, gt_pos, gt_vis = load_sequence(
        FRAMES_JSON, SEQUENCE, JOINT_IDX, COORD
    )

    records = run_filter(
        frame_ids, meas, scores,
        window_size=WINDOW_SIZE,
        n_forecast=N_FORECAST,
        pixel_scale=PIXEL_SCALE,
        gp_length_scale=GP_LENGTH_SCALE,
        gp_signal_var=GP_SIGNAL_VAR,
        precompute_forecast=True,
    )
    print(f"  {len(records)} records ready.\n")

    StepViewer(
        frame_ids, meas, scores, gt_pos, gt_vis, records,
        window_size=WINDOW_SIZE,
        n_forecast=N_FORECAST,
        joint_idx=JOINT_IDX,
        joint_name=KEYPOINT_NAMES[JOINT_IDX],
        coord=COORD,
        sequence=SEQUENCE,
    )
    plt.show()


if __name__ == "__main__":
    main()
