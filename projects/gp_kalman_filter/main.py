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

from filter import WINDOW_SIZE, N_FORECAST

# ── Configuration ──────────────────────────────────────────────────────────────

FRAMES_JSON = (
    "/Users/kofinandi/Documents/ETH/Thesis/mmpose/benchmark/predictions"
    "/20260622_emdb_topdown/ViTPose-small-rfdetr/frames.json"
)
SEQUENCE  = "P1/14_outdoor_climb"

JOINT_IDX = 5       # 0-based COCO joint index  (5 = left_shoulder)
COORD     = "x"     # "x"  or  "y"

# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    frame_ids, meas, scores, gt_pos, gt_vis = load_sequence(
        FRAMES_JSON, SEQUENCE, JOINT_IDX, COORD
    )

    records = run_filter(
        frame_ids, meas, scores,
        precompute_forecast=True,
        verbose=True,
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
