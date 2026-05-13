_base_ = './petr_r50_16x2_100e_coco.py'

# PETR R-101 evaluation on COCO keypoints.
# Only the backbone depth changes relative to R-50.
# Expected performance: ~70.0 AP on COCO val2017.

model = dict(
    petr_model_cfg=dict(
        backbone=dict(depth=101)))
