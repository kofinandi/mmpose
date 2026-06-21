_base_ = '../../../_base_/default_runtime.py'

# Override train_cfg from default_runtime to allow test-only execution.
train_cfg = None

# RF-DETR Keypoint Preview evaluation on COCO keypoints.
# End-to-end multi-person pose estimation via rfdetr.RFDETRKeypointPreview.
# Docs: https://rfdetr.roboflow.com/
#
# This config is for evaluation only (no training).
# Weights auto-download from Roboflow on first use.
# Requires: pip install "rfdetr>=1.8.0"

# model
model = dict(
    type='RFDETRPoseEstimator',
    pretrain_weights='data/models/rf-detr-keypoint-preview-xlarge.pth',
    conf_thr=0.001,
    model_cache_dir='data/models',
    num_keypoints=17,
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[0, 0, 0],
        std=[1, 1, 1],
        bgr_to_rgb=False,
        pad_size_divisor=1),
)

# data pipeline
data_mode = 'bottomup'
data_root = 'data/'

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction')),
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='coco/val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/annotations/person_keypoints_val2017.json',
    score_mode='bbox',
    nms_mode='none',
)
test_evaluator = val_evaluator
