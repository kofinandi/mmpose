_base_ = '../../../_base_/default_runtime.py'

# GroupPose (R50) evaluation on COCO keypoints.
# Paper: https://arxiv.org/abs/2308.07313
# Upstream: https://github.com/Michel-liu/GroupPose
#
# Evaluation-only. Expected README AP on COCO val2017: ~72.0.
# Fidelity: pred_instances.bboxes are derived from keypoint min/max
# (upstream PostProcess has no boxes). Requires compiled
# MultiScaleDeformableAttention (external/GroupPose/models/grouppose/ops).
# Checkpoint (Google Drive): data/models/grouppose_r50.pth

train_cfg = None

model = dict(
    type='GroupPosePoseEstimator',
    config_file='config/grouppose.py',
    checkpoint='data/models/grouppose_r50.pth',
    backbone='resnet50',
    num_body_points=17,
    grouppose_root='external/GroupPose',
    score_thr=0.0,
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
)

data_mode = 'bottomup'
data_root = 'data/'

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupRandomChoiceResize',
        scales=[(800, 1333)],
        keep_ratio=True),
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

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/annotations/person_keypoints_val2017.json',
    score_mode='bbox',
    nms_mode='none',
)
test_evaluator = val_evaluator
