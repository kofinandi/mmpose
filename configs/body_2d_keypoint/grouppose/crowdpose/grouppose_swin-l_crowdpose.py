_base_ = '../../../_base_/default_runtime.py'

# GroupPose (Swin-L) evaluation on CrowdPose (14 keypoints).
# Paper: https://arxiv.org/abs/2308.07313
# Upstream: https://github.com/Michel-liu/GroupPose
#
# Evaluation-only. Expected README AP on CrowdPose test: ~74.1.
# Boxes derived from keypoint min/max. Requires MSDeformAttn CUDA op.
# Checkpoint: data/models/grouppose_crowdpose_swinL.pth

train_cfg = None

model = dict(
    type='GroupPosePoseEstimator',
    config_file='config/grouppose.py',
    checkpoint='data/models/grouppose_crowdpose_swinL.pth',
    backbone='swin_L_384_22k',
    num_body_points=14,
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
data_root = 'data/crowdpose/'

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
        type='CrowdPoseDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/mmpose_crowdpose_test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/mmpose_crowdpose_test.json',
    score_mode='bbox',
    nms_mode='none',
)
test_evaluator = val_evaluator
