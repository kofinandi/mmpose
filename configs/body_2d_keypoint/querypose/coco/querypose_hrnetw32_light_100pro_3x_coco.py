_base_ = '../../../_base_/default_runtime.py'

# QueryPose-light HRNet-W32 evaluation on COCO keypoints.
# Paper: https://arxiv.org/abs/2212.07855
# Upstream: https://github.com/buptxyb666/QueryPose
#
# Evaluation-only. This is the RELEASED LIGHT decoder
# (MODEL.QueryPose.LIGHT_VERSION=True). Expected README AP on COCO val:
# ~69.8. Do NOT report paper Table AP 72.4 (full DynamicConv) under this
# config — those weights were never published.
# Requires: cd external/QueryPose && python setup.py build develop
# Checkpoint (Google Drive): data/models/querypose_hrnet32_light.pth

train_cfg = None

model = dict(
    type='QueryPosePoseEstimator',
    config_file='projects/querypose/configs/querypose.hrnet32.100pro.3x.yaml',
    checkpoint='data/models/querypose_hrnet32_light.pth',
    querypose_root='external/QueryPose',
    score_thr=0.0,
    # Leave RGB float ~[0,255]; QueryPose applies its own pixel_mean/std.
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
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
