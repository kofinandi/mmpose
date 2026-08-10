_base_ = '../../../_base_/default_runtime.py'

# DETRPose-L (HGNetv2) evaluation on COCO keypoints.
# Paper: https://arxiv.org/abs/2506.13027
# Upstream: https://github.com/SebastianJanampa/DETRPose (inference_only)
#
# Evaluation-only config. Expected README AP on COCO val2017: ~72.5.
#
# Fidelity: upstream emits no boxes; pred_instances.bboxes are derived from
# keypoint min/max so CocoMetric(score_mode='bbox') can run. Keypoint
# visibility scores are hard-coded to 1.0 (upstream PostProcess).
# Checkpoint:
#   https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_l.pth

train_cfg = None

model = dict(
    type='DETRPoseEstimator',
    model_name='detrpose_hgnetv2_l',
    checkpoint='data/models/detrpose_hgnetv2_l.pth',
    detrpose_root='external/DETRPose',
    img_size=640,
    score_thr=0.0,
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
        pad_size_divisor=1),
)

data_mode = 'bottomup'
data_root = 'data/'

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupRandomChoiceResize',
        scales=[(640, 640)],
        keep_ratio=False),
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
