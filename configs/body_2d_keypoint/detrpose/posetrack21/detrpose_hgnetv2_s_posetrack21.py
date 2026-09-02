_base_ = '../../../_base_/default_runtime.py'

# DETRPose-S (HGNetv2) trained on PoseTrack21 (15 keypoints).
# Paper: https://arxiv.org/abs/2506.13027
# Upstream inference wrapper: external/DETRPose (inference_only)
# Training config: DETRPose/configs/detrpose/detrpose_hgnetv2_s_posetrack21.py
#
# Evaluation-only. This is a locally trained checkpoint, not an official
# DETRPose release (those are COCO-17 / CrowdPose-14 only).
#
# Native layout is PoseTrack-17 with never-annotated ears dropped
# (nose, head_bottom, head_top, shoulders, elbows, wrists, hips, knees,
# ankles). ``map_to_coco=True`` reprojects onto COCO-17 for
# tools/benchmark_e2e.py, whose GT is always converted to COCO-17.
# left_eye / right_eye / left_ear / right_ear stay at zero confidence;
# head_bottom / head_top have no COCO counterpart and are dropped.
#
# Boxes are derived from keypoint min/max (upstream has no box head).
# Checkpoint (epoch 97):
#   /local/home/nkoefarago/DETRPose/output/detrpose_hgnetv2_s_posetrack21/checkpoint0097.pth
#
# Run (bottomup; PoseTrack21 is large, so prefetch in chunks):
#   python tools/benchmark_e2e.py \
#     configs/body_2d_keypoint/detrpose/posetrack21/detrpose_hgnetv2_s_posetrack21.py \
#     /local/home/nkoefarago/DETRPose/output/detrpose_hgnetv2_s_posetrack21/checkpoint0097.pth \
#     --test-dataset posetrack21 \
#     --kp-batch-size 1 \
#     --prefetch-chunk-size 256 \
#     --include-bad-frames \
#     --device cuda:0

train_cfg = None

model = dict(
    type='DETRPoseEstimator',
    # Architecture skeleton is the COCO-S config; num_keypoints=15 rebuilds
    # the pose head / postprocessor to match the PoseTrack21 checkpoint.
    model_name='detrpose_hgnetv2_s',
    checkpoint='/local/home/nkoefarago/DETRPose/output/'
               'detrpose_hgnetv2_s_posetrack21/checkpoint0097.pth',
    detrpose_root='external/DETRPose',
    img_size=640,
    score_thr=0.0,
    num_keypoints=15,
    map_to_coco=True,
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
        pad_size_divisor=1),
)

data_mode = 'bottomup'
data_root = 'data/posetrack21/'

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
        type='PoseTrack21Dataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/posetrack21_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# tools/benchmark_e2e.py force-adds gt_from_samples=True and appends
# PoseTrack21 temporal metrics. Only CocoMetric accepts that kwarg.
val_evaluator = dict(
    type='CocoMetric',
    gt_from_samples=True,
    score_mode='bbox',
    nms_mode='none')
test_evaluator = val_evaluator
