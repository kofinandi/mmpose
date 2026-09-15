_base_ = '../../../_base_/default_runtime.py'

# DETRPose-MOTIP-S (HGNetv2-B0) trained on PoseTrack21.
#
# MOTIP (Gao et al.) ID decoder + trajectory modeling on top of DETRPose-S.
# This is a different model from configs/body_2d_keypoint/detrpose/... :
# that wrapper is detection-only.  This one assigns track IDs online.
#
# Upstream: external/detrpose-motip
# Training config (architecture source of truth):
#   external/detrpose-motip/configs/detrpose/motip_detrpose_hgnetv2_s_posetrack21.py
# Wrapper: mmpose/models/pose_estimators/detrpose_motip_wrapper.py
#
# Checkpoint (epoch 39, use_ema=False so load ck['model'] not ema):
#   /local/home/nkoefarago/detrpose-motip/output/motip_detrpose_hgnetv2_s_posetrack21_bck/checkpoint0039.pth
#
# Native layout is PoseTrack-17 with never-annotated ears dropped (15
# joints).  map_to_coco=True reprojects onto COCO-17 for
# tools/benchmark_e2e.py.  left_eye / right_eye / left_ear / right_ear stay
# at zero confidence; head_bottom / head_top have no COCO counterpart and
# are dropped.
#
# Boxes are derived from keypoint min/max (upstream has no box head).
#
# Metrics are mmpose CocoMetric + MPJVE/MPJAE + IDSwitch/MOTA/IDF1/HOTA,
# NOT the official PoseTrack21 evaluator (pose mAP + TrackEval HOTA) that
# evaluate_motip reports.  Do not compare these numbers to published
# PoseTrack21 tables or to train_motip.py --eval output.
#
# Requires omegaconf / cloudpickle / iopath.
#
# Canonical run (tracker sees every frame, metrics on labeled frames only):
#   python tools/benchmark_e2e.py \
#     configs/body_2d_keypoint/detrpose_motip/posetrack21/motip_detrpose_hgnetv2_s_posetrack21.py \
#     /local/home/nkoefarago/detrpose-motip/output/motip_detrpose_hgnetv2_s_posetrack21_bck/checkpoint0039.pth \
#     --test-dataset posetrack21 \
#     --include-bad-frames \
#     --eval-good-frames-only \
#     --prefetch-chunk-size 256 \
#     --device cuda:0

train_cfg = None

# Routes tools/benchmark_e2e.py to run_tracking(): frames strictly in
# order, one frame per test_step, reset_tracking() at each sequence
# boundary.  Reported FPS is sequential/unbatched.
emits_track_ids = True

model = dict(
    type='DETRPoseMOTIPEstimator',
    config='external/detrpose-motip/configs/detrpose/'
           'motip_detrpose_hgnetv2_s_posetrack21.py',
    checkpoint='/local/home/nkoefarago/detrpose-motip/output/'
               'motip_detrpose_hgnetv2_s_posetrack21_bck/checkpoint0039.pth',
    motip_root='external/detrpose-motip',
    use_ema=False,
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
# PoseTrack21 temporal metrics plus IDSwitch/MOTA/IDF1/HOTA because
# emits_track_ids is True.  Only CocoMetric accepts that kwarg.
val_evaluator = dict(
    type='CocoMetric',
    gt_from_samples=True,
    score_mode='bbox',
    nms_mode='none')
test_evaluator = val_evaluator
