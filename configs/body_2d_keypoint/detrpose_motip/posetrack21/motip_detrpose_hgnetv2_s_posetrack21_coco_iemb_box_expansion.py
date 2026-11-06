_base_ = '../../../_base_/default_runtime.py'

# DETRPose-MOTIP-S (HGNetv2-B0, depth_mult 0.34, 3 decoder layers),
# PoseTrack21 joint training from the COCO-converted detector, plus
# InstanceEmbedHead.  The head pools the instance token with the 15
# keypoint tokens (pose_encoding="none", residual, zero-init output,
# hidden_dim_ratio=2, 2 layers, no dropout).  Association uses that
# embedding, not the raw instance token.  predict() calls
# apply_instance_embed_head before TrackPostProcess, matching
# evaluate_motip.  No false-positive class.
#
# Architecture source of truth:
#   /local/home/nkoefarago/detrpose-motip/configs/detrpose/motip_detrpose_hgnetv2_s_posetrack21_coco_iemb.py
# Wrapper: mmpose/models/pose_estimators/detrpose_motip_wrapper.py
#
# Checkpoint (epoch 11, best regular eval by official mAP-Total,
# use_ema=False so load ck['model'] not ema).  checkpoint.pth in the
# same directory is the final epoch 39, which scored lower:
#   /local/home/nkoefarago/detrpose-motip/output/2026-10-17_motip_detrpose_hgnetv2_s_posetrack21_coco_iemb/checkpoint_best_regular.pth
#
# Native layout is PoseTrack-17 with never-annotated ears dropped (15
# joints).  map_to_coco=True reprojects onto COCO-17 for
# tools/benchmark_e2e.py.  left_eye / right_eye / left_ear / right_ear stay
# at zero confidence; head_bottom / head_top have no COCO counterpart and
# are dropped.
#
# Boxes are derived from the native keypoint min/max (upstream has no box
# head).  expand_boxes=True then grows that extent by the PoseTrack21-train
# fit: 17% of width on each side, 3% of height above, 11% below.  That
# inflation is wrapper post-processing, applied before the COCO remap, so
# bbox metrics are not the bare keypoint extent.
#
# Default metrics are mmpose CocoMetric + MPJVE/MPJAE +
# IDSwitch/MOTA/IDF1/HOTA.  Those are not the official PoseTrack21
# evaluator.  --official-posetrack21 adds the kit (keypoint HOTA, mAP,
# keypoint MOTA, and box MOTA/IDF1/HOTA with the CoMotion/DETRAM
# ignore-region fix).  --keypoint-space dataset keeps the dataset joint
# layout; joints the model does not predict stay unscored.  With
# map_to_coco=True the emitted layout is COCO-17, so head_bottom /
# head_top are still dropped.  Set map_to_coco=False to pass the native
# head joints through.
#
# Requires omegaconf / cloudpickle / iopath.
#
# Canonical run (tracker sees every frame, metrics on labeled frames only):
#   python tools/benchmark_e2e.py \
#     configs/body_2d_keypoint/detrpose_motip/posetrack21/motip_detrpose_hgnetv2_s_posetrack21_coco_iemb_box_expansion.py \
#     /local/home/nkoefarago/detrpose-motip/output/2026-10-17_motip_detrpose_hgnetv2_s_posetrack21_coco_iemb/checkpoint_best_regular.pth \
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
    config='configs/detrpose/'
           'motip_detrpose_hgnetv2_s_posetrack21_coco_iemb.py',
    checkpoint='/local/home/nkoefarago/detrpose-motip/output/'
               '2026-10-17_motip_detrpose_hgnetv2_s_posetrack21_coco_iemb/'
               'checkpoint_best_regular.pth',
    motip_root='/local/home/nkoefarago/detrpose-motip',
    use_ema=False,
    map_to_coco=True,
    expand_boxes=True,
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
