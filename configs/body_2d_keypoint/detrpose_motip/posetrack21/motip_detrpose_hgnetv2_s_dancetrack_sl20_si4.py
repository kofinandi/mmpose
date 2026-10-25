_base_ = '../../../_base_/default_runtime.py'

# DETRPose-MOTIP-S (HGNetv2-B0) trained on DanceTrack, sample length 20,
# mixed stride 4.  No instance-embedding head: association uses the
# detector's instance token through PoseRuntimeTracker.
#
# Architecture source of truth:
#   /local/home/nkoefarago/detrpose-motip/configs/detrpose/motip_detrpose_hgnetv2_s_dancetrack_sl20_si4.py
# Wrapper: mmpose/models/pose_estimators/detrpose_motip_wrapper.py
#
# Checkpoint (epoch 37, best regular eval, use_ema=False so load ck['model']):
#   /local/home/nkoefarago/detrpose-motip/output/2026-10-02_motip_detrpose_hgnetv2_s_dancetrack_sl20_si4/checkpoint_best_regular.pth
#
# Native layout is PoseTrack-17 with never-annotated ears dropped (15
# joints).  map_to_coco=True reprojects onto COCO-17.  left_eye /
# right_eye / left_ear / right_ear stay at zero confidence; head_bottom /
# head_top have no COCO counterpart and are dropped.  Per-joint scores on
# the native joints are 1 (upstream has no per-joint confidence).  Boxes
# are the keypoint min/max (no box head).
#
# DanceTrack val here is the PCT-huge pseudo-pose JSON
# (data/dancetrack/annotations/dancetrack_pose_val.json), not human pose
# labels and not the official DanceTrack box MOT evaluator.  Metrics are
# mmpose CocoMetric + MPJVE/MPJAE + IDSwitch/MOTA/IDF1/HOTA + PosePR.
# The benchmark loader maps the 19-keypoint annotations onto COCO-17.
#
# There is no train_dataloader, so init_model's dataset_meta falls back
# to COCO-17, which matches the mapped predictions and the converted GT.
#
# Canonical run (every val frame is labeled, so the good-frame flags do
# not change the set; they are kept so the loader path matches PoseTrack):
#   python tools/benchmark_e2e.py \
#     configs/body_2d_keypoint/detrpose_motip/posetrack21/motip_detrpose_hgnetv2_s_dancetrack_sl20_si4.py \
#     /local/home/nkoefarago/detrpose-motip/output/2026-10-02_motip_detrpose_hgnetv2_s_dancetrack_sl20_si4/checkpoint_best_regular.pth \
#     --test-dataset dancetrack \
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
           'motip_detrpose_hgnetv2_s_dancetrack_sl20_si4.py',
    checkpoint='/local/home/nkoefarago/detrpose-motip/output/'
               '2026-10-02_motip_detrpose_hgnetv2_s_dancetrack_sl20_si4/'
               'checkpoint_best_regular.pth',
    motip_root='/local/home/nkoefarago/detrpose-motip',
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
data_root = 'data/dancetrack/'

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
        type='DanceTrackPoseDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/dancetrack_pose_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# tools/benchmark_e2e.py force-adds gt_from_samples=True and, because
# emits_track_ids is True, appends MPJVE/MPJAE, IDSwitch/MOTA/IDF1/HOTA
# and PosePR.  Only CocoMetric accepts gt_from_samples.
val_evaluator = dict(
    type='CocoMetric',
    gt_from_samples=True,
    score_mode='bbox',
    nms_mode='none')
test_evaluator = val_evaluator
