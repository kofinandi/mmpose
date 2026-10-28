_base_ = '../../../_base_/default_runtime.py'

# CoMotion (Newell et al., ICLR 2025): concurrent multi-person SMPL
# detection and tracking.  Benchmarked here as 2D pose + track ids.
#
# Upstream: external/ml-comotion (https://github.com/apple-aiml-research/ml-comotion)
# Wrapper: mmpose/models/pose_estimators/comotion_wrapper.py
#
# The detection and refine checkpoints are loaded by CoMotion itself from
#   external/ml-comotion/src/comotion_demo/data/comotion_detection_checkpoint.pt
#   external/ml-comotion/src/comotion_demo/data/comotion_refine_checkpoint.pt
# The pose_checkpoint argument must be that detection file.  SMPL v1.1.0
# neutral must be at
#   external/ml-comotion/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl
# (basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl from https://smpl.is.tue.mpg.de/,
# renamed).  It is not in the git checkout.
#
# Pipeline, unmodified from demo.py::track_poses:
#   default intrinsics, CoMotion.forward per frame (detection, recurrent
#   update, TrackHandler, PySceneDetect shot reset), then cleanup_tracks
#   with min_matched_frames=1, SMPL-mesh boxes with 5% padding, and
#   joints_coco projected to the original frame.
# tools/benchmark_e2e.py calls finalize_sequence() at each clip boundary
# so that cleanup sees the whole clip, including the last one.
#
# Keypoints are written in PoseTrack-17 order.  joints_coco supplies the
# shared COCO joints (eyes are dropped; they are not in PoseTrack).
# head_bottom and head_top are NOT model outputs.  fill_head=True
# synthesises them from the nose and shoulders:
#   head_bottom = shoulder_mid + 0.296627 * (nose - shoulder_mid)
#   head_top    = nose + 0.397664 * (nose - shoulder_mid)
# Coefficients are least squares on 87,069 PoseTrack21 train persons.
# This is a substitute.  Set fill_head=False to leave both at score 0.
# Score them with --keypoint-space dataset.  --keypoint-space coco drops
# them because COCO has no such names.
#
# Every joint and box is score 1.  The tracker exports no confidence, and
# the demo MOT file hardcodes 1.  Keypoint AP is a single operating point.
#
# The test pipeline does not resize or normalise.  CoMotion crops to
# 512 and applies ImageNet normalisation itself.  Pixels are the
# benchmark's cv2 decode, not the demo's PIL decode.
#
# Requires, in the mmpose env: pypose, tensordict, scenedetect==0.6.4,
# einops, chumpy.  NumPy 2 needs the chumpy alias shim in the wrapper.
# Transformers 5 is fine: the wrapper restores the 4.48 decoder return
# tuple that _call_llama indexes.  Do not downgrade transformers.
#
# Canonical run (tracker sees every frame, metrics on labeled frames,
# official kit so the synthesised head joints are scored):
#   python tools/benchmark_e2e.py \
#     configs/body_2d_keypoint/comotion/posetrack21/comotion_posetrack21.py \
#     external/ml-comotion/src/comotion_demo/data/comotion_detection_checkpoint.pt \
#     --test-dataset posetrack21 \
#     --include-bad-frames \
#     --eval-good-frames-only \
#     --keypoint-space dataset \
#     --official-posetrack21 \
#     --prefetch-chunk-size 256 \
#     --device cuda:0

train_cfg = None

# Routes tools/benchmark_e2e.py to run_tracking(): one frame per
# test_step, finalize_sequence() then reset_tracking() at each video
# boundary.  Reported FPS is sequential and includes cleanup_tracks.
emits_track_ids = True

model = dict(
    type='CoMotionPoseEstimator',
    checkpoint='external/ml-comotion/src/comotion_demo/data/'
               'comotion_detection_checkpoint.pt',
    comotion_root='external/ml-comotion',
    # SUBSTITUTE: head_bottom / head_top are not predicted by CoMotion.
    fill_head=True,
    min_matched_frames=1,
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        bgr_to_rgb=True),
)

data_mode = 'bottomup'
data_root = 'data/posetrack21/'

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
        type='PoseTrack21Dataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/posetrack21_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# tools/benchmark_e2e.py force-adds gt_from_samples=True and, because
# emits_track_ids is True, appends MPJVE/MPJAE, IDSwitch/MOTA/IDF1/HOTA.
# Only CocoMetric accepts gt_from_samples.
val_evaluator = dict(
    type='CocoMetric',
    gt_from_samples=True,
    score_mode='bbox',
    nms_mode='none')
test_evaluator = val_evaluator
