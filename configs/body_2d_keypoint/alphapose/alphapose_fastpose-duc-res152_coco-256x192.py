_base_ = ['../../_base_/default_runtime.py']

# AlphaPose -- FastPose-DUC (SE-ResNet-152 backbone), COCO-17, 256x192, with
# the authors' re-ID pose tracker (`--pose_track`).
#
# Paper: Fang et al., "AlphaPose: Whole-Body Regional Multi-Person Pose
#        Estimation and Tracking in Real-Time", TPAMI 2022.
#        https://ieeexplore.ieee.org/document/9954214
# Code:  https://github.com/MVIG-SJTU/AlphaPose  (external/AlphaPose)
#
# Mirrors the author config `configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml`
# field-for-field -- this file does not restate the architecture, it points
# the wrapper at that YAML, so DATA_PRESET (256x192 input, 64x48 heatmap,
# 17 joints) and MODEL come from upstream.
# MODEL.BACKBONE='se-resnet', NUM_LAYERS=152, DUC stages 4/2/1.
# The most accurate non-DCN model in the author model zoo.
#
# TRACKING: this is the released `--pose_track` tracker, ported as-is.
#   The re-ID embedding comes from a *separate* OSNet-AIN network run over
#   the same 256x192 crops the pose net receives (tracker_api.py:231), NOT
#   from the pose backbone. The pose-backbone variant (`ResModel`) is
#   commented out upstream at tracker_api.py:199 and has no released
#   weights. Association (Kalman motion + embedding cascade @0.7, IoU @0.5,
#   unconfirmed IoU @0.7, track_buffer=240) is inherited verbatim from
#   `trackers.tracker_api.Tracker.update`.
#
# OUTPUT DIFFERENCES vs upstream's JSON writer (neither affects association
# or keypoints; see mmpose/models/pose_estimators/alphapose_wrapper.py):
#   - bboxes are the detector's boxes, not STrack's Kalman-smoothed tlbr;
#   - detections the tracker did not return keep track_id = -1 rather than
#     being dropped.
#
# Checkpoints:
#   pose : data/models/alphapose/fast_421_res152_256x192.pth
#          (author-released, "Fast Pose (DUC) ResNet152" in external/AlphaPose/docs/MODEL_ZOO.md,
#           Google Drive id 1kfyedqyn8exjbbNmYq8XGd2EooQjPtF9)
#   reid : data/models/alphapose/osnet_ain_x1_0_msmt17_256x128_amsgrad_
#          ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth
#          (author-released, linked from external/AlphaPose/trackers/README.md)
#
# Requires the submodule's CUDA extensions to be built:
#   pip install -e external/AlphaPose --no-build-isolation --no-deps
#
# Run (topdown; the model tracks, so do NOT pass --post-config):
#   python tools/benchmark_e2e.py \
#     configs/body_2d_keypoint/alphapose/alphapose_fastpose-duc-res152_coco-256x192.py \
#     data/models/alphapose/fast_421_res152_256x192.pth \
#     --det-config configs/_base_/det_models/rtmdet_m_640-8xb32_coco-person.py \
#     --det-checkpoint https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
#     --test-dataset emdb-mini --device cuda:0 --include-bad-frames

train_cfg = None
val_cfg = None
optim_wrapper = None
param_scheduler = None

# Routes tools/benchmark_e2e.py to run_tracking(): frames strictly in order,
# one frame's detections per test_step, reset_tracking() at each sequence
# boundary.  Reported FPS is sequential/unbatched and is not comparable
# with the batched topdown/bottomup numbers for other models.
emits_track_ids = True

# (W, H), matching the upstream YAML's DATA_PRESET.IMAGE_SIZE = [256, 192]
# (which is [H, W]).  No MMPose codec is configured: the wrapper decodes
# with upstream's own get_func_heatmap_to_coord(cfg), i.e. the released
# argmax + quarter-pixel-offset scheme.
input_size = (192, 256)

model = dict(
    type='AlphaPosePoseEstimator',
    alphapose_root='external/AlphaPose',
    alphapose_cfg='configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml',
    tracker=dict(
        # Defaults below are upstream's trackers/tracker_cfg.py verbatim;
        # only loadmodel is repointed at this repo's data/models/ layout.
        arch='osnet_ain',
        loadmodel='data/models/alphapose/osnet_ain_x1_0_msmt17_256x128_'
                  'amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_'
                  'flip_jitter.pth',
        frame_rate=30,
        track_buffer=240,
        conf_thres=0.5,
        nms_thres=0.4,
        iou_thres=0.5,
    ),
    # Reproduces AlphaPose's SimpleTransform.test_transform exactly:
    # img/255 then subtract (0.406, 0.457, 0.480) per channel, no std
    # division, on RGB input (upstream reads frames RGB, see
    # alphapose/utils/detector.py:162,217).
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[103.53, 116.535, 122.4],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
    ),
)

# Dataset settings below only supply the inference pipeline;
# tools/benchmark_e2e.py's --test-dataset flag decides what is loaded.
# GetBBoxCenterScale's default padding=1.25 matches upstream's
# _box_to_center_scale(..., scale_mult=1.25).
dataset_type = 'EmdbDataset'
data_mode = 'topdown'
data_root = 'data/emdb/'
backend_args = dict(backend='local')

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=input_size),
    dict(type='PackPoseInputs'),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/emdb_all.json',
        data_prefix=dict(img=''),
        emdb1=True,
        emdb2=False,
        good_frame_mask=True,
        test_mode=True,
        pipeline=val_pipeline,
    ))

# benchmark_e2e.py force-adds gt_from_samples=True to every entry here, and
# appends the dataset's temporal metrics plus the tracking suite
# (IDSwitch/MOTA/IDF1/HOTA, enabled automatically for emits_track_ids runs).
test_evaluator = dict(type='CocoMetric', gt_from_samples=True)
