_base_ = ['../../_base_/default_runtime.py']

# OpenPifPaf -- tracking ShuffleNetV2-k30 with the PoseSimilarity decoder.
# This is the paper's own *baseline* for the tracking ablation: poses are
# decoded per frame with CifCaf and then linked by an OKS-style pose
# similarity, WITHOUT using the temporal composite field (Tcaf). Compare
# against openpifpaf_tshufflenetv2k30-trackingpose_posetrack2018.py -- the
# same weights, same preprocessing, only the association differs, which is
# exactly the comparison the paper draws.
#
# Paper: Kreiss et al., "OpenPifPaf: Composite Fields for Semantic Keypoint
#        Detection and Spatio-Temporal Association", T-ITS 2021.
#        https://ieeexplore.ieee.org/document/9617128
# Code:  https://github.com/openpifpaf/openpifpaf  (external/openpifpaf,
#        pinned at v0.13.11)
#
# Preprocessing reproduces the authors' PoseTrack eval setting at batch
# size 1 -- Posetrack2018.common_eval_preprocess:
#   RescaleAbsolute(long_edge=801)  (Posetrack2018.eval_long_edge = 801)
#   CenterPadTight(16)
#   EVAL_TRANSFORM (ImageNet mean/std)
# and coordinates are mapped back with upstream's Annotation.inverse_transform.
#
# Keypoints: native PoseTrack-2018 17-joint layout, reprojected to COCO-17
# via KeypointConverter(src='posetrack18', dst='coco'). left_eye/right_eye
# have no PoseTrack counterpart and stay at zero confidence.
#
# Checkpoint (author-released, 141 MB):
#   data/models/openpifpaf/tshufflenetv2k30-210628-075118-posetrack2018-
#     cocokpst-slurm668247-o25-3d734bb8.pkl
#   It is a pickled *whole model object*, not a state dict, so it is loaded
#   by openpifpaf's network.Factory (see CUSTOM_POSE_WRAPPER_TYPES in
#   mmpose/apis/inference.py), not by MMEngine's load_checkpoint.
#
#   NOTE: `tshufflenetv2k30` is the ONLY released tracking checkpoint.
#   Upstream marks `tshufflenetv2k16` PRETRAINED_UNAVAILABLE
#   (plugins/posetrack/__init__.py:9) despite its own benchmark script
#   listing it as the default, so there is no config for it here.
#
# Requires the submodule's C++ decoder extension to be built (the
# CifCaf/Tcaf decoder *is* that extension):
#   pip install -e external/openpifpaf --no-build-isolation --no-deps
#   pip install pysparkling      # imported by the posetrack plugin
#
# Run (bottomup; the model tracks, so pass no detector and no --post-config):
#   python tools/benchmark_e2e.py \
#     configs/body_2d_keypoint/openpifpaf/openpifpaf_tshufflenetv2k30-posesimilarity_posetrack2018.py \
#     data/models/openpifpaf/tshufflenetv2k30-210628-075118-posetrack2018-cocokpst-slurm668247-o25-3d734bb8.pkl \
#     --test-dataset emdb-mini --device cuda:0 --include-bad-frames

train_cfg = None
val_cfg = None
optim_wrapper = None
param_scheduler = None

# Routes tools/benchmark_e2e.py to run_tracking(): frames strictly in order,
# one per test_step, reset_tracking() at each sequence boundary (which emits
# openpifpaf's 'eval_reset' signal, clearing both the backbone's RunningCache
# and the decoder's track pool).  Reported FPS is sequential/unbatched and is
# not comparable with the batched topdown/bottomup numbers for other models.
emits_track_ids = True

model = dict(
    type='OpenPifPafPoseEstimator',
    openpifpaf_root='external/openpifpaf',
    # Upstream CLI syntax; this is `--decoder=posesimilarity:0`.
    decoder='posesimilarity:0',
    long_edge=801,
    map_to_coco=True,
    # Deliberately does NOT normalise: the wrapper hands the raw pixels to
    # openpifpaf's own transform chain, which does the ImageNet
    # normalisation itself after rescaling and padding (padding uses a
    # non-zero fill, (124, 116, 104), applied before normalisation).
    data_preprocessor=dict(type='PoseDataPreprocessor', bgr_to_rgb=False),
)

# Dataset settings below only supply the inference pipeline;
# tools/benchmark_e2e.py's --test-dataset flag decides what is loaded.
# The pipeline deliberately contains no resize -- all geometry is handled
# by the upstream transform chain inside the wrapper.
dataset_type = 'EmdbDataset'
data_mode = 'bottomup'
data_root = 'data/emdb/'
backend_args = dict(backend='local')

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
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
