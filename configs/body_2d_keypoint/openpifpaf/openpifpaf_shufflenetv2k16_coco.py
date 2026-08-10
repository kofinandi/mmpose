_base_ = ['../../_base_/default_runtime.py']

# OpenPifPaf -- shufflenetv2k16, single-image CifCaf on COCO-17. This is the
# *detection* half of the paper (no tracking): keypoints are grouped within
# a frame by Composite Association Fields only. It is the reference point
# the tracking configs in this directory should be read against.
#
# Paper: Kreiss et al., "OpenPifPaf: Composite Fields for Semantic Keypoint
#        Detection and Spatio-Temporal Association", T-ITS 2021.
#        https://ieeexplore.ieee.org/document/9617128
# Code:  https://github.com/openpifpaf/openpifpaf  (external/openpifpaf,
#        pinned at v0.13.11)
#
# emits_track_ids is NOT set: CifCaf assigns no IDs, so this runs through
# the ordinary batched bottom-up path and its FPS is comparable with the
# other bottom-up models in the benchmark. Every track_id is -1.
#
# Preprocessing keeps the same chain as the tracking configs
# (RescaleAbsolute(801) + CenterPadTight(16) + EVAL_TRANSFORM) so the two
# are directly comparable. The COCO eval default upstream is
# --long-edge=641 with extended scale; 801 is used here for consistency
# within this directory and is noted as a deviation from CocoKp's default.
#
# Keypoints: already COCO-17, so map_to_coco=False (no conversion).
#
# Checkpoint (author-released, 40 MB):
#   data/models/openpifpaf/shufflenetv2k16-210820-232500-cocokp-slurm726069-edge513-o10s-7189450a.pkl
#   A pickled whole model object, loaded by openpifpaf's network.Factory.
#
# Requires the submodule's C++ decoder extension:
#   pip install -e external/openpifpaf --no-build-isolation --no-deps
#
# Run:
#   python tools/benchmark_e2e.py \
#     configs/body_2d_keypoint/openpifpaf/openpifpaf_shufflenetv2k16_coco.py \
#     data/models/openpifpaf/shufflenetv2k16-210820-232500-cocokp-slurm726069-edge513-o10s-7189450a.pkl \
#     --test-dataset emdb-mini --device cuda:0 --kp-batch-size 1 \
#     --include-bad-frames

train_cfg = None
val_cfg = None
optim_wrapper = None
param_scheduler = None

model = dict(
    type='OpenPifPafPoseEstimator',
    openpifpaf_root='external/openpifpaf',
    # Default checkpoint; tools/benchmark_e2e.py takes it as a required
    # positional and overrides this, but pinning it here is what makes
    # this config self-contained (init_model(cfg) needs no checkpoint
    # argument) and what distinguishes the backbone variants -- the
    # architecture lives in the pickled .pkl, not in this file.
    checkpoint='data/models/openpifpaf/shufflenetv2k16-210820-232500-cocokp-'
               'slurm726069-edge513-o10s-7189450a.pkl',
    decoder='cifcaf:0',
    long_edge=801,
    map_to_coco=False,
    # Deliberately does NOT normalise: the wrapper hands the raw pixels to
    # openpifpaf's own transform chain, which normalises after rescaling
    # and padding (padding uses a non-zero fill applied pre-normalisation).
    data_preprocessor=dict(type='PoseDataPreprocessor', bgr_to_rgb=False),
)

dataset_type = 'EmdbDataset'
data_mode = 'bottomup'
data_root = 'data/emdb/'
backend_args = dict(backend='local')

# No resize here -- all geometry is upstream's, inside the wrapper.
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

test_evaluator = dict(type='CocoMetric', gt_from_samples=True)
