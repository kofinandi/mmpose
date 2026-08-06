_base_ = ['../../_base_/default_runtime.py']

# TAR-ViTPose (ViT-B backbone, T=5 frame window), trained on PoseTrack17.
# https://github.com/zgspose/TARViTPose ("Beyond Static Frames: Temporal
# Aggregate-and-Restore Vision Transformer for Human Pose Estimation",
# CVPR 2026). Built directly on the Poseidon codebase (same data/config/
# eval conventions); see configs/body_2d_keypoint/poseidon/ for the sibling
# model.
#
# Matches the author-released `configs/posetrack17/configTARViTPoseVitB.yaml`:
# WINDOWS_SIZE=5, IMAGE_SIZE=[288, 384], HEATMAP_SIZE=[72, 96], SIGMA=3,
# MASK_THRESHOLD=0.2, NUM_LAYERS=6 (JTA/GRA stack depth),
# BBOX_ENLARGE_FACTOR=1.25 (== GetBBoxCenterScale's default padding=1.25).
# This is the only released checkpoint/config (ViT-B only).
#
# Checkpoint (author-released, `tarvitpose_b_17.pt`):
#   data/models/tarvitpose/tarvitpose_b_17.pt
#
# Run (topdown, GT boxes, matching the paper's evaluation protocol):
#   python tools/benchmark_e2e.py \
#     configs/body_2d_keypoint/tarvitpose/tarvitpose_vitb-t5_posetrack17-288x384.py \
#     data/models/tarvitpose/tarvitpose_b_17.pt \
#     --mock-detector --test-dataset emdb-mini --device cuda:0

train_cfg = None
val_cfg = None
optim_wrapper = None
param_scheduler = None

# Temporal clip window size (T), read by tools/benchmark_e2e.py.
num_input_frames = 5

# Heatmap decode codec, configurable. MSRAHeatmap with no refinement is
# numerically equivalent to TAR-ViTPose's own argmax + quarter-pixel-offset
# decoding (inherited from the Poseidon/DCPose codebase). Set
# unbiased=True (DARK) or swap to UDPHeatmap for an alternative
# (non-author) decode.
codec = dict(
    type='MSRAHeatmap',
    input_size=(288, 384),  # (W, H)
    heatmap_size=(72, 96),  # (W, H)
    sigma=3,
)

model = dict(
    type='TARViTPosePoseEstimator',
    tarvitpose_root='external/TARViTPose',
    use_mask=True,
    data_preprocessor=dict(
        type='ClipPoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    tarvitpose_cfg=dict(
        MODEL=dict(
            # In-repo ViTPose-base/COCO config: defines the backbone+head
            # architecture only. Its own (COCO-pretrained) checkpoint is
            # irrelevant here -- the wrapper always builds it with random
            # init and loads the full TAR-ViTPose checkpoint on top.
            CONFIG_FILE='configs/body_2d_keypoint/topdown_heatmap/coco/'
                         'td-hm_ViTPose-base_8xb64-210e_coco-256x192.py',
            EMBED_DIM=768,
            HEATMAP_SIZE=[72, 96],
            NUM_JOINTS=17,
            IMAGE_SIZE=[288, 384],
            MASK_THRESHOLD=0.2,
            NUM_LAYERS=6,
        ),
        WINDOWS_SIZE=5,
    ),
    codec=codec,
    map_to_coco=True,
    synthesize_eyes=False,
)

# Dataset settings below are only used to extract the topdown pipeline
# (LoadImage/GetBBoxCenterScale/TopdownAffine/PackPoseInputs); tools/
# benchmark_e2e.py's --test-dataset flag overrides which images/annotations
# are actually loaded (e.g. --test-dataset emdb-mini).
dataset_type = 'EmdbDataset'
data_mode = 'topdown'
data_root = 'data/emdb/'
backend_args = dict(backend='local')

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
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

# Note: temporal metrics (MPJVE/MPJAE), and throughput, are computed
# separately by tools/benchmark_e2e.py itself (--test-dataset
# extra_metrics / its own timing loop); it force-adds gt_from_samples=True
# to every entry here, so only list metrics that accept that kwarg
# (CocoMetric).
test_evaluator = dict(type='CocoMetric', gt_from_samples=True)
