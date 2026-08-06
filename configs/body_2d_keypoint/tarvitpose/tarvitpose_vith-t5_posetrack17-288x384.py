_base_ = ['../../_base_/default_runtime.py']

# TAR-ViTPose (ViT-H backbone, T=5 frame window), trained on PoseTrack17.
# https://github.com/zgspose/TARViTPose ("Beyond Static Frames: Temporal
# Aggregate-and-Restore Vision Transformer for Human Pose Estimation",
# CVPR 2026). Built directly on the Poseidon codebase (same data/config/
# eval conventions); see configs/body_2d_keypoint/poseidon/ for the sibling
# model.
#
# NOTE -- reconstructed config, no author YAML released for this backbone
# size (the repo only ships `configs/posetrack17/configTARViTPoseVitB.yaml`).
# We do have the author-released `tarvitpose_h_17.pt` checkpoint though, so
# the hyperparameters below were reconstructed as follows:
#   - EMBED_DIM=1280, IMAGE_SIZE/HEATMAP_SIZE/SIGMA, and the ViTPose-huge
#     CONFIG_FILE: same pattern as Poseidon's own (author-released) ViT-H
#     config (configs/body_2d_keypoint/poseidon/poseidon_vith-t5_posetrack21-288x384.py),
#     and independently confirmed against this exact checkpoint --
#     `backbone.patch_embed.projection.weight` is `[1280, 3, 16, 16]`
#     (ViTPose-huge's embed dim/patch size) and `query_feat.weight` is
#     `[17, 1280]`.
#   - NUM_LAYERS=6: *not* guessed -- read directly off this checkpoint's
#     `state_dict` (`model_state_dict`), which has
#     `masked_attention_layers.{0..5}.*`/`self_attention_layers.{0..5}.*`/
#     `ffn_layers.{0..5}.*` keys (6 blocks, indices 0-5), i.e. identical
#     depth to the author-released ViT-B config.
#   - MASK_THRESHOLD=0.2: this is a pure runtime hyperparameter (an
#     attention-mask cutoff applied to the normalized heatmap, see
#     `TAR_ViTPose.generate_attn_mask` in
#     `external/TARViTPose/models/best/TAR_ViTPose.py`) with no
#     corresponding weights in the checkpoint, so it truly cannot be read
#     back out of the checkpoint. It is left at the framework default
#     (`posetimation/config/defaults.py: _C.MODEL.MASK_THRESHOLD = 0.2`),
#     which is also the value the authors used, unchanged, in their one
#     released (ViT-B) config -- i.e. this is the one inferred (not
#     verified) value in this file.
# WINDOWS_SIZE=5, IMAGE_SIZE=[288, 384], HEATMAP_SIZE=[72, 96], SIGMA=3,
# BBOX_ENLARGE_FACTOR=1.25 (== GetBBoxCenterScale's default padding=1.25),
# all carried over unchanged from the ViT-B config (shared across all
# Poseidon/TAR-ViTPose PoseTrack configs, author-released or not).
#
# Checkpoint (author-released, `tarvitpose_h_17.pt`):
#   data/models/tarvitpose/tarvitpose_h_17.pt
#
# Run (topdown, GT boxes, matching the paper's evaluation protocol):
#   python tools/benchmark_e2e.py \
#     configs/body_2d_keypoint/tarvitpose/tarvitpose_vith-t5_posetrack17-288x384.py \
#     data/models/tarvitpose/tarvitpose_h_17.pt \
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
            # In-repo ViTPose-huge/COCO config: defines the backbone+head
            # architecture only. Its own (COCO-pretrained) checkpoint is
            # irrelevant here -- the wrapper always builds it with random
            # init and loads the full TAR-ViTPose checkpoint on top.
            CONFIG_FILE='configs/body_2d_keypoint/topdown_heatmap/coco/'
                         'td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py',
            EMBED_DIM=1280,
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
