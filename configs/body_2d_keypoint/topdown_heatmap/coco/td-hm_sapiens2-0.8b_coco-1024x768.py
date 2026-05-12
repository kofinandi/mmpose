_base_ = ['../../../_base_/default_runtime.py']

# sapiens2-0.8b on COCO 17-keypoint – inference / evaluation only.
#
# Checkpoint:
#   /data/nkoefarago/data/hf_cache/hub/models--facebook--sapiens2-pose-0.8b/
#   snapshots/59ec616493f1986ce6336a53f981c8db4c490180/sapiens2_0.8b_pose.safetensors
#
# Run:
#   python tools/test.py \
#     configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_sapiens2-0.8b_coco-1024x768.py \
#     /data/nkoefarago/data/hf_cache/hub/models--facebook--sapiens2-pose-0.8b/snapshots/59ec616493f1986ce6336a53f981c8db4c490180/sapiens2_0.8b_pose.safetensors

train_cfg = None

# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------
codec = dict(
    type='MSRAHeatmap',
    input_size=(768, 1024),   # (W, H)
    heatmap_size=(192, 256),  # (W/4, H/4)
    sigma=6,
)

# ---------------------------------------------------------------------------
# Keypoint mapping: Goliath-308 channel -> COCO-17 destination index
# ---------------------------------------------------------------------------
coco17_keypoint_mapping = dict(
    num_keypoints=17,
    mapping=[
        (0,  0),   # nose
        (1,  1),   # left_eye
        (2,  2),   # right_eye
        (3,  3),   # left_ear
        (4,  4),   # right_ear
        (5,  5),   # left_shoulder
        (6,  6),   # right_shoulder
        (7,  7),   # left_elbow
        (8,  8),   # right_elbow
        (62, 9),   # left_wrist  (Goliath ch 62)
        (41, 10),  # right_wrist (Goliath ch 41)
        (9,  11),  # left_hip
        (10, 12),  # right_hip
        (11, 13),  # left_knee
        (12, 14),  # right_knee
        (13, 15),  # left_ankle
        (14, 16),  # right_ankle
    ],
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = dict(
    type='Sapiens2PoseEstimator',
    sapiens2_root='external/sapiens2',
    fp16=True,
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    sapiens2_model_cfg=dict(
        type='PoseTopdownEstimator',
        backbone=dict(
            type='Sapiens2',
            arch='sapiens2_0.8b',
            img_size=(1024, 768),   # (H, W)
            patch_size=16,
            final_norm=True,
            use_tokenizer=False,
            with_cls_token=True,
            out_type='featmap',
        ),
        decode_head=dict(
            type='PoseHeatmapHead',
            in_channels=1280,       # embed_dim for 0.8b
            out_channels=308,
            deconv_out_channels=(1024, 768),
            deconv_kernel_sizes=(4, 4),
            conv_out_channels=(512, 512, 256),
            conv_kernel_sizes=(1, 1, 1),
            loss_decode=dict(
                type='KeypointMSELoss',
                use_target_weight=True,
                loss_weight=10.0,
            ),
        ),
    ),
    codec=codec,
    keypoint_mapping=coco17_keypoint_mapping,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
data_root = 'data/coco'
dataset_type = 'CocoDataset'
data_mode = 'topdown'

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/person_keypoints_val2017.json',
)
test_evaluator = val_evaluator
