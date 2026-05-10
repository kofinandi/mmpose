_base_ = ['../../../_base_/default_runtime.py']

# PCT-huge on COCO 256x256 - inference / evaluation only.
train_cfg = None
# Checkpoint: external/PCT/weights/pct/swin_huge.pth
#
# Run:
#   python tools/test.py \
#       configs/body_2d_keypoint/topdown_regression/coco/td-reg_pct-huge_coco-256x256.py \
#       external/PCT/weights/pct/swin_huge.pth

model = dict(
    type='PCTPoseEstimator',
    pct_root='external/PCT',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    pct_model_cfg=dict(
        backbone=dict(
            type='SwinV2TransformerRPE2FC',
            embed_dim=256,
            depths=[2, 2, 42, 2],
            num_heads=[8, 16, 32, 64],
            window_size=[16, 16, 16, 8],
            pretrain_window_size=[12, 12, 12, 6],
            ape=False,
            drop_path_rate=0.5,
            patch_norm=True,
            use_checkpoint=False,
            rpe_interpolation='geo',
            use_shift=[True, True, False, False],
            relative_coords_table_type='norm8_log_bylayer',
            attn_type='cosine_mh',
            rpe_output_type='sigmoid',
            postnorm=True,
            mlp_type='normal',
            out_indices=(3, ),
            patch_embed_type='normal',
            patch_merge_type='normal',
            strid16=False,
            frozen_stages=5,
        ),
        keypoint_head=dict(
            type='PCT_Head',
            stage_pct='classifier',
            in_channels=2048,
            image_size=[256, 256],
            num_joints=17,
            loss_keypoint=dict(
                type='Classifer_loss',
                token_loss=1.0,
                joint_loss=1.0,
            ),
            cls_head=dict(
                conv_num_blocks=2,
                conv_channels=256,
                dilation=1,
                num_blocks=4,
                hidden_dim=64,
                token_inter_dim=64,
                hidden_inter_dim=256,
                dropout=0.0,
            ),
            tokenizer=dict(
                guide_ratio=0.5,
                ckpt='external/PCT/weights/pct/swin_huge.pth',
                encoder=dict(
                    drop_rate=0.2,
                    num_blocks=4,
                    hidden_dim=512,
                    token_inter_dim=64,
                    hidden_inter_dim=512,
                    dropout=0.0,
                ),
                decoder=dict(
                    num_blocks=1,
                    hidden_dim=32,
                    token_inter_dim=64,
                    hidden_inter_dim=64,
                    dropout=0.0,
                ),
                codebook=dict(
                    token_num=34,
                    token_dim=512,
                    token_class_num=2048,
                    ema_decay=0.9,
                ),
                loss_keypoint=dict(
                    type='Tokenizer_loss',
                    joint_loss_w=1.0,
                    e_loss_w=15.0,
                    beta=0.05,
                ),
            ),
        ),
        test_cfg=dict(
            flip_test=True,
            dataset_name='COCO',
        ),
    ),
)

data_root = 'data/coco'
dataset_type = 'CocoDataset'
data_mode = 'topdown'

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs'),
]
test_pipeline = val_pipeline

val_dataloader = dict(
    batch_size=32,
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

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/person_keypoints_val2017.json',
)
test_evaluator = val_evaluator
