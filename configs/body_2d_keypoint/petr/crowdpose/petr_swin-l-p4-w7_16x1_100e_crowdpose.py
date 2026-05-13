_base_ = '../../../../configs/_base_/default_runtime.py'

# PETR Swin-L evaluation on CrowdPose keypoints (14 keypoints).
# Backbone: Swin-Large patch4 window7 224, pre-trained on ImageNet-22K then
# fine-tuned on ImageNet-1K.
# Original paper: https://arxiv.org/abs/2201.02315

# Override train_cfg from default_runtime to allow test-only execution.
train_cfg = None

# model
model = dict(
    type='PETRPoseEstimator',
    petr_root='external/PETR',
    petr_model_cfg=dict(
        type='opera.PETR',
        backbone=dict(
            type='mmdet.SwinTransformer',
            embed_dims=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            patch_norm=True,
            out_indices=(1, 2, 3),
            with_cp=False),
        neck=dict(
            type='mmdet.ChannelMapper',
            in_channels=[384, 768, 1536],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),
        bbox_head=dict(
            type='opera.PETRHead',
            num_query=300,
            num_classes=1,
            in_channels=2048,
            num_keypoints=14,
            sync_cls_avg_factor=True,
            with_kpt_refine=True,
            as_two_stage=True,
            transformer=dict(
                type='opera.PETRTransformer',
                num_keypoints=14,
                encoder=dict(
                    type='mmcv.DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='mmcv.BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='mmcv.MultiScaleDeformableAttention',
                            embed_dims=256),
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn',
                                         'norm'))),
                decoder=dict(
                    type='opera.PetrTransformerDecoder',
                    num_layers=6,
                    num_keypoints=14,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='mmcv.DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='mmcv.MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='opera.MultiScaleDeformablePoseAttention',
                                embed_dims=256,
                                num_points=14)
                        ],
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn',
                                         'norm', 'ffn', 'norm'))),
                hm_encoder=dict(
                    type='mmcv.DetrTransformerEncoder',
                    num_layers=1,
                    transformerlayers=dict(
                        type='mmcv.BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='mmcv.MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=1),
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn',
                                         'norm'))),
                refine_decoder=dict(
                    type='mmcv.DeformableDetrTransformerDecoder',
                    num_layers=3,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='mmcv.DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='mmcv.MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='mmcv.MultiScaleDeformableAttention',
                                embed_dims=256,
                                im2col_step=128)
                        ],
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn',
                                         'norm', 'ffn', 'norm')))),
            positional_encoding=dict(
                type='mmcv.SinePositionalEncoding',
                num_feats=128,
                normalize=True,
                offset=-0.5),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_kpt=dict(type='mmdet.L1Loss', loss_weight=70.0),
            loss_kpt_rpn=dict(type='mmdet.L1Loss', loss_weight=70.0),
            loss_oks=dict(
                type='opera.OKSLoss', loss_weight=2.0, num_keypoints=14),
            loss_hm=dict(type='opera.CenterFocalLoss', loss_weight=4.0),
            loss_kpt_refine=dict(type='mmdet.L1Loss', loss_weight=80.0),
            loss_oks_refine=dict(
                type='opera.OKSLoss', loss_weight=3.0, num_keypoints=14)),
        train_cfg=dict(
            assigner=dict(
                type='opera.PoseHungarianAssigner',
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                kpt_cost=dict(type='opera.KptL1Cost', weight=70.0),
                oks_cost=dict(
                    type='opera.OksCost', weight=7.0, num_keypoints=14)),
            sampler=dict(type='mmdet.PseudoSampler')),
        test_cfg=dict(max_per_img=100)),
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
)

# data pipeline
data_mode = 'bottomup'
data_root = 'data/crowdpose/'

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupRandomChoiceResize',
        scales=[(800, 1333)],
        keep_ratio=True),
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
        type='CrowdPoseDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/mmpose_crowdpose_test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/mmpose_crowdpose_test.json',
    score_mode='bbox',
    nms_mode='none',
)
test_evaluator = val_evaluator
