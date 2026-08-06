_base_ = ['../../_base_/default_runtime.py']

# PAVE-Net (ResNet-50 backbone, T=3 frame window), trained on PoseTrack17.
# https://github.com/yuyonghui/PAVENet (paper repo name: PAVENet)
# Paper: "End-to-End Multi-Person Pose Estimation with Pose-Aware Video
# Transformer" (AAAI 2026), https://arxiv.org/abs/2511.13208
#
# This is a bottom-up, end-to-end *video* model (no external detector, no
# heatmap decode step -- keypoints come directly out of an RLE regression
# head). Matches the only config/checkpoint pair released by the authors,
# `external/PAVENet/configs/PAVE/res50_num_frames_3_posetrack17.py`:
# num_frames=3, num_keypoints=15 (PoseTrack17 layout with ears removed --
# see pavenet_wrapper.py's module docstring), test img_scale=(1333, 800).
#
# Checkpoint (author-released, `resnet50_posetrack.pth`):
#   data/models/pavenet/resnet50_posetrack.pth
#
# Note: the authors also released a Swin-L checkpoint
# (data/models/pavenet/swin_posetrack.pth), but no matching Swin-L config
# was published in the repo (only the R50/T3 config above exists) --
# reconstructing its architecture/hyperparameters would require guessing,
# so it is intentionally not covered by a config here (see fidelity ledger).
#
# Run (bottomup, no detector/mock-detector needed -- PAVE-Net is end-to-end):
#   python tools/benchmark_e2e.py \
#     configs/body_2d_keypoint/pavenet/pavenet_r50-t3_posetrack17.py \
#     data/models/pavenet/resnet50_posetrack.pth \
#     --test-dataset emdb-mini --device cuda:0

train_cfg = None
val_cfg = None
optim_wrapper = None
param_scheduler = None

# Temporal clip window size (T), read by tools/benchmark_e2e.py.
num_input_frames = 3

model = dict(
    type='PAVENetPoseEstimator',
    pavenet_root='external/PAVENet',
    map_to_coco=True,
    data_preprocessor=dict(
        type='ClipPoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),
    pavenet_model_cfg=dict(
        type='opera.PAVE',
        backbone=dict(
            type='mmdet.ResNet',
            input_type='mul_frames',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch'),
        neck=dict(
            type='mmdet.ChannelMapper',
            in_channels=[512, 1024, 2048],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),
        bbox_head=dict(
            type='opera.PAVEHeadMulFrames',
            num_frames=3,
            num_keypoints=15,
            num_query=300,
            num_classes=1,
            in_channels=2048,
            sync_cls_avg_factor=True,
            with_kpt_refine=True,
            as_two_stage=True,
            transformer=dict(
                type='opera.TransformerMulFrames',
                num_keypoints=15,
                num_frames=3,
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
                    type='opera.TransformerDecoderV2',
                    num_keypoints=15,
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
                                type=(
                                    'opera.'
                                    'MulFramesMultiScaleDeformablePoseAttentionNumFrames3'),  # noqa: E501
                                num_points=15,
                                embed_dims=256)
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
                    type='mmcv.DeformableDetrTransformerDecoderV1',
                    num_layers=2,
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
                                type=(
                                    'mmcv.'
                                    'MulFramesMultiScaleDeformableAttentionNumFrames3'),  # noqa: E501
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
                loss_weight=0.5),
            loss_kpt=dict(type='opera.RLELoss', loss_weight=1.0),
            loss_kpt_rpn=dict(type='opera.RLELoss', loss_weight=1.0),
            loss_oks=dict(
                type='opera.OKSLoss', num_keypoints=15, loss_weight=0.0),
            loss_hm=dict(type='opera.CenterFocalLoss', loss_weight=0.0),
            loss_kpt_refine=dict(type='opera.RLELoss', loss_weight=1.0),
            loss_oks_refine=dict(
                type='opera.OKSLoss', num_keypoints=15, loss_weight=0.0)),
        train_cfg=dict(
            assigner=dict(
                type='opera.PoseHungarianAssigner',
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                kpt_cost=dict(type='opera.KptL1Cost', weight=70.0),
                oks_cost=dict(
                    type='opera.OksCost', num_keypoints=15, weight=7.0))),
        test_cfg=dict(max_per_img=20)),
)

# Dataset settings below are only used to extract the bottomup pipeline
# (LoadImage/BottomupResize/PackPoseInputs); tools/benchmark_e2e.py's
# --test-dataset flag overrides which images/annotations are actually
# loaded (e.g. --test-dataset emdb-mini).
dataset_type = 'EmdbDataset'
data_mode = 'bottomup'
data_root = 'data/emdb/'
backend_args = dict(backend='local')

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    # Matches the author test pipeline's `MultiScaleFlipAug(img_scale=
    # (1333, 800), ...)` (keep_ratio=True, Pad size_divisor=1): a single
    # fixed-scale, aspect-ratio-preserving resize with no further padding
    # requirement (size_factor=1). See BottomupResize's clip-list support
    # in mmpose/datasets/transforms/bottomup_transforms.py.
    dict(
        type='BottomupResize',
        input_size=(1333, 800),
        size_factor=1,
        resize_mode='fit'),
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

# Note: temporal metrics (MPJVE/MPJAE) for EMDB/3DPW/PoseTrack21, and
# throughput, are computed separately by tools/benchmark_e2e.py itself
# (--test-dataset extra_metrics / its own timing loop); it force-adds
# gt_from_samples=True to every entry here, so only list metrics that
# accept that kwarg (CocoMetric).
test_evaluator = dict(type='CocoMetric', gt_from_samples=True)
