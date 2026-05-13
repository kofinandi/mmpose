_base_ = ['../../../_base_/default_runtime.py']

# test-only config: disable training components inherited from default_runtime
train_cfg = None
val_cfg = None
optim_wrapper = None
param_scheduler = None

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'  # noqa
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# base dataset settings
dataset_type = 'ThreeDPWDataset'
data_mode = 'topdown'
data_root = 'data/3dpw/'

backend_args = dict(backend='local')

# test pipeline
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# test dataloader
test_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/threedpw_test.json',
        data_prefix=dict(img='imageFiles/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

# evaluators
test_evaluator = [
    dict(type='PCKAccuracy', thr=0.05, prefix='threedpw_pck05'),
    dict(type='PCKAccuracy', thr=0.1, prefix='threedpw_pck10'),
    dict(type='AUC', prefix='threedpw'),
    dict(type='EPE', prefix='threedpw'),
    dict(type='MPJVE', prefix='threedpw'),
    dict(type='MPJAE', prefix='threedpw'),
    dict(type='MPJVE', norm_item=['bbox', 'torso'], prefix='threedpw'),
    dict(type='MPJAE', norm_item=['bbox', 'torso'], prefix='threedpw'),
]
