_base_ = './petr_r50_16x2_100e_coco.py'

# PETR Swin-L evaluation on COCO keypoints.
# Backbone: Swin-Large patch4 window7 224, pre-trained on ImageNet-22K then
# fine-tuned on ImageNet-1K.
# Expected performance: ~73.1 AP on COCO val2017.

model = dict(
    petr_model_cfg=dict(
        backbone=dict(
            _delete_=True,
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
        neck=dict(in_channels=[384, 768, 1536])))
