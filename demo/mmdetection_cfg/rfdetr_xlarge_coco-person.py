default_scope = 'mmdet'

custom_imports = dict(imports=['mmpose.models.detectors'])

model = dict(
    type='RFDETRDetector',
    model_class='RFDETRXLarge',
    pretrain_weights='rf-detr-xlarge.pth',
    conf_thr=0.05,
    model_cache_dir='data/models',
    inference_batch_size=16,
)
