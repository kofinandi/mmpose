default_scope = 'mmdet'

custom_imports = dict(imports=['mmpose.models.detectors'])

model = dict(
    type='RFDETRDetector',
    model_class='RFDETRSmall',
    pretrain_weights='rf-detr-small.pth',
    conf_thr=0.05,
    model_cache_dir='data/models',
)
