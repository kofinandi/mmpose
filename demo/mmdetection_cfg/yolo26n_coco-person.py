default_scope = 'mmdet'

custom_imports = dict(imports=['mmpose.models.detectors'])

model = dict(
    type='UltralyticsYOLODetector',
    weights='data/models/yolo26n.pt',
    conf_thr=0.05,
    iou_thr=0.7,
    imgsz=640,
)
