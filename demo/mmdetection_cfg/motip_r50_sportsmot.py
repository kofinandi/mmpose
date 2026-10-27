# Official MOTIP (CVPR 2025) ResNet-50, SportsMOT, trajectory
# augmentation on. Same wrapper as motip_r50_dancetrack.py. The SportsMOT
# yaml inherits DanceTrack and overrides miss tolerance and relative
# position length to 60. This is the published box tracker, not
# DETRPose-MOTIP.
#
# emits_track_ids tells tools/benchmark_e2e.py to step the detector one
# frame at a time, keep these track ids on the pose instances, and score
# HOTA/MOTA/IDF1 from them. NMS is not applied a second time. Do not pass
# --post-config, and do not pair this with a pose config that also sets
# emits_track_ids.
#
# Shorter-side resize 800 is upstream submit_and_evaluate's function
# default. The remaining inference settings come from the yaml after
# super-config inheritance. TF32 stays at the process default.
#
# A PoseTrack21 number from this checkpoint is a transfer run. It is not
# the paper's SportsMOT HOTA.

emits_track_ids = True

default_scope = 'mmdet'

custom_imports = dict(imports=['mmpose.models.detectors'])

model = dict(
    type='MOTIPDetector',
    config='external/MOTIP/configs/r50_deformable_detr_motip_sportsmot.yaml',
    checkpoint='data/models/r50_deformable_detr_motip_sportsmot.pth',
    motip_root='external/MOTIP',
)
