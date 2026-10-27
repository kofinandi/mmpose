# Official MOTIP (CVPR 2025) ResNet-50, DanceTrack, trajectory
# augmentation on. This is the published box tracker from
# https://github.com/MCG-NJU/MOTIP, not DETRPose-MOTIP.
#
# emits_track_ids tells tools/benchmark_e2e.py to step the detector one
# frame at a time, keep these track ids on the pose instances, and score
# HOTA/MOTA/IDF1 from them. NMS is not applied a second time. Do not pass
# --post-config, and do not pair this with a pose config that also sets
# emits_track_ids.
#
# Shorter-side resize 800 is upstream submit_and_evaluate's function
# default. Longer side, det/newborn/id thresholds, miss tolerance (30),
# and object-max assignment come from the yaml. TF32 stays at the process
# default; upstream disables it in the submit script, which would also
# change a pose model running alongside this detector.
#
# A PoseTrack21 number from this checkpoint is a transfer run. It is not
# the paper's DanceTrack HOTA.

emits_track_ids = True

default_scope = 'mmdet'

custom_imports = dict(imports=['mmpose.models.detectors'])

model = dict(
    type='MOTIPDetector',
    config='external/MOTIP/configs/r50_deformable_detr_motip_dancetrack.yaml',
    checkpoint='data/models/r50_deformable_detr_motip_dancetrack.pth',
    motip_root='external/MOTIP',
)
