# BoxMOT BoT-SORT - bbox tracker with ReID and camera-motion compensation.
#
#   Broström, "BoxMOT: Pluggable SOTA tracking modules"
#   https://github.com/mikel-brostrom/boxmot
#   BoT-SORT: Aharon et al., "BoT-SORT: Robust Associations Multi-
#   Pedestrian Tracking", arXiv 2022.  https://arxiv.org/abs/2206.14651
#
# Association, CMC and ReID all run unmodified inside BoxMOT's Python
# BoT-SORT.  This is the published BoT-SORT pipeline wrapped as a
# post-processor: appearance embeddings come from BoxMOT's OSNet
# checkpoint, not a substitute.
#
# Detections and poses come from the prediction bundle rather than a
# BoxMOT-bundled YOLO.  Keypoints are carried through via det_ind and
# are never re-estimated.  Kalman-only tracks (no matching detection)
# are dropped because there is no pose to attach.

post_processor = dict(
    type='PostProcessingPipeline',
    needs_images=True,
    filters=[
        dict(
            type='OKSNMS',
            score_thr=0.3,
            oks_thr=0.9,
            score_mode='auto',
        ),
        dict(
            type='BoxMOTTracker',
            tracker='occluboost',
            reid_weights='data/models/boxmot/lmbn_n_duke.pt',  # or osnet_x0_25_msmt17.pt
            device='cuda:0',
        ),
    ],
)
