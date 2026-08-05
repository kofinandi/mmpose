# Detect-and-Track (CVPR 2018) - box IoU + CNN appearance cost.
#
#   Girdhar et al., "Detect-and-Track: Efficient Pose Estimation in Videos"
#   https://github.com/facebookresearch/DetectAndTrack
#
# Adds the paper's `cnn-cosdist` cost: each detection's crop is resized to
# 224x224 and pushed through an ImageNet-pretrained ResNet-18 up to
# `layer3` (upstream `TRACKING.CNN_MATCHING_LAYER`), and pairs are compared
# by cosine distance.
#
# NOT a substitution: upstream's appearance model is literally
# `torchvision.models.resnet18(pretrained=True)`, so the shipped feature is
# the published one. The preprocessing port keeps upstream's normalisation
# constants including its apparent typo in the third std channel
# (0.224 rather than 0.225); pass `keep_upstream_std=False` for the
# standard ImageNet value. Crops are batched into one forward pass per
# frame rather than one per box - equivalent, just faster.
#
# TUNED - `conf_filter_initial_dets` is 0.5 rather than upstream's 0.95,
# which was calibrated for the paper's Mask R-CNN and here would discard
# most detections before matching ever runs. Compare against
# detect_and_track_iou_tuned.py, which is this config minus the appearance
# cost, to isolate what the CNN term contributes.
#
# This config needs the source frames, hence `needs_images=True`. Run it
# with tools/postprocess_predictions.py (benchmark_e2e --post-config
# cannot supply images).
#
# Usage:
#   python tools/postprocess_predictions.py PRED_DIR \
#       --post-config configs/post_processing/detect_and_track_cnn.py \
#       --postproc-name dat_cnn

post_processor = dict(
    type='PostProcessingPipeline',
    needs_images=True,
    filters=[
        dict(
            type='DetectAndTrackLinker',
            cost_types=('bbox-overlap', 'cnn-cosdist'),
            cost_weights=(1.0, 1.0),
            bipart_match_algo='hungarian',
            conf_filter_initial_dets=0.5,   # TUNED, upstream uses 0.95
            min_box_area=50.0,
            max_track_ids=999,
            appearance_embedder=dict(
                type='TorchvisionCNNEmbedder',
                model='resnet18',
                layer='layer3',
                device='cuda:0',
                keep_upstream_std=True,
            ),
        ),
    ],
)
