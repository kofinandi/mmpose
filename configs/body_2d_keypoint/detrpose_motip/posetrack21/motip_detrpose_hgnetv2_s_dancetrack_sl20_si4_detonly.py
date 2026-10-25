_base_ = ['./motip_detrpose_hgnetv2_s_dancetrack_sl20_si4.py']

# Same checkpoint and detector as
# motip_detrpose_hgnetv2_s_dancetrack_sl20_si4.py, with the tracking head
# cut off.  predict() runs part='detr' and TrackPostProcess (top-60
# queries) and does not call trajectory modeling or the ID decoder, so
# the outputs have no track ids.
#
# emits_track_ids stays False so tools/benchmark_e2e.py takes the
# bottom-up path and can attach an external tracker with --post-config.
# The two comparison runs are:
#
#   python tools/benchmark_e2e.py \
#     configs/body_2d_keypoint/detrpose_motip/posetrack21/motip_detrpose_hgnetv2_s_dancetrack_sl20_si4_detonly.py \
#     /local/home/nkoefarago/detrpose-motip/output/2026-10-02_motip_detrpose_hgnetv2_s_dancetrack_sl20_si4/checkpoint_best_regular.pth \
#     --test-dataset dancetrack \
#     --include-bad-frames --eval-good-frames-only \
#     --prefetch-chunk-size 256 --kp-batch-size 8 \
#     --post-config configs/post_processing/oks_nms_pgpt_geom.py \
#     --postproc-name oks_nms_pgpt_geom --device cuda:0
#
#   ... --post-config configs/post_processing/oks_nms_pgpt.py \
#       --postproc-name oks_nms_pgpt
#
# Per-joint scores on the native joints are 1, so PGPT's joint-count
# admission gate sees every copied body joint as confident.  Eyes and
# ears stay at score 0 and do not count.  That is a property of this
# detector, not of the PGPT configs.

emits_track_ids = False

model = dict(detector_only=True)
