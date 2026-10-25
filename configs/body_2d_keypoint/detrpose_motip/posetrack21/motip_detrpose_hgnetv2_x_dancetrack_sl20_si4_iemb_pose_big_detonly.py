_base_ = ['./motip_detrpose_hgnetv2_x_dancetrack_sl20_si4_iemb_pose_big.py']

# Same DETRPose-MOTIP-X checkpoint as
# motip_detrpose_hgnetv2_x_dancetrack_sl20_si4_iemb_pose_big.py, with the
# tracking head cut off.  predict() runs part='detr' and TrackPostProcess
# and does not call the instance-embedding head, trajectory model, or ID
# decoder.  The full checkpoint is still loaded so the detector weights
# match the tracking run.
#
# emits_track_ids stays False so tools/benchmark_e2e.py can attach an
# external tracker with --post-config.  DanceTrack eval:
#
#   python tools/benchmark_e2e.py \
#     configs/body_2d_keypoint/detrpose_motip/posetrack21/motip_detrpose_hgnetv2_x_dancetrack_sl20_si4_iemb_pose_big_detonly.py \
#     /local/home/nkoefarago/detrpose-motip/output/2026-10-05_motip_detrpose_hgnetv2_x_dancetrack_sl20_si4_iemb_pose_big/checkpoint_best_regular.pth \
#     --test-dataset dancetrack \
#     --include-bad-frames --eval-good-frames-only \
#     --prefetch-chunk-size 256 --kp-batch-size 4 \
#     --post-config configs/post_processing/oks_nms_pgpt_geom.py \
#     --postproc-name oks_nms_pgpt_geom --device cuda:0

emits_track_ids = False

model = dict(detector_only=True)
