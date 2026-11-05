#!/bin/bash
# Run a pose-tracking sweep: each CSV row is one bottom-up model, one
# top-down model plus a detector, or one end-to-end tracker, with an
# optional online post-processing pipeline on top.
#
# Mode (column 5):
#   bottomup  Pose model only. DETRPose, YOLO-Pose, and the same class
#             of model. Detector columns must be empty.
#   topdown   Pose model on boxes from DetConfig. This is also how a
#             box tracker is paired with a top-down pose model
#             (MOTIP + ViTPose): the detector config sets
#             emits_track_ids, and tools/benchmark_e2e.py keeps those
#             ids. Person class id is per detector (RF-DETR COCO
#             person is 1; single-class MOTIP is 0).
#   tracker   End-to-end pose tracker. The pose config sets
#             emits_track_ids (CoMotion, DETRPose-MOTIP, AlphaPose,
#             OpenPifPaf tracking). No detector. tools/benchmark_e2e.py
#             steps one frame at a time and refuses --post-config,
#             because a second tracker would replace the model's ids.
#
# PostConfig is any online pipeline under configs/post_processing/
# (PGPT, PGPT geometric, ByteTrack, Deep OC-SORT, BoT-SORT, GP-Kalman
# SORT, ...). It runs inside the timed loop. Offline filters
# (SmoothNet) are rejected by the driver; run
# those afterwards with scripts/run_postprocess_predictions.sh.
# Leave PostConfig and PostprocName empty for a raw run. A detector
# or pose config that sets emits_track_ids cannot take PostConfig.
#
# configs/post_processing/pgpt.py pins PoseGCN to cuda:0. Run PGPT
# rows with DEVICE=cuda:0, or the script prints a warning.
#
# Test set defaults to PoseTrack21, with unlabeled frames kept in the
# tracker state and metrics computed on labeled frames only. PoseTrack21
# runs also pass --official-posetrack21. The official kit still scores
# predictions on unlabeled frames; the mmpose metrics do not.
# Bottom-up models and end-to-end trackers use keypoint batch size 1.
# Top-down rows use detector batch size 1 and pose batch size 32.
# Frames are streamed in chunks of 256; the dataset is not loaded up front.
#   ./scripts/run_pose_tracking_benchmarks.sh
#   DEVICE=cuda:0 ./scripts/run_pose_tracking_benchmarks.sh
#   NUM_FRAMES=64 ./scripts/run_pose_tracking_benchmarks.sh
#   TEST_DATASET=emdb-mini ./scripts/run_pose_tracking_benchmarks.sh
#   CSV=scripts/benchmark_configs_pose_tracking.csv \
#     SUFFIX=pose_tracking ./scripts/run_pose_tracking_benchmarks.sh
#
# CSV columns:
#   Config, Checkpoint, Name, Variant, Mode, KpBatchSize,
#   DetConfig, DetCheckpoint, DetCatId, DetBatchSize, NmsThr,
#   PostConfig, PostprocName, KeypointSpace, OfficialPT21
# Empty KpBatchSize is 1 for bottomup/tracker and 32 for topdown.
# Empty DetBatchSize is 1. Empty NmsThr uses NMS_THR.
# KeypointSpace is coco or dataset (default coco). OfficialPT21 is
# unused on PoseTrack21, where the official kit is always enabled.
# OfficialPT21=1 on any other test set is an error.
# Lines whose first field is empty or starts with # are skipped.

set -uo pipefail

TIMESTAMP="$(date '+%Y%m%d')"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV="${CSV:-${ROOT_DIR}/scripts/benchmark_configs_pose_tracking.csv}"
SUFFIX="${SUFFIX:-pose_tracking}"
TEST_DATASET="${TEST_DATASET:-posetrack21}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/benchmark/logs/${TIMESTAMP}_${TEST_DATASET}_${SUFFIX}}"
RESULTS_FILE="${RESULTS_FILE:-${ROOT_DIR}/benchmark/results/${TIMESTAMP}_${TEST_DATASET}_${SUFFIX}.json}"
DEVICE="${DEVICE:-cuda:7}"
# Pose batch for top-down rows. Bottom-up and tracker rows use
# KP_BATCH_SIZE_SEQUENTIAL.
KP_BATCH_SIZE="${KP_BATCH_SIZE:-32}"
KP_BATCH_SIZE_SEQUENTIAL="${KP_BATCH_SIZE_SEQUENTIAL:-1}"
DET_BATCH_SIZE="${DET_BATCH_SIZE:-1}"
NMS_THR="${NMS_THR:-0.95}"
BBOX_THR="${BBOX_THR:-0.3}"
# PoseTrack21 / EMDB / 3DPW: the tracker sees every frame, metrics stay
# on frames that have reliable GT. Set either variable to 0 to drop the
# flag. Both are a no-op on datasets without that notion.
INCLUDE_BAD_FRAMES="${INCLUDE_BAD_FRAMES:-1}"
EVAL_GOOD_FRAMES_ONLY="${EVAL_GOOD_FRAMES_ONLY:-1}"
# Stream the dataset in chunks. 0 eager-loads every frame up front.
PREFETCH_CHUNK_SIZE="${PREFETCH_CHUNK_SIZE:-256}"
NUM_FRAMES="${NUM_FRAMES:-}"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

FAILED=()
TOTAL=0
PASSED=0

strip_cr() {
    printf '%s' "$1" | tr -d '\r'
}

config_emits_track_ids() {
    local path="$1"
    [[ -f "$path" ]] && grep -qE '^emits_track_ids[[:space:]]*=[[:space:]]*True' "$path"
}

fail_row() {
    local label="$1"
    local reason="$2"
    FAILED+=("${label} (${reason})")
    TOTAL=$((TOTAL + 1))
    echo "Error: ${label}: ${reason}" >&2
}

run_benchmark() {
    local config="$1"
    local checkpoint="$2"
    local name="$3"
    local variant="$4"
    local mode="$5"
    local kp_batch_size="$6"
    local det_config="$7"
    local det_checkpoint="$8"
    local det_cat_id="$9"
    local det_batch_size="${10}"
    local nms_thr="${11}"
    local post_config="${12}"
    local postproc_name="${13}"
    local keypoint_space="${14}"
    local official_pt21="${15}"

    local log_file="${LOG_DIR}/${name}-${variant}.log"
    local pred_dir="benchmark/predictions/${TIMESTAMP}_${TEST_DATASET}_${SUFFIX}/${name}-${variant}"
    local pred_out="$pred_dir"
    if [[ -n "$post_config" ]]; then
        pred_out="${pred_dir%/*}_${postproc_name}/$(basename "$pred_dir")"
    fi

    local extra_args=()
    if [[ "$PREFETCH_CHUNK_SIZE" -gt 0 ]]; then
        extra_args+=(--prefetch-chunk-size "$PREFETCH_CHUNK_SIZE")
    fi
    if [[ -n "$NUM_FRAMES" ]]; then
        extra_args+=(--num-frames "$NUM_FRAMES")
    fi
    if [[ "$INCLUDE_BAD_FRAMES" != "0" ]]; then
        extra_args+=(--include-bad-frames)
    fi
    if [[ "$EVAL_GOOD_FRAMES_ONLY" != "0" ]]; then
        extra_args+=(--eval-good-frames-only)
    fi
    if [[ -n "$keypoint_space" ]]; then
        extra_args+=(--keypoint-space "$keypoint_space")
    fi
    if [[ "$official_pt21" == "1" ]]; then
        extra_args+=(--official-posetrack21)
    fi
    if [[ -n "$post_config" ]]; then
        extra_args+=(--post-config "$post_config" --postproc-name "$postproc_name")
    fi
    if [[ "$mode" == "topdown" ]]; then
        extra_args+=(
            --det-config "$det_config"
            --det-checkpoint "$det_checkpoint"
            --det-batch-size "$det_batch_size"
            --det-metrics
            --det-cat-id "$det_cat_id"
            --nms-thr "$nms_thr"
            --bbox-thr "$BBOX_THR"
        )
    fi

    local label="${name} / ${variant}"
    if [[ -n "$post_config" ]]; then
        label="${label} + ${postproc_name}"
    fi

    TOTAL=$((TOTAL + 1))
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${label}"
    echo "  Test set:   ${TEST_DATASET}"
    echo "  Mode:       ${mode}"
    echo "  Config:     ${config}"
    echo "  Checkpoint: ${checkpoint}"
    echo "  KP batch:   ${kp_batch_size}"
    if [[ "$mode" == "topdown" ]]; then
        echo "  Detector:   ${det_config}"
        echo "  Det ckpt:   ${det_checkpoint}"
        echo "  Det cat:    ${det_cat_id}"
        echo "  Det batch:  ${det_batch_size}"
        echo "  NMS thr:    ${nms_thr}"
    fi
    if [[ "$official_pt21" == "1" ]]; then
        echo "  Official:   PoseTrack21 kit"
    fi
    echo "  Post-config:${post_config:-"(none)"}"
    echo "  Results:    ${RESULTS_FILE}"
    echo "  Log file:   ${log_file}"
    echo "============================================================"

    if [[ -n "$post_config" && "$DEVICE" != "cuda:0" ]] && \
        grep -qE "device=['\"]cuda:0['\"]" "$post_config"; then
        echo "Warning: ${post_config} pins an embedder to cuda:0, while DEVICE=${DEVICE}."
    fi

    if python tools/benchmark_e2e.py \
        "$config" \
        "$checkpoint" \
        --test-dataset "$TEST_DATASET" \
        --kp-batch-size "$kp_batch_size" \
        --queue-strategy full_batch \
        --device "$DEVICE" \
        --results-file "$RESULTS_FILE" \
        --model-name "$name" \
        --model-variant "$variant" \
        --pred-dir "$pred_dir" \
        "${extra_args[@]}" \
        >"$log_file" 2>&1; then
        PASSED=$((PASSED + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${label}"
        echo "  Predictions: ${pred_out}/"
    else
        FAILED+=("${name}/${variant}")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ${label} (see ${log_file})"
    fi
}

if [[ ! -f "$CSV" ]]; then
    echo "Error: CSV not found at ${CSV}" >&2
    exit 1
fi

while IFS=',' read -r config checkpoint name variant mode kp_batch_size \
    det_config det_checkpoint det_cat_id det_batch_size nms_thr \
    post_config postproc_name keypoint_space official_pt21 \
    || [[ -n "${config:-}" ]]; do
    config="$(strip_cr "$config")"
    checkpoint="$(strip_cr "$checkpoint")"
    name="$(strip_cr "$name")"
    variant="$(strip_cr "$variant")"
    mode="$(strip_cr "$mode")"
    kp_batch_size="$(strip_cr "${kp_batch_size:-}")"
    det_config="$(strip_cr "${det_config:-}")"
    det_checkpoint="$(strip_cr "${det_checkpoint:-}")"
    det_cat_id="$(strip_cr "${det_cat_id:-}")"
    det_batch_size="$(strip_cr "${det_batch_size:-}")"
    nms_thr="$(strip_cr "${nms_thr:-}")"
    post_config="$(strip_cr "${post_config:-}")"
    postproc_name="$(strip_cr "${postproc_name:-}")"
    keypoint_space="$(strip_cr "${keypoint_space:-}")"
    official_pt21="$(strip_cr "${official_pt21:-}")"
    official_pt21="${official_pt21,,}"

    if [[ -z "$config" || "$config" == \#* ]]; then
        continue
    fi

    local_label="${name:-?}/${variant:-?}"

    if [[ -z "$name" || -z "$variant" ]]; then
        fail_row "$local_label" "Name and Variant are required"
        continue
    fi
    case "$mode" in
        bottomup|tracker)
            if [[ -z "$kp_batch_size" ]]; then
                kp_batch_size="$KP_BATCH_SIZE_SEQUENTIAL"
            fi
            ;;
        topdown)
            if [[ -z "$kp_batch_size" ]]; then
                kp_batch_size="$KP_BATCH_SIZE"
            fi
            ;;
        *)
            if [[ -z "$kp_batch_size" ]]; then
                kp_batch_size="$KP_BATCH_SIZE_SEQUENTIAL"
            fi
            ;;
    esac
    if [[ -z "$det_batch_size" ]]; then
        det_batch_size="$DET_BATCH_SIZE"
    fi
    if [[ -z "$nms_thr" ]]; then
        nms_thr="$NMS_THR"
    fi
    if [[ "$official_pt21" == "true" || "$official_pt21" == "yes" ]]; then
        official_pt21="1"
    fi

    if [[ ! -f "$config" ]]; then
        fail_row "$local_label" "pose config not found at ${config}"
        continue
    fi
    if [[ -n "$post_config" && -z "$postproc_name" ]]; then
        fail_row "$local_label" "PostprocName is required when PostConfig is set"
        continue
    fi
    if [[ -z "$post_config" && -n "$postproc_name" ]]; then
        fail_row "$local_label" "PostConfig is required when PostprocName is set"
        continue
    fi
    if [[ -n "$post_config" && ! -f "$post_config" ]]; then
        fail_row "$local_label" "post-config not found at ${post_config}"
        continue
    fi
    if [[ -n "$keypoint_space" && "$keypoint_space" != "coco" && "$keypoint_space" != "dataset" ]]; then
        fail_row "$local_label" "KeypointSpace must be coco or dataset (got '${keypoint_space}')"
        continue
    fi
    if [[ -n "$official_pt21" && "$official_pt21" != "0" && "$official_pt21" != "1" ]]; then
        fail_row "$local_label" "OfficialPT21 must be 0 or 1 (got '${official_pt21}')"
        continue
    fi
    if [[ "$official_pt21" == "1" && "$TEST_DATASET" != "posetrack21" ]]; then
        fail_row "$local_label" "OfficialPT21=1 requires TEST_DATASET=posetrack21"
        continue
    fi
    if [[ "$TEST_DATASET" == "posetrack21" ]]; then
        official_pt21="1"
    fi

    pose_tracks=0
    det_tracks=0
    if config_emits_track_ids "$config"; then
        pose_tracks=1
    fi
    if [[ -n "$det_config" ]] && config_emits_track_ids "$det_config"; then
        det_tracks=1
    fi
    if [[ "$pose_tracks" == "1" && "$det_tracks" == "1" ]]; then
        fail_row "$local_label" "pose config and detector both set emits_track_ids"
        continue
    fi
    if [[ -n "$post_config" && ( "$pose_tracks" == "1" || "$det_tracks" == "1" ) ]]; then
        fail_row "$local_label" \
            "PostConfig would replace track ids already assigned by the model or detector"
        continue
    fi

    case "$mode" in
        bottomup)
            if [[ -n "$det_config" || -n "$det_checkpoint" ]]; then
                fail_row "$local_label" "bottomup rows must leave DetConfig and DetCheckpoint empty"
                continue
            fi
            ;;
        tracker)
            if [[ -n "$det_config" || -n "$det_checkpoint" ]]; then
                fail_row "$local_label" "tracker rows must leave DetConfig and DetCheckpoint empty"
                continue
            fi
            if [[ -n "$post_config" ]]; then
                fail_row "$local_label" \
                    "tracker rows cannot take PostConfig; the pose model assigns its own track ids"
                continue
            fi
            if [[ "$pose_tracks" != "1" ]]; then
                fail_row "$local_label" \
                    "tracker mode requires emits_track_ids = True in ${config}"
                continue
            fi
            ;;
        topdown)
            if [[ -z "$det_config" || -z "$det_checkpoint" ]]; then
                fail_row "$local_label" "topdown rows require DetConfig and DetCheckpoint"
                continue
            fi
            if [[ -z "$det_cat_id" ]]; then
                fail_row "$local_label" \
                    "topdown rows require DetCatId (RF-DETR person is 1, single-class MOTIP is 0)"
                continue
            fi
            if [[ ! -f "$det_config" ]]; then
                fail_row "$local_label" "detector config not found at ${det_config}"
                continue
            fi
            ;;
        *)
            fail_row "$local_label" \
                "unknown Mode '${mode}' (expected bottomup, topdown, or tracker)"
            continue
            ;;
    esac

    run_benchmark "$config" "$checkpoint" "$name" "$variant" "$mode" \
        "$kp_batch_size" "$det_config" "$det_checkpoint" "$det_cat_id" \
        "$det_batch_size" "$nms_thr" "$post_config" "$postproc_name" \
        "$keypoint_space" "$official_pt21"
done < <(tail -n +2 "$CSV")

echo ""
echo "============================================================"
echo "Pose-tracking sweep complete"
echo "  Test set:    ${TEST_DATASET}"
echo "  Total runs:  ${TOTAL}"
echo "  Passed:      ${PASSED}"
echo "  Failed:      $((TOTAL - PASSED))"
echo "  Logs:        ${LOG_DIR}"
echo "  Results:     ${RESULTS_FILE}"
if ((${#FAILED[@]} > 0)); then
    echo ""
    echo "Failed runs:"
    for entry in "${FAILED[@]}"; do
        echo "  - ${entry}"
    done
fi
echo "============================================================"

if ((${#FAILED[@]} > 0)); then
    exit 1
fi
