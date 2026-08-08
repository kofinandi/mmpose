#!/bin/bash
# Run benchmarks for every (video) model in benchmark_configs_video.csv, on
# any video (or image) test set supported by tools/benchmark_e2e.py.
#
# Each CSV row is one multi-frame video model config + checkpoint, tagged
# with a `Mode` of either:
#   - topdown:  needs person detections. Swept against two detector
#               strategies, mirroring run_topdown_coco_benchmarks.sh:
#                 - "rfdetr" : RF-DETR-medium
#                 - "rtmdet" : RTMDet-medium
#   - bottomup: end-to-end (e.g. PAVE-Net), run once with no detector,
#               mirroring run_e2e_coco_benchmarks.sh.
#
# All predictions are saved under one {date}_{dataset}_video/ folder
# (rather than tools/benchmark_e2e.py's own topdown/e2e split), since every
# row here is a video model regardless of Mode.
#
# Test set is configurable (default: emdb-mini) since these are video
# models -- e.g.:
#   TEST_DATASET=emdb ./scripts/run_video_benchmarks.sh
#   TEST_DATASET=3dpw ./scripts/run_video_benchmarks.sh
#   TEST_DATASET=posetrack21 ./scripts/run_video_benchmarks.sh
#
# For large test sets, stream frames in chunks instead of eager-loading
# everything upfront, e.g.:
#   PREFETCH_CHUNK_SIZE=256 TEST_DATASET=emdb ./scripts/run_video_benchmarks.sh

set -uo pipefail

TIMESTAMP="$(date '+%Y%m%d')"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV="${CSV:-${ROOT_DIR}/scripts/benchmark_configs_video.csv}"
TEST_DATASET="${TEST_DATASET:-emdb-mini}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/benchmark/logs/${TIMESTAMP}_${TEST_DATASET}_video}"
RESULTS_FILE="${RESULTS_FILE:-${ROOT_DIR}/benchmark/results/${TIMESTAMP}_${TEST_DATASET}_video.json}"
DEVICE="${DEVICE:-cuda:7}"
KP_BATCH_SIZE="${KP_BATCH_SIZE:-8}"
# 0 (default) = eager-load the whole dataset upfront, matching
# run_e2e_coco_benchmarks.sh/run_topdown_coco_benchmarks.sh. Set e.g.
# PREFETCH_CHUNK_SIZE=256 to stream in chunks instead (recommended for
# large video test sets such as full EMDB/3DPW/PoseTrack21).
PREFETCH_CHUNK_SIZE="${PREFETCH_CHUNK_SIZE:-0}"

RFDET_CONFIG="demo/mmdetection_cfg/rfdetr_medium_coco-person.py"
RFDET_CHECKPOINT="rf-detr-medium.pth"
RTMDET_CONFIG="demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
RTMDET_CHECKPOINT="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

FAILED=()
TOTAL=0
PASSED=0

# Runs one topdown/bottomup config once, optionally against a detector.
# Common args ($1-$5); detector args ($6-$9) are only passed through when
# non-empty (bottomup models take none of them).
run_benchmark() {
    local config="$1"
    local checkpoint="$2"
    local name="$3"
    local full_variant="$4"
    local kp_batch_size="$5"
    local det_config="${6:-}"
    local det_checkpoint="${7:-}"
    local det_cat_id="${8:-}"

    local log_file="${LOG_DIR}/${name}-${full_variant}.log"
    local pred_dir="benchmark/predictions/${TIMESTAMP}_${TEST_DATASET}_video/${name}-${full_variant}"

    local extra_args=()
    if [[ "$PREFETCH_CHUNK_SIZE" -gt 0 ]]; then
        extra_args+=(--prefetch-chunk-size "$PREFETCH_CHUNK_SIZE")
    fi
    if [[ -n "$det_config" ]]; then
        extra_args+=(
            --det-config "$det_config"
            --det-checkpoint "$det_checkpoint"
            --det-batch-size 32
            --det-metrics
            --det-cat-id "$det_cat_id"
            --nms-thr 0.95
        )
    fi

    TOTAL=$((TOTAL + 1))
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${name} / ${full_variant}"
    echo "  Test set:   ${TEST_DATASET}"
    echo "  Config:     ${config}"
    echo "  Checkpoint: ${checkpoint}"
    echo "  KP batch:   ${kp_batch_size}"
    [[ -n "$det_config" ]] && echo "  Detector:   ${det_config}"
    echo "  Log file:   ${log_file}"
    echo "============================================================"

    if python tools/benchmark_e2e.py \
        "$config" \
        "$checkpoint" \
        --test-dataset "$TEST_DATASET" \
        --kp-batch-size "$kp_batch_size" \
        --queue-strategy full_batch \
        --device "$DEVICE" \
        --results-file "$RESULTS_FILE" \
        --model-name "$name" \
        --model-variant "$full_variant" \
        --include-bad-frames \
        --pred-dir "$pred_dir" \
        "${extra_args[@]}" \
        >"$log_file" 2>&1; then
        PASSED=$((PASSED + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${name} / ${full_variant}"
        echo "  Predictions: ${pred_dir}/"
    else
        FAILED+=("${name}/${full_variant}")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ${name} / ${full_variant} (see ${log_file})"
    fi
}

strip_cr() {
    printf '%s' "$1" | tr -d '\r'
}

if [[ ! -f "$CSV" ]]; then
    echo "Error: CSV not found at ${CSV}" >&2
    exit 1
fi

while IFS=',' read -r config checkpoint name variant mode kp_batch_size \
    || [[ -n "${config:-}" ]]; do
    config="$(strip_cr "$config")"
    checkpoint="$(strip_cr "$checkpoint")"
    name="$(strip_cr "$name")"
    variant="$(strip_cr "$variant")"
    mode="$(strip_cr "$mode")"
    kp_batch_size="$(strip_cr "${kp_batch_size:-}")"

    # Skip blank lines and "# ..." comment rows (e.g. entries with a known-
    # unusable checkpoint, documented inline in the CSV).
    if [[ -z "$config" || "$config" == \#* ]]; then
        continue
    fi

    if [[ -z "$kp_batch_size" ]]; then
        kp_batch_size="$KP_BATCH_SIZE"
    fi

    case "$mode" in
        topdown)
            run_benchmark "$config" "$checkpoint" "$name" "${variant}-rfdetr" \
                "$kp_batch_size" "$RFDET_CONFIG" "$RFDET_CHECKPOINT" 1
            run_benchmark "$config" "$checkpoint" "$name" "${variant}-rtmdet" \
                "$kp_batch_size" "$RTMDET_CONFIG" "$RTMDET_CHECKPOINT" 0
            ;;
        bottomup)
            run_benchmark "$config" "$checkpoint" "$name" "$variant" \
                "$kp_batch_size"
            ;;
        *)
            echo "Error: unknown Mode '${mode}' for ${name}/${variant} " \
                "(expected 'topdown' or 'bottomup')" >&2
            FAILED+=("${name}/${variant} (bad Mode '${mode}')")
            TOTAL=$((TOTAL + 1))
            ;;
    esac
done < <(tail -n +2 "$CSV")

echo ""
echo "============================================================"
echo "Benchmark sweep complete"
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
