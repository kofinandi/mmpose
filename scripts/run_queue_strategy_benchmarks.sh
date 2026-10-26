#!/bin/bash
# Run the pose models in benchmark_configs_detector_sweep.csv once per
# top-down queue strategy.
#
# 4 pose estimators (ViTPose-large, PCT-large, UDP HRNet-W48 large-input,
# RTMPose-large) x 3 strategies (full_batch, any, same_frame) = 12 runs.
# The detector stays RF-DETR medium so the comparison is the queue
# strategy. Each run is appended to one results JSON, keyed by pose name
# and "{pose variant}-{queue strategy}".
#
#   TEST_DATASET=coco DEVICE=cuda:7 ./scripts/run_queue_strategy_benchmarks.sh

set -uo pipefail

TIMESTAMP="$(date '+%Y%m%d')"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSE_CSV="${POSE_CSV:-${ROOT_DIR}/scripts/benchmark_configs_detector_sweep.csv}"
TEST_DATASET="${TEST_DATASET:-coco}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/benchmark/logs/${TIMESTAMP}_${TEST_DATASET}_queue_sweep}"
RESULTS_FILE="${RESULTS_FILE:-${ROOT_DIR}/benchmark/results/${TIMESTAMP}_${TEST_DATASET}_queue_sweep.json}"
DEVICE="${DEVICE:-cuda:7}"
KP_BATCH_SIZE="${KP_BATCH_SIZE:-32}"
DET_BATCH_SIZE="${DET_BATCH_SIZE:-32}"

DET_CONFIG="${DET_CONFIG:-demo/mmdetection_cfg/rfdetr_medium_coco-person.py}"
DET_CHECKPOINT="${DET_CHECKPOINT:-rf-detr-medium.pth}"
DET_CAT_ID="${DET_CAT_ID:-1}"

QUEUE_STRATEGIES=(full_batch any same_frame)

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

FAILED=()
TOTAL=0
PASSED=0

run_benchmark() {
    local config="$1"
    local checkpoint="$2"
    local name="$3"
    local variant="$4"
    local queue_strategy="$5"

    local full_variant="${variant}-${queue_strategy}"
    local log_file="${LOG_DIR}/${name}-${full_variant}.log"
    local pred_dir="benchmark/predictions/${TIMESTAMP}_${TEST_DATASET}_queue_sweep/${name}-${full_variant}"

    TOTAL=$((TOTAL + 1))
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${name} / ${full_variant}"
    echo "  Test set:   ${TEST_DATASET}"
    echo "  Config:     ${config}"
    echo "  Detector:   ${DET_CONFIG}"
    echo "  Queue:      ${queue_strategy}"
    echo "  KP batch:   ${KP_BATCH_SIZE}"
    echo "  Log file:   ${log_file}"
    echo "============================================================"

    if python tools/benchmark_e2e.py \
        "$config" \
        "$checkpoint" \
        --test-dataset "$TEST_DATASET" \
        --det-config "$DET_CONFIG" \
        --det-checkpoint "$DET_CHECKPOINT" \
        --det-batch-size "$DET_BATCH_SIZE" \
        --kp-batch-size "$KP_BATCH_SIZE" \
        --queue-strategy "$queue_strategy" \
        --det-metrics \
        --det-cat-id "$DET_CAT_ID" \
        --nms-thr 0.95 \
        --device "$DEVICE" \
        --results-file "$RESULTS_FILE" \
        --model-name "$name" \
        --model-variant "$full_variant" \
        --pred-dir "$pred_dir" \
        --include-bad-frames \
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

if [[ ! -f "$POSE_CSV" ]]; then
    echo "Error: pose CSV not found at ${POSE_CSV}" >&2
    exit 1
fi

echo "Results file: ${RESULTS_FILE}"

while IFS=',' read -r config checkpoint name variant || [[ -n "${config:-}" ]]; do
    config="$(strip_cr "$config")"
    checkpoint="$(strip_cr "$checkpoint")"
    name="$(strip_cr "$name")"
    variant="$(strip_cr "$variant")"

    if [[ -z "$config" || "$config" == \#* || "$config" == "Config" ]]; then
        continue
    fi

    for queue_strategy in "${QUEUE_STRATEGIES[@]}"; do
        run_benchmark "$config" "$checkpoint" "$name" "$variant" "$queue_strategy"
    done
done < "$POSE_CSV"

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
