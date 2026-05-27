#!/bin/bash
# Run top-down pose benchmarks for every model in benchmark_configs_topdown.csv,
# using RF-DETR and RTMDet as person detectors.

set -uo pipefail

TIMESTAMP="$(date '+%Y%m%d')"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV="${CSV:-${ROOT_DIR}/scripts/benchmark_configs_topdown.csv}"
TEST_DATASET="${TEST_DATASET:-coco}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/benchmark/logs/${TIMESTAMP}_${TEST_DATASET}_topdown}"
RESULTS_FILE="${RESULTS_FILE:-${ROOT_DIR}/benchmark/results/${TIMESTAMP}_${TEST_DATASET}_topdown.json}"
DEVICE="${DEVICE:-cuda:7}"

RFDET_CONFIG="demo/mmdetection_cfg/rfdetr_medium_coco-person.py"
RFDET_CHECKPOINT="rf-detr-medium.pth"
RTMDET_CONFIG="demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
RTMDET_CHECKPOINT="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"

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
    local det_suffix="$5"
    local det_config="$6"
    local det_checkpoint="$7"
    local det_cat_id="$8"

    local full_variant="${variant}-${det_suffix}"
    local log_file="${LOG_DIR}/${name}-${full_variant}.log"

    TOTAL=$((TOTAL + 1))
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${name} / ${full_variant}"
    echo "  Test set:   ${TEST_DATASET}"
    echo "  Config:     ${config}"
    echo "  Detector:   ${det_config}"
    echo "  Log file:   ${log_file}"
    echo "============================================================"

    if python tools/benchmark_e2e.py \
        "$config" \
        "$checkpoint" \
        --test-dataset "$TEST_DATASET" \
        --det-config "$det_config" \
        --det-checkpoint "$det_checkpoint" \
        --det-batch-size 32 \
        --kp-batch-size 32 \
        --queue-strategy full_batch \
        --det-metrics \
        --det-cat-id "$det_cat_id" \
        --device "$DEVICE" \
        --results-file "$RESULTS_FILE" \
        --model-name "$name" \
        --model-variant "$full_variant" \
        >"$log_file" 2>&1; then
        PASSED=$((PASSED + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${name} / ${full_variant}"
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

while IFS=',' read -r config checkpoint name variant || [[ -n "${config:-}" ]]; do
    config="$(strip_cr "$config")"
    checkpoint="$(strip_cr "$checkpoint")"
    name="$(strip_cr "$name")"
    variant="$(strip_cr "$variant")"

    if [[ -z "$config" ]]; then
        continue
    fi

    run_benchmark "$config" "$checkpoint" "$name" "$variant" \
        "rfdetr" "$RFDET_CONFIG" "$RFDET_CHECKPOINT" 1
    run_benchmark "$config" "$checkpoint" "$name" "$variant" \
        "rtmdet" "$RTMDET_CONFIG" "$RTMDET_CHECKPOINT" 0
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
