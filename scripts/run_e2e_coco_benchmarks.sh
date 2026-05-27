#!/bin/bash
# Run end-to-end pose benchmarks for every model in benchmark_configs_e2e.csv.
# These are bottom-up models (e.g. PETR, YOLO-Pose) that do not use a detector.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV="${ROOT_DIR}/scripts/benchmark_configs_e2e.csv"
TEST_DATASET="${TEST_DATASET:-coco}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/benchmark_logs_e2e_${TEST_DATASET}}"
RESULTS_FILE="${RESULTS_FILE:-${ROOT_DIR}/${TEST_DATASET}_benchmark_e2e.json}"
DEVICE="${DEVICE:-cuda:7}"
KP_BATCH_SIZE="${KP_BATCH_SIZE:-32}"

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

    local log_file="${LOG_DIR}/${name}-${variant}.log"

    TOTAL=$((TOTAL + 1))
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${name} / ${variant}"
    echo "  Test set:   ${TEST_DATASET}"
    echo "  Config:     ${config}"
    echo "  Checkpoint: ${checkpoint}"
    echo "  Log file:   ${log_file}"
    echo "============================================================"

    if python tools/benchmark_e2e.py \
        "$config" \
        "$checkpoint" \
        --test-dataset "$TEST_DATASET" \
        --kp-batch-size "$KP_BATCH_SIZE" \
        --device "$DEVICE" \
        --results-file "$RESULTS_FILE" \
        --model-name "$name" \
        --model-variant "$variant" \
        >"$log_file" 2>&1; then
        PASSED=$((PASSED + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${name} / ${variant}"
    else
        FAILED+=("${name}/${variant}")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ${name} / ${variant} (see ${log_file})"
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

    run_benchmark "$config" "$checkpoint" "$name" "$variant"
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
