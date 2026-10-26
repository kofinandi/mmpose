#!/bin/bash
# Cross every pose model in benchmark_configs_detector_sweep.csv with every
# detector in benchmark_configs_detectors.csv.
#
# 4 pose estimators (ViTPose-large, PCT-large, UDP HRNet-W48 large-input,
# RTMPose-large) x 17 detectors = 68 runs. Each run is appended to one
# results JSON, keyed by pose name and "{pose variant}-{detector suffix}".
#
# RF-DETR emits COCO category ids, so person is CatId 1. YOLO26, the
# MMDetection COCO RTMDet models (tiny/s/m/l/x), and Faster R-CNN R50-FPN
# use contiguous labels, so person is CatId 0.
# RF-DETR large / xlarge are compiled at batch 16. RF-DETR 2xlarge is
# compiled at batch 4, and Faster R-CNN uses detection batch 4, so both
# fit next to the pose model on a 24 GB GPU.
#
#   TEST_DATASET=coco DEVICE=cuda:7 ./scripts/run_detector_sweep_benchmarks.sh

set -uo pipefail

TIMESTAMP="$(date '+%Y%m%d')"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSE_CSV="${POSE_CSV:-${ROOT_DIR}/scripts/benchmark_configs_detector_sweep.csv}"
DET_CSV="${DET_CSV:-${ROOT_DIR}/scripts/benchmark_configs_detectors.csv}"
TEST_DATASET="${TEST_DATASET:-coco}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/benchmark/logs/${TIMESTAMP}_${TEST_DATASET}_detector_sweep}"
RESULTS_FILE="${RESULTS_FILE:-${ROOT_DIR}/benchmark/results/${TIMESTAMP}_${TEST_DATASET}_detector_sweep.json}"
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
    local det_suffix="$5"
    local det_config="$6"
    local det_checkpoint="$7"
    local det_cat_id="$8"
    local det_batch_size="$9"

    local full_variant="${variant}-${det_suffix}"
    local log_file="${LOG_DIR}/${name}-${full_variant}.log"
    local pred_dir="benchmark/predictions/${TIMESTAMP}_${TEST_DATASET}_detector_sweep/${name}-${full_variant}"

    TOTAL=$((TOTAL + 1))
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${name} / ${full_variant}"
    echo "  Test set:   ${TEST_DATASET}"
    echo "  Config:     ${config}"
    echo "  Detector:   ${det_config}"
    echo "  Det ckpt:   ${det_checkpoint}"
    echo "  Det cat:    ${det_cat_id}"
    echo "  Det batch:  ${det_batch_size}"
    echo "  Log file:   ${log_file}"
    echo "============================================================"

    if python tools/benchmark_e2e.py \
        "$config" \
        "$checkpoint" \
        --test-dataset "$TEST_DATASET" \
        --det-config "$det_config" \
        --det-checkpoint "$det_checkpoint" \
        --det-batch-size "$det_batch_size" \
        --kp-batch-size "$KP_BATCH_SIZE" \
        --queue-strategy full_batch \
        --det-metrics \
        --det-cat-id "$det_cat_id" \
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
if [[ ! -f "$DET_CSV" ]]; then
    echo "Error: detector CSV not found at ${DET_CSV}" >&2
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

    while IFS=',' read -r det_config det_checkpoint det_suffix det_cat_id det_batch_size \
        || [[ -n "${det_config:-}" ]]; do
        det_config="$(strip_cr "$det_config")"
        det_checkpoint="$(strip_cr "$det_checkpoint")"
        det_suffix="$(strip_cr "$det_suffix")"
        det_cat_id="$(strip_cr "$det_cat_id")"
        det_batch_size="$(strip_cr "${det_batch_size:-}")"

        if [[ -z "$det_config" || "$det_config" == \#* || "$det_config" == "Config" ]]; then
            continue
        fi
        if [[ -z "$det_batch_size" ]]; then
            det_batch_size=32
        fi

        run_benchmark "$config" "$checkpoint" "$name" "$variant" \
            "$det_suffix" "$det_config" "$det_checkpoint" \
            "$det_cat_id" "$det_batch_size"
    done < "$DET_CSV"
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
