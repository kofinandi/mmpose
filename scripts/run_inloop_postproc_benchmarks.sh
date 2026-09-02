#!/bin/bash
# Sweep pose models on PoseTrack21, once raw and once per in-loop
# post-processing pipeline.
#
# 5 models x (1 raw + 3 post-processing configs) = 20 sequential runs.
# Topdown models use a single detector (RF-DETR-medium) so the count
# stays at 20.
#
# Each condition writes its own results JSON:
#   benchmark/results/{date}_posetrack21_raw.json
#   benchmark/results/{date}_posetrack21_{postproc_name}.json
#
# Raw predictions land under the shared run folder; post-processed
# bundles are siblings with the postproc name appended, e.g.:
#   benchmark/predictions/{date}_posetrack21_inloop/DETRPose-m
#   benchmark/predictions/{date}_posetrack21_inloop_oks_nms_pgpt_geom/DETRPose-m
#
# Usage:
#   ./scripts/run_inloop_postproc_benchmarks.sh
#   DEVICE=cuda:0 ./scripts/run_inloop_postproc_benchmarks.sh
#   NUM_FRAMES=64 ./scripts/run_inloop_postproc_benchmarks.sh   # quick test

set -uo pipefail

TIMESTAMP="$(date '+%Y%m%d')"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV="${CSV:-${ROOT_DIR}/scripts/benchmark_configs_inloop_postproc.csv}"
TEST_DATASET="${TEST_DATASET:-posetrack21}"
SUFFIX="${SUFFIX:-inloop}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/benchmark/logs/${TIMESTAMP}_${TEST_DATASET}_${SUFFIX}}"
DEVICE="${DEVICE:-cuda:7}"
KP_BATCH_SIZE="${KP_BATCH_SIZE:-1}"
DET_BATCH_SIZE="${DET_BATCH_SIZE:-1}"
PREFETCH_CHUNK_SIZE="${PREFETCH_CHUNK_SIZE:-256}"
NUM_FRAMES="${NUM_FRAMES:-}"

RFDET_CONFIG="demo/mmdetection_cfg/rfdetr_medium_coco-person.py"
RFDET_CHECKPOINT="rf-detr-medium.pth"
RFDET_CAT_ID=1

# post-config path -> --postproc-name (also names the results file)
POST_CONFIGS=(
    "configs/post_processing/oks_nms_pgpt_geom.py|oks_nms_pgpt_geom"
    "configs/post_processing/oks_nms_pgpt.py|oks_nms_pgpt"
    "configs/post_processing/oks_nms_boxmot_occluboost.py|oks_nms_boxmot_occluboost"
)

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

FAILED=()
TOTAL=0
PASSED=0

run_benchmark() {
    local config="$1"
    local checkpoint="$2"
    local name="$3"
    local full_variant="$4"
    local mode="$5"
    local post_config="${6:-}"
    local postproc_name="${7:-raw}"

    local log_file="${LOG_DIR}/${name}-${full_variant}-${postproc_name}.log"
    local pred_dir="benchmark/predictions/${TIMESTAMP}_${TEST_DATASET}_${SUFFIX}/${name}-${full_variant}"
    local results_file="${ROOT_DIR}/benchmark/results/${TIMESTAMP}_${TEST_DATASET}_${postproc_name}.json"
    local pred_out="$pred_dir"
    if [[ -n "$post_config" ]]; then
        pred_out="${pred_dir%/*}_${postproc_name}/${name}-${full_variant}"
    fi

    local extra_args=(--prefetch-chunk-size "$PREFETCH_CHUNK_SIZE")
    if [[ -n "$post_config" ]]; then
        extra_args+=(--post-config "$post_config" --postproc-name "$postproc_name")
    fi
    if [[ -n "$NUM_FRAMES" ]]; then
        extra_args+=(--num-frames "$NUM_FRAMES")
    fi
    if [[ "$mode" == "topdown" ]]; then
        extra_args+=(
            --det-config "$RFDET_CONFIG"
            --det-checkpoint "$RFDET_CHECKPOINT"
            --det-batch-size "$DET_BATCH_SIZE"
            --det-metrics
            --det-cat-id "$RFDET_CAT_ID"
            --nms-thr 0.95
        )
    fi

    local label="${name} / ${full_variant}"
    if [[ -n "$post_config" ]]; then
        label="${label} + ${postproc_name}"
    else
        label="${label} (raw)"
    fi

    TOTAL=$((TOTAL + 1))
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${label}"
    echo "  Test set:   ${TEST_DATASET}"
    echo "  Config:     ${config}"
    echo "  Checkpoint: ${checkpoint}"
    echo "  Mode:       ${mode}"
    echo "  Post-config:${post_config:-"(none)"}"
    echo "  KP batch:   ${KP_BATCH_SIZE}"
    echo "  Results:    ${results_file}"
    echo "  Log file:   ${log_file}"
    echo "============================================================"

    if python tools/benchmark_e2e.py \
        "$config" \
        "$checkpoint" \
        --test-dataset "$TEST_DATASET" \
        --kp-batch-size "$KP_BATCH_SIZE" \
        --queue-strategy full_batch \
        --device "$DEVICE" \
        --results-file "$results_file" \
        --model-name "$name" \
        --model-variant "$full_variant" \
        --include-bad-frames \
        --pred-dir "$pred_dir" \
        "${extra_args[@]}" \
        >"$log_file" 2>&1; then
        PASSED=$((PASSED + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${label}"
        echo "  Predictions: ${pred_out}/"
    else
        FAILED+=("${name}/${full_variant}/${postproc_name}")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ${label} (see ${log_file})"
    fi
}

strip_cr() {
    printf '%s' "$1" | tr -d '\r'
}

if [[ ! -f "$CSV" ]]; then
    echo "Error: CSV not found at ${CSV}" >&2
    exit 1
fi

for_each_model() {
    local post_config="${1:-}"
    local postproc_name="${2:-}"

    while IFS=',' read -r config checkpoint name variant mode \
        || [[ -n "${config:-}" ]]; do
        config="$(strip_cr "$config")"
        checkpoint="$(strip_cr "$checkpoint")"
        name="$(strip_cr "$name")"
        variant="$(strip_cr "$variant")"
        mode="$(strip_cr "$mode")"

        if [[ -z "$config" || "$config" == \#* ]]; then
            continue
        fi

        local_variant="$variant"
        if [[ "$mode" == "topdown" ]]; then
            local_variant="${variant}-rfdetr"
        fi

        case "$mode" in
            topdown|bottomup)
                run_benchmark "$config" "$checkpoint" "$name" \
                    "$local_variant" "$mode" "$post_config" "$postproc_name"
                ;;
            *)
                echo "Error: unknown Mode '${mode}' for ${name}/${variant} " \
                    "(expected 'topdown' or 'bottomup')" >&2
                FAILED+=("${name}/${variant} (bad Mode '${mode}')")
                TOTAL=$((TOTAL + 1))
                ;;
        esac
    done < <(tail -n +2 "$CSV")
}

for_each_model
for post_entry in "${POST_CONFIGS[@]}"; do
    post_config="${post_entry%%|*}"
    postproc_name="${post_entry##*|}"
    if [[ ! -f "$post_config" ]]; then
        echo "Error: post-config not found at ${post_config}" >&2
        exit 1
    fi
    for_each_model "$post_config" "$postproc_name"
done

echo ""
echo "============================================================"
echo "In-loop post-processing sweep complete"
echo "  Test set:    ${TEST_DATASET}"
echo "  Total runs:  ${TOTAL}"
echo "  Passed:      ${PASSED}"
echo "  Failed:      $((TOTAL - PASSED))"
echo "  Logs:        ${LOG_DIR}"
echo "  Results:"
echo "    ${ROOT_DIR}/benchmark/results/${TIMESTAMP}_${TEST_DATASET}_raw.json"
for post_entry in "${POST_CONFIGS[@]}"; do
    postproc_name="${post_entry##*|}"
    echo "    ${ROOT_DIR}/benchmark/results/${TIMESTAMP}_${TEST_DATASET}_${postproc_name}.json"
done
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
