#!/bin/bash
# Run postprocess_predictions.py on every prediction bundle under one or more
# directories. Each bundle must contain manifest.json and frames.json.
# Output goes to a sibling of the run folder containing each bundle, with
# "_<postproc-name>" appended to the run folder's name, e.g.:
#   benchmark/predictions/20260715_emdb_e2e/YOLO-Pose-tiny
#   -->
#   benchmark/predictions/20260715_emdb_e2e_smoothnetw8/YOLO-Pose-tiny

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date '+%Y%m%d')"
POST_CONFIG="${POST_CONFIG:-}"
POSTPROC_NAME="${POSTPROC_NAME:-}"
NUM_FRAMES="${NUM_FRAMES:-}"
RESULTS_FILE="${RESULTS_FILE:-}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/benchmark/logs/${TIMESTAMP}_postproc}"
METRIC_TYPES=(CocoMetric MPJVE MPJAE IDSwitch)
METRICS_WERE_SET=0

usage() {
    cat <<EOF
Usage: $(basename "$0") --post-config CONFIG --postproc-name NAME [OPTIONS] PRED_PARENT_DIR [PRED_PARENT_DIR ...]

Apply the same post-processing pipeline to every prediction bundle found under
the given directories. A bundle is a directory containing manifest.json and
frames.json (e.g. benchmark/predictions/DATE_DATASET_MODE/MODEL-VARIANT/).

Options:
  --post-config PATH    Post-processing config (required unless POST_CONFIG is set)
  --postproc-name NAME  Name of this post-processing run, e.g. "smoothnetw8"
                        (required unless POSTPROC_NAME is set). Used to build
                        the output dir by appending "_NAME" to the run folder
                        name containing each bundle.
  --metrics M1 M2 ...   Metrics to evaluate (default: CocoMetric MPJVE MPJAE IDSwitch)
  --num-frames N        Limit to the first N frames (for quick tests)
  --results-file PATH   Append metrics to this JSON file after each run
  -h, --help            Show this help

Environment:
  POST_CONFIG, POSTPROC_NAME, METRICS, NUM_FRAMES, RESULTS_FILE, LOG_DIR

Examples:
  $(basename "$0") \\
      --post-config configs/post_processing/oks_track_one_euro.py \\
      --postproc-name one_euro \\
      --results-file benchmark/results/${TIMESTAMP}_postproc.json \\
      benchmark/predictions/20260615_coco_topdown

  POST_CONFIG=configs/post_processing/oks_track_one_euro.py \\
      POSTPROC_NAME=one_euro \\
      RESULTS_FILE=benchmark/results/${TIMESTAMP}_postproc.json \\
      $(basename "$0") benchmark/predictions/20260615_coco_topdown \\
      benchmark/predictions/20260615_coco_e2e
EOF
}

is_prediction_bundle() {
    local dir="$1"
    [[ -f "${dir}/manifest.json" && -f "${dir}/frames.json" ]]
}

collect_bundles() {
    local dir="$1"
    local sub bundle

    if is_prediction_bundle "$dir"; then
        printf '%s\n' "$dir"
        return 0
    fi

    shopt -s nullglob
    for sub in "$dir"/*; do
        [[ -d "$sub" ]] || continue
        if is_prediction_bundle "$sub"; then
            printf '%s\n' "$sub"
        fi
    done
    shopt -u nullglob
}

FAILED=()
TOTAL=0
PASSED=0
PRED_DIRS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --post-config)
            [[ $# -ge 2 ]] || { echo "Error: --post-config requires a path" >&2; exit 1; }
            POST_CONFIG="$2"
            shift 2
            ;;
        --postproc-name)
            [[ $# -ge 2 ]] || { echo "Error: --postproc-name requires a value" >&2; exit 1; }
            POSTPROC_NAME="$2"
            shift 2
            ;;
        --metrics)
            shift
            METRIC_TYPES=()
            while [[ $# -gt 0 && "$1" != --* && "$1" != -* ]]; do
                METRIC_TYPES+=("$1")
                shift
            done
            if ((${#METRIC_TYPES[@]} == 0)); then
                echo "Error: --metrics requires at least one metric name" >&2
                exit 1
            fi
            METRICS_WERE_SET=1
            ;;
        --num-frames)
            [[ $# -ge 2 ]] || { echo "Error: --num-frames requires a value" >&2; exit 1; }
            NUM_FRAMES="$2"
            shift 2
            ;;
        --results-file)
            [[ $# -ge 2 ]] || { echo "Error: --results-file requires a path" >&2; exit 1; }
            RESULTS_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            PRED_DIRS+=("$1")
            shift
            ;;
    esac
done

if [[ "$METRICS_WERE_SET" -eq 0 && -n "${METRICS:-}" ]]; then
    read -r -a METRIC_TYPES <<< "$METRICS"
fi

if [[ -z "$POST_CONFIG" ]]; then
    echo "Error: --post-config is required (or set POST_CONFIG)" >&2
    usage >&2
    exit 1
fi

if [[ -z "$POSTPROC_NAME" ]]; then
    echo "Error: --postproc-name is required (or set POSTPROC_NAME)" >&2
    usage >&2
    exit 1
fi

if ((${#PRED_DIRS[@]} == 0)); then
    echo "Error: at least one prediction directory is required" >&2
    usage >&2
    exit 1
fi

if [[ ! -f "$POST_CONFIG" ]]; then
    echo "Error: post-config not found at ${POST_CONFIG}" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

BUNDLES=()
for parent_dir in "${PRED_DIRS[@]}"; do
    if [[ ! -d "$parent_dir" ]]; then
        echo "Error: directory not found: ${parent_dir}" >&2
        exit 1
    fi

    mapfile -t found < <(collect_bundles "$parent_dir")
    if ((${#found[@]} == 0)); then
        echo "Warning: no prediction bundles found under ${parent_dir}" >&2
        continue
    fi
    BUNDLES+=("${found[@]}")
done

if ((${#BUNDLES[@]} == 0)); then
    echo "Error: no prediction bundles to process" >&2
    exit 1
fi

run_postprocess() {
    local pred_dir="$1"
    local rel="${pred_dir#${ROOT_DIR}/}"
    local log_slug="${rel//\//__}"
    local log_file="${LOG_DIR}/${log_slug}.log"
    local model_label run_dir run_name predictions_dir out_dir
    model_label="$(basename "$pred_dir")"
    run_dir="$(dirname "$pred_dir")"
    run_name="$(basename "$run_dir")"
    predictions_dir="$(dirname "$run_dir")"
    out_dir="${predictions_dir}/${run_name}_${POSTPROC_NAME}/${model_label}"
    local -a cmd=(
        python tools/postprocess_predictions.py
        "$pred_dir"
        --post-config "$POST_CONFIG"
        --postproc-name "$POSTPROC_NAME"
        --metrics "${METRIC_TYPES[@]}"
    )

    if [[ -n "$NUM_FRAMES" ]]; then
        cmd+=(--num-frames "$NUM_FRAMES")
    fi

    if [[ -n "$RESULTS_FILE" ]]; then
        cmd+=(--results-file "$RESULTS_FILE")
    fi

    TOTAL=$((TOTAL + 1))
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${pred_dir}"
    echo "  Post-config: ${POST_CONFIG}"
    echo "  Postproc:    ${POSTPROC_NAME}"
    echo "  Metrics:     ${METRIC_TYPES[*]}"
    echo "  Output:      ${out_dir}"
    if [[ -n "$RESULTS_FILE" ]]; then
        echo "  Results:     ${RESULTS_FILE}"
    fi
    echo "  Log file:    ${log_file}"
    echo "============================================================"

    if "${cmd[@]}" >"$log_file" 2>&1; then
        PASSED=$((PASSED + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${pred_dir}"
        echo "  Post-processed bundle: ${out_dir}/"
    else
        FAILED+=("$pred_dir")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ${pred_dir} (see ${log_file})"
    fi
}

for pred_dir in "${BUNDLES[@]}"; do
    run_postprocess "$pred_dir"
done

echo ""
echo "============================================================"
echo "Post-processing sweep complete"
echo "  Post-config: ${POST_CONFIG}"
echo "  Postproc:    ${POSTPROC_NAME}"
echo "  Total runs:  ${TOTAL}"
echo "  Passed:      ${PASSED}"
echo "  Failed:      $((TOTAL - PASSED))"
echo "  Logs:        ${LOG_DIR}"
if [[ -n "$RESULTS_FILE" ]]; then
    echo "  Results:     ${RESULTS_FILE}"
fi
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
