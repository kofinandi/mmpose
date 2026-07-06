#!/usr/bin/env bash
# Runs the GP-Kalman postprocessing evaluation with a hard timeout, then
# saves the runtime and the key quality metrics (+ their sum as "score")
# to a logfile instead of printing them to the terminal.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TIMEOUT_SECONDS=900
RESULT_LOG="$SCRIPT_DIR/run_evaluation.log"
RAW_LOG="$(mktemp)"
TIMEOUT_FLAG="$(mktemp -u)"

cleanup() {
    rm -f "$RAW_LOG" "$TIMEOUT_FLAG"
}
trap cleanup EXIT

START_TIME=$(date +%s)

python tools/postprocess_predictions.py \
    benchmark/predictions/20260622_emdb_topdown/ViTPose-small-rfdetr \
    --post-config configs/post_processing/gp_kalman.py > "$RAW_LOG" 2>&1 &
CMD_PID=$!

(
    sleep "$TIMEOUT_SECONDS"
    if kill -0 "$CMD_PID" 2>/dev/null; then
        touch "$TIMEOUT_FLAG"
        kill -TERM "$CMD_PID" 2>/dev/null
        sleep 2
        kill -KILL "$CMD_PID" 2>/dev/null
    fi
) &
WATCHER_PID=$!

wait "$CMD_PID" 2>/dev/null
EXIT_CODE=$?

kill "$WATCHER_PID" 2>/dev/null
wait "$WATCHER_PID" 2>/dev/null

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ -f "$TIMEOUT_FLAG" ]; then
    echo "Command timed out after ${TIMEOUT_SECONDS} seconds." > "$RESULT_LOG"
    exit 1
fi

if [ "$EXIT_CODE" -ne 0 ]; then
    {
        echo "Command failed with exit code ${EXIT_CODE}."
        echo "--- Full output ---"
        cat "$RAW_LOG"
    } > "$RESULT_LOG"
    exit "$EXIT_CODE"
fi

AR=$(grep -oE 'coco/AR: [0-9.]+' "$RAW_LOG" | awk '{print $2}')
BMPJVE=$(grep -oE 'emdb/bMPJVE: [0-9.]+' "$RAW_LOG" | awk '{print $2}')
BMPJAE=$(grep -oE 'emdb/bMPJAE: [0-9.]+' "$RAW_LOG" | awk '{print $2}')
SCORE=$(awk "BEGIN { print ${AR:-0} - 10*${BMPJVE:-0} - 10*${BMPJAE:-0} }")

{
    echo "Runtime: ${ELAPSED} s"
    echo "coco/AR: ${AR}"
    echo "emdb/bMPJVE: ${BMPJVE}"
    echo "emdb/bMPJAE: ${BMPJAE}"
    echo "score: ${SCORE}"
} > "$RESULT_LOG"
