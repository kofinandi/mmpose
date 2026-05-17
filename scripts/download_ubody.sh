#!/bin/bash

# UBody (CVPR 2023) — expected layout matches MMPose docs:
#   docs/en/user_guides/dataset_tools.md (UBody2D)
#
# Google Drive sources (file ids from /file/d/<id>/view links):
#   annotations.zip: https://drive.google.com/file/d/13TqjIWQgR6lErAJX7tzV1WzuPvkYqJj3/view
#   splits.zip:      https://drive.google.com/file/d/191ysR6Puxg4lv2pn8pA1QpqDe3hjPhkZ/view
#   videos.zip:      https://drive.google.com/file/d/19Pntpx4WVu96Btw4G4-W6s0oAE75LD3N/view
#
# Official project: https://github.com/IDEA-Research/OSX
#
# Requires: pip install -U gdown, unzip
#
# Usage:
#   bash scripts/download_ubody.sh [BASE_DIR] [optional videos Google Drive file id]
#
# Optional env overrides (same pattern as scripts/download_ochuman.sh):
#   UBODY_ANNOTATIONS_ZIP_ID, UBODY_SPLITS_ZIP_ID, UBODY_VIDEOS_ZIP_ID (defaults match URLs above)
#   UBODY_SKIP_VIDEOS=1  — skip videos.zip (e.g. videos already on disk)

BASE_DIR=${1:-"data/ubody"}

if [[ -n "${2:-}" ]]; then
    UBODY_VIDEOS_ZIP_ID="${UBODY_VIDEOS_ZIP_ID:-$2}"
fi

# Override file ids if mirrors change (values match the Drive URLs in the header).
: "${UBODY_ANNOTATIONS_ZIP_ID:=13TqjIWQgR6lErAJX7tzV1WzuPvkYqJj3}"
: "${UBODY_SPLITS_ZIP_ID:=191ysR6Puxg4lv2pn8pA1QpqDe3hjPhkZ}"
: "${UBODY_VIDEOS_ZIP_ID:=19Pntpx4WVu96Btw4G4-W6s0oAE75LD3N}"

echo "Targeting directory: $BASE_DIR"

mkdir -p "$BASE_DIR"

gdown_file() {
    local file_id=$1
    local output=$2
    if command -v gdown >/dev/null 2>&1; then
        gdown "https://drive.google.com/uc?id=${file_id}" -O "$output"
    elif python3 -m gdown --help >/dev/null 2>&1; then
        python3 -m gdown "https://drive.google.com/uc?id=${file_id}" -O "$output"
    else
        echo "Error: gdown is required for Google Drive downloads." >&2
        echo "Install with: pip install -U gdown" >&2
        exit 1
    fi
}

if ! command -v unzip >/dev/null 2>&1; then
    echo "Error: unzip not found. Install unzip (e.g. apt install unzip)." >&2
    exit 1
fi

fetch_zip() {
    local file_id=$1
    local filename=$2

    echo "------------------------------------------------"
    echo "Downloading $filename..."
    local zip_path="$BASE_DIR/$filename"
    gdown_file "$file_id" "$zip_path"

    echo "Extracting $filename..."
    unzip -q -o "$zip_path" -d "$BASE_DIR"
    rm -f "$zip_path"
}

echo "------------------------------------------------"
echo "Downloading annotations.zip..."
fetch_zip "$UBODY_ANNOTATIONS_ZIP_ID" "annotations.zip"

echo "------------------------------------------------"
echo "Downloading splits.zip..."
fetch_zip "$UBODY_SPLITS_ZIP_ID" "splits.zip"

if [[ "${UBODY_SKIP_VIDEOS:-0}" == "1" ]]; then
    echo "------------------------------------------------"
    echo "Skipping videos.zip (UBODY_SKIP_VIDEOS=1)."
else
    if [[ "$UBODY_VIDEOS_ZIP_ID" == "$UBODY_SPLITS_ZIP_ID" ]]; then
        echo "Error: UBODY_VIDEOS_ZIP_ID matches UBODY_SPLITS_ZIP_ID (splits.zip)." >&2
        echo "Use the file id from the separate videos.zip link on Google Drive." >&2
        exit 1
    fi
    echo "------------------------------------------------"
    echo "Downloading videos.zip (large)..."
    fetch_zip "$UBODY_VIDEOS_ZIP_ID" "videos.zip"
fi

echo "------------------------------------------------"
echo "Done! UBody dataset is ready in $BASE_DIR"
echo ""
echo "Expected structure (see docs/en/user_guides/dataset_tools.md):"
echo "  $BASE_DIR/"
echo "  ├── annotations/"
echo "  ├── splits/"
echo "  │   ├── inter_scene_test_list.npy"
echo "  │   └── intra_scene_test_list.npy"
echo "  └── videos/"
echo ""
echo "Convert videos to COCO-style train/val images and JSON with:"
echo "  python tools/dataset_converters/ubody_kpts_to_coco.py --data-root $BASE_DIR"
echo ""
