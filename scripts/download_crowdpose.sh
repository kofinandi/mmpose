#!/bin/bash

# Use the first argument as BASE_DIR; default to "data/crowdpose" if not provided
BASE_DIR=${1:-"data/crowdpose"}

echo "Targeting directory: $BASE_DIR"

mkdir -p "$BASE_DIR"

# --- Annotations + bbox detections (OpenMMLab, same as dataset zoo) ---
ANNO_URL="https://download.openmmlab.com/mmpose/datasets/crowdpose_annotations.tar"

echo "------------------------------------------------"
echo "Downloading and extracting MMPose CrowdPose annotations..."
wget -c "$ANNO_URL" -P "$BASE_DIR"
tar -xf "$BASE_DIR/crowdpose_annotations.tar" -C "$BASE_DIR"
rm -f "$BASE_DIR/crowdpose_annotations.tar"

# Tar unpacks to crowdpose_annotations/; configs expect annotations/
if [[ -d "$BASE_DIR/crowdpose_annotations" ]]; then
    if [[ ! -d "$BASE_DIR/annotations" ]]; then
        mv "$BASE_DIR/crowdpose_annotations" "$BASE_DIR/annotations"
    else
        mv "$BASE_DIR/crowdpose_annotations"/* "$BASE_DIR/annotations/"
        rmdir "$BASE_DIR/crowdpose_annotations" 2>/dev/null || rm -rf "$BASE_DIR/crowdpose_annotations"
    fi
fi

# --- Images (OpenXLab; replaces deprecated mim / odl flow) ---
# Requires: pip install -U openxlab && openxlab login
echo "------------------------------------------------"
echo "Downloading CrowdPose images from OpenXLab..."
echo "Note: requires 'openxlab login' to be completed beforehand."

if ! command -v openxlab >/dev/null 2>&1; then
    echo "Error: 'openxlab' not found. Install with: pip install -U openxlab" >&2
    exit 1
fi

OPENXLAB_DATASET="OpenDataLab/CrowdPose"
OPENXLAB_RAW_FILE="raw/CrowdPose.tar.gz"
ARCHIVE="$BASE_DIR/OpenDataLab___CrowdPose/raw/CrowdPose.tar.gz"

openxlab dataset download \
    --dataset-repo "$OPENXLAB_DATASET" \
    --source-path "/$OPENXLAB_RAW_FILE" \
    --target-path "$BASE_DIR"

if [[ ! -f "$ARCHIVE" ]]; then
    echo "Error: expected archive not found at $ARCHIVE" >&2
    exit 1
fi

echo "------------------------------------------------"
echo "Extracting images into $BASE_DIR/images ..."

EXTRACT_TMP=$(mktemp -d)
trap 'rm -rf "$EXTRACT_TMP"' EXIT

tar -xzf "$ARCHIVE" -C "$EXTRACT_TMP"

IMG_DIR=""
if [[ -d "$EXTRACT_TMP/crowdpose/images" ]]; then
    IMG_DIR="$EXTRACT_TMP/crowdpose/images"
else
    IMG_DIR=$(find "$EXTRACT_TMP" -type d -name images 2>/dev/null | head -n 1)
fi

if [[ -z "$IMG_DIR" || ! -d "$IMG_DIR" ]]; then
    echo "Error: could not locate an images/ directory inside the CrowdPose archive." >&2
    echo "Top-level paths in archive (for debugging):" >&2
    find "$EXTRACT_TMP" -mindepth 1 -maxdepth 3 -type d >&2
    exit 1
fi

mkdir -p "$BASE_DIR/images"
cp -a "$IMG_DIR"/. "$BASE_DIR/images/"

rm -f "$ARCHIVE"
rm -rf "$BASE_DIR/OpenDataLab___CrowdPose"

echo "------------------------------------------------"
echo "Done! CrowdPose dataset is ready in $BASE_DIR"
echo ""
echo "Expected structure:"
echo "  $BASE_DIR/"
echo "  ├── annotations/"
echo "  │   ├── mmpose_crowdpose_train.json"
echo "  │   ├── mmpose_crowdpose_val.json"
echo "  │   ├── mmpose_crowdpose_trainval.json"
echo "  │   ├── mmpose_crowdpose_test.json"
echo "  │   └── det_for_crowd_test_0.1_0.5.json"
echo "  └── images/"
echo "      └── *.jpg"
echo ""
