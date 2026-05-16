#!/bin/bash

# OCHuman (CVPR 2019) — layout matches MMPose dataset zoo:
#   https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#ochuman
#
# Google Drive sources (same file IDs as the /file/d/<id>/view links):
#   images.zip: https://drive.google.com/file/d/1H_FltQ2SL4qz-Vhf5HFz-rirEyAZwcWM/view
#   ochuman_coco_format_val_range_0.00_1.00.json:
#       https://drive.google.com/file/d/1L4puWqEU5CjhwljM_phvLf5lEo0f87rG/view
#   ochuman_coco_format_test_range_0.00_1.00.json:
#       https://drive.google.com/file/d/1VXtTnUQ9Aeq87W1MO1sIs4qGVXMPbcoQ/view
#   ochuman.json (native annotations):
#       https://drive.google.com/file/d/19hH7fGIyVgszmdSVlzvm0jCDAeRic5a4/view
#
# Native ochuman.json is optional for typical MMPose COCO-format eval but is
# downloaded by default. Override id with OCHUMAN_JSON_GDRIVE_ID or the second CLI
# argument; change the default id with OCHUMAN_NATIVE_JSON_ID. Set
# OCHUMAN_SKIP_NATIVE_JSON=1 to skip that download.
#
# Requires: pip install -U gdown unzip
#
# Use the first argument as BASE_DIR; default to "data/ochuman" if not provided.
# Optional second argument: Google Drive file id for ochuman.json (only if
# OCHUMAN_JSON_GDRIVE_ID is unset or empty).
BASE_DIR=${1:-"data/ochuman"}

# Override file ids if mirrors change (values match the Drive URLs above).
: "${OCHUMAN_IMAGES_ZIP_ID:=1H_FltQ2SL4qz-Vhf5HFz-rirEyAZwcWM}"
: "${OCHUMAN_VAL_JSON_ID:=1L4puWqEU5CjhwljM_phvLf5lEo0f87rG}"
: "${OCHUMAN_TEST_JSON_ID:=1VXtTnUQ9Aeq87W1MO1sIs4qGVXMPbcoQ}"
: "${OCHUMAN_NATIVE_JSON_ID:=19hH7fGIyVgszmdSVlzvm0jCDAeRic5a4}"

NATIVE_JSON_ID="${OCHUMAN_JSON_GDRIVE_ID:-${2:-$OCHUMAN_NATIVE_JSON_ID}}"

echo "Targeting directory: $BASE_DIR"

mkdir -p "$BASE_DIR/annotations"

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

echo "------------------------------------------------"
echo "Downloading COCO-format annotation JSON files..."
gdown_file "$OCHUMAN_VAL_JSON_ID" \
    "$BASE_DIR/annotations/ochuman_coco_format_val_range_0.00_1.00.json"
gdown_file "$OCHUMAN_TEST_JSON_ID" \
    "$BASE_DIR/annotations/ochuman_coco_format_test_range_0.00_1.00.json"

if [[ "${OCHUMAN_SKIP_NATIVE_JSON:-0}" != "1" ]]; then
    echo "------------------------------------------------"
    echo "Downloading native ochuman.json..."
    gdown_file "$NATIVE_JSON_ID" "$BASE_DIR/annotations/ochuman.json"
else
    echo "------------------------------------------------"
    echo "Skipping ochuman.json (OCHUMAN_SKIP_NATIVE_JSON=1)."
fi

echo "------------------------------------------------"
echo "Downloading images.zip (large)..."
IMAGES_ZIP="$BASE_DIR/images.zip"
gdown_file "$OCHUMAN_IMAGES_ZIP_ID" "$IMAGES_ZIP"

echo "------------------------------------------------"
echo "Extracting images into $BASE_DIR/images ..."

EXTRACT_TMP=$(mktemp -d)
trap 'rm -rf "$EXTRACT_TMP"' EXIT

unzip -q -o "$IMAGES_ZIP" -d "$EXTRACT_TMP"
rm -f "$IMAGES_ZIP"

IMG_DIR=""
if [[ -d "$EXTRACT_TMP/images" ]]; then
    IMG_DIR="$EXTRACT_TMP/images"
else
    IMG_DIR=$(find "$EXTRACT_TMP" -type d -name images 2>/dev/null | head -n 1)
fi

mkdir -p "$BASE_DIR/images"

if [[ -n "$IMG_DIR" && -d "$IMG_DIR" ]]; then
    cp -a "$IMG_DIR"/. "$BASE_DIR/images/"
else
    shopt -s nullglob
    jpgs=( "$EXTRACT_TMP"/*.jpg "$EXTRACT_TMP"/*.jpeg "$EXTRACT_TMP"/*.JPG )
    if [[ ${#jpgs[@]} -gt 0 ]]; then
        cp -a "${jpgs[@]}" "$BASE_DIR/images/"
    else
        echo "Error: could not find images/ or JPEG files inside images.zip." >&2
        echo "Top-level entries under extracted archive:" >&2
        find "$EXTRACT_TMP" -mindepth 1 -maxdepth 2 >&2
        shopt -u nullglob
        exit 1
    fi
    shopt -u nullglob
fi

echo "------------------------------------------------"
echo "Done! OCHuman dataset is ready in $BASE_DIR"
echo ""
echo "Expected structure:"
echo "  $BASE_DIR/"
echo "  ├── annotations/"
echo "  │   ├── ochuman_coco_format_val_range_0.00_1.00.json"
echo "  │   ├── ochuman_coco_format_test_range_0.00_1.00.json"
echo "  │   └── ochuman.json   (skipped if OCHUMAN_SKIP_NATIVE_JSON=1)"
echo "  └── images/"
echo "      └── *.jpg"
echo ""
