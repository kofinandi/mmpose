#!/bin/bash

# Use the first argument as BASE_DIR; default to "data/aic" if not provided
BASE_DIR=${1:-"data/aic"}
ANNO_DIR="$BASE_DIR/annotations"

echo "Targeting directory: $BASE_DIR"

mkdir -p "$ANNO_DIR"

# Download annotations from OpenMMLab
ANNO_URL="https://download.openmmlab.com/mmpose/datasets/aic_annotations.tar"

echo "------------------------------------------------"
echo "Downloading and extracting Annotations..."
wget -c "$ANNO_URL" -P "$BASE_DIR"
tar -xf "$BASE_DIR/aic_annotations.tar" -C "$BASE_DIR"
mv "$BASE_DIR"/aic_*.json "$ANNO_DIR"/
rm "$BASE_DIR/aic_annotations.tar"

# Download images from OpenXLab (~34 GB)
# Requires: conda activate mmpose && openxlab login
echo "------------------------------------------------"
echo "Downloading Images from OpenXLab (~34 GB)..."
echo "Note: requires 'openxlab login' to be completed beforehand."

OPENXLAB_DATASET="OpenDataLab/AI_Challenger"
OPENXLAB_RAW_FILE="raw/AI_Challenger.tar.gz"
# openxlab saves to <target>/<repo-slug>/<source-path>
ARCHIVE="$BASE_DIR/OpenDataLab___AI_Challenger/raw/AI_Challenger.tar.gz"

openxlab dataset download \
    --dataset-repo "$OPENXLAB_DATASET" \
    --source-path "/$OPENXLAB_RAW_FILE" \
    --target-path "$BASE_DIR"

echo "------------------------------------------------"
echo "Extracting Images (~34 GB)..."
tar -xzf "$ARCHIVE" -C "$BASE_DIR"
rm "$ARCHIVE"
rmdir -p "$BASE_DIR/OpenDataLab___AI_Challenger/raw" 2>/dev/null || true

echo "------------------------------------------------"
echo "Done! AIC dataset is ready in $BASE_DIR"
echo ""
echo "Expected structure:"
echo "  $BASE_DIR/"
echo "  ├── annotations/"
echo "  │   ├── aic_train.json"
echo "  │   └── aic_val.json"
echo "  ├── ai_challenger_keypoint_train_20170902/"
echo "  │   └── keypoint_train_images_20170902/"
echo "  └── ai_challenger_keypoint_validation_20170911/"
echo "      └── keypoint_validation_images_20170911/"
