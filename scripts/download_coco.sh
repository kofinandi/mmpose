#!/bin/bash

# Use the first argument as BASE_DIR; default to "data/coco" if not provided
BASE_DIR=${1:-"data/coco"}

echo "Targeting directory: $BASE_DIR"

# Create directories
mkdir -p "$BASE_DIR"

# URLs
TRAIN_URL="http://images.cocodataset.org/zips/train2017.zip"
VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
ANNO_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

fetch_data() {
    local url=$1
    local target_path=$2
    local filename=$(basename "$url")

    echo "------------------------------------------------"
    echo "Downloading $filename..."
    wget -c "$url" -P "$target_path"

    echo "Extracting $filename..."
    unzip -q "$target_path/$filename" -d "$target_path"
    rm "$target_path/$filename"
}

fetch_data "$ANNO_URL" "$BASE_DIR"
fetch_data "$TRAIN_URL" "$BASE_DIR"
fetch_data "$VAL_URL" "$BASE_DIR"

echo "------------------------------------------------"
echo "Done! COCO dataset is ready in $BASE_DIR"