#!/bin/bash

# Download 3DPW ("3D People in the Wild", MPI-INF).
#
# The official release extracts to two top-level folders. This matches the
# layout used across common 3D pose / SMPL pipelines; MMPose does not ship a
# built-in 3DPW dataset adapter, so there is nothing extra to normalize here—
# point your own converters or external projects at this tree.
#
# Reference: https://virtualhumans.mpi-inf.mpg.de/3DPW/

# Use the first argument as BASE_DIR; default to "data/3dpw" if not provided
BASE_DIR=${1:-"data/3dpw"}

IMAGE_ZIP_URL="https://virtualhumans.mpi-inf.mpg.de/3DPW/imageFiles.zip"
SEQUENCE_ZIP_URL="https://virtualhumans.mpi-inf.mpg.de/3DPW/sequenceFiles.zip"

echo "Targeting directory: $BASE_DIR"

mkdir -p "$BASE_DIR"

fetch_zip() {
    local url=$1
    local target_path=$2
    local filename
    filename=$(basename "$url")

    echo "------------------------------------------------"
    echo "Downloading $filename..."
    wget -c "$url" -P "$target_path"

    echo "Extracting $filename..."
    unzip -q "$target_path/$filename" -d "$target_path"
    rm "$target_path/$filename"
}

echo "------------------------------------------------"
echo "Note: imageFiles.zip is large (multi-GB). sequenceFiles.zip is smaller."
echo "------------------------------------------------"

fetch_zip "$IMAGE_ZIP_URL" "$BASE_DIR"
fetch_zip "$SEQUENCE_ZIP_URL" "$BASE_DIR"

echo "------------------------------------------------"
echo "Done! 3DPW is ready in $BASE_DIR"
echo ""
echo "Expected structure (official MPI layout):"
echo "  $BASE_DIR/"
echo "  ├── imageFiles/"
echo "  │   └── <sequence_name>/"
echo "  │       └── image_*.jpg"
echo "  └── sequenceFiles/"
echo "      ├── train/*.pkl"
echo "      ├── test/*.pkl"
echo "      └── validation/*.pkl"
