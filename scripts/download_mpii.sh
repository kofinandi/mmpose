#!/bin/bash

# Use the first argument as BASE_DIR; default to "data/mpii" if not provided
BASE_DIR=${1:-"data/mpii"}
IMAGE_DIR="$BASE_DIR/images"
ANNO_DIR="$BASE_DIR/annotations"

echo "Targeting directory: $BASE_DIR"

mkdir -p "$IMAGE_DIR"
mkdir -p "$ANNO_DIR"

# URLs
IMAGE_URL="https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz"
ANNO_URL="https://download.openmmlab.com/mmpose/datasets/mpii_annotations.tar"

echo "------------------------------------------------"
echo "Downloading and extracting Annotations..."
wget -c "$ANNO_URL" -P "$BASE_DIR"
tar -xf "$BASE_DIR/mpii_annotations.tar" -C "$BASE_DIR"
mv "$BASE_DIR"/mpii_*.json "$ANNO_DIR"/
mv "$BASE_DIR"/mpii_gt_val.mat "$ANNO_DIR"/
rm "$BASE_DIR/mpii_annotations.tar"

echo "------------------------------------------------"
echo "Downloading and extracting Images (~13 GB)..."
wget -c "$IMAGE_URL" -P "$BASE_DIR"
tar -xzf "$BASE_DIR/mpii_human_pose_v1.tar.gz" -C "$BASE_DIR"
rm "$BASE_DIR/mpii_human_pose_v1.tar.gz"

echo "------------------------------------------------"
echo "Done! MPII dataset is ready in $BASE_DIR"