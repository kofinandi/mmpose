#!/usr/bin/env bash
#
# DanceTrack (CVPR 2022) — layout matches the official MOT-style release:
#   https://github.com/DanceTrack/DanceTrack
#   https://huggingface.co/datasets/noahcao/dancetrack
#
# Hugging Face hosts the dataset as split zip archives (~18 GB compressed):
#   train1.zip, train2.zip, val.zip, test1.zip, test2.zip
#
# Requires: unzip, and either huggingface_hub (pulled in by transformers) or wget.
#
# Usage:
#   bash scripts/download_dancetrack.sh [BASE_DIR] [splits...]
#
# Examples:
#   bash scripts/download_dancetrack.sh
#   bash scripts/download_dancetrack.sh data/dancetrack
#   bash scripts/download_dancetrack.sh data/dancetrack train val
#   DANCETRACK_SKIP_TEST=1 bash scripts/download_dancetrack.sh
#
# Optional env:
#   DANCETRACK_SKIP_TEST=1   — skip test1.zip / test2.zip (~6.5 GB)
#   DANCETRACK_KEEP_ZIPS=1   — keep archives after extracting
#   HF_TOKEN                 — forwarded to huggingface_hub if the Hub needs auth

set -euo pipefail

HF_REPO="noahcao/dancetrack"
HF_RESOLVE="https://huggingface.co/datasets/noahcao/dancetrack/resolve/main"

BASE_DIR=${1:-"data/dancetrack"}
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ $# -gt 0 ]]; then
    SPLITS=("$@")
else
    SPLITS=(train val test)
fi

if [[ "${DANCETRACK_SKIP_TEST:-0}" == "1" ]]; then
    FILTERED=()
    for split in "${SPLITS[@]}"; do
        if [[ "$split" != "test" ]]; then
            FILTERED+=("$split")
        fi
    done
    SPLITS=("${FILTERED[@]}")
fi

if [[ ${#SPLITS[@]} -eq 0 ]]; then
    echo "Error: no splits selected." >&2
    exit 1
fi

# Each official split is one or two zip files on the Hub.
zip_files_for_split() {
    case "$1" in
        train) echo "train1.zip train2.zip" ;;
        val)   echo "val.zip" ;;
        test)  echo "test1.zip test2.zip" ;;
        *)
            echo "Error: unknown split '$1' (expected train, val, or test)." >&2
            return 1
            ;;
    esac
}

echo "Targeting directory: $BASE_DIR"
echo "Splits: ${SPLITS[*]}"
echo "Source: https://huggingface.co/datasets/${HF_REPO}"
echo "Note: compressed download is ~18 GB for all splits (train+val+test)."

mkdir -p "$BASE_DIR"

if ! command -v unzip >/dev/null 2>&1; then
    echo "Error: unzip not found. Install unzip (e.g. apt install unzip)." >&2
    exit 1
fi

HAVE_HF_HUB=0
if python3 -c "from huggingface_hub import hf_hub_download" >/dev/null 2>&1; then
    HAVE_HF_HUB=1
elif ! command -v wget >/dev/null 2>&1; then
    echo "Error: need huggingface_hub or wget to download from Hugging Face." >&2
    echo "Install with: pip install huggingface_hub" >&2
    echo "(huggingface_hub is already pulled in by the transformers dependency.)" >&2
    exit 1
fi

download_file() {
    local filename=$1
    local dest="$BASE_DIR/$filename"

    echo "------------------------------------------------"
    echo "Downloading $filename..."

    if [[ "$HAVE_HF_HUB" -eq 1 ]]; then
        HF_REPO="$HF_REPO" HF_FILENAME="$filename" HF_LOCAL_DIR="$BASE_DIR" python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id=os.environ["HF_REPO"],
    filename=os.environ["HF_FILENAME"],
    repo_type="dataset",
    local_dir=os.environ["HF_LOCAL_DIR"],
    cache_dir=os.path.join(os.environ["HF_LOCAL_DIR"], ".hf_cache"),
)
PY
    else
        wget -c "${HF_RESOLVE}/${filename}" -O "$dest"
    fi

    if [[ ! -f "$dest" ]]; then
        echo "Error: expected $dest after download." >&2
        exit 1
    fi
}

extract_zip() {
    local zip_path=$1
    local filename
    filename=$(basename "$zip_path")

    echo "Extracting $filename..."
    unzip -q -o "$zip_path" -d "$BASE_DIR"

    if [[ "${DANCETRACK_KEEP_ZIPS:-0}" != "1" ]]; then
        rm -f "$zip_path"
    fi
}

# Official zips usually unpack train/ val/ test/ at the archive root. Some
# mirrors wrap that tree in an extra dancetrack/ (or DanceTrack/) folder.
flatten_nested_root() {
    local nested
    for nested in "$BASE_DIR/dancetrack" "$BASE_DIR/DanceTrack"; do
        if [[ ! -d "$nested" ]]; then
            continue
        fi
        echo "------------------------------------------------"
        echo "Flattening nested $(basename "$nested")/ into $BASE_DIR ..."
        local item
        for item in train val test train_seqmap.txt val_seqmap.txt test_seqmap.txt; do
            if [[ -e "$nested/$item" && ! -e "$BASE_DIR/$item" ]]; then
                mv "$nested/$item" "$BASE_DIR/"
            elif [[ -d "$nested/$item" && -d "$BASE_DIR/$item" ]]; then
                mv "$nested/$item"/* "$BASE_DIR/$item/" 2>/dev/null || true
                rmdir "$nested/$item" 2>/dev/null || true
            fi
        done
        rmdir "$nested" 2>/dev/null || true
    done
}

write_seqmap() {
    local split=$1
    local split_dir="$BASE_DIR/$split"
    local seqmap="$BASE_DIR/${split}_seqmap.txt"

    if [[ -f "$seqmap" || ! -d "$split_dir" ]]; then
        return
    fi

    echo "Writing ${split}_seqmap.txt ..."
    {
        echo "name"
        find "$split_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
    } > "$seqmap"
}

for split in "${SPLITS[@]}"; do
    # shellcheck disable=SC2046
    for zip_name in $(zip_files_for_split "$split"); do
        download_file "$zip_name"
        extract_zip "$BASE_DIR/$zip_name"
    done
done

flatten_nested_root
rm -rf "$BASE_DIR/.hf_cache" "$BASE_DIR/.cache"

for split in "${SPLITS[@]}"; do
    if [[ ! -d "$BASE_DIR/$split" ]]; then
        echo "Error: expected $BASE_DIR/$split after extracting ${split} archives." >&2
        echo "Top-level entries under $BASE_DIR:" >&2
        find "$BASE_DIR" -mindepth 1 -maxdepth 2 >&2
        exit 1
    fi
    write_seqmap "$split"
done

echo "------------------------------------------------"
echo "Done! DanceTrack is ready in $BASE_DIR"
echo ""
echo "Expected structure:"
echo "  $BASE_DIR/"
echo "  ├── train/"
echo "  │   └── dancetrackXXXX/"
echo "  │       ├── img1/"
echo "  │       ├── gt/"
echo "  │       └── seqinfo.ini"
echo "  ├── val/"
echo "  ├── test/                  (omitted if DANCETRACK_SKIP_TEST=1)"
echo "  ├── train_seqmap.txt"
echo "  ├── val_seqmap.txt"
echo "  └── test_seqmap.txt"
echo ""
