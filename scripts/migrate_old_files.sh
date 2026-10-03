#!/usr/bin/env bash
# migrate_old_files.sh
# Copies the trained models, hex files and MNIST data from the OLD folder
# layout into this repository layout. Nothing in the old folder is changed.
#
# Usage (from the repository root):
#   bash scripts/migrate_old_files.sh [path/to/old/software]
# Default old folder: ~/rv-sparsemac_old/software

set -e

OLD="${1:-$HOME/rv-sparsemac_old/software}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$OLD" ]; then
    echo "Old folder not found: $OLD"
    exit 1
fi

echo "Old folder : $OLD"
echo "New repo   : $REPO"
echo

copy_file() {
    if [ -f "$1" ]; then
        cp "$1" "$2"
        echo "copied : $1  ->  $2"
    else
        echo "MISSING: $1"
    fi
}

mkdir -p "$REPO/models" "$REPO/hardware/mem"

# Trained models
copy_file "$OLD/mnist_model.pth" "$REPO/models/baseline_fp32.pth"
for p in 70 80 90; do
    copy_file "$OLD/pruned_fc1_$p.pth" "$REPO/models/pruned_fc1_$p.pth"
done

# Hex files and golden vectors
for f in fc1_weights.hex fc1_bias.hex fc2_weights.hex fc2_bias.hex \
         test_images.hex test_labels.txt golden_logits.txt golden_pred.txt params.txt; do
    copy_file "$OLD/export/$f" "$REPO/hardware/mem/$f"
done

# MNIST dataset (saves a new download)
if [ -d "$OLD/data" ] && [ ! -d "$REPO/data" ]; then
    cp -r "$OLD/data" "$REPO/data"
    echo "copied : $OLD/data  ->  $REPO/data"
fi

echo
echo "Done."
