#!/usr/bin/env bash
# Run DAG seg + det attacks on several VOC2007 images at paper-budget iter counts.
set -e
cd "$(dirname "$0")/.."

VOC_ROOT=${VOC_ROOT:-/tmp/VOCdevkit/VOC2007}
OUT=${OUT:-/home/adsl-1-2/GitHub/DAG/outputs/multi}
DEVICE=${DEVICE:-cuda}
SEG_ITER=${SEG_ITER:-200}
DET_ITER=${DET_ITER:-150}

# Five VOC2007 segmentation-trainval images with diverse VOC classes.
IDS=(000033 000042 000061 000063 000121)

for id in "${IDS[@]}"; do
    echo
    echo "================ $id (seg) ================"
    time uv run python -m dag.cli seg --config configs/seg_voc.yaml \
        --voc-root "$VOC_ROOT" --device "$DEVICE" \
        --image-id "$id" --output-dir "$OUT/seg" \
        --max-iter "$SEG_ITER" 2>&1 | tail -3

    echo "================ $id (det) ================"
    time uv run python -m dag.cli det --config configs/det_voc.yaml \
        --voc-root "$VOC_ROOT" --device "$DEVICE" \
        --image-id "$id" --output-dir "$OUT/det" \
        --max-iter "$DET_ITER" 2>&1 | tail -3
done

echo
echo "==== ALL DONE ===="
