#!/bin/sh
# Rebuild docs/hero/hero.jpg: the full workflow's model steps, with no review,
# on a CC0 photo. Needs the model environment and an NVIDIA GPU.
#   sh docs/hero/make.sh [cuda:N]
# Photo: "Corn ready for removing kernels from cob.jpg", Wikimedia Commons, CC0
# (https://commons.wikimedia.org/wiki/File:Corn_ready_for_removing_kernels_from_cob.jpg),
# 960 x 640 thumbnail. config.json is configs/corn-starting-point.json with a
# larger budget, because this photo is bigger than the original corn photo.
set -eu
here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
S=$here/../../skills/sam3-segmentation
device=${1:-cuda:0}
work=$(mktemp -d)
curl -sSL -A "sam3-segmentation-docs" -o "$work/corn.jpg" \
  "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Corn_ready_for_removing_kernels_from_cob.jpg/960px-Corn_ready_for_removing_kernels_from_cob.jpg"
sh "$S/scripts/run" base segment.py prepare "$work/corn.jpg" --out "$work/run" --config "$here/config.json" --orientation raw
sh "$S/scripts/run" base segment.py plan "$work/run"
sh "$S/scripts/run" model segment.py infer "$work/run" --device "$device" --batch-size 24
sh "$S/scripts/run" base segment.py select "$work/run"
sh "$S/scripts/run" model segment.py refine "$work/run" --device "$device"
sh "$S/scripts/run" base segment.py validate "$work/run"
sh "$S/scripts/run" base segment.py export "$work/run"
sh "$S/scripts/run" base python "$here/render.py" "$work/run/export" "$here/hero.jpg"
echo "Run kept in $work"
