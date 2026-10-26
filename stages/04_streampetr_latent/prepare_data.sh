#!/usr/bin/env bash
# Stage StreamPETR's data layout against our existing nuScenes data.
#
# StreamPETR's config expects:
#   StreamPETR/data/nuscenes/  containing v1.0-trainval/, samples/, sweeps/, maps/
#   StreamPETR/data/nuscenes/nuscenes2d_temporal_infos_{train,val}.pkl
#
# We have all the raw data at /ibex/user/khangea/multisensor/data_full/.
# We just need a symlink and to run StreamPETR's converter once.
#
# Run from any cwd, AFTER activating streampetr_env:
#   conda activate /ibex/user/khangea/conda-environments/streampetr_env
#   bash stages/04_streampetr_latent/prepare_data.sh

set -e

ROOT=/ibex/user/khangea/multisensor
SP=${ROOT}/StreamPETR
DATA=${ROOT}/data_full

# 1) Symlink data into StreamPETR's expected location
mkdir -p "${SP}/data"
if [ ! -e "${SP}/data/nuscenes" ]; then
    ln -s "${DATA}" "${SP}/data/nuscenes"
fi
ls -la "${SP}/data/nuscenes/" | head -5

# 2) Run StreamPETR's converter to emit nuscenes2d_temporal_infos_{train,val}.pkl
#    NOTE: nuscenes_converter.py is patched to skip TRAIN samples early
#    (val is all we need for Phase 0/1/2). With the patch:
#      ~6k val samples × ~2.5 samples/s ≈ 40 min.
#    Without the patch, ~34k samples ≈ 4 hours.
#    The train pkl will still be written but will be empty. When/if we need
#    train infos for Phase 3, revert the patch in
#    StreamPETR/tools/data_converter/nuscenes_converter.py (search "PATCH").
cd "${SP}"
python tools/create_data_nusc.py \
    --root-path ./data/nuscenes \
    --version v1.0 \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes2d
# Note: the converter inserts "_temporal" into the filename itself, so
# passing --extra-tag nuscenes2d gives `nuscenes2d_temporal_infos_{train,val}.pkl`
# which matches what the StreamPETR config expects.

echo
echo "=== Data prep done ==="
ls -la "${SP}/data/nuscenes/nuscenes2d_temporal_infos"*.pkl 2>&1 || \
    echo "Note: if filenames differ, point dump_latents.py at the actual outputs."
