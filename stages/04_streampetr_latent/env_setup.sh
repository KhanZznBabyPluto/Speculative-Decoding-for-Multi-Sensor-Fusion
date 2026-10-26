#!/usr/bin/env bash
# StreamPETR environment setup recipe.
#
# Why a separate env: StreamPETR is built on the legacy OpenMMLab stack
# (mmcv 1.6 / mmdet 2.28 / mmdet3d 1.0rc6), which cannot coexist with the
# `sensor` env's mmcv 2.x. We keep `sensor` untouched so stage 03 stays runnable.
#
# Run this on a GPU node (an A100/V100 salloc is fine, mmcv-full compile may
# fall back to building from source which needs nvcc). ~30-45 minutes.
#
# Versions chosen to match StreamPETR's docs/setup.md while pinning to wheels
# that have prebuilt distributions:
#   torch 1.13.0 + cu117 (closer to our cluster than cu111; mmcv-full prebuilt)
#   mmcv-full 1.7.0 (newer than the 1.6.0 in StreamPETR README, but compatible)
#   mmdet 2.28.2, mmsegmentation 0.30.0, mmdet3d 1.0.0rc6 (from source)
#
# After this script: clone StreamPETR (if not already cloned), then run
# stages/04_streampetr_latent/install_mmdet3d.sh inside it.

set -e

ENV_NAME=streampetr_env
ENV_PREFIX=/ibex/user/khangea/conda-environments/${ENV_NAME}

# --- 1. Create env ---
conda create -p "${ENV_PREFIX}" python=3.8 -y
source activate "${ENV_PREFIX}"

# --- 2. PyTorch 1.13.0 + cu117 ---
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# --- 3. mmcv-full 1.7.0 (prebuilt wheel for this torch/cuda combo) ---
pip install mmcv-full==1.7.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# --- 4. mmdet + mmsegmentation pinned versions ---
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0

# --- 5. Other deps StreamPETR uses ---
pip install \
    nuscenes-devkit==1.1.10 \
    numba==0.53.0 \
    numpy==1.23.5 \
    plyfile \
    scikit-image \
    tensorboard \
    trimesh \
    yapf==0.40.1 \
    ninja \
    einops \
    spconv-cu117==2.3.6

# --- 6. flash-attn is optional; skipping. We will disable flash-attn in the
#        StreamPETR config when we run inference.

echo
echo "=== Environment ${ENV_NAME} created at ${ENV_PREFIX} ==="
echo "Next: clone StreamPETR (if not already), then run install_mmdet3d.sh"
