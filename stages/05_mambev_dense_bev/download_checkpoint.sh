#!/usr/bin/env bash
# Stage 05 — fetch MamBEV checkpoint(s).
#
# README ([MamBEV/README.md](../../MamBEV/README.md)) is mixed on availability:
#
#   | row              | "Backbone" link              | "model" link                   |
#   |------------------|------------------------------|--------------------------------|
#   | MamBEV-tiny  R50 | R50 backbone (gdrive, public)| MISSING — text says [model]/[log] but no URL |
#   | MamBEV-small R101-DCN | r101_dcn_fcos3d_pretrain.pth (github releases) | Google Drive FOLDER (manual) |
#
# Catalog explicitly lists "[ ] 3D Detection checkpoints" as TODO — so the
# Drive folder for MamBEV-small is the only public trained-detector
# checkpoint, and it requires manual download (gdrive auth from cluster
# is annoying).
#
# What this script does automatically:
#   1. R50 backbone (for MamBEV-tiny) — direct gdrive file, gdown
#   2. R101-DCN backbone (BEVFormer's storage) — straight curl
#   3. Prints manual instructions for the MamBEV-small detector folder
#
# Run on a node with network access (login or job that allows egress).

set -e

CKPT_DIR=/ibex/user/khangea/multisensor/checkpoints
MAMBEV_DIR=${CKPT_DIR}/mambev
mkdir -p "${MAMBEV_DIR}"

# -------------------------------------------------------------------------
# 1) R50 backbone for MamBEV-tiny
# -------------------------------------------------------------------------
R50_ID=1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7
R50_OUT=${MAMBEV_DIR}/fcos_r50_coco_2mmdet.pth
if [ ! -f "${R50_OUT}" ]; then
    echo "[1/3] Downloading R50 backbone (gdrive)..."
    if ! command -v gdown >/dev/null 2>&1; then
        echo "  gdown not installed. Install with: pip install gdown"
        exit 1
    fi
    gdown "https://drive.google.com/uc?id=${R50_ID}" -O "${R50_OUT}"
else
    echo "[1/3] R50 backbone already at ${R50_OUT}"
fi

# -------------------------------------------------------------------------
# 2) R101-DCN backbone for MamBEV-small (BEVFormer's storage)
# -------------------------------------------------------------------------
R101_URL=https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
R101_OUT=${MAMBEV_DIR}/r101_dcn_fcos3d_pretrain.pth
if [ ! -f "${R101_OUT}" ]; then
    echo "[2/3] Downloading R101-DCN backbone..."
    curl -L "${R101_URL}" -o "${R101_OUT}"
else
    echo "[2/3] R101-DCN backbone already at ${R101_OUT}"
fi

# -------------------------------------------------------------------------
# 3) MamBEV-small TRAINED DETECTOR — manual
# -------------------------------------------------------------------------
SMALL_DET=${MAMBEV_DIR}/mambev_small_detector.pth
if [ ! -f "${SMALL_DET}" ]; then
    cat <<EOF
[3/3] MamBEV-small detector checkpoint — MANUAL STEP
    Google Drive folder:
      https://drive.google.com/drive/folders/1-MXxszU80aSG4P1_hQnC0qlQOHnPb4iM
    Download the .pth there, then place it at:
      ${SMALL_DET}
    (Note: the MamBEV-tiny detector checkpoint is NOT published as of
    README revision in repo. If only the tiny exists in the Drive folder,
    rename it to mambev_tiny_detector.pth and pass --variant tiny to
    verify_checkpoint.py.)
EOF
else
    echo "[3/3] MamBEV-small detector already at ${SMALL_DET}"
fi

echo
echo "Done. Files under ${MAMBEV_DIR}:"
ls -la "${MAMBEV_DIR}"
