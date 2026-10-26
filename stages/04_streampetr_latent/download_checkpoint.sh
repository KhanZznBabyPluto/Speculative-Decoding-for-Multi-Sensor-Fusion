#!/usr/bin/env bash
# Download the StreamPETR R50-900q nuScenes checkpoint.
#
# This is the simplest config in the repo: ResNet-50 backbone (no extra
# FCOS3D pretrained backbone to fetch), 900 queries, 90-epoch schedule,
# 53.7 NDS / 43.2 mAP on nuScenes val.
#
# Trained with flash-attn — we'll disable flash-attn at inference time
# via a small config override (TBD in a follow-up script).

set -e

CKPT_DIR=/ibex/user/khangea/multisensor/checkpoints
URL=https://github.com/exiawsh/storage/releases/download/v1.0/stream_petr_r50_flash_704_bs2_seq_90e.pth

mkdir -p "${CKPT_DIR}"
curl -L "${URL}" -o "${CKPT_DIR}/stream_petr_r50_flash_704_bs2_seq_90e.pth"
ls -la "${CKPT_DIR}/stream_petr_r50_flash_704_bs2_seq_90e.pth"
