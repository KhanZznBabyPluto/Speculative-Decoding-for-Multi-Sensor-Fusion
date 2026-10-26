#!/usr/bin/env bash
#
# submit_mambev_baseline.sh — measure MamBEV's own mAP/NDS on our 23-scene val.
#
# This closes the project's biggest unverified assumption: stage 05 concluded
# "MamBEV's dense BEV is a query-position cache, not a metric world grid" from
# the encoder's internal tensors, without ever checking that the model actually
# detects anything on our data. Published MamBEV-small is 42.2 mAP / 52.5 NDS.
#
# a100 only: mamba-ssm / causal-conv1d kernels are compiled for sm_80+, and the
# stage-05 array crashed silently on V100 nodes for exactly this reason. MamBEV
# also calls torch.cuda.is_bf16_supported() at import time, so nothing about
# this config can be validated on a login node — the first real check is here.
#
# Usage:   sbatch stages/06_gated_bev_psm/submit_mambev_baseline.sh
# Monitor: squeue -u "$USER"
#
#SBATCH --job-name=mambev_base
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/ibex/user/khangea/multisensor/stages/06_gated_bev_psm/logs/mambev_base_%j.out
#SBATCH --error=/ibex/user/khangea/multisensor/stages/06_gated_bev_psm/logs/mambev_base_%j.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ibex/user/khangea/multisensor}"
STAGE_DIR="${PROJECT_DIR}/stages/06_gated_bev_psm"
CONDA_ENV="${CONDA_ENV:-/ibex/user/khangea/conda-environments/mambev_env}"
CONDA_SH="${CONDA_SH:-/ibex/user/khangea/miniforge/etc/profile.d/conda.sh}"

CONFIG="${CONFIG:-projects/MAMBEV/configs/MamBEV/mambev_small_t3.py}"
CKPT="${CKPT:-${PROJECT_DIR}/checkpoints/mambev/mambev_small_detector.pth}"
WORK_DIR="${WORK_DIR:-${STAGE_DIR}/dumps/mambev_baseline}"

main() {
  [[ -f "$CKPT" ]] || { echo "error: checkpoint not found: $CKPT" >&2; exit 1; }
  [[ -f "$CONDA_SH" ]] || { echo "error: conda.sh not found: $CONDA_SH" >&2; exit 1; }

  # MamBEV's config uses a relative data_root ('data/nuscenes/'), so the symlink
  # must resolve or the run dies minutes in with an opaque error.
  local pkl="${PROJECT_DIR}/MamBEV/data/nuscenes/nuscenes_infos_temporal_val.pkl"
  [[ -f "$pkl" ]] || {
    echo "error: dataroot does not resolve: $pkl" >&2
    echo "  fix: ln -s ${PROJECT_DIR}/data_full ${PROJECT_DIR}/MamBEV/data/nuscenes" >&2
    exit 1
  }

  # conda's init block is not `set -u`/`set -e` clean; guard the activation.
  set +e
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  local rc=$?
  set -e
  [[ $rc -eq 0 ]] || { echo "error: could not activate $CONDA_ENV" >&2; exit 1; }

  nvidia-smi --query-gpu=name --format=csv,noheader || true

  mkdir -p "${STAGE_DIR}/logs" "$WORK_DIR"
  cd "${PROJECT_DIR}/MamBEV"
  PYTHONPATH="${PWD}" python "${STAGE_DIR}/measure_mambev_baseline.py" \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --work-dir "$WORK_DIR"

  echo "[mambev] done -> ${WORK_DIR}"
}

main "$@"
