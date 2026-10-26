#!/usr/bin/env bash
#
# submit_async_cams.sh — StreamPETR with stale 12 Hz camera sweeps (stage 09).
#
# Two questions, one array:
#   A. Sub-keyframe staleness: all 6 cameras J sweeps old (J=1,2,3,4,6 ≈ 76,
#      164, 249, ~330, 496 ms), naive vs honest geometry. Does the honest lift
#      close a LARGER share of the gap at realistic 80-250 ms delays than the
#      16.7% it closed at 500 ms?
#   B. Per-camera asynchrony at equal compute: round-robin (rr:P, 6/P cameras
#      refreshed per tick) vs whole-frame every P ticks (whole:P). Same mean age,
#      different age pattern.
#
# Task 0 is the gate: uniform:0 must reproduce fresh mAP 0.4448 exactly.
# ~4-6 min per task on one A100 (full backbone on real images).
#
# Usage:   sbatch stages/09_subframe_async/submit_async_cams.sh
#          sbatch --array=0 stages/09_subframe_async/submit_async_cams.sh   # gate only
# Monitor: squeue -u "$USER"
#
#SBATCH --job-name=async_cams
#SBATCH --array=0-16%4
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/ibex/user/khangea/multisensor/stages/09_subframe_async/logs/async_cams_%A_%a.out
#SBATCH --error=/ibex/user/khangea/multisensor/stages/09_subframe_async/logs/async_cams_%A_%a.err

# -u omitted: conda's activate scripts reference unset variables.
set -eo pipefail
IFS=$'\n\t'

PROJECT_DIR="${PROJECT_DIR:-/ibex/user/khangea/multisensor}"
STAGE_DIR="${PROJECT_DIR}/stages/09_subframe_async"
CONDA_ENV="${CONDA_ENV:-/ibex/user/khangea/conda-environments/streampetr_env}"
CONDA_SH="${CONDA_SH:-/ibex/user/khangea/miniforge/etc/profile.d/conda.sh}"

# One entry per array task: "<schedule> <geometry>"
CONFIGS=(
  "uniform:0 naive"
  "uniform:1 naive"  "uniform:1 honest"
  "uniform:2 naive"  "uniform:2 honest"
  "uniform:3 naive"  "uniform:3 honest"
  "uniform:4 naive"  "uniform:4 honest"
  "uniform:6 naive"  "uniform:6 honest"
  "rr:2 honest"      "whole:2 honest"
  "rr:3 honest"      "whole:3 honest"
  "rr:2 naive"       "rr:3 naive"
)

main() {
  local task="${SLURM_ARRAY_TASK_ID:-0}"
  [[ -n "${CONFIGS[$task]+x}" ]] || { echo "error: no config for task $task" >&2; exit 1; }
  local schedule geometry
  IFS=' ' read -r schedule geometry <<< "${CONFIGS[$task]}"

  local out="${STAGE_DIR}/dumps/async_${schedule/:/}_${geometry}"
  echo "[async-cams] task=${task} schedule=${schedule} geometry=${geometry} out=${out}"

  [[ -f "${STAGE_DIR}/dumps/sweep_index_val.pkl" ]] || {
    echo "error: run build_sweep_index.py first" >&2; exit 1; }
  [[ -f "$CONDA_SH" ]] || { echo "error: conda.sh not found: $CONDA_SH" >&2; exit 1; }
  [[ ! -e "$out" ]] || { echo "error: $out exists; refusing to overwrite" >&2; exit 1; }

  set +e
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  local rc=$?
  set -e
  [[ $rc -eq 0 ]] || { echo "error: could not activate $CONDA_ENV" >&2; exit 1; }

  nvidia-smi --query-gpu=name --format=csv,noheader || true

  cd "${PROJECT_DIR}/StreamPETR"
  PYTHONPATH="${PWD}" python "${STAGE_DIR}/eval_async_cams.py" \
    --schedule "$schedule" \
    --geometry "$geometry" \
    --out "$out"

  echo "[async-cams] task ${task} done -> ${out}"
}

main "$@"
