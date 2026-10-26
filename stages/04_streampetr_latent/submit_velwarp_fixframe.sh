#!/usr/bin/env bash
#
# submit_velwarp_fixframe.sh — re-measure vel_warp with the velocity frame fixed.
#
# Every earlier vel_warp number added a lidar(t)-frame velocity to a global-frame
# reference point (see velo_lidar_to_global in measure_stale_map_mitigation.py),
# so "vel_warp is noise" tested a broken implementation. This array re-runs it
# with the rotation applied. Outputs go to NEW dirs (*_fixframe); the old dumps
# are left untouched.
#
# Tasks (honest-mode, delay, mitigation):
#   0-2  key   {1,2,3}  vel_warp  -> compare to honest key none: 0.1849 / 0.1007 / 0.0731
#   3    none  1        vel_warp  -> compare to naive delay1 0.1331 (the paper's
#                                    original, retracted "+3-5%" condition)
#
# ~3-5 min per task on one A100. a100 only: flash-attn kernels need sm_80+.
#
# Usage:   sbatch stages/04_streampetr_latent/submit_velwarp_fixframe.sh
# Monitor: squeue -u "$USER"
# After:   python stages/04_streampetr_latent/bootstrap_ci.py \
#            --configs honest_d1_key honestmit_d1_vel_warp_fixframe --baseline honest_d1_key
#
#SBATCH --job-name=velwarp_fix
#SBATCH --array=0-3
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/ibex/user/khangea/multisensor/stages/04_streampetr_latent/logs/velwarp_fix_%A_%a.out
#SBATCH --error=/ibex/user/khangea/multisensor/stages/04_streampetr_latent/logs/velwarp_fix_%A_%a.err

# -u omitted: conda's activate scripts reference unset variables.
set -eo pipefail
IFS=$'\n\t'

PROJECT_DIR="${PROJECT_DIR:-/ibex/user/khangea/multisensor}"
STAGE_DIR="${PROJECT_DIR}/stages/04_streampetr_latent"
CONDA_ENV="${CONDA_ENV:-/ibex/user/khangea/conda-environments/streampetr_env}"
CONDA_SH="${CONDA_SH:-/ibex/user/khangea/miniforge/etc/profile.d/conda.sh}"

CONFIG="${CONFIG:-projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py}"
CKPT="${CKPT:-${PROJECT_DIR}/checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth}"

# One entry per array task: "<honest-mode> <delay> <out-name>"
CONFIGS=(
  "key 1 honestmit_d1_vel_warp_fixframe"
  "key 2 honestmit_d2_vel_warp_fixframe"
  "key 3 honestmit_d3_vel_warp_fixframe"
  "none 1 stale_map_d1_velwarp_fixframe"
)

main() {
  local task="${SLURM_ARRAY_TASK_ID:-0}"
  [[ -n "${CONFIGS[$task]+x}" ]] || { echo "error: no config for task $task" >&2; exit 1; }
  local honest delay name
  IFS=' ' read -r honest delay name <<< "${CONFIGS[$task]}"

  local out="${STAGE_DIR}/dumps/${name}"
  echo "[velwarp-fix] task=${task} honest=${honest} delay=${delay} out=${out}"

  [[ -f "$CKPT" ]] || { echo "error: checkpoint not found: $CKPT" >&2; exit 1; }
  [[ -f "$CONDA_SH" ]] || { echo "error: conda.sh not found: $CONDA_SH" >&2; exit 1; }
  [[ ! -e "$out" ]] || { echo "error: $out exists; refusing to overwrite" >&2; exit 1; }

  local pkl="${PROJECT_DIR}/StreamPETR/data/nuscenes/nuscenes2d_temporal_infos_val.pkl"
  [[ -f "$pkl" ]] || { echo "error: dataroot does not resolve: $pkl" >&2; exit 1; }

  set +e
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  local rc=$?
  set -e
  [[ $rc -eq 0 ]] || { echo "error: could not activate $CONDA_ENV" >&2; exit 1; }

  nvidia-smi --query-gpu=name --format=csv,noheader || true

  cd "${PROJECT_DIR}/StreamPETR"
  PYTHONPATH="${PWD}:${STAGE_DIR}" python \
    "${STAGE_DIR}/measure_stale_map_mitigation.py" \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --delay-all "$delay" \
    --mitigation vel_warp \
    --honest-mode "$honest" \
    --out "$out"

  echo "[velwarp-fix] task ${task} done -> ${out}"
}

main "$@"
