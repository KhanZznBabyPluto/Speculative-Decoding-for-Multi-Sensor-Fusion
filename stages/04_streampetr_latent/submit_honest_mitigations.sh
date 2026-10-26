#!/usr/bin/env bash
#
# submit_honest_mitigations.sh — re-measure the mitigation table honestly.
#
# Every number in the paper's mitigation table was measured under the naive
# harness, i.e. stale image features lifted with the CURRENT frame's pose and
# stamped age 0. The 2026-07-17 audit showed that artefact accounts for ~1/6 of
# the delay=1 cliff, so both the mitigations AND the `none` baseline they were
# compared against are contaminated. This array re-runs them on top of the
# honest key-side protocol.
#
# The 9 tasks are delay {1,2,3} x mitigation {zero, freeze, vel_warp}.
# The `none` column is already measured (dumps/honest_d{1,2,3}_key):
#     0.1849 / 0.1007 / 0.0731   vs naive 0.1331 / 0.0774 / 0.0610
#
# LTC v1/v2 are deliberately NOT re-run here: their training pairs were
# generated under the coordinate lie, so an honest evaluation of a contaminated
# model would answer nothing. Re-dumping and retraining is a separate decision.
#
# a100 only: streampetr_env's flash-attn kernels need sm_80+.
#
# Usage:   sbatch stages/04_streampetr_latent/submit_honest_mitigations.sh
#          sbatch --array=0-2 stages/04_streampetr_latent/submit_honest_mitigations.sh
# Monitor: squeue -u "$USER"
#
#SBATCH --job-name=honest_mit
#SBATCH --array=0-8%4
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/ibex/user/khangea/multisensor/stages/04_streampetr_latent/logs/honest_mit_%A_%a.out
#SBATCH --error=/ibex/user/khangea/multisensor/stages/04_streampetr_latent/logs/honest_mit_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ibex/user/khangea/multisensor}"
STAGE_DIR="${PROJECT_DIR}/stages/04_streampetr_latent"
CONDA_ENV="${CONDA_ENV:-/ibex/user/khangea/conda-environments/streampetr_env}"
CONDA_SH="${CONDA_SH:-/ibex/user/khangea/miniforge/etc/profile.d/conda.sh}"

CONFIG="${CONFIG:-projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py}"
CKPT="${CKPT:-${PROJECT_DIR}/checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth}"

# One line per array task: "<delay> <mitigation>"
CONFIGS=(
  "1 zero"     "1 freeze"     "1 vel_warp"
  "2 zero"     "2 freeze"     "2 vel_warp"
  "3 zero"     "3 freeze"     "3 vel_warp"
)

main() {
  local task="${SLURM_ARRAY_TASK_ID:-0}"
  local spec="${CONFIGS[$task]}"
  local delay mit
  delay="$(echo "$spec" | cut -d' ' -f1)"
  mit="$(echo "$spec" | cut -d' ' -f2)"

  local out="${STAGE_DIR}/dumps/honestmit_d${delay}_${mit}"
  echo "[honest-mit] task=${task} delay=${delay} mitigation=${mit} (honest key)"
  echo "[honest-mit] out=${out}"

  [[ -f "$CKPT" ]] || { echo "error: checkpoint not found: $CKPT" >&2; exit 1; }
  [[ -f "$CONDA_SH" ]] || { echo "error: conda.sh not found: $CONDA_SH" >&2; exit 1; }

  local pkl="${PROJECT_DIR}/StreamPETR/data/nuscenes/nuscenes2d_temporal_infos_val.pkl"
  [[ -f "$pkl" ]] || {
    echo "error: dataroot does not resolve: $pkl" >&2
    echo "  fix: ln -sfn ${PROJECT_DIR}/data_full ${PROJECT_DIR}/StreamPETR/data/nuscenes" >&2
    exit 1
  }

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
    --mitigation "$mit" \
    --honest-mode key \
    --out "$out"

  echo "[honest-mit] task ${task} done -> ${out}"
}

main "$@"
