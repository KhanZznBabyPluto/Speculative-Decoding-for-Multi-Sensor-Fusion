#!/usr/bin/env bash
#
# submit_switch.sh — stage 10d: mode switching between pretrained and delay-FT weights.
#
# The stack knows each camera's age, so it can run the pretrained weights when
# every camera is fresh and the delay-FT weights otherwise. Under a constant delay
# that is just picking a model per condition (exact from existing dumps). The
# real test is a delay that JUMPS frame to frame: weights then swap mid-scene and
# each model reads temporal memory written by the other.
#
# Schedule rand:3 = every frame all cameras share one age U{0..3} sweeps
# (0 / ~76 / ~164 / ~249 ms), seeded by sample token; honest geometry.
#   0  pretrained only
#   1  delay-FT only
#   2  switch: delay-FT when max age > 0 ms, else pretrained   (pre-registered rule)
#   3  switch: delay-FT when max age > 100 ms                   (exploratory threshold)
#
# ~5 min per task on one A100.
# Usage:   sbatch stages/10_delay_finetune/submit_switch.sh
# Monitor: squeue -u "$USER"
#
#SBATCH --job-name=ft_switch
#SBATCH --array=0-3
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/ibex/user/khangea/multisensor/stages/10_delay_finetune/logs/switch_%A_%a.out
#SBATCH --error=/ibex/user/khangea/multisensor/stages/10_delay_finetune/logs/switch_%A_%a.err

# -u omitted: conda's activate scripts reference unset variables.
set -eo pipefail
IFS=$'\n\t'

PROJECT_DIR="${PROJECT_DIR:-/ibex/user/khangea/multisensor}"
STAGE_DIR="${PROJECT_DIR}/stages/10_delay_finetune"
CONDA_ENV="${CONDA_ENV:-/ibex/user/khangea/conda-environments/streampetr_env}"
CONDA_SH="${CONDA_SH:-/ibex/user/khangea/miniforge/etc/profile.d/conda.sh}"
PRETRAINED="${PROJECT_DIR}/checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth"
DELAY_FT="${STAGE_DIR}/work_dirs/ft_delay_p0.5/iter_7380.pth"
SCHEDULE="${SCHEDULE:-rand:3}"

# One entry per array task: "<name> <ckpt> <switch-ckpt|-> <switch-above-ms>"
CONFIGS=(
  "pretrained ${PRETRAINED} - 0"
  "delayft ${DELAY_FT} - 0"
  "switch0 ${PRETRAINED} ${DELAY_FT} 0"
  "switch100 ${PRETRAINED} ${DELAY_FT} 100"
)

main() {
  local task="${SLURM_ARRAY_TASK_ID:-0}"
  [[ -n "${CONFIGS[$task]+x}" ]] || { echo "error: no config for task $task" >&2; exit 1; }
  local name ckpt switch_ckpt above
  IFS=' ' read -r name ckpt switch_ckpt above <<< "${CONFIGS[$task]}"

  local out="${STAGE_DIR}/dumps/switch_${SCHEDULE/:/}_honest_${name}"
  echo "[switch] task=${task} name=${name} schedule=${SCHEDULE} out=${out}"

  [[ -f "$ckpt" ]] || { echo "error: checkpoint not found: $ckpt" >&2; exit 1; }
  [[ "$switch_ckpt" == "-" || -f "$switch_ckpt" ]] || { echo "error: not found: $switch_ckpt" >&2; exit 1; }
  [[ ! -e "$out" ]] || { echo "error: $out exists; refusing to overwrite" >&2; exit 1; }
  [[ -f "$CONDA_SH" ]] || { echo "error: conda.sh not found: $CONDA_SH" >&2; exit 1; }

  local switch_args=()
  [[ "$switch_ckpt" != "-" ]] && switch_args=(--switch-ckpt "$switch_ckpt" --switch-above-ms "$above")

  set +e
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  local rc=$?
  set -e
  [[ $rc -eq 0 ]] || { echo "error: could not activate $CONDA_ENV" >&2; exit 1; }
  nvidia-smi --query-gpu=name --format=csv,noheader || true

  cd "${PROJECT_DIR}/StreamPETR"
  PYTHONPATH="${PWD}:${STAGE_DIR}" python "${PROJECT_DIR}/stages/09_subframe_async/eval_async_cams.py" \
    --schedule "$SCHEDULE" --geometry honest --ckpt "$ckpt" --out "$out" "${switch_args[@]}"

  echo "[switch] task ${task} done -> ${out}"
}

main "$@"
