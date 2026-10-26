#!/usr/bin/env bash
#
# submit_eval.sh — evaluate the two stage-10 fine-tunes on the stage-09 delay grid.
#
# Pre-registered: the checkpoint is the FINAL one (iter_7380) for both arms — no
# checkpoint selection on the 23 val scenes. The pretrained model's numbers for
# the same grid already exist in stages/09_subframe_async/dumps/async_*.
#
# 16 tasks = {delay, control} x {uniform:0, uniform:1/2/3 x naive/honest, rr:2 honest}
# ~5 min each on one A100.
#
# Usage:   sbatch stages/10_delay_finetune/submit_eval.sh
#   age arm (8 tasks): sbatch --array=0-7 --export=ALL,ARMS=ft_age_delay_p0.5,CONFIG=/ibex/user/khangea/multisensor/stages/10_delay_finetune/configs/ft_age_r50.py stages/10_delay_finetune/submit_eval.sh
# Monitor: squeue -u "$USER"
#
#SBATCH --job-name=ft_eval
#SBATCH --array=0-15%4
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/ibex/user/khangea/multisensor/stages/10_delay_finetune/logs/eval_%A_%a.out
#SBATCH --error=/ibex/user/khangea/multisensor/stages/10_delay_finetune/logs/eval_%A_%a.err

# -u omitted: conda's activate scripts reference unset variables.
set -eo pipefail
IFS=$'\n\t'

PROJECT_DIR="${PROJECT_DIR:-/ibex/user/khangea/multisensor}"
STAGE_DIR="${PROJECT_DIR}/stages/10_delay_finetune"
CONDA_ENV="${CONDA_ENV:-/ibex/user/khangea/conda-environments/streampetr_env}"
CONDA_SH="${CONDA_SH:-/ibex/user/khangea/miniforge/etc/profile.d/conda.sh}"
CKPT_ITER="${CKPT_ITER:-7380}"
CONFIG="${CONFIG:-projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py}"

# ARMS env (space-separated) overrides the default pair, e.g. ARMS=ft_age_delay_p0.5
IFS=' ' read -r -a ARMS <<< "${ARMS:-ft_delay_p0.5 ft_control_p0}"
GRID=(
  "uniform:0 naive"
  "uniform:1 naive" "uniform:1 honest"
  "uniform:2 naive" "uniform:2 honest"
  "uniform:3 naive" "uniform:3 honest"
  "rr:2 honest"
)

main() {
  local task="${SLURM_ARRAY_TASK_ID:-0}"
  local arm="${ARMS[$(( task / ${#GRID[@]} ))]}"
  local spec="${GRID[$(( task % ${#GRID[@]} ))]}"
  [[ -n "$arm" && -n "$spec" ]] || { echo "error: no config for task $task" >&2; exit 1; }
  local schedule geometry
  IFS=' ' read -r schedule geometry <<< "$spec"

  local ckpt="${STAGE_DIR}/work_dirs/${arm}/iter_${CKPT_ITER}.pth"
  local out="${STAGE_DIR}/dumps/eval_${arm}_i${CKPT_ITER}_${schedule/:/}_${geometry}"
  echo "[ft-eval] task=${task} arm=${arm} schedule=${schedule} geometry=${geometry}"
  echo "[ft-eval] ckpt=${ckpt} out=${out}"

  [[ -s "$ckpt" ]] || { echo "error: checkpoint missing/empty: $ckpt" >&2; exit 1; }
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
  PYTHONPATH="${PWD}:${STAGE_DIR}" python "${PROJECT_DIR}/stages/09_subframe_async/eval_async_cams.py" \
    --config "$CONFIG" --schedule "$schedule" --geometry "$geometry" --ckpt "$ckpt" --out "$out"

  echo "[ft-eval] task ${task} done -> ${out}"
}

main "$@"
