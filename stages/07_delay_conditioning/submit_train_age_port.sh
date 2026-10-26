#!/usr/bin/env bash
#
# submit_train_age_port.sh — fine-tune the age-conditioning port, then evaluate.
#
# Answers: is the conditioning chimera a property of the representation, or just
# an untrained input port? Trains ~422K params (1.1% of the model; everything
# else frozen) on 62 train scenes under per-scene delay augmentation, then
# evaluates on the 23 val scenes at delay=1.
#
#   DECISION RULE, fixed in advance:
#     mAP > 0.1849  -> chimera dissolves with training; engineering problem
#     mAP <= 0.1849 -> chimera survives a trained port; structural
#   (0.1849 = honest key-side baseline, 23/23 scenes, dumps/honest_d1_key)
#
# Two array tasks:
#   0  SMOKE: 40 steps, then eval. Proves the loop runs and the port is a no-op
#      at init before committing to a full run. Expect eval ~= 0.1849.
#   1  FULL: 2 epochs over 2462 frames, then eval.
#
# Run task 0 first and check it, THEN task 1:
#     sbatch --array=0 stages/07_delay_conditioning/submit_train_age_port.sh
#     sbatch --array=1 stages/07_delay_conditioning/submit_train_age_port.sh
#
# a100 only: streampetr_env's flash-attn kernels need sm_80+.
#
#SBATCH --job-name=age_port
#SBATCH --array=0-1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/ibex/user/khangea/multisensor/stages/07_delay_conditioning/logs/age_port_%A_%a.out
#SBATCH --error=/ibex/user/khangea/multisensor/stages/07_delay_conditioning/logs/age_port_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ibex/user/khangea/multisensor}"
STAGE_DIR="${PROJECT_DIR}/stages/07_delay_conditioning"
S04_DIR="${PROJECT_DIR}/stages/04_streampetr_latent"
CONDA_ENV="${CONDA_ENV:-/ibex/user/khangea/conda-environments/streampetr_env}"
CONDA_SH="${CONDA_SH:-/ibex/user/khangea/miniforge/etc/profile.d/conda.sh}"

CONFIG="${CONFIG:-projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py}"
CKPT="${CKPT:-${PROJECT_DIR}/checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth}"
TRAIN_PKL="${TRAIN_PKL:-${PROJECT_DIR}/data_withtrain/nuscenes2d_temporal_infos_train.pkl}"

main() {
  local task="${SLURM_ARRAY_TASK_ID:-0}"
  local tag epochs max_steps
  if [[ "$task" == "0" ]]; then
    tag="smoke"; epochs=1; max_steps=40
  else
    tag="full";  epochs=2; max_steps=0
  fi

  local port_ckpt="${STAGE_DIR}/dumps/age_port_${tag}.pt"
  local eval_out="${STAGE_DIR}/dumps/eval_age_port_${tag}_d1"
  echo "[age-port] task=${task} mode=${tag} epochs=${epochs} max_steps=${max_steps}"

  [[ -f "$CKPT" ]] || { echo "error: checkpoint not found: $CKPT" >&2; exit 1; }
  [[ -f "$TRAIN_PKL" ]] || {
    echo "error: train pkl not found: $TRAIN_PKL" >&2
    echo "  run submit_regen_train_pkl.sh first" >&2; exit 1; }
  [[ -f "$CONDA_SH" ]] || { echo "error: conda.sh not found: $CONDA_SH" >&2; exit 1; }

  local val_pkl="${PROJECT_DIR}/StreamPETR/data/nuscenes/nuscenes2d_temporal_infos_val.pkl"
  [[ -f "$val_pkl" ]] || {
    echo "error: dataroot does not resolve: $val_pkl" >&2
    echo "  fix: ln -sfn ${PROJECT_DIR}/data_full ${PROJECT_DIR}/StreamPETR/data/nuscenes" >&2
    exit 1; }

  set +e
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  local rc=$?
  set -e
  [[ $rc -eq 0 ]] || { echo "error: could not activate $CONDA_ENV" >&2; exit 1; }

  nvidia-smi --query-gpu=name --format=csv,noheader || true
  mkdir -p "${STAGE_DIR}/logs" "${STAGE_DIR}/dumps"

  cd "${PROJECT_DIR}/StreamPETR"
  export PYTHONPATH="${PWD}:${S04_DIR}:${STAGE_DIR}"

  echo "=== TRAIN ==="
  python "${STAGE_DIR}/train_age_port.py" \
    --config "$CONFIG" --ckpt "$CKPT" --train-pkl "$TRAIN_PKL" \
    --epochs "$epochs" --max-steps "$max_steps" --out "$port_ckpt"

  echo
  echo "=== EVAL (delay=1, honest key + trained port) ==="
  echo "    baseline to beat: 0.1849 (honest key alone)"
  python "${S04_DIR}/measure_stale_map_mitigation.py" \
    --config "$CONFIG" --ckpt "$CKPT" \
    --delay-all 1 --mitigation none \
    --honest-mode key --age-port "$port_ckpt" \
    --out "$eval_out"

  echo "[age-port] task ${task} done -> ${eval_out}"
}

main "$@"
