#!/usr/bin/env bash
#
# train_chain.sh — self-resubmitting 1 h SLURM chain: StreamPETR delay fine-tune.
#
# Two arms, identical except the per-camera delay probability:
#   ARM=delay    P_STALE=0.5  (each camera 1-3 sweeps ≈ 80-250 ms old, honest geometry)
#   ARM=control  P_STALE=0    (same fine-tune, no delay: separates "learned the delay"
#                              from "just fine-tuned on these 62 scenes")
#
# Each job: find the newest non-empty iter_*.pth -> stop if done -> queue the next
# job (afterany) -> train under a soft time limit -> warn if no new checkpoint.
# W&B run id is fixed per arm so the whole chain is one run.
#
# Usage (you launch the first job; the chain resubmits itself):
#   sbatch --job-name=ft_delay   --export=ALL,ARM=delay   stages/10_delay_finetune/train_chain.sh
#   sbatch --job-name=ft_control --export=ALL,ARM=control stages/10_delay_finetune/train_chain.sh
#   age-conditioned delay arm:
#   sbatch --job-name=ft_age --export=ALL,ARM=delay,RUN_NAME=ft_age_delay_p0.5,CONFIG=/ibex/user/khangea/multisensor/stages/10_delay_finetune/configs/ft_age_r50.py stages/10_delay_finetune/train_chain.sh
#   age arm, lr sweep on age_embed only (M = 10 / 30 / 100):
#   sbatch --job-name=ft_age_lrx<M> --export=ALL,ARM=delay,AGE_LR_MULT=<M>,RUN_NAME=ft_age_delay_p0.5_lrx<M>,CONFIG=/ibex/user/khangea/multisensor/stages/10_delay_finetune/configs/ft_age_lr_r50.py stages/10_delay_finetune/train_chain.sh
# Monitor: squeue -u "$USER";  tail -f stages/10_delay_finetune/work_dirs/<arm>/slurm/*.out
# Stop:    scancel --name=<the --job-name you launched with>  (successors inherit it)
#
#SBATCH --job-name=ft_chain
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/ibex/user/khangea/multisensor/stages/10_delay_finetune/logs/ft_%j.out
#SBATCH --error=/ibex/user/khangea/multisensor/stages/10_delay_finetune/logs/ft_%j.err

# -u omitted: conda's activate scripts reference unset variables.
set -eo pipefail
IFS=$'\n\t'

PROJECT_DIR="${PROJECT_DIR:-/ibex/user/khangea/multisensor}"
STAGE_DIR="${PROJECT_DIR}/stages/10_delay_finetune"
CONDA_ENV="${CONDA_ENV:-/ibex/user/khangea/conda-environments/streampetr_env}"
CONDA_SH="${CONDA_SH:-/ibex/user/khangea/miniforge/etc/profile.d/conda.sh}"
CONFIG="${CONFIG:-${STAGE_DIR}/configs/ft_stale_r50.py}"
NGPUS="${NGPUS:-2}"
MAX_ITERS="${MAX_ITERS:-7380}"                        # must match the config
SOFT_TIME_LIMIT_SEC="${SOFT_TIME_LIMIT_SEC:-3300}"   # 55 min of a 1 h job
ARM="${ARM:?set ARM=delay or ARM=control}"

case "$ARM" in
  delay)   P_STALE="${P_STALE:-0.5}" ;;
  control) P_STALE="${P_STALE:-0}" ;;
  *) echo "error: ARM must be delay or control, got '$ARM'" >&2; exit 1 ;;
esac
RUN_NAME="${RUN_NAME:-ft_${ARM}_p${P_STALE}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-stage10-${RUN_NAME}}"
WORK_DIR="${WORK_DIR:-${STAGE_DIR}/work_dirs/${RUN_NAME}}"

latest_checkpoint() {
  # Newest non-empty iter_*.pth; a wall-time kill mid-save can leave a 0-byte file.
  LATEST=""; LATEST_ITER=-1
  shopt -s nullglob
  local f iter
  for f in "${WORK_DIR}"/iter_*.pth; do
    [[ -s "$f" ]] || { echo "[chain] skipping empty ckpt $f"; continue; }
    iter="$(basename "$f" | sed -E 's/iter_0*([0-9]+)\.pth/\1/')"
    [[ "$iter" =~ ^[0-9]+$ ]] && (( iter > LATEST_ITER )) && { LATEST_ITER=$iter; LATEST="$f"; }
  done
  shopt -u nullglob
}

main() {
  mkdir -p "${WORK_DIR}/slurm" "${STAGE_DIR}/logs"
  echo "[chain] job=${SLURM_JOB_ID:-local} arm=${ARM} p_stale=${P_STALE} work_dir=${WORK_DIR}"

  latest_checkpoint
  echo "[chain] latest checkpoint: ${LATEST:-none} (iter ${LATEST_ITER})"
  if (( LATEST_ITER >= MAX_ITERS )); then
    echo "[chain] reached ${MAX_ITERS} iters; not resubmitting."; exit 0
  fi

  # Watchdog: queue the successor BEFORE training so a crash/hang/kill still continues.
  local next_jid
  next_jid="$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" \
      --job-name="${SLURM_JOB_NAME:-ft_${ARM}}" \
      --output="${WORK_DIR}/slurm/%j.out" --error="${WORK_DIR}/slurm/%j.err" \
      --export="ALL,ARM=${ARM},P_STALE=${P_STALE},RUN_NAME=${RUN_NAME},WANDB_RUN_ID=${WANDB_RUN_ID},WORK_DIR=${WORK_DIR},CONFIG=${CONFIG}" \
      "${STAGE_DIR}/train_chain.sh")"
  echo "[chain] queued next job: ${next_jid}"

  set +e
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  local rc=$?
  set -e
  [[ $rc -eq 0 ]] || { echo "error: could not activate $CONDA_ENV" >&2; exit 1; }
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

  # W&B: online if the node can reach the API, otherwise offline + sync later.
  if curl -s -m 10 -o /dev/null https://api.wandb.ai; then
    export WANDB_MODE="${WANDB_MODE:-online}"
  else
    export WANDB_MODE=offline
    echo "[chain] no route to api.wandb.ai -> WANDB_MODE=offline; later: wandb sync ${PROJECT_DIR}/StreamPETR/wandb/offline-run-*"
  fi
  export P_STALE RUN_NAME WANDB_RUN_ID

  local resume_arg=()
  [[ -n "$LATEST" ]] && resume_arg=(--resume-from "$LATEST")

  cd "${PROJECT_DIR}/StreamPETR"
  local port=$(( 29500 + ${SLURM_JOB_ID:-0} % 1000 ))
  set +e
  PYTHONPATH="${PWD}:${STAGE_DIR}" timeout --preserve-status --signal=SIGTERM "${SOFT_TIME_LIMIT_SEC}" \
    python -m torch.distributed.launch --nproc_per_node="${NGPUS}" --master_port="${port}" --use_env \
      tools/train.py "$CONFIG" --work-dir "$WORK_DIR" --seed 0 --launcher pytorch \
      --no-validate "${resume_arg[@]}"
  rc=$?
  set -e
  echo "[chain] training exited rc=${rc}"

  local before=$LATEST_ITER
  latest_checkpoint
  if (( LATEST_ITER <= before )); then
    echo "[chain] WARNING: no new checkpoint this job (still iter ${LATEST_ITER})." \
         "Two of these in a row = broken, not slow: scancel --name=${SLURM_JOB_NAME:-ft_${ARM}}" >&2
  else
    echo "[chain] progress: iter ${before} -> ${LATEST_ITER} / ${MAX_ITERS}"
  fi
}

main "$@"
