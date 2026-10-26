#!/usr/bin/env bash
#
# submit_honest_staleness.sh — SLURM array: honest-protocol staleness sweep.
#
# Runs honest_staleness.py once per array task (one config each), evaluating
# StreamPETR under stale camera features with the true capture pose and age
# declared to the head, instead of the hardcoded (identity, dt=0).
#
# The 7 tasks, in order:
#   0  delay0 both   -- NO-OP SELF-TEST. Applied delay is 0, so T=I and dt=0;
#                       this MUST reproduce the fresh baseline mAP 0.4448.
#                       If it does not, the patch is wrong and tasks 2-6 are
#                       meaningless. Check this one first.
#   1  delay1 none   -- patch installed but inactive; MUST reproduce 0.1331.
#   2  delay1 key    -- 3D-lift correction only
#   3  delay1 query  -- MLN/time conditioning correction only
#   4  delay1 both   -- the full honest protocol  <-- the headline number
#   5  delay2 both      -- SUPERSEDED, see tasks 7/8
#   6  delay3 both      -- SUPERSEDED, see tasks 7/8
#   7  delay2 key       -- the deeper-staleness numbers that actually matter
#   8  delay3 key
#
# Tasks 2/3 vs 4 attribute the recovery to the two halves independently, and
# that attribution is why 7/8 exist: the first run showed 'key' helps (+0.052 at
# delay=1, 23/23 scenes) while 'query' HURTS (-0.042, 1/23), so 'both' is the
# two cancelling out. Tasks 5/6 were run in 'both' mode and are therefore
# contaminated by the harmful half; re-run the delay sweep in 'key' mode.
#
# Re-run only the new tasks with:  sbatch --array=7,8 <this script>
#
# a100 only: streampetr_env's flash-attn kernels need sm_80+; V100 nodes fail
# at runtime (same foot-gun that bit the stage-05 array on V100).
#
# Usage:   sbatch stages/04_streampetr_latent/submit_honest_staleness.sh
#          sbatch --array=0,4 stages/04_streampetr_latent/submit_honest_staleness.sh
# Monitor: squeue -u "$USER" ; then compare mAP across dumps/honest_*/
#
#SBATCH --job-name=honest_stale
#SBATCH --array=0-6%4
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/ibex/user/khangea/multisensor/stages/04_streampetr_latent/logs/honest_stale_%A_%a.out
#SBATCH --error=/ibex/user/khangea/multisensor/stages/04_streampetr_latent/logs/honest_stale_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ibex/user/khangea/multisensor}"
STAGE_DIR="${PROJECT_DIR}/stages/04_streampetr_latent"
CONDA_ENV="${CONDA_ENV:-/ibex/user/khangea/conda-environments/streampetr_env}"
CONDA_SH="${CONDA_SH:-/ibex/user/khangea/miniforge/etc/profile.d/conda.sh}"

CONFIG="${CONFIG:-projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py}"
CKPT="${CKPT:-${PROJECT_DIR}/checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth}"

# One line per array task: "<delay> <honest-mode>"
CONFIGS=(
  "0 both"
  "1 none"
  "1 key"
  "1 query"
  "1 both"
  "2 both"
  "3 both"
  "2 key"
  "3 key"
)

main() {
  local task="${SLURM_ARRAY_TASK_ID:-0}"
  local spec="${CONFIGS[$task]}"
  local delay mode
  delay="$(echo "$spec" | cut -d' ' -f1)"
  mode="$(echo "$spec" | cut -d' ' -f2)"

  local out="${STAGE_DIR}/dumps/honest_d${delay}_${mode}"
  echo "[honest] task=${task} delay=${delay} mode=${mode}"
  echo "[honest] out=${out}"

  [[ -f "$CKPT" ]] || { echo "error: checkpoint not found: $CKPT" >&2; exit 1; }
  [[ -f "$CONDA_SH" ]] || { echo "error: conda.sh not found: $CONDA_SH" >&2; exit 1; }

  # The config uses a relative data_root ('./data/nuscenes/'), and the pkl stores
  # relative image paths, so everything hinges on this symlink resolving. It was
  # left dangling (pointing at poc/data_full, which no longer exists) and killed
  # a whole array 15 s in with an opaque FileNotFoundError. Fail loudly instead.
  local pkl="${PROJECT_DIR}/StreamPETR/data/nuscenes/nuscenes2d_temporal_infos_val.pkl"
  [[ -f "$pkl" ]] || {
    echo "error: dataroot does not resolve: $pkl" >&2
    echo "  StreamPETR/data/nuscenes -> $(readlink -f "${PROJECT_DIR}/StreamPETR/data/nuscenes" || true)" >&2
    echo "  fix: ln -sfn ${PROJECT_DIR}/data_full ${PROJECT_DIR}/StreamPETR/data/nuscenes" >&2
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

  # honest_staleness.py imports measure_stale_map, so the stage dir must be on
  # the path too; it must run from StreamPETR/ for the plugin imports to resolve.
  cd "${PROJECT_DIR}/StreamPETR"
  PYTHONPATH="${PWD}:${STAGE_DIR}" python "${STAGE_DIR}/honest_staleness.py" \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --delay-all "$delay" \
    --honest-mode "$mode" \
    --out "$out"

  echo "[honest] task ${task} done -> ${out}"
}

main "$@"
