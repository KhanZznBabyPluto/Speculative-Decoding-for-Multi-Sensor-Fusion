#!/usr/bin/env bash
#SBATCH --job-name=mambev_bev_stats
#SBATCH --array=0-22%4                       # 23 scenes (partial blob), max 4 concurrent
#SBATCH --gres=gpu:a100:1                    # mmcv+mamba-ssm were compiled for
                                             # sm_80+sm_90 only — V100 (sm_70)
                                             # crashes with "no kernel image"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/khangea/multisensor/stages/05_mambev_dense_bev/dumps/scale/logs/%A_%a.out
#SBATCH --error=/ibex/user/khangea/multisensor/stages/05_mambev_dense_bev/dumps/scale/logs/%A_%a.err
#
# Stage 05 Scalable AI deliverable — fire one task per scene on one GPU.
#
# Usage:
#   1) Build the scene list (already prepared at dumps/scale/scene_list.txt):
#         ls /ibex/.../stages/04_streampetr_latent/dumps/scene_list.txt
#      The list is 23 scene tokens (one per line) from the partial-blob val.
#   2) Submit the array:
#         sbatch submit_bev_stats.sh
#   3) Watch:
#         squeue -u $USER -A <account>
#         tail -f dumps/scale/logs/<JOBID>_*.out
#   4) Aggregate:
#         python aggregate_bev_stats.py
#
# Adjust --array=0-22%N to throttle concurrency to your allocation (e.g. %4
# for 4-GPU runs, %8 for 8-GPU runs). The "%N" caps concurrent array tasks
# without changing total work — this is how you do the strong-scaling plot
# in 4 separate submissions.

set -euo pipefail

ROOT=/ibex/user/khangea/multisensor
STAGE=${ROOT}/stages/05_mambev_dense_bev
SCENE_LIST=${STAGE}/dumps/scale/scene_list.txt
OUT_DIR=${STAGE}/dumps/scale

mkdir -p "${OUT_DIR}/logs"

if [ ! -f "${SCENE_LIST}" ]; then
    echo "ERROR: scene list not at ${SCENE_LIST}" >&2
    exit 1
fi

# Pick the scene for THIS array slot
SCENE_IDX=${SLURM_ARRAY_TASK_ID}
SCENE=$(sed -n "$((SCENE_IDX + 1))p" "${SCENE_LIST}")
if [ -z "${SCENE}" ]; then
    echo "ERROR: no scene at index ${SCENE_IDX} in ${SCENE_LIST}" >&2
    exit 1
fi

echo "Array task ${SLURM_ARRAY_TASK_ID} → scene ${SCENE}"
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-unset}, start: $(date)"

# Env — avoid `source activate` because conda isn't always initialized in
# sbatch's subshell. Use the env's python binary directly and set the env
# vars the CUDA build needs.
ENV_PREFIX=/ibex/user/khangea/conda-environments/mambev_env
export PATH=${ENV_PREFIX}/bin:${PATH:-}
export CUDA_HOME=${ENV_PREFIX}
export LD_LIBRARY_PATH=${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}

OUT_FILE=${OUT_DIR}/scene_${SCENE:0:8}.jsonl
${ENV_PREFIX}/bin/python "${STAGE}/run_bev_stats.py" \
    --scene "${SCENE}" \
    --anchor-stride 1 \
    --k-list 1 \
    --out   "${OUT_FILE}"

echo "Done. end: $(date)"
