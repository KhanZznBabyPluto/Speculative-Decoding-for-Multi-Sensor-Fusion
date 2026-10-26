#!/usr/bin/env bash
#
# submit_regen_train_pkl.sh — regenerate nuScenes infos INCLUDING train scenes.
#
# The delay-conditioning fine-tune needs train infos, but the stage-04 converter
# patch skips train samples (val-only, for speed) and left the train pkl empty.
# That patch is now switchable; this job runs the converter with
# STREAMPETR_INCLUDE_TRAIN=1 to emit both splits.
#
# Blob 01 covers 62 train scenes (2462 keyframes) + 23 val scenes (914). The
# partial-blob tolerance patch skips samples whose files are absent, so this
# processes ~3376 samples rather than all ~34k.
#
# SAFETY: writes to a NEW directory. The existing
# data_full/nuscenes2d_temporal_infos_val.pkl is what every measurement in the
# project so far was computed against; it is NOT touched. The freshly generated
# val pkl lands beside the new train pkl, which also gives a free consistency
# check — it should contain the same 914 samples as the old one.
#
# CPU only; no GPU is requested (the converter does 3D->2D projection and sweep
# backtracking on CPU). Wall time is generous because this is a one-off: the
# val-only run took ~40 min for 914 samples, so ~3376 samples is ~2-2.5 h.
#
# Usage:   sbatch stages/07_delay_conditioning/submit_regen_train_pkl.sh
# Monitor: squeue -u "$USER"
#
#SBATCH --job-name=regen_pkl
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/ibex/user/khangea/multisensor/stages/07_delay_conditioning/logs/regen_pkl_%j.out
#SBATCH --error=/ibex/user/khangea/multisensor/stages/07_delay_conditioning/logs/regen_pkl_%j.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ibex/user/khangea/multisensor}"
STAGE_DIR="${PROJECT_DIR}/stages/07_delay_conditioning"
CONDA_ENV="${CONDA_ENV:-/ibex/user/khangea/conda-environments/streampetr_env}"
CONDA_SH="${CONDA_SH:-/ibex/user/khangea/miniforge/etc/profile.d/conda.sh}"

# New output tree: symlinks the raw data in, so the converter finds samples/ and
# v1.0-trainval/ while writing its pkls somewhere that clobbers nothing.
OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/data_withtrain}"

main() {
  [[ -f "$CONDA_SH" ]] || { echo "error: conda.sh not found: $CONDA_SH" >&2; exit 1; }

  mkdir -p "${STAGE_DIR}/logs" "$OUT_ROOT"

  # Link the read-only inputs the converter needs. -n so we never follow an
  # existing symlink and write through it.
  for d in samples sweeps maps v1.0-trainval; do
    [[ -e "${PROJECT_DIR}/data_full/${d}" ]] || {
      echo "error: missing input: data_full/${d}" >&2; exit 1; }
    ln -sfn "${PROJECT_DIR}/data_full/${d}" "${OUT_ROOT}/${d}"
  done

  # Refuse to clobber a previous run's output.
  if [[ -f "${OUT_ROOT}/nuscenes2d_temporal_infos_train.pkl" ]]; then
    echo "error: ${OUT_ROOT}/nuscenes2d_temporal_infos_train.pkl already exists." >&2
    echo "  Move it aside first; this script will not overwrite it." >&2
    exit 1
  fi

  set +e
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  local rc=$?
  set -e
  [[ $rc -eq 0 ]] || { echo "error: could not activate $CONDA_ENV" >&2; exit 1; }

  cd "${PROJECT_DIR}/StreamPETR"
  echo "[regen] writing to ${OUT_ROOT} (data_full is untouched)"
  STREAMPETR_INCLUDE_TRAIN=1 python tools/create_data_nusc.py \
      --root-path "$OUT_ROOT" \
      --version v1.0 \
      --out-dir "$OUT_ROOT" \
      --extra-tag nuscenes2d

  echo
  echo "=== result ==="
  ls -la "${OUT_ROOT}"/nuscenes2d_temporal_infos_*.pkl

  python - <<PY
import pickle
for split in ("train", "val"):
    p = "${OUT_ROOT}/nuscenes2d_temporal_infos_%s.pkl" % split
    infos = pickle.load(open(p, "rb"))["infos"]
    scenes = {i["scene_token"] for i in infos}
    print(f"  {split}: {len(infos)} samples across {len(scenes)} scenes")
# Consistency check against the pkl every prior measurement used.
old = pickle.load(open("${PROJECT_DIR}/data_full/nuscenes2d_temporal_infos_val.pkl", "rb"))["infos"]
new = pickle.load(open("${OUT_ROOT}/nuscenes2d_temporal_infos_val.pkl", "rb"))["infos"]
same = {i["token"] for i in old} == {i["token"] for i in new}
print(f"  val token set matches the original pkl: {same}")
PY
}

main "$@"
