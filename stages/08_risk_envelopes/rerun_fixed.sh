#!/usr/bin/env bash
#
# rerun_fixed.sh — re-run every stage-08 analysis after the audit fixes (CPU only).
# Writes *_fixed outputs next to the originals; nothing existing is overwritten.
# Sequential on purpose: each script loads the NuScenes metadata (~several GB) and
# parallel loads have been OOM-killed before. ~2 h total.
#
# Usage: bash stages/08_risk_envelopes/rerun_fixed.sh
#
set -eo pipefail
IFS=$'\n\t'

STAGE_DIR="/ibex/user/khangea/multisensor/stages/08_risk_envelopes"
D="${STAGE_DIR}/dumps"

# shellcheck disable=SC1091
source /ibex/user/khangea/miniforge/etc/profile.d/conda.sh
conda activate /ibex/user/khangea/conda-environments/streampetr_env
cd "$STAGE_DIR"

step() {   # step <name> <args...>: run one script, log to dumps/<name>.log
  local name="$1"; shift
  [[ ! -e "${D}/${name}.json" ]] || { echo "exists: ${name}.json; skipping"; return; }
  echo "[$(date +%H:%M)] ${name}"
  python "$@" --out "${D}/${name}.json" > "${D}/${name}.log" 2>&1
}

step conformal_d1_key_fixed     conformal_envelopes.py --config d1_key
step envelope_coverage_fixed    envelope_coverage.py
step precision_d1_key_car_fixed envelope_precision.py --config d1_key --classes car
step precision_fresh_car_fixed  envelope_precision.py --config fresh --classes car
step refresh_policy_fixed       refresh_policy.py
echo "[$(date +%H:%M)] ALL DONE" > "${D}/rerun_fixed.DONE"
