#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paired seeds. Override, for example:
#   SEEDS="101 202 303 404" ./run_ishizawa_seed_ensemble.sh
SEEDS="${SEEDS:-101 202 303 404 505 606}"

MODE="${MODE:-linear}"
PPC="${PPC:-49}"
COULOMB_LOG="${COULOMB_LOG:-15.0}"
COLLISION_SUPERCYCLE="${COLLISION_SUPERCYCLE:-10}"
FORCE="${FORCE:-0}"

# both | nocoll | coll
CASES="${CASES:-both}"

NOCOLL_RUNNER="${SCRIPT_DIR}/run_ishizawa_scale_pilot_seeded.sh"
COLL_RUNNER="${SCRIPT_DIR}/run_ishizawa_scale_pilot_coll_ei_seeded.sh"

[[ -x "${NOCOLL_RUNNER}" ]] || { echo "ERROR: not executable: ${NOCOLL_RUNNER}" >&2; exit 1; }
[[ -x "${COLL_RUNNER}" ]] || { echo "ERROR: not executable: ${COLL_RUNNER}" >&2; exit 1; }

case "${CASES}" in
    both|nocoll|coll) ;;
    *) echo "ERROR: CASES must be both|nocoll|coll" >&2; exit 1 ;;
esac

read -r -a SEED_ARRAY <<< "${SEEDS}"

echo "=============================================================================="
echo "Paired Ishizawa-scale random-seed ensemble"
echo "seeds                  : ${SEEDS}"
echo "number of seeds        : ${#SEED_ARRAY[@]}"
echo "cases                  : ${CASES}"
echo "mode                   : ${MODE}"
echo "PPC/species            : ${PPC}"
echo "CoulombLog             : ${COULOMB_LOG}"
echo "collision supercycle   : ${COLLISION_SUPERCYCLE}"
echo "=============================================================================="

idx=0
for seed in "${SEED_ARRAY[@]}"; do
    idx=$((idx+1))
    echo
    echo "##############################################################################"
    echo "# SEED PAIR ${idx}/${#SEED_ARRAY[@]} : ${seed}"
    echo "##############################################################################"

    if [[ "${CASES}" == "both" || "${CASES}" == "nocoll" ]]; then
        SEED="${seed}" MODE="${MODE}" PPC="${PPC}" FORCE="${FORCE}" \
            "${NOCOLL_RUNNER}"
    fi

    if [[ "${CASES}" == "both" || "${CASES}" == "coll" ]]; then
        SEED="${seed}" MODE="${MODE}" PPC="${PPC}" FORCE="${FORCE}" \
        COULOMB_LOG="${COULOMB_LOG}" \
        COLLISION_SUPERCYCLE="${COLLISION_SUPERCYCLE}" \
            "${COLL_RUNNER}"
    fi
done

echo
echo "=============================================================================="
echo "Finished requested ensemble sequence."
echo "Root: ${SCRIPT_DIR}/runs_ishizawa_scale_ensemble"
echo "=============================================================================="
