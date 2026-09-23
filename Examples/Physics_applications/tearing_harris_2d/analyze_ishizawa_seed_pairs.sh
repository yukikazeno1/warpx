#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEEDS="${SEEDS:-101 202 303 404 505 606}"
PPC="${PPC:-49}"
COULOMB_LOG="${COULOMB_LOG:-15.0}"
COLLISION_SUPERCYCLE="${COLLISION_SUPERCYCLE:-10}"

COMPARE_SCRIPT="${COMPARE_SCRIPT:-${SCRIPT_DIR}/compare_ishizawa_collisionless_collisional_fixed_v2.py}"

[[ -f "${COMPARE_SCRIPT}" ]] || {
    echo "ERROR: comparison script not found: ${COMPARE_SCRIPT}" >&2
    exit 1
}

CLOG_TAG="$(printf '%s' "${COULOMB_LOG}" | tr '.' 'p' | tr '-' 'm')"

read -r -a SEED_ARRAY <<< "${SEEDS}"

for seed in "${SEED_ARRAY[@]}"; do
    SEED_TAG="$(printf '%06d' "${seed}")"

    NOCOLL="${SCRIPT_DIR}/runs_ishizawa_scale_ensemble/seed${SEED_TAG}/nocoll_ppc${PPC}"
    COLL="${SCRIPT_DIR}/runs_ishizawa_scale_ensemble/seed${SEED_TAG}/coll_ei_ppc${PPC}_clog${CLOG_TAG}_sc${COLLISION_SUPERCYCLE}"
    OUT="${SCRIPT_DIR}/runs_ishizawa_scale_ensemble/seed${SEED_TAG}/comparison"

    echo
    echo "=============================================================================="
    echo "Analyzing seed ${seed}"
    echo "nocoll : ${NOCOLL}"
    echo "coll   : ${COLL}"
    echo "out    : ${OUT}"
    echo "=============================================================================="

    [[ -d "${NOCOLL}" ]] || { echo "WARNING: missing ${NOCOLL}; skipping seed ${seed}" >&2; continue; }
    [[ -d "${COLL}" ]] || { echo "WARNING: missing ${COLL}; skipping seed ${seed}" >&2; continue; }

    python3 "${COMPARE_SCRIPT}" \
        --collisionless "${NOCOLL}" \
        --collisional "${COLL}" \
        --output-dir "${OUT}"
done

echo
echo "Per-seed comparisons completed."
