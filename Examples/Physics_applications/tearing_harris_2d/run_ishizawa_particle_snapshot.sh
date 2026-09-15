#!/usr/bin/env bash
set -euo pipefail

# Write one particle-rich plotfile from the latest long-run checkpoint.
# This is intended for Ishizawa-style species-flow / pressure-tensor analysis.
# It does NOT rerun the simulation from t=0.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_INPUT="${SCRIPT_DIR}/inputs_2d_tearing_mi800"
WARPX_EXE="${WARPX_EXE:-${REPO_ROOT}/build/bin/warpx.2d}"
PPC="${PPC:-64}"
SOURCE_RUN="${SOURCE_RUN:-${SCRIPT_DIR}/runs_long/ppc${PPC}}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs_ishizawa/ppc${PPC}}"

export AMREX_DEFAULT_INIT="${AMREX_DEFAULT_INIT:-amrex.the_arena_init_size=0}"

if [[ "${PPC}" != "64" && "${PPC}" != "16" ]]; then
    echo "ERROR: supported PPC values are 16 or 64" >&2
    exit 1
fi

case "${PPC}" in
    16) DIMS="4 4" ;;
    64) DIMS="8 8" ;;
esac

latest_chk="$(find "${SOURCE_RUN}/diags" -maxdepth 1 -type d -name 'chk*' -print 2>/dev/null | sort -V | tail -n 1)"
if [[ -z "${latest_chk}" ]]; then
    echo "ERROR: no checkpoint found under ${SOURCE_RUN}/diags" >&2
    exit 1
fi

base="$(basename "${latest_chk}")"
stepstr="${base#chk}"
# Strip leading zeros safely via base-10 expansion.
step=$((10#${stepstr}))
target=$((step + 1))

mkdir -p "${OUT_DIR}"
input="${OUT_DIR}/inputs_particle_snapshot"

sed \
    -e "s/^electrons\.num_particles_per_cell_each_dim *=.*/electrons.num_particles_per_cell_each_dim = ${DIMS}/" \
    -e "s/^ions\.num_particles_per_cell_each_dim *=.*/ions.num_particles_per_cell_each_dim = ${DIMS}/" \
    -e "s/^my_constants\.epsb *=.*/my_constants.epsb = 0.0/" \
    -e "s/^max_step *=.*/max_step = ${target}/" \
    -e "s/^diag1\.intervals *=.*/diag1.intervals = 1/" \
    -e "s/^diag1\.write_species *=.*/diag1.write_species = 1/" \
    "${BASE_INPUT}" > "${input}"

# Use an absolute checkpoint path so the snapshot can be written in a separate
# directory without touching the production run.
chk_abs="$(cd "$(dirname "${latest_chk}")" && pwd)/$(basename "${latest_chk}")"

echo "=============================================================================="
echo "Ishizawa particle-rich diagnostic snapshot"
echo "PPC              : ${PPC}"
echo "restart checkpoint: ${chk_abs}"
echo "restart step      : ${step}"
echo "target step       : ${target}"
echo "output directory  : ${OUT_DIR}"
echo "=============================================================================="

(
    cd "${OUT_DIR}"
    rm -rf diags
    "${WARPX_EXE}" inputs_particle_snapshot "amr.restart=${chk_abs}" 2>&1 | tee snapshot.log
)

echo
echo "Particle-rich snapshot completed."
echo "Inspect: ${OUT_DIR}/diags/diag1*"
