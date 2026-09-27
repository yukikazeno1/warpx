#!/usr/bin/env bash
set -euo pipefail

# Generate particle-rich plotfiles at four representative phases of the
# saturated Ishizawa-scale breathing cycle.  Each target is regenerated from
# the nearest valid preceding checkpoint so particles need not be written
# throughout the production run.
#
# Default targets:
#   expanded     omega_ci t=3.275 -> step 131000
#   contracting  omega_ci t=3.400 -> step 136000
#   contracted   omega_ci t=3.550 -> step 142000
#   expanding    omega_ci t=3.725 -> step 149000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_INPUT="${SCRIPT_DIR}/inputs_2d_ishizawa_scale_pilot_mi800"
SOURCE_RUN="${SOURCE_RUN:-${SCRIPT_DIR}/runs_ishizawa_scale/ppc49}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/runs_ishizawa_particle_phases/ppc49}"
TARGET_STEPS="${TARGET_STEPS:-131000 136000 142000 149000}"

if [[ -n "${WARPX_EXE:-}" ]]; then
    exe="${WARPX_EXE}"
elif [[ -x "${REPO_ROOT}/build/bin/warpx.2d.NOMPI.CUDA.DP.PDP.EB" ]]; then
    exe="${REPO_ROOT}/build/bin/warpx.2d.NOMPI.CUDA.DP.PDP.EB"
elif [[ -x "${REPO_ROOT}/build/bin/warpx.2d" ]]; then
    exe="${REPO_ROOT}/build/bin/warpx.2d"
else
    echo "ERROR: WarpX executable not found under ${REPO_ROOT}/build/bin" >&2
    exit 1
fi

export AMREX_DEFAULT_INIT="${AMREX_DEFAULT_INIT:-amrex.the_arena_init_size=0}"

valid_checkpoint() {
    local d="$1"
    [[ -s "${d}/WarpXHeader" ]] &&
    [[ -s "${d}/electrons/Header" ]] &&
    [[ -s "${d}/ions/Header" ]]
}

find_checkpoint_le() {
    local target="$1"
    local best=""
    local beststep=-1
    local d base s step
    shopt -s nullglob
    for d in "${SOURCE_RUN}"/diags/chk*; do
        [[ -d "${d}" ]] || continue
        base="$(basename "${d}")"
        s="${base#chk}"
        [[ "${s}" =~ ^[0-9]+$ ]] || continue
        step=$((10#${s}))
        if (( step <= target && step > beststep )); then
            if valid_checkpoint "${d}"; then
                best="${d}"
                beststep="${step}"
            fi
        fi
    done
    shopt -u nullglob
    [[ -n "${best}" ]] || return 1
    printf '%s|%d\n' "${best}" "${beststep}"
}

mkdir -p "${OUT_ROOT}"

echo "=============================================================================="
echo "Ishizawa-scale particle-rich breathing-phase snapshots"
echo "source run   : ${SOURCE_RUN}"
echo "output root  : ${OUT_ROOT}"
echo "targets      : ${TARGET_STEPS}"
echo "WarpX        : ${exe}"
echo "=============================================================================="

for target in ${TARGET_STEPS}; do
    found="$(find_checkpoint_le "${target}")" || {
        echo "ERROR: no valid checkpoint <= ${target}" >&2
        exit 1
    }
    chk="${found%%|*}"
    chkstep="${found##*|}"

    if (( chkstep == target )); then
        echo "WARNING: target ${target} equals checkpoint ${chkstep};"
        echo "         prefer a nearby non-checkpoint target if no diagnostic is emitted."
    fi

    out="${OUT_ROOT}/step${target}"
    mkdir -p "${out}"
    input="${out}/inputs_particle_phase"

    sed \
        -e "s/^max_step *=.*/max_step = ${target}/" \
        -e "s/^diag1\.intervals *=.*/diag1.intervals = ${target}/" \
        -e "s/^diag1\.write_species *=.*/diag1.write_species = 1/" \
        "${BASE_INPUT}" > "${input}"

    chk_abs="$(cd "$(dirname "${chk}")" && pwd)/$(basename "${chk}")"

    echo
    echo "------------------------------------------------------------------------------"
    echo "target step       : ${target}"
    echo "restart checkpoint: ${chk_abs}"
    echo "restart step      : ${chkstep}"
    echo "steps to recompute: $((target-chkstep))"
    echo "output            : ${out}"
    echo "------------------------------------------------------------------------------"

    (
        cd "${out}"
        rm -rf diags
        "${exe}" inputs_particle_phase "amr.restart=${chk_abs}" 2>&1 | tee snapshot.log
    )

    nplot="$(find "${out}/diags" -maxdepth 1 -type d -name 'diag1*' 2>/dev/null | wc -l)"
    if (( nplot < 1 )); then
        echo "ERROR: no particle-rich plotfile written for target ${target}" >&2
        exit 1
    fi
done

echo
echo "Completed particle-rich phase snapshots."
find "${OUT_ROOT}" -maxdepth 3 -type d -name 'diag1*' -print | sort -V
