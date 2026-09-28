#!/usr/bin/env bash
set -euo pipefail

# Short restart probe for validating field gathering.
#
# The production executable used for the Ishizawa run was built without
# OpenPMD support, so this probe uses a normal plotfile diagnostic and writes
# both a random particle sample and the raw staggered Yee fields.
#
# This branch extends the ordinary plotfile writer so Ex/Ey/Ez/Bx/By/Bz can
# be stored directly on sampled particles using WarpX's own native gather
# helper.  Raw Yee fields are retained as an independent cross-check.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_INPUT="${SCRIPT_DIR}/inputs_2d_ishizawa_scale_pilot_mi800"

DENSE_RUN="${DENSE_RUN:-${SCRIPT_DIR}/runs_ishizawa_particle_dense/ppc49_tau4p0_6p0}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs_ishizawa_native_gather_probe/ppc49}"

START_STEP="${START_STEP:-200000}"
END_STEP="${END_STEP:-212000}"
INTERVAL="${INTERVAL:-2000}"
RANDOM_FRACTION="${RANDOM_FRACTION:-0.02}"
FORCE="${FORCE:-0}"

if [[ -n "${WARPX_EXE:-}" ]]; then
    exe="${WARPX_EXE}"
elif [[ -x "${REPO_ROOT}/build/bin/warpx.2d.NOMPI.CUDA.DP.PDP.EB" ]]; then
    exe="${REPO_ROOT}/build/bin/warpx.2d.NOMPI.CUDA.DP.PDP.EB"
elif [[ -x "${REPO_ROOT}/build/bin/warpx.2d" ]]; then
    exe="${REPO_ROOT}/build/bin/warpx.2d"
else
    echo "ERROR: WarpX executable not found." >&2
    exit 1
fi

export AMREX_DEFAULT_INIT="${AMREX_DEFAULT_INIT:-amrex.the_arena_init_size=0}"

valid_checkpoint () {
    local d="$1"
    [[ -s "${d}/WarpXHeader" ]] &&
    [[ -s "${d}/electrons/Header" ]] &&
    [[ -s "${d}/ions/Header" ]]
}

tag="$(printf '%06d' "${START_STEP}")"
candidates=(
    "${DENSE_RUN}/diags/chk_dense${tag}"
    "${DENSE_RUN}/diags/chk${tag}"
    "${SCRIPT_DIR}/runs_ishizawa_scale/ppc49/diags/chk${tag}"
)

SOURCE_CHK="${SOURCE_CHK:-}"
if [[ -z "${SOURCE_CHK}" ]]; then
    for d in "${candidates[@]}"; do
        if [[ -d "${d}" ]] && valid_checkpoint "${d}"; then
            SOURCE_CHK="${d}"
            break
        fi
    done
fi

if [[ -z "${SOURCE_CHK}" || ! -d "${SOURCE_CHK}" ]]; then
    echo "ERROR: no valid checkpoint found for step ${START_STEP}." >&2
    echo "Tried:" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    echo "Set SOURCE_CHK=/absolute/path/to/checkpoint if needed." >&2
    exit 1
fi
if ! valid_checkpoint "${SOURCE_CHK}"; then
    echo "ERROR: incomplete checkpoint: ${SOURCE_CHK}" >&2
    exit 1
fi

if [[ -e "${OUT_DIR}" && "${FORCE}" != "1" ]]; then
    echo "ERROR: output exists: ${OUT_DIR}" >&2
    echo "Use a fresh OUT_DIR or FORCE=1." >&2
    exit 1
fi
if [[ "${FORCE}" == "1" && -d "${OUT_DIR}" ]]; then
    rm -rf "${OUT_DIR}"
fi
mkdir -p "${OUT_DIR}"

INPUT="${OUT_DIR}/inputs_native_gather_probe"

# Remove the production diagnostic and replace it with a raw-Yee plotfile
# diagnostic. Explicitly synchronize particle momentum with position.
sed \
    -e "s/^max_step *=.*/max_step = ${END_STEP}/" \
    -e "s/^diagnostics\.diags_names *=.*/diagnostics.diags_names = native/" \
    "${BASE_INPUT}" > "${INPUT}"

cat >> "${INPUT}" <<EOF

###############################################################################
# Raw-Yee gather probe (compatible with the current non-OpenPMD WarpX build)
###############################################################################
warpx.synchronize_velocity_for_diagnostics = 1

native.diag_type = Full
native.format = plotfile
native.intervals = ${INTERVAL}
native.file_prefix = diags/native
native.species = electrons ions
native.write_species = 1
native.fields_to_plot = Ex Ey Ez Bx By Bz

# Save the native staggered arrays as raw fields in addition to the usual
# cell-centered diagnostic fields.
native.plot_raw_fields = 1
native.plot_raw_fields_guards = 1

# Request fields on sampled particles.  The plotfile writer on this branch
# calls storeFieldOnParticles(), i.e. WarpX's native gather, before writing.
native.electrons.additional_variables = Ex Ey Ez Bx By Bz
native.ions.additional_variables      = Ex Ey Ez Bx By Bz

# Keep a small unbiased particle sample. Standard particle attributes
# (positions, weighting, momenta) are written by default.
native.electrons.random_fraction = ${RANDOM_FRACTION}
native.ions.random_fraction      = ${RANDOM_FRACTION}
EOF
chk_abs="$(cd "$(dirname "${SOURCE_CHK}")" && pwd)/$(basename "${SOURCE_CHK}")"

echo "=============================================================================="
echo "Ishizawa native field-gather probe"
echo "checkpoint       : ${chk_abs}"
echo "start/end        : ${START_STEP} -> ${END_STEP}"
echo "output interval  : ${INTERVAL}"
echo "particle fraction: ${RANDOM_FRACTION}"
echo "output           : ${OUT_DIR}"
echo "WarpX            : ${exe}"
echo "=============================================================================="

(
    cd "${OUT_DIR}"
    "${exe}" inputs_native_gather_probe "amr.restart=${chk_abs}" 2>&1 | tee native_gather_probe.log
)

echo
echo "Probe completed. Files:"
find "${OUT_DIR}/diags" -maxdepth 2 -type f -o -type d 2>/dev/null | sort | tail -80
