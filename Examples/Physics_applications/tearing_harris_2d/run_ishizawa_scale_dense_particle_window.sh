#!/usr/bin/env bash
set -euo pipefail

# Dense particle-rich continuation for the saturated Ishizawa-scale breathing
# state.  The run is restarted from an *exact* production checkpoint and is
# written to a separate directory, so the original production run is not
# modified.
#
# Default window:
#   step 160000 -> 240000
#   omega_ci t  4.0 -> 6.0
#   particle-rich output every 2000 steps
#   Delta(omega_ci t)=0.05, about 12.6 samples per T=0.63 breathing period
#
# This is intentionally I/O heavy.  It exists to make the weak-form momentum
# closure temporally well conditioned; do not use it as the ordinary
# production runner.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_INPUT="${SCRIPT_DIR}/inputs_2d_ishizawa_scale_pilot_mi800"

SOURCE_RUN="${SOURCE_RUN:-${SCRIPT_DIR}/runs_ishizawa_scale/ppc49}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs_ishizawa_particle_dense/ppc49_tau4p0_6p0}"

START_STEP="${START_STEP:-160000}"
END_STEP="${END_STEP:-240000}"
PARTICLE_INTERVAL="${PARTICLE_INTERVAL:-2000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20000}"
FORCE="${FORCE:-0}"

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

for v in START_STEP END_STEP PARTICLE_INTERVAL CHECKPOINT_INTERVAL; do
    val="${!v}"
    if ! [[ "${val}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${v} must be a non-negative integer; got '${val}'" >&2
        exit 1
    fi
done

if (( END_STEP <= START_STEP )); then
    echo "ERROR: END_STEP must be > START_STEP" >&2
    exit 1
fi
if (( PARTICLE_INTERVAL <= 0 || CHECKPOINT_INTERVAL <= 0 )); then
    echo "ERROR: intervals must be positive" >&2
    exit 1
fi
if (( START_STEP % PARTICLE_INTERVAL != 0 || END_STEP % PARTICLE_INTERVAL != 0 )); then
    echo "ERROR: START_STEP and END_STEP should be exact multiples of PARTICLE_INTERVAL" >&2
    exit 1
fi

valid_checkpoint() {
    local d="$1"
    [[ -s "${d}/WarpXHeader" ]] &&
    [[ -s "${d}/electrons/Header" ]] &&
    [[ -s "${d}/ions/Header" ]]
}

start_tag="$(printf '%06d' "${START_STEP}")"
SOURCE_CHK="${SOURCE_CHK:-${SOURCE_RUN}/diags/chk${start_tag}}"

if [[ ! -d "${SOURCE_CHK}" ]]; then
    echo "ERROR: exact start checkpoint not found: ${SOURCE_CHK}" >&2
    echo "This dense run intentionally refuses to fall back to an earlier checkpoint." >&2
    echo "Set SOURCE_CHK=/absolute/path/to/a/valid/chk${start_tag} if it exists elsewhere." >&2
    exit 1
fi
if ! valid_checkpoint "${SOURCE_CHK}"; then
    echo "ERROR: checkpoint is incomplete/corrupted: ${SOURCE_CHK}" >&2
    echo "Required non-empty files: WarpXHeader, electrons/Header, ions/Header" >&2
    exit 1
fi

if [[ -e "${OUT_DIR}" && "${FORCE}" != "1" ]]; then
    echo "ERROR: output directory already exists: ${OUT_DIR}" >&2
    echo "Use a new OUT_DIR, or FORCE=1 to replace it." >&2
    exit 1
fi
if [[ "${FORCE}" == "1" && -d "${OUT_DIR}" ]]; then
    echo "FORCE=1: removing ${OUT_DIR}"
    rm -rf "${OUT_DIR}"
fi
mkdir -p "${OUT_DIR}"

INPUT_FILE="${OUT_DIR}/inputs_dense_particle_window"

sed \
    -e "s/^max_step *=.*/max_step = ${END_STEP}/" \
    -e "s/^diag1\\.intervals *=.*/diag1.intervals = ${PARTICLE_INTERVAL}/" \
    -e "s/^diag1\\.write_species *=.*/diag1.write_species = 1/" \
    -e "s/^diagnostics\\.diags_names *=.*/diagnostics.diags_names = diag1 chk_dense/" \
    "${BASE_INPUT}" > "${INPUT_FILE}"

cat >> "${INPUT_FILE}" <<EOF

###############################################################################
# Dense particle-window restart support
###############################################################################
diag1.dump_last_timestep = 1

chk_dense.intervals = ${CHECKPOINT_INTERVAL}
chk_dense.diag_type = Full
chk_dense.format = checkpoint
chk_dense.dump_last_timestep = 1
EOF

OMEGA_CI_T0="$(python3 - <<PY
print(${START_STEP} * 0.02 / 800.0)
PY
)"
OMEGA_CI_T1="$(python3 - <<PY
print(${END_STEP} * 0.02 / 800.0)
PY
)"
DELTA_TAU="$(python3 - <<PY
print(${PARTICLE_INTERVAL} * 0.02 / 800.0)
PY
)"
NSNAP=$(( (END_STEP - START_STEP) / PARTICLE_INTERVAL + 1 ))

chk_abs="$(cd "$(dirname "${SOURCE_CHK}")" && pwd)/$(basename "${SOURCE_CHK}")"

echo "=============================================================================="
echo "Ishizawa dense particle-rich breathing window"
echo "source checkpoint      : ${chk_abs}"
echo "start step             : ${START_STEP}  (omega_ci t=${OMEGA_CI_T0})"
echo "end step               : ${END_STEP}  (omega_ci t=${OMEGA_CI_T1})"
echo "particle interval      : ${PARTICLE_INTERVAL}"
echo "Delta(omega_ci t)      : ${DELTA_TAU}"
echo "expected snapshots     : about ${NSNAP}"
echo "checkpoint interval    : ${CHECKPOINT_INTERVAL}"
echo "output directory       : ${OUT_DIR}"
echo "WarpX                   : ${exe}"
echo "=============================================================================="
echo "WARNING: write_species=1 for ~${NSNAP} outputs is disk intensive."
echo "         Check available disk space before continuing."
echo

(
    cd "${OUT_DIR}"
    "${exe}" inputs_dense_particle_window "amr.restart=${chk_abs}" 2>&1 | tee dense_particle.log
)

echo
echo "Dense particle-rich continuation completed."
echo "Particle-rich plotfiles:"
find "${OUT_DIR}/diags" -maxdepth 1 -type d -name 'diag1*' -print 2>/dev/null | sort -V
