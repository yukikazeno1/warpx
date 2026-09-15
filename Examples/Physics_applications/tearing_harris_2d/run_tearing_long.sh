#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_INPUT="${SCRIPT_DIR}/inputs_2d_tearing_mi800"
WARPX_EXE="${WARPX_EXE:-${REPO_ROOT}/build/bin/warpx.2d}"

# Long-time production defaults.
# Override on the command line, e.g.
#   CASES="64" TARGET_STEPS=20000 ./run_tearing_long.sh
CASES="${CASES:-16 64}"
TARGET_STEPS="${TARGET_STEPS:-10000}"
DIAG_INTERVAL="${DIAG_INTERVAL:-100}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2000}"
FORCE="${FORCE:-0}"

export AMREX_DEFAULT_INIT="${AMREX_DEFAULT_INIT:-amrex.the_arena_init_size=0}"

if [[ ! -x "${WARPX_EXE}" ]]; then
    echo "ERROR: WarpX executable not found or not executable:" >&2
    echo "  ${WARPX_EXE}" >&2
    echo "Set WARPX_EXE=/path/to/warpx.2d and rerun." >&2
    exit 1
fi

if [[ ! -f "${BASE_INPUT}" ]]; then
    echo "ERROR: base input not found: ${BASE_INPUT}" >&2
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/runs_long"

ppc_dims() {
    case "$1" in
        16) echo "4 4" ;;
        64) echo "8 8" ;;
        *)
            echo "ERROR: unsupported production PPC '$1' (supported: 16, 64)" >&2
            return 1
            ;;
    esac
}

latest_checkpoint() {
    local run_dir="$1"
    if [[ ! -d "${run_dir}/diags" ]]; then
        return 0
    fi
    find "${run_dir}/diags" -maxdepth 1 -type d -name 'chk*' -print 2>/dev/null \
        | sort -V | tail -n 1
}

run_case() {
    local ppc="$1"
    local dims
    dims="$(ppc_dims "${ppc}")"

    local run_dir="${SCRIPT_DIR}/runs_long/ppc${ppc}"
    local input_file="${run_dir}/inputs_long"
    local log_file="${run_dir}/run.log"

    if [[ "${FORCE}" == "1" && -d "${run_dir}" ]]; then
        echo "[ppc${ppc}] FORCE=1: removing previous long-run directory"
        rm -rf "${run_dir}"
    fi

    mkdir -p "${run_dir}"

    # Create an exact per-case production input.  Compared with the short PPC
    # convergence test, field output is reduced by a factor of 10 (100 vs 10
    # steps).  Checkpoints remain relatively sparse and contain restart data.
    sed \
        -e "s/^electrons\.num_particles_per_cell_each_dim *=.*/electrons.num_particles_per_cell_each_dim = ${dims}/" \
        -e "s/^ions\.num_particles_per_cell_each_dim *=.*/ions.num_particles_per_cell_each_dim = ${dims}/" \
        -e "s/^my_constants\.epsb *=.*/my_constants.epsb = 0.0/" \
        -e "s/^max_step *=.*/max_step = ${TARGET_STEPS}/" \
        -e "s/^diag1\.intervals *=.*/diag1.intervals = ${DIAG_INTERVAL}/" \
        -e "s/^diagnostics\.diags_names *=.*/diagnostics.diags_names = diag1 chk/" \
        "${BASE_INPUT}" > "${input_file}"

    cat >> "${input_file}" <<EOF

###############################################################################
# Long-run restart checkpoint diagnostic
###############################################################################
chk.intervals = ${CHECKPOINT_INTERVAL}
chk.diag_type = Full
chk.format = checkpoint
chk.dump_last_timestep = 1
EOF

    local restart
    restart="$(latest_checkpoint "${run_dir}")"

    echo
    echo "=============================================================================="
    echo "Long-time no-seed Harris tearing run"
    echo "PPC/species          : ${ppc} (${dims})"
    echo "Target total steps   : ${TARGET_STEPS}"
    echo "Field diag interval  : ${DIAG_INTERVAL}"
    echo "Checkpoint interval  : ${CHECKPOINT_INTERVAL}"
    echo "Run directory        : ${run_dir}"
    echo "=============================================================================="

    if [[ -n "${restart}" ]]; then
        # Use a path relative to the run directory, as expected by WarpX.
        local restart_rel="${restart#${run_dir}/}"
        echo "[ppc${ppc}] restarting from ${restart_rel}"
        (
            cd "${run_dir}"
            "${WARPX_EXE}" inputs_long "amr.restart=${restart_rel}" 2>&1 | tee -a run.log
        )
    else
        echo "[ppc${ppc}] no checkpoint found; starting from t=0"
        (
            cd "${run_dir}"
            "${WARPX_EXE}" inputs_long 2>&1 | tee run.log
        )
    fi

    echo "[ppc${ppc}] completed target=${TARGET_STEPS}"
}

for ppc in ${CASES}; do
    run_case "${ppc}"
done

echo
echo "All requested long-time tearing cases completed."
echo "Analyze with:"
echo "  python ${SCRIPT_DIR}/analyze_tearing_long.py"
