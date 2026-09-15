#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_INPUT="${SCRIPT_DIR}/inputs_2d_tearing_mi800"
WARPX_EXE="${WARPX_EXE:-${REPO_ROOT}/build/bin/warpx.2d}"

# Space-separated subset is allowed, e.g. CASES="16 64" ./run_ppc_convergence.sh
CASES="${CASES:-4 16 64}"
FORCE="${FORCE:-0}"

# Avoid AMReX's large default GPU arena pre-allocation on the 12 GB TITAN V.
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

mkdir -p "${SCRIPT_DIR}/runs"

ppc_dims() {
    case "$1" in
        4)  echo "2 2" ;;
        16) echo "4 4" ;;
        64) echo "8 8" ;;
        *)
            echo "ERROR: unsupported PPC '$1' (supported: 4, 16, 64)" >&2
            return 1
            ;;
    esac
}

run_case() {
    local ppc="$1"
    local dims
    dims="$(ppc_dims "${ppc}")"

    local run_dir="${SCRIPT_DIR}/runs/ppc${ppc}"
    local input_file="${run_dir}/inputs"
    local log_file="${run_dir}/run.log"

    if [[ -e "${run_dir}/diags" || -e "${log_file}" ]]; then
        if [[ "${FORCE}" == "1" ]]; then
            echo "[ppc${ppc}] FORCE=1: removing previous run directory"
            rm -rf "${run_dir}"
        else
            echo "[ppc${ppc}] existing output found; skipping."
            echo "           Use FORCE=1 to rerun from scratch."
            return 0
        fi
    fi

    mkdir -p "${run_dir}"

    # Materialize the exact per-case input so every run is self-documenting.
    sed \
        -e "s/^electrons\.num_particles_per_cell_each_dim *=.*/electrons.num_particles_per_cell_each_dim = ${dims}/" \
        -e "s/^ions\.num_particles_per_cell_each_dim *=.*/ions.num_particles_per_cell_each_dim = ${dims}/" \
        -e "s/^my_constants\.epsb *=.*/my_constants.epsb = 0.0/" \
        -e "s/^max_step *=.*/max_step = 2000/" \
        -e "s/^diag1\.intervals *=.*/diag1.intervals = 10/" \
        "${BASE_INPUT}" > "${input_file}"

    echo
    echo "======================================================================"
    echo "Starting no-seed Harris tearing convergence case"
    echo "PPC/species     : ${ppc} (${dims})"
    echo "Run directory   : ${run_dir}"
    echo "WarpX executable: ${WARPX_EXE}"
    echo "======================================================================"

    (
        cd "${run_dir}"
        "${WARPX_EXE}" inputs 2>&1 | tee run.log
    )

    echo "[ppc${ppc}] completed"
}

for ppc in ${CASES}; do
    run_case "${ppc}"
done

echo
echo "All requested convergence cases completed."
echo "Next: source your yt environment and run:"
echo "  python ${SCRIPT_DIR}/analyze_ppc_convergence.py"
