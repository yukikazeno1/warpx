#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_INPUT="${SCRIPT_DIR}/inputs_2d_harris_500ld_80ld_mi800"
WARPX_EXE="${WARPX_EXE:-${REPO_ROOT}/build/bin/warpx.2d}"

MODE="${MODE:-smoke}"
PPC="${PPC:-49}"
FORCE="${FORCE:-0}"

export AMREX_DEFAULT_INIT="${AMREX_DEFAULT_INIT:-amrex.the_arena_init_size=0}"

case "${PPC}" in
    49) DIMS="7 7" ;;
    25) DIMS="5 5" ;;
    *)
        echo "ERROR: PPC must be 49 or 25." >&2
        exit 1
        ;;
esac

case "${MODE}" in
    memory)
        TARGET_STEPS="${TARGET_STEPS:-1}"
        DIAG_INTERVAL="${DIAG_INTERVAL:-1000}"
        CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
        ;;
    smoke)
        TARGET_STEPS="${TARGET_STEPS:-100}"
        DIAG_INTERVAL="${DIAG_INTERVAL:-100}"
        CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
        ;;
    early)
        TARGET_STEPS="${TARGET_STEPS:-10000}"
        DIAG_INTERVAL="${DIAG_INTERVAL:-1000}"
        CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
        ;;
    linear)
        TARGET_STEPS="${TARGET_STEPS:-40000}"
        DIAG_INTERVAL="${DIAG_INTERVAL:-1000}"
        CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"
        ;;
    onset)
        TARGET_STEPS="${TARGET_STEPS:-100000}"
        DIAG_INTERVAL="${DIAG_INTERVAL:-2500}"
        CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20000}"
        ;;
    *)
        echo "ERROR: MODE must be memory|smoke|early|linear|onset" >&2
        exit 1
        ;;
esac

if [[ ! -x "${WARPX_EXE}" ]]; then
    echo "ERROR: WarpX executable not found: ${WARPX_EXE}" >&2
    exit 1
fi

RUN_DIR="${SCRIPT_DIR}/runs_harris_500ld_80ld/ppc${PPC}"
INPUT_FILE="${RUN_DIR}/inputs_harris_500ld_80ld"

if [[ "${FORCE}" == "1" && -d "${RUN_DIR}" ]]; then
    echo "FORCE=1: removing ${RUN_DIR}"
    rm -rf "${RUN_DIR}"
fi
mkdir -p "${RUN_DIR}"

sed \
    -e "s/^max_step *=.*/max_step = ${TARGET_STEPS}/" \
    -e "s/^electrons\.num_particles_per_cell_each_dim *=.*/electrons.num_particles_per_cell_each_dim = ${DIMS}/" \
    -e "s/^ions\.num_particles_per_cell_each_dim *=.*/ions.num_particles_per_cell_each_dim = ${DIMS}/" \
    -e "s/^diag1\.intervals *=.*/diag1.intervals = ${DIAG_INTERVAL}/" \
    -e "s/^diagnostics\.diags_names *=.*/diagnostics.diags_names = diag1 chk/" \
    "${BASE_INPUT}" > "${INPUT_FILE}"

cat >> "${INPUT_FILE}" <<EOF

###############################################################################
# Restart checkpoint diagnostic added by run_harris_500ld_80ld.sh
###############################################################################
chk.intervals = ${CHECKPOINT_INTERVAL}
chk.diag_type = Full
chk.format = checkpoint
chk.dump_last_timestep = 1
EOF

latest_checkpoint() {
    if [[ ! -d "${RUN_DIR}/diags" ]]; then
        return 0
    fi
    find "${RUN_DIR}/diags" -maxdepth 1 -type d -name 'chk*' -print 2>/dev/null \
        | sort -V | tail -n 1
}

OMEGA_CI_T=$(python3 - <<PY
print(${TARGET_STEPS} * 0.02 / 800.0)
PY
)

echo "=============================================================================="
echo "Harris 500 lambda_D / 80 lambda_D sheet test"
echo "mode                  : ${MODE}"
echo "grid                  : 512 x 512"
echo "box                   : 500 lambda_D x 500 lambda_D"
echo "Harris sheet_l        : 80 lambda_D (tanh scale length)"
echo "PPC/species           : ${PPC} (${DIMS})"
echo "target steps          : ${TARGET_STEPS}"
echo "omega_ci * t target   : ${OMEGA_CI_T}"
echo "field diag interval   : ${DIAG_INTERVAL}"
echo "checkpoint interval   : ${CHECKPOINT_INTERVAL}"
echo "run directory         : ${RUN_DIR}"
echo "=============================================================================="

restart="$(latest_checkpoint)"
if [[ -n "${restart}" ]]; then
    restart_rel="${restart#${RUN_DIR}/}"
    echo "Restarting from ${restart_rel}"
    (
        cd "${RUN_DIR}"
        "${WARPX_EXE}" inputs_harris_500ld_80ld "amr.restart=${restart_rel}" 2>&1 | tee -a run.log
    )
else
    echo "No checkpoint found; starting from t=0"
    (
        cd "${RUN_DIR}"
        "${WARPX_EXE}" inputs_harris_500ld_80ld 2>&1 | tee run.log
    )
fi

echo
echo "Completed MODE=${MODE}, PPC=${PPC}, target=${TARGET_STEPS}."
