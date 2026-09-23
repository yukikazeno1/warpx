#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_INPUT="${SCRIPT_DIR}/inputs_2d_ishizawa_scale_pilot_mi800_coll_ei"
WARPX_EXE="${WARPX_EXE:-${REPO_ROOT}/build/bin/warpx.2d}"

MODE="${MODE:-linear}"
PPC="${PPC:-49}"
SEED="${SEED:-101}"
FORCE="${FORCE:-0}"
COULOMB_LOG="${COULOMB_LOG:-15.0}"
COLLISION_SUPERCYCLE="${COLLISION_SUPERCYCLE:-10}"

export AMREX_DEFAULT_INIT="${AMREX_DEFAULT_INIT:-amrex.the_arena_init_size=0}"

case "${SEED}" in
    ''|*[!0-9]*) echo "ERROR: SEED must be a positive integer." >&2; exit 1 ;;
esac
(( SEED > 0 )) || { echo "ERROR: SEED must be > 0." >&2; exit 1; }

case "${PPC}" in
    49) DIMS="7 7" ;;
    25) DIMS="5 5" ;;
    *) echo "ERROR: PPC must be 49 or 25." >&2; exit 1 ;;
esac

case "${MODE}" in
    memory)
        TARGET_STEPS="${TARGET_STEPS:-1}"
        DIAG_INTERVAL="${DIAG_INTERVAL:-1000}"
        CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}" ;;
    smoke)
        TARGET_STEPS="${TARGET_STEPS:-100}"
        DIAG_INTERVAL="${DIAG_INTERVAL:-100}"
        CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}" ;;
    early)
        TARGET_STEPS="${TARGET_STEPS:-10000}"
        DIAG_INTERVAL="${DIAG_INTERVAL:-1000}"
        CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}" ;;
    linear)
        TARGET_STEPS="${TARGET_STEPS:-40000}"
        DIAG_INTERVAL="${DIAG_INTERVAL:-1000}"
        CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}" ;;
    onset)
        TARGET_STEPS="${TARGET_STEPS:-100000}"
        DIAG_INTERVAL="${DIAG_INTERVAL:-2500}"
        CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20000}" ;;
    *)
        echo "ERROR: MODE must be memory|smoke|early|linear|onset" >&2
        exit 1 ;;
esac

[[ -f "${BASE_INPUT}" ]] || { echo "ERROR: missing ${BASE_INPUT}" >&2; exit 1; }
[[ -x "${WARPX_EXE}" ]] || { echo "ERROR: missing WarpX executable ${WARPX_EXE}" >&2; exit 1; }

SEED_TAG="$(printf '%06d' "${SEED}")"
CLOG_TAG="$(printf '%s' "${COULOMB_LOG}" | tr '.' 'p' | tr '-' 'm')"
RUN_DIR="${SCRIPT_DIR}/runs_ishizawa_scale_ensemble/seed${SEED_TAG}/coll_ei_ppc${PPC}_clog${CLOG_TAG}_sc${COLLISION_SUPERCYCLE}"
INPUT_FILE="${RUN_DIR}/inputs_scale_pilot_coll_ei_seeded"

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
    -e "s/^collision_ei\.CoulombLog *=.*/collision_ei.CoulombLog = ${COULOMB_LOG}/" \
    -e "s/^collision_ei\.ndt_supercycle *=.*/collision_ei.ndt_supercycle = ${COLLISION_SUPERCYCLE}/" \
    "${BASE_INPUT}" > "${INPUT_FILE}"

cat >> "${INPUT_FILE}" <<EOF

###############################################################################
# Paired-ensemble random seed and checkpoint diagnostic
###############################################################################
warpx.random_seed = ${SEED}

chk.intervals = ${CHECKPOINT_INTERVAL}
chk.diag_type = Full
chk.format = checkpoint
chk.dump_last_timestep = 1
EOF

latest_checkpoint() {
    [[ -d "${RUN_DIR}/diags" ]] || return 0
    find "${RUN_DIR}/diags" -maxdepth 1 -type d -name 'chk*' -print 2>/dev/null \
        | sort -V | tail -n 1
}

OMEGA_CI_T=$(python3 - <<PY
print(${TARGET_STEPS} * 0.02 / 800.0)
PY
)

echo "=============================================================================="
echo "Ishizawa-scale SEEDED e-i collisional run"
echo "seed                    : ${SEED}"
echo "mode                    : ${MODE}"
echo "PPC/species             : ${PPC} (${DIMS})"
echo "CoulombLog              : ${COULOMB_LOG}"
echo "collision supercycle    : ${COLLISION_SUPERCYCLE}"
echo "target steps            : ${TARGET_STEPS}"
echo "omega_ci * t target     : ${OMEGA_CI_T}"
echo "field diag interval     : ${DIAG_INTERVAL}"
echo "checkpoint interval     : ${CHECKPOINT_INTERVAL}"
echo "run directory           : ${RUN_DIR}"
echo "=============================================================================="

restart="$(latest_checkpoint)"
if [[ -n "${restart}" ]]; then
    restart_rel="${restart#${RUN_DIR}/}"
    echo "Restarting from ${restart_rel}"
    (
        cd "${RUN_DIR}"
        "${WARPX_EXE}" inputs_scale_pilot_coll_ei_seeded "amr.restart=${restart_rel}" 2>&1 | tee -a run.log
    )
else
    echo "No checkpoint found; starting from t=0"
    (
        cd "${RUN_DIR}"
        "${WARPX_EXE}" inputs_scale_pilot_coll_ei_seeded 2>&1 | tee run.log
    )
fi
