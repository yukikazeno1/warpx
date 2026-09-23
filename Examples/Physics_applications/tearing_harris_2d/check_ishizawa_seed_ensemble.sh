#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEEDS="${SEEDS:-101 202 303 404}"
PPC="${PPC:-49}"
COULOMB_LOG="${COULOMB_LOG:-15.0}"
COLLISION_SUPERCYCLE="${COLLISION_SUPERCYCLE:-10}"
TARGET_STEPS="${TARGET_STEPS:-40000}"

ROOT="${SCRIPT_DIR}/runs_ishizawa_scale_ensemble"
CLOG_TAG="$(printf '%s' "${COULOMB_LOG}" | tr '.' 'p' | tr '-' 'm')"

read -r -a SEED_ARRAY <<< "${SEEDS}"

canonical_diag_stats() {
    local run_dir="$1"
    python3 - "${run_dir}" <<'PY'
import re, sys
from pathlib import Path
run = Path(sys.argv[1])
steps = []
diag = run / "diags"
if diag.is_dir():
    for p in diag.iterdir():
        if p.is_dir():
            m = re.fullmatch(r"diag1(\d+)", p.name)
            if m:
                steps.append(int(m.group(1)))
print((max(steps) if steps else -1), len(steps))
PY
}

check_case() {
    local seed="$1"
    local case_name="$2"
    local run_dir="$3"
    local input_file="$4"
    local expect_collision="$5"

    local seed_ok="NO"
    local input_ok="NO"
    local final_ok="NO"
    local status="FAIL"

    if [[ -f "${input_file}" ]]; then
        if grep -Eq "^[[:space:]]*warpx\.random_seed[[:space:]]*=[[:space:]]*${seed}([[:space:]]*#.*)?$" "${input_file}"; then
            seed_ok="YES"
        fi
        if grep -Eq "^[[:space:]]*max_step[[:space:]]*=[[:space:]]*${TARGET_STEPS}([[:space:]]*#.*)?$" "${input_file}"; then
            input_ok="YES"
        fi
        if [[ "${expect_collision}" == "1" ]]; then
            grep -Eq "^[[:space:]]*collision_ei\.species[[:space:]]*=[[:space:]]*electrons[[:space:]]+ions" "${input_file}" || input_ok="NO"
            grep -Eq "^[[:space:]]*collision_ei\.CoulombLog[[:space:]]*=[[:space:]]*${COULOMB_LOG}" "${input_file}" || input_ok="NO"
            grep -Eq "^[[:space:]]*collision_ei\.ndt_supercycle[[:space:]]*=[[:space:]]*${COLLISION_SUPERCYCLE}" "${input_file}" || input_ok="NO"
        fi
    fi

    read -r latest_step nplots < <(canonical_diag_stats "${run_dir}")
    if (( latest_step >= TARGET_STEPS )); then
        final_ok="YES"
    fi

    if [[ "${seed_ok}" == "YES" && "${input_ok}" == "YES" && "${final_ok}" == "YES" ]]; then
        status="PASS"
    else
        overall=1
    fi

    printf "%-8s %-8s %-9s %-9s %-9s %-12s %-12s %-10s\n" \
        "${seed}" "${case_name}" "${seed_ok}" "${input_ok}" "${final_ok}" \
        "${latest_step}" "${nplots}" "${status}"
}

echo "=============================================================================="
echo "Ishizawa random-seed ensemble integrity check"
echo "root                  : ${ROOT}"
echo "seeds                 : ${SEEDS}"
echo "target steps          : ${TARGET_STEPS}"
echo "PPC/species           : ${PPC}"
echo "CoulombLog            : ${COULOMB_LOG}"
echo "collision supercycle  : ${COLLISION_SUPERCYCLE}"
echo "=============================================================================="
printf "%-8s %-8s %-9s %-9s %-9s %-12s %-12s %-10s\n" \
    "seed" "case" "seed_ok" "input_ok" "final_ok" "latest_step" "n_plotfiles" "status"

overall=0

for seed in "${SEED_ARRAY[@]}"; do
    tag="$(printf '%06d' "${seed}")"
    no_dir="${ROOT}/seed${tag}/nocoll_ppc${PPC}"
    co_dir="${ROOT}/seed${tag}/coll_ei_ppc${PPC}_clog${CLOG_TAG}_sc${COLLISION_SUPERCYCLE}"
    check_case "${seed}" "nocoll" "${no_dir}" "${no_dir}/inputs_scale_pilot_seeded" 0
    check_case "${seed}" "coll"   "${co_dir}" "${co_dir}/inputs_scale_pilot_coll_ei_seeded" 1
done

echo
echo "Pairwise seed consistency:"
pair_fail=0
for seed in "${SEED_ARRAY[@]}"; do
    tag="$(printf '%06d' "${seed}")"
    no_in="${ROOT}/seed${tag}/nocoll_ppc${PPC}/inputs_scale_pilot_seeded"
    co_in="${ROOT}/seed${tag}/coll_ei_ppc${PPC}_clog${CLOG_TAG}_sc${COLLISION_SUPERCYCLE}/inputs_scale_pilot_coll_ei_seeded"

    no_seed="$(grep -E '^[[:space:]]*warpx\.random_seed[[:space:]]*=' "${no_in}" 2>/dev/null | tail -1 | awk -F= '{gsub(/[[:space:]]/,"",$2); print $2}' || true)"
    co_seed="$(grep -E '^[[:space:]]*warpx\.random_seed[[:space:]]*=' "${co_in}" 2>/dev/null | tail -1 | awk -F= '{gsub(/[[:space:]]/,"",$2); print $2}' || true)"

    if [[ "${no_seed}" == "${seed}" && "${co_seed}" == "${seed}" ]]; then
        echo "  seed ${seed}: PASS  (nocoll=${no_seed}, coll=${co_seed})"
    else
        echo "  seed ${seed}: FAIL  (nocoll=${no_seed:-missing}, coll=${co_seed:-missing})"
        pair_fail=1
    fi
done

echo
echo "Disk usage by seed:"
for seed in "${SEED_ARRAY[@]}"; do
    tag="$(printf '%06d' "${seed}")"
    [[ -d "${ROOT}/seed${tag}" ]] && du -sh "${ROOT}/seed${tag}"
done

if (( overall != 0 || pair_fail != 0 )); then
    echo
    echo "ENSEMBLE INTEGRITY CHECK: FAIL"
    exit 1
fi

echo
echo "ENSEMBLE INTEGRITY CHECK: PASS"
