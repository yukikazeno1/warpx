# Harris-sheet tearing PPC convergence study

This directory contains a small convergence workflow for the 2D fully kinetic Harris-sheet tearing pilot discussed in the local WarpX setup.

The convergence study is designed to distinguish three effects:

1. finite-particle PIC noise, expected to scale approximately as `Nppc^(-1/2)`;
2. broadband fluctuations that seed tearing even when no explicit perturbation is applied;
3. genuine mode selection/growth of the `m=1` tearing mode.

## Cases

All three cases are **no-seed** runs (`epsb=0`) and use the same physical and numerical parameters except for particles per cell per species:

- `ppc4`: `2 x 2 = 4` PPC/species
- `ppc16`: `4 x 4 = 16` PPC/species
- `ppc64`: `8 x 8 = 64` PPC/species

The default run length is 2000 steps. Diagnostics are written every 10 steps so that the early noise-injection stage can be resolved.

## Run

From this directory:

```bash
export AMREX_DEFAULT_INIT="amrex.the_arena_init_size=0"
./run_ppc_convergence.sh
```

By default the script expects the executable at:

```text
../../../build/bin/warpx.2d
```

Set `WARPX_EXE` to override it:

```bash
WARPX_EXE=/path/to/warpx.2d ./run_ppc_convergence.sh
```

Each case is written to its own directory:

```text
runs/ppc4/
runs/ppc16/
runs/ppc64/
```

The three runs are intentionally serial. This avoids simultaneously loading multiple large PIC jobs on a 12 GB GPU.

## Analyze

After all cases finish:

```bash
source ~/venvs/warpx-vis/bin/activate
python analyze_ppc_convergence.py
```

This produces:

```text
ppc_convergence_noise_scaling.png
ppc_convergence_m1_history.png
ppc_convergence_spectrum.png
ppc_convergence_summary.txt
```

The analysis reports, for each PPC:

- total `Bz_rms/B0`;
- core `Bz_rms/B0` in `|z| <= L`;
- phase-independent `m=1` Fourier amplitude in the core;
- the early-time broadband noise level;
- a log-log fit of noise floor versus PPC;
- the core Fourier spectrum `m=1..20` at selected times.

For pure PIC shot-noise scaling one expects approximately

```text
noise ~ Nppc^(-1/2)
```

so changing 4 -> 16 -> 64 PPC should reduce the noise by roughly factors `1`, `1/2`, and `1/4`, respectively.

A stronger indication of physical noise-seeded tearing is obtained if higher-PPC runs start from a lower noise floor but later show the same exponential `m=1` slope, shifted later in time.
