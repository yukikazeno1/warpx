# Quick start

```bash
cd ~/warpx
git fetch origin tearing-ppc-convergence
git switch tearing-ppc-convergence

cd Examples/Physics_applications/tearing_harris_2d
export AMREX_DEFAULT_INIT="amrex.the_arena_init_size=0"
bash run_ppc_convergence.sh
```

The runner executes no-seed 4, 16, and 64 PPC/species cases serially for 2000 steps and writes diagnostics every 10 steps.

To run only the two new higher-PPC cases after an existing 4-PPC pilot:

```bash
CASES="16 64" bash run_ppc_convergence.sh
```

After the runs finish:

```bash
source ~/venvs/warpx-vis/bin/activate
python analyze_ppc_convergence.py
```

Main outputs:

- `ppc_convergence_noise_scaling.png`
- `ppc_convergence_m1_history.png`
- `ppc_convergence_m1_fraction.png`
- `ppc_convergence_spectrum_early.png`
- `ppc_convergence_spectrum_final.png`
- `ppc_convergence_summary.txt`
- `ppc_convergence_summary.csv`

The main convergence discriminator is whether the early magnetic-noise level follows approximately `PPC^(-1/2)`. The Fourier spectra then test whether the later signal selectively transfers into `m=1` instead of remaining broadband.
