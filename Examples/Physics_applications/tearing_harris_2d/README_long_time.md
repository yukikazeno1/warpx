# Long-time Harris tearing runs

This workflow extends the no-seed PPC study to longer times after the early-noise scaling has already been established.

## Recommended production cases

Use only:

- 16 PPC/species (`4 x 4`)
- 64 PPC/species (`8 x 8`)

The 4 PPC case is retained as a pilot/noise reference and is not recommended for quantitative tearing growth rates.

## Output cadence

The short convergence study used `diag1.intervals = 10` to resolve the early noise-injection stage.

The long-time runs use by default:

```text
diag1.intervals = 100
```

This reduces plotfile output by a factor of 10 while retaining about 45 samples per Alfvén time for the current setup.

Restart checkpoints are written every 2000 steps by default:

```text
chk.intervals = 2000
chk.diag_type = Full
chk.format = checkpoint
```

WarpX restart is performed with `amr.restart=<checkpoint path>`.

## First production stage: 10000 total steps

```bash
cd ~/warpx/Examples/Physics_applications/tearing_harris_2d
export AMREX_DEFAULT_INIT="amrex.the_arena_init_size=0"
bash run_tearing_long.sh
```

Defaults:

```text
CASES="16 64"
TARGET_STEPS=10000
DIAG_INTERVAL=100
CHECKPOINT_INTERVAL=2000
```

Run only 64 PPC:

```bash
CASES="64" bash run_tearing_long.sh
```

## Extend to 20000 total steps

After the 10000-step run completes, simply rerun:

```bash
TARGET_STEPS=20000 bash run_tearing_long.sh
```

The script searches `runs_long/ppc*/diags/chk*` and automatically restarts from the latest checkpoint rather than starting from zero.

To change the output cadence, for example one field plotfile every 200 steps:

```bash
DIAG_INTERVAL=200 TARGET_STEPS=20000 bash run_tearing_long.sh
```

For growth-rate work, 100 is the recommended default; 200 is acceptable if disk usage is the primary concern.

## Analyze the long-time evolution

```bash
source ~/venvs/warpx-vis/bin/activate
python analyze_tearing_long.py
```

The analysis tracks x-Fourier modes `m=1..20` in the current-sheet core `|z|<=L` and produces:

```text
tearing_long_m1_history.png
tearing_long_spectrum_map_ppc16.png
tearing_long_spectrum_map_ppc64.png
tearing_long_gamma_vs_m.png
tearing_long_dominant_mode.png
tearing_long_growth_summary.txt
tearing_long_mode_history_ppc16.txt
tearing_long_mode_history_ppc64.txt
```

The `m-t` maps are intended to show whether one Fourier mode emerges from the broadband PIC noise. The growth-rate screening does not assume that `m=1` is the fastest mode.

Default acceptance criteria for an exponential growth window are conservative:

```text
t/tau_A >= 0.5
at least 12 points
window span >= 0.35 tau_A
R^2 >= 0.95
fitted growth factor >= 2
```

These fits are diagnostics, not a substitute for physical convergence. A convincing tearing result should show comparable late-time growth slopes in the 16 and 64 PPC cases, even if their noise-seeded onset times differ.
