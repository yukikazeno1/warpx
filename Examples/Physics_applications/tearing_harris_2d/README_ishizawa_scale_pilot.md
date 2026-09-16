# Ishizawa-scale Harris pilot

This case is the next step after the small `16 d_e x 8 d_e` tearing pilot.
The earlier 64-PPC run showed coherent low-m tearing-like fluctuations but no
clean exponential growth or developed X/O topology by about `4.4 tau_A`.
Particle-moment reconstruction from four closely spaced particle snapshots also
had poor generalized-Ohm closure (electron relRMS ~3.4, ion ~8.8), so those
curves must not be used to claim pressure/inertia dominance.

The dominant geometric limitation is clear: for `mi/me=800`,

```
d_i/d_e = sqrt(800) = 28.284
```

while the old half-height was only `4 d_e`.  It did not contain the complete
`l_mi < |z| < d_i` two-fluid region discussed by Ishizawa & Horiuchi.

## What is matched in this pilot

The 2005 PRL explicitly uses:

- `mi/me = 800`
- a square open 2-D region
- `512 x 512` grid points
- `25.6 million` particles
- `omega_ce0 * dt = 0.02`
- Harris initial equilibrium
- no guide field

The 2004 predecessor setup additionally documents:

- `omega_pe0/omega_ce0 = 3.5`
- `Ti0 = Te0`
- shifted-Maxwellian Harris particles
- upstream driving with `E0/B0 = -0.04`
- an open driven geometry

The present input uses the first four scale/particle targets plus
`omega_pe/omega_ce=3.5` and `Ti=Te`.  It chooses a square side of `64 d_e`.
With the resulting pressure-balanced thermodynamics,

```
lambda_D = d_e/7
Delta x = Delta z = (64 d_e)/512 = d_e/8 = 0.875 lambda_D
half-box = 32 d_e = 1.131 d_i
```

Thus the full ion-skin-depth scale is inside the box instead of far outside it.
The `64 d_e` side is an inference/design choice, not a box size explicitly
quoted by the 2005 PRL.  It is chosen because it gives a 512-cell square with
Debye-scale resolution and a half-box slightly larger than `d_i`.

The default particle loading is `7 x 7 = 49 PPC/species`, so

```
512^2 * 2 * 49 = 25,690,112 macroparticles
```

which is essentially the quoted 25.6-million-particle count if the two species
are represented in roughly equal numbers.  The paper does not explicitly state
the species split, so this is a matching inference rather than a quoted fact.

## What is NOT matched yet

This is deliberately a **scale-matched closed/periodic pilot**, not a complete
Ishizawa reproduction.  It retains the earlier WarpX boundary model:

- x: periodic fields/particles
- z: PEC fields + reflecting particles

Ishizawa instead uses an open driven system.  Upstream incoming particles are a
shifted Maxwellian with mean E x B drift from an imposed out-of-plane electric
field; downstream plasma can freely flow in/out.  The drive becomes a uniform
`E0/B0 = -0.04` in the steady state.  Implementing that reservoir/open-boundary
physics is a separate stage and must not be replaced by PEC/reflecting
boundaries while claiming an exact reproduction.

The exact Harris scale height `y_h` is not given in the short 2005 PRL.  This
pilot retains `sheet_l = d_e` so that the sheet stays well resolved.  This is a
provisional modeling choice, not a recovered paper parameter.

## Derived SI values for n0 = 1e19 m^-3

Using `omega_pe/omega_ce = 3.5`, `Ti=Te`, and Harris pressure balance gives
approximately:

```
omega_pe = 1.783986e11 rad/s
d_e      = 1.680464e-3 m
B0       = 2.898025e-1 T
Te = Ti  = 1.042855e4 eV
u_th,e   = sqrt(Te/(m_e c^2)) = 1/7 = 0.142857
u_th,i   = 0.00505076
lambda_D = 2.400663e-4 m = d_e/7
```

WarpX is forced to the paper timestep:

```
warpx.const_dt = 0.02 / omega_ce0
```

For this mesh, `c dt / dx = 0.56`, below the 2-D Yee light-wave CFL limit.
WarpX supports `warpx.const_dt`; the runner also adds standard checkpoint
FullDiagnostics for restart.

## Staged run procedure

Do **not** start immediately with the 320k-step target on a single 12-GB TITAN V.
First test whether 25.7 million macroparticles fit.

```bash
cd ~/warpx/Examples/Physics_applications/tearing_harris_2d
export AMREX_DEFAULT_INIT="amrex.the_arena_init_size=0"

MODE=memory PPC=49 bash run_ishizawa_scale_pilot.sh
```

If that succeeds, continue progressively using the same run directory and its
latest checkpoint:

```bash
MODE=smoke PPC=49 bash run_ishizawa_scale_pilot.sh
MODE=early PPC=49 bash run_ishizawa_scale_pilot.sh
```

Stages are:

| MODE | steps | omega_ci t | default mesh-output cadence |
|---|---:|---:|---:|
| memory | 1 | 2.5e-5 | final only in practice |
| smoke | 100 | 0.0025 | 100 |
| early | 10,000 | 0.25 | 500 |
| onset | 100,000 | 2.5 | 1,000 |
| steady | 320,000 | 8.0 | 2,000 |

The last two are large production jobs.  On the current NOMPI/single-GPU build,
measure the wall time of `early` before attempting them.  A multi-GPU/HPC build
is more appropriate for the full steady-state target.

If 49 PPC does not fit in 12 GB, keep the **512x512 grid and 64 d_e box** and
reduce particle count first:

```bash
FORCE=1 MODE=memory PPC=25 bash run_ishizawa_scale_pilot.sh
```

This gives 13,107,200 total macroparticles.  For the domain-size hypothesis it
is more informative to preserve the ion-scale box and spatial resolution than
to shrink the domain again.  The final production result should return to the
49-PPC target on suitable hardware.

## Scientific decision point

The first comparison should be between the old small-box run and this enlarged
box, not immediately against the paper's steady driven state.  Check whether the
large box now develops:

1. an actual X/O magnetic topology,
2. a growing reconnection flux above the noise-selection floor,
3. an ion-scale/two-fluid structure extending toward `d_i`,
4. a broad ion current shoulder plus a narrower electron peak.

Only after this size-isolation test is understood should the project replace the
closed/periodic boundaries with Ishizawa-style driven open boundaries.
