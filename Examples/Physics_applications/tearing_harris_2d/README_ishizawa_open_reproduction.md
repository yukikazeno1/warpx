# Ishizawa & Horiuchi (2005): open/driven WarpX reproduction plan

This branch isolates the transition from the existing closed/noise-seeded
Harris tearing work to the **open, externally driven, quasi-steady collisionless
reconnection** studied by Ishizawa & Horiuchi (PRL 95, 045003, 2005).

## What the 2005 Letter states directly

The Letter specifies:

- a 2D full electromagnetic particle-in-cell model;
- a square open reconnection domain;
- no guide field;
- `mi/me = 800`;
- a Harris-type initial equilibrium
  `Bx(y)=B0 tanh(y/yh)` and `P(y)=B0^2/(8 pi) sech^2(y/yh)`;
- `omega_ce0 * dt = 0.02`;
- a `512 x 512` grid and `25.6e6` particles;
- upstream inflow driven by an externally applied out-of-plane electric field;
- free inflow/outflow at the downstream boundary;
- a steady-state applied field `E0 = -0.04 B0 c` in the paper's coordinates.

The four-page Letter does **not** state every box-size, sheet-width, thermal,
background-density, or numerical boundary implementation detail.  Those must
not be silently presented as exact paper values.

## Coordinate/sign mapping

WarpX here uses the X-Z simulation plane with Y out of plane.  A right-handed
mapping to the paper is

- `x_paper = x_WarpX`,
- `y_paper = z_WarpX`,
- `z_paper = -y_WarpX`.

Consequences:

- the positive quantity `-Jz` plotted by Ishizawa corresponds to `+Jy` in
  this WarpX setup;
- the paper's steady `Ez = -0.04 B0 c` corresponds to a **positive** WarpX
  out-of-plane drive `Ey = +0.04 B0 c` if the same inward E x B inflow is to
  be produced with `Bx(z)=B0 tanh(z/L)`.

This sign conversion is important and should be preserved in all later
analysis scripts.

## Stage A committed in this branch

Files:

- `inputs_2d_ishizawa_open_equilibrium_mi800`
- `run_ishizawa_open_equilibrium.sh`
- `analyze_ishizawa_open_equilibrium.py`

Stage A changes **only the boundary physics** relative to the previous
Ishizawa-scale pilot:

- field boundaries: `absorbing_silver_mueller` on x and z;
- particle boundaries: `absorbing` on x and z;
- external drive: OFF;
- reinjection: OFF.

The purpose is to answer one question before adding more physics:

> Can the native WarpX open boundaries preserve the interior Harris sheet for
> long enough that a driven/open reconnection calculation is meaningful?

The present `64 de x 64 de`, `L=de`, `omega_pe/omega_ce=3.5`, `Ti=Te`
choices are inherited controlled-design choices from the previous scale pilot.
They are **not claimed to be exact values from the 2005 Letter**.

## Stage-A run sequence

From `Examples/Physics_applications/tearing_harris_2d`:

```bash
MODE=memory ./run_ishizawa_open_equilibrium.sh
MODE=smoke  ./run_ishizawa_open_equilibrium.sh
MODE=relax  ./run_ishizawa_open_equilibrium.sh
MODE=boundary ./run_ishizawa_open_equilibrium.sh
```

The runner automatically restarts from the newest checkpoint.  Use
`FORCE=1` only when deliberately deleting the previous run directory.

Analyze with:

```bash
python3 analyze_ishizawa_open_equilibrium.py \
  --run-dir runs_ishizawa_open_equilibrium/ppc49
```

Stage A is considered usable only if the interior `Bx`, `Jy`, density and
charge-neutrality profiles remain close to Harris equilibrium while any
particle loss or wave absorption remains localized near the open boundaries.

## Stage B: next implementation target

After Stage A is validated, add the two ingredients required for the paper's
steady driven system:

1. **upstream reservoir/injection** at both z boundaries;
2. **out-of-plane drive** ramping to the mapped WarpX value
   `Ey = +0.04 B0 c`.

WarpX already has the required building blocks:

- `NFluxPerCell` injection through x/z planes;
- `gaussianflux` momentum distributions;
- absorbing particle boundaries;
- absorbing electromagnetic boundaries;
- parser-based external particle fields as a possible controlled drive.

The first Stage-B implementation should keep injection and drive separately
switchable so that particle-reservoir balance and E x B driving can be tested
independently.

## Final reproduction criteria

A successful reproduction is not just a magnetic island.  The target is the
paper's quasi-steady structure:

1. an X point and a two-scale out-of-plane current layer;
2. an ion shoulder controlled by the ion meandering scale `l_mi`;
3. electron-scale central current peak;
4. in `l_mi < |z| < d_i`, ion inertia counterbalanced by the ion pressure
   tensor (gyroviscous cancellation);
5. strong ion frozen-in breaking only for `|z| < l_mi`;
6. electron pressure tensor supporting the reconnection electric field near
   the X point.

Particle-rich steady snapshots are therefore mandatory.  Mesh fields alone
cannot reproduce the pressure-tensor decomposition in Figs. 3 and 4.


## Initial div(B) projection and Silver-Mueller

WarpX automatically enables the MLMG projection-based initial div(B) cleaner
when a non-constant parsed external magnetic field is loaded with the Yee
solver.  That projection solver currently accepts only periodic, PEC, PMC, or
Neumann field boundaries and therefore aborts when Stage A uses
`absorbing_silver_mueller`.

For this Stage-A Harris field,

```text
Bx = B0*tanh(z/L)
By = 0
Bz = 0
```

the magnetic field is divergence-free because Bx has no x dependence and Bz
has no z dependence.  Stage A therefore explicitly sets

```text
warpx.do_initial_div_cleaning = 0
```

instead of running the incompatible MLMG projection.  The full diagnostic now
writes `divB`, and `analyze_ishizawa_open_equilibrium.py` reports the
interior RMS div(B), normalized by `B0/L`.  This must remain small during the
open-boundary pilot before Stage B is enabled.


## Stage-A boundary revision after first smoke test

The first Stage-A smoke test used Silver-Mueller field boundaries on both x and
z.  Although the code ran and div(B) remained small, the interior equilibrium
was strongly perturbed by 100 steps: Bx and Jy profile errors were several to
ten percent and E_rms/(c B0) reached O(1e-2).

This is expected because the upstream z faces carry a nonzero *tangential*
static Harris field Bx ~= +/- B0, whereas a Silver-Mueller radiation boundary
is designed for outgoing electromagnetic waves rather than for holding a DC
tangential equilibrium field.

The revised Stage A therefore uses:

- x / downstream: `absorbing_silver_mueller`;
- z / upstream: `pec_insulator`, with the whole face treated as insulator;
- prescribed upstream tangential magnetic field
  `Bx(z_lo)=-B0`, `Bx(z_hi)=+B0`;
- Stage-A drive `Ey=0` on both upstream faces;
- absorbing particle boundaries on all physical sides.

This layout also prepares Stage B cleanly: the same upstream boundary can later
replace `Ey=0` by a time-ramped reconnection drive without changing the
downstream boundary implementation.

The revised pilot also uses `amr.max_grid_size=256` (four boxes on one GPU)
instead of 64, because the first smoke test showed a severe 64-box/GPU
decomposition overhead.

The controlled-upstream revision writes to a new run directory:

```text
runs_ishizawa_open_equilibrium/controlled_upstream_ppc49
```

so that checkpoints from the earlier all-Silver-Mueller pilot cannot be
accidentally restarted.
