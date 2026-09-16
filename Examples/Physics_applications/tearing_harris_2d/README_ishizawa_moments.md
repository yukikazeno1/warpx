# Ishizawa-style particle-moment diagnostics

This directory contains `analyze_ishizawa_particle_moments.py`, which analyzes
particle-rich WarpX plotfiles written by `run_ishizawa_particle_snapshot.sh`.

## Purpose

The field-only diagnostics can identify current-sheet structure, candidate X/O
points, reconnecting flux, and Hall-like fields.  The particle-rich diagnostics
add species bulk velocities and pressure-tensor moments in order to evaluate the
out-of-plane species momentum balance used in Ishizawa & Horiuchi (2005).

For species `s`, the code evaluates

```text
E_y + (u_s x B)_y
  = (div P_s)_y/(q_s n_s)
    + (m_s/q_s) [d u_sy/dt + u_s . grad u_sy]
```

in the WarpX X-Z plane, where WarpX `y` is the out-of-plane direction and maps
to the paper's out-of-plane `z`.

Unlike the steady-state Ishizawa calculation, this spontaneous tearing pilot is
unsteady.  Therefore the `d u_sy/dt` term is retained explicitly.

## Recommended command

From the tearing example directory:

```bash
source ~/venvs/warpx-vis/bin/activate
python analyze_ishizawa_particle_moments.py \
  --runs-dir runs_ishizawa \
  --ppc 64 \
  --coarsen 4 \
  --density-cut 0.02
```

The default `--coarsen 4` turns the 256x128 fine grid into 64x32 moment cells.
For a 64-PPC/species fine-grid run this gives roughly 1024 macro-particles per
species per coarse cell before density weighting, which significantly reduces
noise in off-diagonal pressure moments.

For a resolution/noise sensitivity check, rerun with

```bash
python analyze_ishizawa_particle_moments.py \
  --runs-dir runs_ishizawa \
  --ppc 64 \
  --coarsen 2 \
  --density-cut 0.02
```

The pressure-gradient terms should not change qualitatively if the signal is
physical.

## Outputs

- `ishizawa_particle_moments_summary.txt`
- `ishizawa_ion_ohm_profile.png`
- `ishizawa_electron_ohm_profile.png`
- `ishizawa_frozen_in_compare.png`
- `ishizawa_pressure_terms.png`
- `ishizawa_species_flow_profile.png`
- `ishizawa_moment_closure.txt`

## Interpretation

The most important comparison is between the frozen-in violation
`E_y + (u x B)_y` and the sum of pressure-tensor, convective-inertia, and
temporal-inertia terms.

A clean Ishizawa-like electron diffusion region would show the electron
frozen-in violation supported primarily by the electron pressure-tensor term
near the X point.  Ion behavior should be interpreted more cautiously in this
pilot because the present box only spans `z/d_e=[-4,4]`, whereas
`d_i/d_e=sqrt(800)=28.284`.  Thus the full `l_mi < |z| < d_i` two-fluid region
of Ishizawa & Horiuchi (2005) is outside the present domain.

## Numerical caution

Pressure-tensor divergence is much noisier than density or bulk velocity.  Do
not interpret single-cell spikes.  Compare `--coarsen 4` and `--coarsen 2`, and
focus on spatially coherent structures around `l_me` and `l_mi`.
