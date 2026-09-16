# Ishizawa & Horiuchi (2005) reference notes for the WarpX Harris study

This note separates parameters explicitly stated in PRL 95, 045003 (2005)
from design choices for the present WarpX study.

## Parameters stated explicitly in the paper

- 2D fully electromagnetic full-particle PIC.
- No guide field.
- Electron-ion mass ratio `mi/me = 800`.
- Square open simulation region in the reconnection plane.
- Upstream inflow is externally driven by an out-of-plane electric field;
  downstream plasma can flow freely in or out.
- 1D Harris equilibrium:
  `Bx(y) = B0 tanh(y/yh)` and `P(y) = B0^2/(8 pi) sech^2(y/yh)`.
- Time step: `omega_ce0 * dt = 0.02`.
- Grid: `512 x 512`.
- Total particle count: `25.6e6`.
- In the steady state the imposed electric field is reported as
  `E0 = -0.04 B0 c`.

The PRL does **not** state an explicit physical box length or grid spacing in
`d_e`, `d_i`, or Debye-length units.  Those should not be invented from the
512x512 grid alone.  The exact loading procedure and electron/ion particle
split are also not given in this short Letter.

## Particle-count comparison

`25.6e6 / 512^2 = 97.65625` total macro-particles per cell on average.
If the total was split equally between electrons and ions, this corresponds to
about `48.83 PPC/species`.  The equal split is an inference, not a statement in
the Letter.

The present WarpX `64 PPC/species` case therefore has `128` total
macro-particles/cell, which is already comparable to or somewhat above the
paper's average total particle sampling per cell.  The larger mismatch with the
paper is currently spatial scale/boundary physics, not simply PPC.

## Scale hierarchy emphasized by the paper

The paper distinguishes

- electron meandering scale `l_me`,
- electron skin depth `d_e`,
- ion meandering scale `l_mi`,
- ion skin depth `d_i`.

Its central ion result concerns the two-fluid region

`l_mi < |y| < d_i`,

where ion inertia is counteracted by the ion pressure tensor (gyroviscous
cancellation), suppressing the Hall-term effect.  Inside `l_mi`, nongyrotropic
ion pressure associated with ion meandering strongly breaks ion frozen-in.

For the current WarpX parameters with `mi/me=800` and `L=d_e`,

- `d_i/d_e = sqrt(800) = 28.284`,
- initial estimate `l_me/d_e ~ 0.749`,
- initial estimate `l_mi/d_e ~ 1.631`,
- current box is only `z/d_e in [-4,4]`.

Therefore the present box contains the inner meandering scales but not the ion
skin depth.  It cannot reproduce the full Ishizawa `l_mi < |y| < d_i`
gyroviscous-cancellation region.

## Recommended staged use of the paper

1. Finish the present small-box run as an inner-kinetic tearing/reconnection
   pilot and diagnose X/O topology, reconnecting flux, current profile,
   coherence/parity, and particle pressure tensors near the center.
2. Do not claim a reproduction of Ishizawa Fig. 2/3 until the box includes
   `d_i` plus an outer region.
3. For a later Ishizawa-oriented production run, use open/driven boundaries and
   a transverse half-size comfortably larger than `d_i`.  A working design
   choice such as a full transverse size of order `4 d_i` is a **new simulation
   design choice**, not a value stated by the PRL.
4. Preserve `mi/me=800` if the goal is direct comparison.
5. Use particle sampling around the paper's inferred average (~49
   PPC/species if equally split) as a reference; `64 PPC/species` is a sensible
   high-quality target, but PPC should still be convergence-tested.

For an exact reproduction of the original physical box size, cell size,
particle loading, and boundary implementation, the earlier code/setup papers
cited as Refs. 17-19 need to be consulted; those details are not all provided
in this four-page PRL.
