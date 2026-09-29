# Ishizawa-scale late-time breathing: presentation figure commands

This guide maps the current analysis pipeline to the slide figures for the
late-time island-breathing presentation.  All commands assume:

```bash
cd ~/warpx/Examples/Physics_applications/tearing_harris_2d
source ~/venvs/warpx-vis/bin/activate
```

The validated breathing period used throughout is

```text
T = 0.58628938 / omega_ci
```

and the q-fixed normalization window is

```text
4.05 <= omega_ci*t <= 5.95
```.

---

## 1. Long-time saturation / discovery of breathing

Use the dense 4--6 ion-time run:

```bash
mkdir -p analysis_ishizawa_dense_saturation_240000
cd analysis_ishizawa_dense_saturation_240000

python ../analyze_ishizawa_saturation.py \
  --run-dir ../runs_ishizawa_particle_dense/ppc49_tau4p0_6p0 \
  --tmin 4.00 \
  --large-scale-max-mode 3 \
  --current-core-de 4.0 \
  --current-percentile 99.5 \
  --plateau-tmin 4.05 \
  --slope-window 0.50
```

Outputs used for slides:

```text
ishizawa_saturation_three_curves.png
ishizawa_saturation_psi.png
ishizawa_saturation_width.png
ishizawa_saturation_history.txt
```

For the presentation, prefer `three_curves.png` for the discovery slide,
and `width.png` when one clean observable is needed.

---

## 2. Breathing period / coherence

```bash
cd ~/warpx/Examples/Physics_applications/tearing_harris_2d
mkdir -p analysis_ishizawa_breathing_long_qfixed
cd analysis_ishizawa_breathing_long_qfixed

python ../analyze_ishizawa_breathing_long.py \
  --history ../analysis_ishizawa_dense_saturation_240000/ishizawa_saturation_history.txt \
  --tmin 4.05 \
  --tmax 5.95 \
  --period-min 0.54 \
  --period-max 0.64 \
  --min-peak-distance 0.45 \
  --peak-prominence-frac 0.20 \
  --Lx-de 64 \
  --min-samples 24
```

Slide figures:

```text
ishizawa_breathing_long_timeseries.png
ishizawa_breathing_long_spectrum.png
ishizawa_breathing_long_phase.png
```

Use `timeseries.png` as the main figure.  Use `spectrum.png` as a small
inset or secondary panel.

---

## 3. q-fixed direct particle weak-form closure

This rereads all dense particle snapshots and is expensive.  If the q-fixed
outputs already exist, do not rerun it only for the presentation.

```bash
cd ~/warpx/Examples/Physics_applications/tearing_harris_2d
mkdir -p analysis_ishizawa_particle_quadrature_dense_240000_qfixed
cd analysis_ishizawa_particle_quadrature_dense_240000_qfixed

python ../analyze_ishizawa_modal_particle_quadrature.py \
  --phase-root ../runs_ishizawa_particle_dense/ppc49_tau4p0_6p0 \
  --em-history ../analysis_ishizawa_dense_saturation_240000/ishizawa_saturation_history.txt \
  --period 0.58628938 \
  --tmin 4.05 \
  --tmax 5.95 \
  --mode-coarsen 4 \
  --field-coarsen 1 \
  --core-z-de 12 \
  --mode-density-cut 0.02 \
  --smooth-passes 2 \
  --x-edge-taper-de 0 \
  --periodic-x-gradient \
  --boundary-strip-de 2
```

Slide figures:

```text
ishizawa_particle_quadrature_closure_time.png
ishizawa_particle_quadrature_closure_complex.png
```

Use the time figure as the main result.  The complex-plane figure works well
as a small inset showing that volume RHS and LHS nearly overlap.

---

## 4. Closure robustness and integrated check

These read only the small closure-history text file and are cheap.

```bash
cd ~/warpx/Examples/Physics_applications/tearing_harris_2d/analysis_ishizawa_particle_quadrature_dense_240000_qfixed

PERIOD=0.58628938
WINDOW=1.17257876

python ../analyze_ishizawa_particle_quadrature_robustness.py \
  ishizawa_particle_quadrature_closure_history.txt \
  --period "$PERIOD" \
  --period-min 0.54 \
  --period-max 0.64 \
  --period-n 201 \
  --window-width "$WINDOW" \
  --window-step 0.10 \
  --min-window-points 20 \
  --out-prefix ishizawa_qfixed_robustness

python ../analyze_ishizawa_particle_quadrature_integral.py \
  ishizawa_particle_quadrature_closure_history.txt \
  --out-prefix ishizawa_qfixed_integral
```

Slide figures:

```text
ishizawa_qfixed_robustness.png
ishizawa_qfixed_robustness_sliding.png
ishizawa_qfixed_integral.png
```

For a compact validation slide, use `ishizawa_qfixed_integral.png` plus
a small numerical table with the 0.33% fundamental closure and 4.9%
integrated relRMS.

---

## 5. Native WarpX gather versus cell-centered bilinear gather

This test uses the existing 2% native-gather probe.  Do not rerun WarpX if
`ppc49_native_cycle` already exists.

```bash
cd ~/warpx/Examples/Physics_applications/tearing_harris_2d
mkdir -p analysis_ishizawa_native_vs_bilinear_multitime
cd analysis_ishizawa_native_vs_bilinear_multitime

python ../analyze_ishizawa_native_vs_bilinear_qem_multitime.py \
  --probe-glob '../runs_ishizawa_native_gather_probe/ppc49_native_cycle/diags/native*' \
  --phase-root ../runs_ishizawa_particle_dense/ppc49_tau4p0_6p0 \
  --em-history ../analysis_ishizawa_dense_saturation_240000/ishizawa_saturation_history.txt \
  --period 0.58628938 \
  --tmin 4.05 \
  --tmax 5.95 \
  --mode-coarsen 4 \
  --core-z-de 12 \
  --mode-density-cut 0.02 \
  --smooth-passes 2 \
  --x-edge-taper-de 0 \
  --sample-fraction 0.02 \
  --out-prefix ishizawa_native_vs_bilinear_qem_multitime_qfixed
```

Slide figure:

```text
ishizawa_native_vs_bilinear_qem_multitime_qfixed.png
```

The total curve is the key visual result: it remains at approximately zero
difference over a breathing cycle.

---

## 6. Exact ion kinetic tensor decomposition and bulk/random split

This rereads ion particles and is moderately expensive.  Reuse existing
q-matched outputs when possible.

```bash
cd ~/warpx/Examples/Physics_applications/tearing_harris_2d
mkdir -p analysis_ishizawa_ion_kinetic_decomposition_qfixed
cd analysis_ishizawa_ion_kinetic_decomposition_qfixed

python ../analyze_ishizawa_ion_kinetic_decomposition.py \
  --phase-root ../runs_ishizawa_particle_dense/ppc49_tau4p0_6p0 \
  --em-history ../analysis_ishizawa_dense_saturation_240000/ishizawa_saturation_history.txt \
  --period 0.58628938 \
  --tmin 4.05 \
  --tmax 5.90 \
  --q-tmin 4.05 \
  --q-tmax 5.95 \
  --mode-coarsen 4 \
  --core-z-de 12 \
  --mode-density-cut 0.02 \
  --smooth-passes 2 \
  --x-edge-taper-de 0 \
  --periodic-x-gradient \
  --out-prefix ishizawa_ion_kinetic_decomposition_qmatched
```

Tensor slide:

```text
ishizawa_ion_kinetic_decomposition_qmatched_tensor_time.png
ishizawa_ion_kinetic_decomposition_qmatched_tensor_complex.png
```

Bulk/random slide:

```text
ishizawa_ion_kinetic_decomposition_qmatched_bulk_random_time.png
ishizawa_ion_kinetic_decomposition_qmatched_bulk_random_complex.png
```

Spatial phasor data used by the next diagnostics:

```text
ishizawa_ion_kinetic_decomposition_qmatched_spatial_phasors.npz
```

---

## 7. O/X topology localization

This is lightweight: it reuses the spatial phasor NPZ and reads only the two
q~0 crossing magnetic fields.

```bash
cd ~/warpx/Examples/Physics_applications/tearing_harris_2d/analysis_ishizawa_ion_kinetic_decomposition_qfixed

python ../analyze_ishizawa_ion_restoring_localization.py \
  --npz ishizawa_ion_kinetic_decomposition_qmatched_spatial_phasors.npz \
  --summary ishizawa_ion_kinetic_decomposition_qmatched_summary.txt \
  --phase-root ../runs_ishizawa_particle_dense/ppc49_tau4p0_6p0 \
  --mode-coarsen 4 \
  --large-scale-max-mode 3 \
  --out-prefix ishizawa_ion_restoring_localization
```

Slide figures:

```text
ishizawa_ion_restoring_localization_rand_xx_restoring_topology.png
ishizawa_ion_restoring_localization_rand_total_restoring_topology.png
ishizawa_ion_restoring_localization_rand_xx_damping_topology.png
```

Prefer the rand-xx restoring topology for the first spatial-localization slide.

---

## 8. O-centered magnetic-flux shell localization

This is also lightweight.

```bash
cd ~/warpx/Examples/Physics_applications/tearing_harris_2d/analysis_ishizawa_ion_kinetic_decomposition_qfixed

python ../analyze_ishizawa_ion_restoring_flux_shells.py \
  --npz ishizawa_ion_kinetic_decomposition_qmatched_spatial_phasors.npz \
  --summary ishizawa_ion_kinetic_decomposition_qmatched_summary.txt \
  --phase-root ../runs_ishizawa_particle_dense/ppc49_tau4p0_6p0 \
  --mode-coarsen 4 \
  --large-scale-max-mode 3 \
  --out-prefix ishizawa_ion_restoring_flux_shells
```

Best presentation figures:

```text
ishizawa_ion_restoring_flux_shells_rand_xx_restoring_Ocentered.png
ishizawa_ion_restoring_flux_shells_rand_total_restoring_Ocentered.png
ishizawa_ion_restoring_flux_shells_rand_xx_damping_Ocentered.png
```

The rand-xx O-centered figure is the strongest visual proof that the restoring
budget is deep inside the closed island rather than near the separatrix.

---

## 9. Density/occupancy versus specific-stress mechanism

This rereads ion particles.  Reuse the existing output if already generated.

```bash
cd ~/warpx/Examples/Physics_applications/tearing_harris_2d
mkdir -p analysis_ishizawa_ion_randxx_mechanism
cd analysis_ishizawa_ion_randxx_mechanism

python ../analyze_ishizawa_ion_randxx_mechanism.py \
  --phase-root ../runs_ishizawa_particle_dense/ppc49_tau4p0_6p0 \
  --em-history ../analysis_ishizawa_dense_saturation_240000/ishizawa_saturation_history.txt \
  --period 0.58628938 \
  --tmin 4.05 \
  --tmax 5.90 \
  --q-tmin 4.05 \
  --q-tmax 5.95 \
  --mode-coarsen 4 \
  --core-z-de 12 \
  --mode-density-cut 0.02 \
  --smooth-passes 2 \
  --x-edge-taper-de 0 \
  --periodic-x-gradient \
  --large-scale-max-mode 3 \
  --out-prefix ishizawa_ion_randxx_mechanism
```

Best presentation figures:

```text
ishizawa_ion_randxx_mechanism_time.png
ishizawa_ion_randxx_mechanism_complex.png
ishizawa_ion_randxx_mechanism_occupancy_restoring_map.png
ishizawa_ion_randxx_mechanism_specific_stress_restoring_map.png
```

The time plot is usually the cleanest main slide figure.  The two spatial maps
work as side-by-side supporting panels.

---

## 10. Conceptual schematics for the presentation

These are not data plots.  They explain the research logic and the current
physical interpretation.

```bash
cd ~/warpx/Examples/Physics_applications/tearing_harris_2d
mkdir -p presentation_schematics

python plot_ishizawa_presentation_schematics.py \
  --out-dir presentation_schematics \
  --prefix ishizawa_ppt
```

Outputs are written as both PNG and SVG:

```text
ishizawa_ppt_research_story_flow.png/.svg
ishizawa_ppt_breathing_cycle.png/.svg
ishizawa_ppt_weak_form_budget.png/.svg
ishizawa_ppt_restoring_mechanism_chain.png/.svg
```

Use SVG in PowerPoint when possible for vector-quality labels and lines.

---

## Suggested slide-to-file mapping

| Slide purpose | Recommended figure |
|---|---|
| Initial steady-state question | `ishizawa_saturation_three_curves.png` |
| Discovery of breathing | `ishizawa_breathing_long_timeseries.png` |
| Coherent period | `ishizawa_breathing_long_spectrum.png` |
| What provides the restoring force? | `ishizawa_ppt_breathing_cycle.svg` |
| Weak-form method | `ishizawa_ppt_weak_form_budget.svg` |
| q-definition correction | before/after closure numbers + closure figures |
| Validated closure | `ishizawa_particle_quadrature_closure_time.png` |
| Numerical checks | native-vs-bilinear + integral check |
| Ion tensor channel | tensor time / tensor complex |
| Bulk vs random | bulk-random time / complex |
| Where is restoring stress? | O-centered rand-xx flux-shell map |
| Core localization | same flux-shell map + shell percentages |
| Density vs specific stress | randxx mechanism time + two spatial maps |
| Physical cycle | `ishizawa_ppt_restoring_mechanism_chain.svg` |
| Final summary | story-flow or mechanism-chain schematic |

---

## Presentation export recommendations

For all analysis figures already generated as PNG, use the existing 210 dpi
version directly unless it becomes visibly soft after cropping.

For conceptual schematics, use the SVG files in PowerPoint.

For any figure that needs to be regenerated at higher resolution, change its
`fig.savefig(..., dpi=210)` to `dpi=300` or `dpi=360`.  Do not rescale a
small raster screenshot from the terminal.

For slide use, crop empty margins but do not crop the color bar, O/X markers,
or axis labels.
