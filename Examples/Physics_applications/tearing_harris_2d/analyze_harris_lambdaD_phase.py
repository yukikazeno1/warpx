#!/usr/bin/env python3
"""Phase/coherence diagnostics for Debye-length-normalized Harris-sheet runs.

This complements analyze_harris_lambdaD_early.py once a candidate low-m tearing
mode begins to emerge.  It tracks:

* coherent complex Bz Fourier coefficients for selected x modes;
* unwrapped temporal phase and z-direction phase coherence;
* effective Harris scale length inferred from the x-averaged Jy FWHM;
* time-dependent k L_eff and heuristic Harris Delta' L_eff.

The effective-width diagnostics are deliberately labeled heuristic: once the
sheet departs strongly from a sech^2 profile, a single L_eff is no longer a
complete description of the current layer.
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_harris_lambdaD_early import (
    params,
    load_fields,
    corrected_jy,
    fwhm,
)

FWHM_FACTOR = 1.762747174039086


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--box-lambdaD", type=float, default=500.0)
    p.add_argument("--sheet-lambdaD", type=float, default=40.0)
    p.add_argument("--modes", default="1,2,3,4")
    p.add_argument("--core-sheet-halfwidth", type=float, default=1.0)
    p.add_argument("--phase-tmin", type=float, default=0.30)
    return p.parse_args()


def numeric_key(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def circular_resultant(phi):
    if len(phi) == 0:
        return np.nan
    return float(np.abs(np.mean(np.exp(1j * phi))))


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params(args.sheet_lambdaD)
    modes = [int(x) for x in args.modes.split(",") if x.strip()]

    files = sorted(
        glob.glob(str(Path(args.run_dir) / "diags" / "diag1*")),
        key=numeric_key,
    )
    if not files:
        raise FileNotFoundError(f"No diag1* under {args.run_dir}/diags")

    tci = []
    coeff = {m: [] for m in modes}
    spatial_coh = {m: [] for m in modes}
    fwhm_ld = []
    leff_ld = []

    core_half = args.core_sheet_halfwidth * P["L"]

    for i, fn in enumerate(files, 1):
        F = load_fields(fn)
        t = P["wci"] * F["ds"].current_time.to_value("s")
        tci.append(t)

        z = F["z"]
        core = np.abs(z) <= core_half
        ft = np.fft.rfft(F["Bz"], axis=0) / F["Bz"].shape[0]
        for m in modes:
            if m >= ft.shape[0]:
                coeff[m].append(np.nan + 1j*np.nan)
                spatial_coh[m].append(np.nan)
                continue
            fm = ft[m, core]
            # Coherent one-sided complex amplitude, normalized to B0.
            cm = 2.0 * np.mean(fm) / P["B0"]
            coeff[m].append(cm)
            denom = np.mean(np.abs(fm))
            spatial_coh[m].append(
                float(np.abs(np.mean(fm)) / denom) if denom > 0 else np.nan
            )

        jyc = corrected_jy(F["Bx"], F["Bz"], F["dx"], F["dz"])
        jy_mean = np.mean(jyc, axis=0)
        width = fwhm(z, jy_mean) / P["lambdaD"]
        fwhm_ld.append(width)
        leff_ld.append(width / FWHM_FACTOR if np.isfinite(width) else np.nan)

        print(
            f"{i:3d}/{len(files)} omega_ci*t={t:.5f} "
            f"FWHM/lambdaD={width:.3f} L_eff/lambdaD={leff_ld[-1]:.3f}"
        )

    tci = np.asarray(tci)
    fwhm_ld = np.asarray(fwhm_ld)
    leff_ld = np.asarray(leff_ld)
    for m in modes:
        coeff[m] = np.asarray(coeff[m], dtype=complex)
        spatial_coh[m] = np.asarray(spatial_coh[m], dtype=float)

    # Phase/coherence figure.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for m in modes:
        c = coeff[m]
        ax1.semilogy(tci, np.abs(c), "o-", ms=3, label=f"m={m}")
        ax2.plot(tci, np.unwrap(np.angle(c)), "o-", ms=3, label=f"m={m}")
        ax3.plot(tci, spatial_coh[m], "o-", ms=3, label=f"m={m}")
    ax1.set_ylabel(r"coherent $|\hat B_{z,m}|/B_0$")
    ax2.set_ylabel("unwrapped phase [rad]")
    ax3.set_ylabel("z-phase coherence")
    ax3.set_xlabel(r"$\omega_{ci}t$")
    ax3.set_ylim(0.0, 1.05)
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.25)
    ax1.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig("harris_500ld_40ld_phase_coherence.png", dpi=200)
    plt.close(fig)

    # Effective-sheet evolution and dynamic kL_eff.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(tci, leff_ld, "o-", label=r"$L_{eff}$ from $J_y$ FWHM")
    ax1.axhline(args.sheet_lambdaD, ls="--", lw=1.0, label="initial L")
    ax1.set_ylabel(r"$L_{eff}/\lambda_D$")
    ax1.grid(alpha=0.25)
    ax1.legend()

    for m in modes:
        kL_eff = 2.0 * np.pi * m * leff_ld / args.box_lambdaD
        ax2.plot(tci, kL_eff, "o-", ms=3, label=f"m={m}")
    ax2.axhline(1.0, ls="--", lw=1.0, label=r"$kL=1$")
    ax2.set_ylabel(r"$kL_{eff}$")
    ax2.set_xlabel(r"$\omega_{ci}t$")
    ax2.grid(alpha=0.25)
    ax2.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig("harris_500ld_40ld_effective_sheet.png", dpi=200)
    plt.close(fig)

    mask_phase = tci >= args.phase_tmin
    with open("harris_500ld_40ld_phase_summary.txt", "w") as f:
        f.write("Harris 500 lambda_D / 40 lambda_D phase diagnostics\n")
        f.write("=================================================\n\n")
        f.write(f"final omega_ci*t = {tci[-1]:.8f}\n")
        f.write(f"initial L/lambda_D = {args.sheet_lambdaD:.8f}\n")
        f.write(f"final Jy FWHM/lambda_D = {fwhm_ld[-1]:.8f}\n")
        f.write(f"final L_eff/lambda_D = {leff_ld[-1]:.8f}\n")
        f.write(f"phase statistics use omega_ci*t >= {args.phase_tmin:g}\n\n")
        f.write("mode final_coherent_amp mean_z_coherence phase_resultant phase_span_rad final_kLeff final_DeltaPrimeLeff\n")
        for m in modes:
            c = coeff[m]
            good = mask_phase & np.isfinite(c.real) & np.isfinite(c.imag) & (np.abs(c) > 0)
            phi = np.angle(c[good])
            up = np.unwrap(phi)
            span = float(np.max(up) - np.min(up)) if len(up) else np.nan
            kLe = 2.0 * np.pi * m * leff_ld[-1] / args.box_lambdaD
            dpLe = 2.0 * (1.0 / kLe - kLe) if np.isfinite(kLe) and kLe > 0 else np.nan
            f.write(
                f"{m:3d} {abs(c[-1]):.8e} "
                f"{np.nanmean(spatial_coh[m][good]):.6f} "
                f"{circular_resultant(phi):.6f} {span:.6f} "
                f"{kLe:.8f} {dpLe:.8f}\n"
            )

    # Machine-readable width/kL history for follow-on fitting.
    cols = [tci, fwhm_ld, leff_ld]
    hdr = ["omega_ci_t", "Jy_FWHM_lambdaD", "L_eff_lambdaD"]
    for m in modes:
        kLe = 2.0 * np.pi * m * leff_ld / args.box_lambdaD
        dpLe = 2.0 * (1.0 / kLe - kLe)
        cols.extend([kLe, dpLe])
        hdr.extend([f"kL_eff_m{m}", f"DeltaPrimeL_eff_m{m}"])
    np.savetxt(
        "harris_500ld_40ld_effective_sheet_history.txt",
        np.column_stack(cols),
        header=" ".join(hdr),
    )

    print("Saved harris_500ld_40ld_phase_coherence.png")
    print("Saved harris_500ld_40ld_effective_sheet.png")
    print("Saved harris_500ld_40ld_phase_summary.txt")
    print("Saved harris_500ld_40ld_effective_sheet_history.txt")


if __name__ == "__main__":
    main()
