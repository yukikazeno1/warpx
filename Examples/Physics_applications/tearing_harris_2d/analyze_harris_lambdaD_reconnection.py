#!/usr/bin/env python3
"""Reconnection/topology diagnostics for the 500 lambda_D / 40 lambda_D Harris run.

Adds diagnostics that complement the growth and phase scripts:
  * continuous sech^2 fit to the x-averaged curl-B Jy profile;
  * full-Ay magnetic topology on top of Jy/J0;
  * perturbation-flux topology delta Ay;
  * raw and lightly smoothed out-of-plane By/B0 maps for Hall-structure checks.

The Gaussian smoothing is visualization-only; all reported amplitudes and
sheet-width fits use unsmoothed fields.
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
    reconstruct_Ay,
)

try:
    from scipy.optimize import curve_fit
    from scipy.ndimage import gaussian_filter
except Exception as exc:
    raise RuntimeError("This diagnostic requires scipy (optimize + ndimage).") from exc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--box-lambdaD", type=float, default=500.0)
    p.add_argument("--sheet-lambdaD", type=float, default=40.0)
    p.add_argument("--fit-halfwidth-L", type=float, default=2.5)
    p.add_argument("--zoom-sheet", type=float, default=2.0)
    p.add_argument("--by-sigma", type=float, default=1.2,
                   help="Gaussian sigma in grid cells for visualization only")
    return p.parse_args()


def numeric_key(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def sech2_model(z, jp, z0, L, c0):
    q = np.clip((z - z0) / L, -30.0, 30.0)
    return jp / np.cosh(q) ** 2 + c0


def fit_sheet(z, jy, L0, J0, halfwidth_L):
    mask = np.isfinite(z) & np.isfinite(jy) & (np.abs(z) <= halfwidth_L * L0)
    zz = z[mask]
    yy = jy[mask]
    if zz.size < 20:
        return (np.nan,) * 6
    p0 = [float(np.nanmax(yy)), 0.0, L0, 0.0]
    lo = [0.1 * J0, -0.5 * L0, 0.2 * L0, -0.3 * J0]
    hi = [3.0 * J0, 0.5 * L0, 2.0 * L0, 0.3 * J0]
    try:
        popt, _ = curve_fit(
            sech2_model, zz, yy, p0=p0, bounds=(lo, hi), maxfev=20000
        )
        pred = sech2_model(zz, *popt)
        rms = np.sqrt(np.mean((yy - pred) ** 2)) / J0
        ssr = np.sum((yy - pred) ** 2)
        sst = np.sum((yy - np.mean(yy)) ** 2)
        r2 = 1.0 - ssr / sst if sst > 0 else np.nan
        jp, z0, Lfit, c0 = [float(x) for x in popt]
        return jp, z0, Lfit, c0, float(rms), float(r2)
    except Exception:
        return (np.nan,) * 6


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params(args.sheet_lambdaD)
    files = sorted(
        glob.glob(str(Path(args.run_dir) / "diags" / "diag1*")),
        key=numeric_key,
    )
    if not files:
        raise FileNotFoundError(f"No diag1* under {args.run_dir}/diags")

    tci = []
    Lfit_ld = []
    Jpeak_J0 = []
    z0_ld = []
    fit_rms = []
    fit_r2 = []
    final = None

    for i, fn in enumerate(files, 1):
        F = load_fields(fn)
        t = P["wci"] * F["ds"].current_time.to_value("s")
        jyc = corrected_jy(F["Bx"], F["Bz"], F["dx"], F["dz"])
        jym = np.mean(jyc, axis=0)
        fit = fit_sheet(
            F["z"], jym, P["L"], P["J0"], args.fit_halfwidth_L
        )
        jp, z0, Lfit, c0, rms, r2 = fit
        tci.append(t)
        Lfit_ld.append(Lfit / P["lambdaD"] if np.isfinite(Lfit) else np.nan)
        Jpeak_J0.append(jp / P["J0"] if np.isfinite(jp) else np.nan)
        z0_ld.append(z0 / P["lambdaD"] if np.isfinite(z0) else np.nan)
        fit_rms.append(rms)
        fit_r2.append(r2)
        final = (F, jyc, jym)
        print(
            f"{i:3d}/{len(files)} omega_ci*t={t:.5f} "
            f"Lfit/lambdaD={Lfit_ld[-1]:.4f} R2={r2:.6f}"
        )

    tci = np.asarray(tci)
    Lfit_ld = np.asarray(Lfit_ld)
    Jpeak_J0 = np.asarray(Jpeak_J0)
    z0_ld = np.asarray(z0_ld)
    fit_rms = np.asarray(fit_rms)
    fit_r2 = np.asarray(fit_r2)

    np.savetxt(
        "harris_500ld_40ld_sech2_fit_history.txt",
        np.column_stack([tci, Lfit_ld, Jpeak_J0, z0_ld, fit_rms, fit_r2]),
        header="omega_ci_t Lfit_lambdaD Jpeak_J0 z0_lambdaD fit_rms_J0 fit_R2",
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax1.plot(tci, Lfit_ld, "o-", ms=3)
    ax1.axhline(args.sheet_lambdaD, ls="--", lw=1.0, label="initial L")
    ax1.set_ylabel(r"$L_{fit}/\lambda_D$")
    ax1.legend()
    ax2.plot(tci, Jpeak_J0, "o-", ms=3)
    ax2.set_ylabel(r"$J_{peak}/J_0$")
    ax3.plot(tci, fit_r2, "o-", ms=3, label=r"$R^2$")
    ax3.set_ylabel(r"fit $R^2$")
    ax3.set_xlabel(r"$\omega_{ci}t$")
    ax3.set_ylim(0.9, 1.001)
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("harris_500ld_40ld_sech2_fit_history.png", dpi=200)
    plt.close(fig)

    F, jyc, jym = final
    xld = F["x"] / P["lambdaD"]
    zld = F["z"] / P["lambdaD"]
    X, Z = np.meshgrid(xld, zld, indexing="ij")
    zoom = np.abs(zld) <= args.zoom_sheet * args.sheet_lambdaD

    Ay = reconstruct_Ay(F["Bx"], F["Bz"], F["x"], F["z"])
    Ay -= np.nanmean(Ay)
    dAy = Ay - np.mean(Ay, axis=0, keepdims=True)
    AyN = Ay / (P["B0"] * P["L"])
    dAyN = dAy / (P["B0"] * P["L"])
    JyN = jyc / P["J0"]
    ByN = F["By"] / P["B0"]

    # Full magnetic topology over current density.
    fig, ax = plt.subplots(figsize=(11, 5.7))
    vmaxj = np.nanpercentile(np.abs(JyN[:, zoom]), 99.5)
    pcm = ax.pcolormesh(
        X[:, zoom], Z[:, zoom], JyN[:, zoom], shading="auto",
        vmin=-0.05 * vmaxj, vmax=vmaxj
    )
    arr = AyN[:, zoom]
    amin, amax = np.nanpercentile(arr, [1.0, 99.0])
    if amax > amin:
        ax.contour(
            X[:, zoom], Z[:, zoom], arr,
            levels=np.linspace(amin, amax, 45),
            colors="k", linewidths=0.55, alpha=0.75
        )
    fig.colorbar(pcm, ax=ax, label=r"$J_y/J_0$")
    ax.set_xlabel(r"$x/\lambda_D$")
    ax.set_ylabel(r"$z/\lambda_D$")
    ax.set_title(fr"full magnetic topology, $\omega_{{ci}}t={tci[-1]:.3f}$")
    fig.tight_layout()
    fig.savefig("harris_500ld_40ld_full_Ay_topology.png", dpi=220)
    plt.close(fig)

    # Perturbation topology with a symmetric color range.
    fig, ax = plt.subplots(figsize=(11, 5.7))
    vmax = np.nanpercentile(np.abs(dAyN[:, zoom]), 99.5)
    pcm = ax.pcolormesh(
        X[:, zoom], Z[:, zoom], dAyN[:, zoom], shading="auto",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    if vmax > 0:
        ax.contour(
            X[:, zoom], Z[:, zoom], dAyN[:, zoom],
            levels=np.linspace(-vmax, vmax, 31),
            colors="k", linewidths=0.45, alpha=0.6
        )
    fig.colorbar(pcm, ax=ax, label=r"$\delta A_y/(B_0L)$")
    ax.set_xlabel(r"$x/\lambda_D$")
    ax.set_ylabel(r"$z/\lambda_D$")
    ax.set_title(fr"perturbation flux, $\omega_{{ci}}t={tci[-1]:.3f}$")
    fig.tight_layout()
    fig.savefig("harris_500ld_40ld_deltaAy_topology_reconnection.png", dpi=220)
    plt.close(fig)

    # Raw By map.
    vby = np.nanpercentile(np.abs(ByN[:, zoom]), 99.0)
    fig, ax = plt.subplots(figsize=(11, 5.7))
    pcm = ax.pcolormesh(
        X[:, zoom], Z[:, zoom], ByN[:, zoom], shading="auto",
        cmap="RdBu_r", vmin=-vby, vmax=vby
    )
    fig.colorbar(pcm, ax=ax, label=r"$B_y/B_0$")
    ax.set_xlabel(r"$x/\lambda_D$")
    ax.set_ylabel(r"$z/\lambda_D$")
    ax.set_title(fr"raw out-of-plane $B_y$, $\omega_{{ci}}t={tci[-1]:.3f}$")
    fig.tight_layout()
    fig.savefig("harris_500ld_40ld_By_raw.png", dpi=220)
    plt.close(fig)

    # Lightly smoothed By map, visualization only.
    ByS = gaussian_filter(ByN, sigma=args.by_sigma)
    vbys = np.nanpercentile(np.abs(ByS[:, zoom]), 99.0)
    fig, ax = plt.subplots(figsize=(11, 5.7))
    pcm = ax.pcolormesh(
        X[:, zoom], Z[:, zoom], ByS[:, zoom], shading="auto",
        cmap="RdBu_r", vmin=-vbys, vmax=vbys
    )
    if amax > amin:
        ax.contour(
            X[:, zoom], Z[:, zoom], arr,
            levels=np.linspace(amin, amax, 25),
            colors="k", linewidths=0.45, alpha=0.5
        )
    fig.colorbar(pcm, ax=ax, label=r"smoothed $B_y/B_0$")
    ax.set_xlabel(r"$x/\lambda_D$")
    ax.set_ylabel(r"$z/\lambda_D$")
    ax.set_title(
        fr"$B_y$ Hall-structure check (visual smoothing only), "
        fr"$\omega_{{ci}}t={tci[-1]:.3f}$"
    )
    fig.tight_layout()
    fig.savefig("harris_500ld_40ld_By_hall_check.png", dpi=220)
    plt.close(fig)

    final_L = Lfit_ld[-1]
    k1 = 2.0 * np.pi * final_L / args.box_lambdaD
    k2 = 2.0 * k1
    dp1 = 2.0 * (1.0 / k1 - k1)
    dp2 = 2.0 * (1.0 / k2 - k2)
    by_rms = np.sqrt(np.mean(ByN[:, zoom] ** 2))
    bz_rms = np.sqrt(np.mean((F["Bz"][:, zoom] / P["B0"]) ** 2))
    j0 = int(np.argmin(np.abs(F["z"])))
    flux_span = (
        np.nanmax(dAyN[:, j0]) - np.nanmin(dAyN[:, j0])
    )

    with open("harris_500ld_40ld_reconnection_summary.txt", "w") as f:
        f.write("Harris 500 lambda_D / 40 lambda_D reconnection diagnostics\n")
        f.write("========================================================\n\n")
        f.write(f"final omega_ci*t = {tci[-1]:.8f}\n")
        f.write(f"final Lfit/lambda_D = {final_L:.8f}\n")
        f.write(f"final Jpeak/J0 = {Jpeak_J0[-1]:.8f}\n")
        f.write(f"final fit R2 = {fit_r2[-1]:.8f}\n")
        f.write(f"final k1*Lfit = {k1:.8f}\n")
        f.write(f"final DeltaPrime1*Lfit = {dp1:.8f}\n")
        f.write(f"final k2*Lfit = {k2:.8f}\n")
        f.write(f"final DeltaPrime2*Lfit = {dp2:.8f}\n")
        f.write(f"zoom By_rms/B0 = {by_rms:.8e}\n")
        f.write(f"zoom Bz_rms/B0 = {bz_rms:.8e}\n")
        f.write(f"midplane deltaAy span/(B0 L0) = {flux_span:.8e}\n")
        f.write(f"By smoothing sigma (visualization only) = {args.by_sigma:.3f} cells\n")

    print("Saved harris_500ld_40ld_reconnection_summary.txt")
    print("Saved harris_500ld_40ld_sech2_fit_history.txt/.png")
    print("Saved harris_500ld_40ld_full_Ay_topology.png")
    print("Saved harris_500ld_40ld_deltaAy_topology_reconnection.png")
    print("Saved harris_500ld_40ld_By_raw.png")
    print("Saved harris_500ld_40ld_By_hall_check.png")


if __name__ == "__main__":
    main()
