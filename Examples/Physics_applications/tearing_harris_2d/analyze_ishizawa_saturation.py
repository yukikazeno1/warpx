#!/usr/bin/env python3
"""Nonlinear saturation diagnostics for the Ishizawa-scale Harris-sheet run.

This script supplements the Fourier-mode diagnostics with three direct
nonlinear/topological observables:

  1) reconnected / island flux
       Psi_rec = A_y(O) - A_y(X_sep)

  2) dominant-island full width in z
       w_island = z_upper - z_lower
     measured at the dominant O-point from the full A_y separatrix level,
     not from a small-island analytic approximation;

  3) peak current density in the current-sheet region
       J_y,max / J0

The largest accepted island in each frame is used as the dominant island.
O/X detection is performed on a Fourier-low-pass midplane A_y signal to avoid
counting PIC cell-scale extrema, but island width and current are measured from
the UNSMOOTHED full reconstructed A_y and corrected curl-B current.

Outputs
-------
ishizawa_saturation_history.txt
ishizawa_saturation_psi.png
ishizawa_saturation_width.png
ishizawa_saturation_jymax.png
ishizawa_saturation_three_curves.png
ishizawa_saturation_summary.txt
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import (
    params,
    load_fields,
    reconstruct_Ay,
    corrected_jy,
)
from analyze_ishizawa_island_coalescence import detect_islands


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs_ishizawa_scale/ppc49")
    p.add_argument("--tmin", type=float, default=0.5,
                   help="minimum omega_ci*t written to saturation plots")
    p.add_argument("--topology-max-mode", type=int, default=20,
                   help="maximum x Fourier mode retained for O/X detection")
    p.add_argument("--relative-flux-threshold", type=float, default=0.12)
    p.add_argument("--absolute-flux-threshold", type=float, default=1.0e-3,
                   help="island-flux floor in B0*d_e")
    p.add_argument("--min-separation-de", type=float, default=2.0)
    p.add_argument("--current-core-de", type=float, default=4.0,
                   help="half-width |z|/de used for J_y,max")
    p.add_argument("--plateau-tmin", type=float, default=1.5,
                   help="start time for quasi-saturation statistics")
    p.add_argument("--slope-window", type=float, default=0.50,
                   help="late-time window in omega_ci*t for normalized slopes")
    return p.parse_args()


def numeric_key(path):
    name = os.path.basename(path.rstrip("/"))
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


def step_from_path(path):
    name = os.path.basename(path.rstrip("/"))
    m = re.search(r"(\d+)$", name)
    if not m:
        return -1
    s = m.group(1)
    # plotfile convention here is diag1<step>, e.g. diag1100000
    if name.startswith("diag1") and s.startswith("1") and len(s) > 1:
        return int(s[1:])
    return int(s)


def periodic_interp_crossing(z1, y1, z2, y2, target):
    """Linear interpolation of y(z)=target between two bracketing samples."""
    dy = y2 - y1
    if dy == 0:
        return 0.5 * (z1 + z2)
    a = (target - y1) / dy
    return z1 + a * (z2 - z1)


def island_width_at_O(Ay, z, io, sep_value, j0):
    """Full vertical separatrix width at x-index io.

    For an O-point that is a local maximum of A_y on z~0, closed contours
    exist for A_y >= A_sep.  Starting from the midplane, find the first
    crossing of A_y=A_sep above and below z=0 and interpolate the crossing.
    """
    prof = np.asarray(Ay[io, :], dtype=float)

    if not np.isfinite(sep_value) or prof[j0] <= sep_value:
        return np.nan, np.nan, np.nan

    # Upward crossing
    z_up = np.nan
    for j in range(j0, len(z) - 1):
        y1 = prof[j] - sep_value
        y2 = prof[j + 1] - sep_value
        if not (np.isfinite(y1) and np.isfinite(y2)):
            continue
        if y1 >= 0.0 and y2 <= 0.0:
            z_up = periodic_interp_crossing(
                z[j], prof[j], z[j + 1], prof[j + 1], sep_value
            )
            break

    # Downward crossing
    z_dn = np.nan
    for j in range(j0, 0, -1):
        y1 = prof[j] - sep_value
        y2 = prof[j - 1] - sep_value
        if not (np.isfinite(y1) and np.isfinite(y2)):
            continue
        if y1 >= 0.0 and y2 <= 0.0:
            z_dn = periodic_interp_crossing(
                z[j], prof[j], z[j - 1], prof[j - 1], sep_value
            )
            break

    if np.isfinite(z_up) and np.isfinite(z_dn):
        return float(z_up - z_dn), float(z_dn), float(z_up)
    return np.nan, z_dn, z_up


def linear_slope(x, y, xmin):
    """Least-squares dy/dx over x>=xmin; returns slope and R^2."""
    mask = np.isfinite(x) & np.isfinite(y) & (x >= xmin)
    xx = x[mask]
    yy = y[mask]
    if len(xx) < 3:
        return np.nan, np.nan
    c = np.polyfit(xx, yy, 1)
    pred = np.polyval(c, xx)
    ssr = np.sum((yy - pred) ** 2)
    sst = np.sum((yy - np.mean(yy)) ** 2)
    r2 = 1.0 - ssr / sst if sst > 0 else np.nan
    return float(c[0]), float(r2)


def plateau_stats(t, q, tmin):
    mask = np.isfinite(t) & np.isfinite(q) & (t >= tmin)
    if np.count_nonzero(mask) < 2:
        return np.nan, np.nan, np.nan
    a = q[mask]
    mean = float(np.mean(a))
    std = float(np.std(a))
    cv = std / abs(mean) if mean != 0 else np.nan
    return mean, std, cv


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params()

    files = sorted(
        glob.glob(str(Path(args.run_dir) / "diags" / "diag1*")),
        key=numeric_key
    )
    if not files:
        raise FileNotFoundError(
            f"No plotfiles found under {args.run_dir}/diags/diag1*"
        )

    rows = []

    print("=" * 82)
    print("Ishizawa-scale nonlinear saturation diagnostics")
    print("=" * 82)

    for ik, fn in enumerate(files, 1):
        F = load_fields(fn)
        tci = P["wci"] * F["ds"].current_time.to_value("s")
        if tci < args.tmin:
            continue

        Ay = reconstruct_Ay(F["Bx"], F["Bz"], F["x"], F["z"])
        det = detect_islands(
            Ay, F["x"], F["z"], P["B0"], P["de"],
            topology_max_mode=args.topology_max_mode,
            relative_threshold=args.relative_flux_threshold,
            absolute_threshold=args.absolute_flux_threshold,
            min_separation_de=args.min_separation_de,
        )

        jy = corrected_jy(
            F["Bx"], F["Bz"], F["dx"], F["dz"]
        )
        core = np.abs(F["z"]) <= args.current_core_de * P["de"]
        jy_core = jy[:, core] / P["J0"]
        jy_max = float(np.nanmax(jy_core))
        jy_absmax = float(np.nanmax(np.abs(jy_core)))

        islands = det["islands"]
        NO = len(islands)

        psi_rec = np.nan
        width = np.nan
        xO_de = np.nan
        zlo_de = np.nan
        zhi_de = np.nan

        if islands:
            # Dominant island = largest topological flux.
            q = max(islands, key=lambda a: a["psi"])
            io = q["io"]
            il = q["il"]
            ir = q["ir"]

            # A_sep is the higher neighbouring X-point flux; then
            # A_O - A_sep = min(A_O-A_XL, A_O-A_XR) = q["psi"]*B0*de.
            j0 = det["j0"]
            AXL = Ay[il, j0]
            AXR = Ay[ir, j0]
            Asep = max(AXL, AXR)

            psi_rec = float((Ay[io, j0] - Asep) / (P["B0"] * P["de"]))
            w, zdn, zup = island_width_at_O(
                Ay, F["z"], io, Asep, j0
            )
            width = w / P["de"] if np.isfinite(w) else np.nan
            xO_de = F["x"][io] / P["de"]
            zlo_de = zdn / P["de"] if np.isfinite(zdn) else np.nan
            zhi_de = zup / P["de"] if np.isfinite(zup) else np.nan

        step = step_from_path(fn)
        rows.append([
            step, tci, NO, psi_rec, width,
            jy_max, jy_absmax, xO_de, zlo_de, zhi_de,
            det["threshold"],
        ])

        print(
            f"{ik:3d}/{len(files)}  step={step:6d}  "
            f"wci*t={tci:7.4f}  N_O={NO:2d}  "
            f"Psi={psi_rec:10.4e}  w/de={width:8.3f}  "
            f"Jymax/J0={jy_max:8.3f}"
        )

    if not rows:
        raise RuntimeError("No frames satisfy --tmin")

    a = np.asarray(rows, dtype=float)
    step = a[:, 0]
    t = a[:, 1]
    NO = a[:, 2]
    psi = a[:, 3]
    width = a[:, 4]
    jymax = a[:, 5]
    jyabs = a[:, 6]

    np.savetxt(
        "ishizawa_saturation_history.txt",
        a,
        header=(
            "step omega_ci_t N_O "
            "Psi_rec_B0de island_full_width_de "
            "Jy_max_J0 absJy_max_J0 "
            "dominant_O_x_de separatrix_zlo_de separatrix_zhi_de "
            "island_threshold_B0de"
        ),
    )

    def save_curve(filename, y, ylabel, title):
        fig, ax = plt.subplots(figsize=(8.6, 5.5))
        ax.plot(t, y, "o-", ms=3.5)
        ax.axvline(
            args.plateau_tmin, ls="--", lw=1.0,
            label=fr"plateau test start: $\omega_{{ci}}t={args.plateau_tmin:g}$"
        )
        ax.set_xlabel(r"$\omega_{ci}t$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(filename, dpi=210)
        plt.close(fig)

    save_curve(
        "ishizawa_saturation_psi.png",
        psi,
        r"$\Psi_{\rm rec}/(B_0d_e)$",
        "Dominant-island reconnected flux",
    )
    save_curve(
        "ishizawa_saturation_width.png",
        width,
        r"$w_{\rm island}/d_e$",
        "Dominant-island full separatrix width",
    )
    save_curve(
        "ishizawa_saturation_jymax.png",
        jymax,
        r"$J_{y,\max}/J_0$",
        "Peak current density in the sheet core",
    )

    # One compact 3-panel figure.
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.0), sharex=True)
    axes[0].plot(t, psi, "o-", ms=3.2)
    axes[0].set_ylabel(r"$\Psi_{\rm rec}/(B_0d_e)$")
    axes[1].plot(t, width, "o-", ms=3.2)
    axes[1].set_ylabel(r"$w_{\rm island}/d_e$")
    axes[2].plot(t, jymax, "o-", ms=3.2, label=r"$J_{y,\max}/J_0$")
    axes[2].plot(t, jyabs, "s-", ms=3.0, label=r"$\max|J_y|/J_0$")
    axes[2].set_ylabel(r"current / $J_0$")
    axes[2].set_xlabel(r"$\omega_{ci}t$")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axvline(args.plateau_tmin, ls="--", lw=1.0)
        ax.grid(alpha=0.25)
    fig.suptitle("Ishizawa-scale nonlinear saturation diagnostics")
    fig.tight_layout()
    fig.savefig("ishizawa_saturation_three_curves.png", dpi=210)
    plt.close(fig)

    # Quantitative late-time plateau diagnostics.
    final_t = float(t[-1])
    late_start = max(args.plateau_tmin, final_t - args.slope_window)

    psi_mean, psi_std, psi_cv = plateau_stats(t, psi, args.plateau_tmin)
    w_mean, w_std, w_cv = plateau_stats(t, width, args.plateau_tmin)
    j_mean, j_std, j_cv = plateau_stats(t, jymax, args.plateau_tmin)

    psi_slope, psi_r2 = linear_slope(t, psi, late_start)
    w_slope, w_r2 = linear_slope(t, width, late_start)
    j_slope, j_r2 = linear_slope(t, jymax, late_start)

    # Normalize slopes by late-window mean so quantities with different units
    # can be compared as fractional change per omega_ci*t.
    def norm_slope(s, q, xmin):
        m = np.isfinite(q) & (t >= xmin)
        qbar = np.nanmean(q[m]) if np.any(m) else np.nan
        return s / abs(qbar) if np.isfinite(s) and np.isfinite(qbar) and qbar != 0 else np.nan

    psi_ns = norm_slope(psi_slope, psi, late_start)
    w_ns = norm_slope(w_slope, width, late_start)
    j_ns = norm_slope(j_slope, jymax, late_start)

    with open("ishizawa_saturation_summary.txt", "w") as f:
        f.write("Ishizawa-scale nonlinear saturation diagnostics\n")
        f.write("================================================\n\n")
        f.write(f"frames analysed = {len(t)}\n")
        f.write(f"omega_ci*t range = {t[0]:.8f} .. {t[-1]:.8f}\n")
        f.write(f"plateau statistics start = {args.plateau_tmin:.8f}\n")
        f.write(f"late slope window = {late_start:.8f} .. {final_t:.8f}\n\n")

        f.write("Final values\n")
        f.write("------------\n")
        f.write(f"N_O = {int(NO[-1])}\n")
        f.write(f"Psi_rec/(B0 de) = {psi[-1]:.8e}\n")
        f.write(f"island full width/de = {width[-1]:.8e}\n")
        f.write(f"Jy_max/J0 = {jymax[-1]:.8e}\n")
        f.write(f"max|Jy|/J0 = {jyabs[-1]:.8e}\n\n")

        f.write("Plateau statistics (mean, std, coefficient of variation)\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"Psi_rec : {psi_mean:.8e} {psi_std:.8e} {psi_cv:.8e}\n")
        f.write(f"width   : {w_mean:.8e} {w_std:.8e} {w_cv:.8e}\n")
        f.write(f"Jy_max  : {j_mean:.8e} {j_std:.8e} {j_cv:.8e}\n\n")

        f.write("Late-time linear slopes\n")
        f.write("-----------------------\n")
        f.write("# slope is dq/d(omega_ci*t); normalized slope is slope/<q>\n")
        f.write(f"Psi_rec slope = {psi_slope:.8e}, R2={psi_r2:.6f}, normalized={psi_ns:.8e}\n")
        f.write(f"width   slope = {w_slope:.8e}, R2={w_r2:.6f}, normalized={w_ns:.8e}\n")
        f.write(f"Jy_max  slope = {j_slope:.8e}, R2={j_r2:.6f}, normalized={j_ns:.8e}\n\n")

        f.write("Interpretation guide\n")
        f.write("--------------------\n")
        f.write("A quasi-saturated state is supported when Psi_rec and island width\n")
        f.write("show no sustained secular increase over a long late-time interval,\n")
        f.write("and Jy_max fluctuates around a stationary mean rather than drifting.\n")
        f.write("Use these diagnostics together with the m=1 Fourier plateau and full-Ay\n")
        f.write("topology; no single numerical threshold is a universal saturation criterion.\n")

    print()
    print("Saved:")
    print("  ishizawa_saturation_history.txt")
    print("  ishizawa_saturation_psi.png")
    print("  ishizawa_saturation_width.png")
    print("  ishizawa_saturation_jymax.png")
    print("  ishizawa_saturation_three_curves.png")
    print("  ishizawa_saturation_summary.txt")


if __name__ == "__main__":
    main()
