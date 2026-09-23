#!/usr/bin/env python3
"""Robust nonlinear saturation diagnostics for the Ishizawa-scale Harris run.

This version is designed specifically for the late, system-scale m=1 state.
The first implementation selected the largest local island from an m<=20
midplane signal, then measured flux/width on the unfiltered Ay field.  During
strong nonlinear coalescence that can switch between small local extrema and
can even make the raw-field O/X ordering inconsistent.

Here we instead measure the LARGE-SCALE magnetic topology consistently from an
x-low-pass filtered full Ay field:

  Psi_LS = A_y(O) - A_y(X)      (signed by the selected O/X orientation)
  w_LS   = full vertical separatrix width through the system-scale O point
  Jy,max = peak corrected curl-B current in the sheet core

The low pass is used only for the topology diagnostic.  Jy is always measured
from the unfiltered fields.  A one-mode estimate from Bz,m=1 is also written
as an independent consistency check:
  Psi_m1/(B0 de) = 2 (Bz,m1/B0) / (k1 de).

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs_ishizawa_scale/ppc49")
    p.add_argument("--tmin", type=float, default=0.50)
    p.add_argument(
        "--large-scale-max-mode", type=int, default=3,
        help="retain x Fourier modes 0..M in Ay for system-scale O/X topology",
    )
    p.add_argument("--current-core-de", type=float, default=4.0)
    p.add_argument(
        "--current-percentile", type=float, default=99.5,
        help="robust percentile of positive Jy/J0 in the current-sheet core",
    )
    p.add_argument("--plateau-tmin", type=float, default=1.50)
    p.add_argument("--slope-window", type=float, default=0.50)

    # Accepted for backward compatibility with the first version.  They are
    # deliberately not used by the new large-scale topology measurement.
    p.add_argument("--topology-max-mode", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--relative-flux-threshold", type=float, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--absolute-flux-threshold", type=float, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--min-separation-de", type=float, default=None,
                   help=argparse.SUPPRESS)
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
    if name.startswith("diag1") and s.startswith("1") and len(s) > 1:
        return int(s[1:])
    return int(s)


def lowpass_x(a, max_mode):
    """Periodic x low pass of a 2-D x-z array."""
    ft = np.fft.rfft(np.asarray(a, dtype=float), axis=0)
    if max_mode + 1 < ft.shape[0]:
        ft[max_mode + 1:, :] = 0.0
    return np.fft.irfft(ft, n=a.shape[0], axis=0)


def interp_crossing(z1, q1, z2, q2):
    """Linear zero crossing of q(z)."""
    dq = q2 - q1
    if dq == 0:
        return 0.5 * (z1 + z2)
    a = -q1 / dq
    return z1 + a * (z2 - z1)


def separatrix_width(profile, z, j0, sep, orientation):
    """Width between first upper/lower crossings of the separatrix.

    orientation=+1 means O has Ay > Asep; -1 means Ay < Asep.
    """
    q = orientation * (np.asarray(profile, dtype=float) - sep)
    if not np.isfinite(q[j0]) or q[j0] <= 0:
        return np.nan, np.nan, np.nan

    zup = np.nan
    for j in range(j0, len(z) - 1):
        if not (np.isfinite(q[j]) and np.isfinite(q[j + 1])):
            continue
        if q[j] >= 0 and q[j + 1] <= 0:
            zup = interp_crossing(z[j], q[j], z[j + 1], q[j + 1])
            break

    zdn = np.nan
    for j in range(j0, 0, -1):
        if not (np.isfinite(q[j]) and np.isfinite(q[j - 1])):
            continue
        if q[j] >= 0 and q[j - 1] <= 0:
            zdn = interp_crossing(z[j], q[j], z[j - 1], q[j - 1])
            break

    if np.isfinite(zup) and np.isfinite(zdn):
        return float(zup - zdn), float(zdn), float(zup)
    return np.nan, zdn, zup


def choose_system_scale_OX(Ay_lp, z):
    """Choose global low-mode O/X pair and its orientation.

    The late state can, in principle, flip the sign of the local flux
    extremum.  We therefore test whether the global maximum is also a maximum
    in z, or the global minimum is also a minimum in z.  If neither test is
    clean because of discretization, fall back to the extremum with the larger
    matching-curvature magnitude.
    """
    j0 = int(np.argmin(np.abs(z)))
    mid = Ay_lp[:, j0]
    imax = int(np.nanargmax(mid))
    imin = int(np.nanargmin(mid))

    def zcurv(i):
        if j0 <= 0 or j0 >= len(z) - 1:
            return np.nan
        dz1 = z[j0] - z[j0 - 1]
        dz2 = z[j0 + 1] - z[j0]
        if not np.isclose(dz1, dz2, rtol=1e-4, atol=0):
            # only sign is used; this is adequate for nearly uniform z
            return Ay_lp[i, j0 + 1] - 2*Ay_lp[i, j0] + Ay_lp[i, j0 - 1]
        return (Ay_lp[i, j0 + 1] - 2*Ay_lp[i, j0] + Ay_lp[i, j0 - 1]) / dz1**2

    cmax = zcurv(imax)
    cmin = zcurv(imin)

    max_is_O = np.isfinite(cmax) and cmax < 0
    min_is_O = np.isfinite(cmin) and cmin > 0

    if max_is_O and not min_is_O:
        io, ix, orientation = imax, imin, +1.0
    elif min_is_O and not max_is_O:
        io, ix, orientation = imin, imax, -1.0
    else:
        # For the original positive-current Harris sheet the O point is a
        # maximum of Ay.  Prefer that convention unless curvature strongly
        # favours the opposite choice.
        score_max = -cmax if np.isfinite(cmax) else -np.inf
        score_min = cmin if np.isfinite(cmin) else -np.inf
        if score_min > 1.5 * score_max:
            io, ix, orientation = imin, imax, -1.0
        else:
            io, ix, orientation = imax, imin, +1.0

    A_O = float(mid[io])
    A_X = float(mid[ix])
    psi_signed = orientation * (A_O - A_X)
    return j0, io, ix, orientation, A_O, A_X, float(psi_signed)


def linear_slope(x, y, xmin):
    mask = np.isfinite(x) & np.isfinite(y) & (x >= xmin)
    xx, yy = x[mask], y[mask]
    if len(xx) < 3:
        return np.nan, np.nan
    c = np.polyfit(xx, yy, 1)
    pred = np.polyval(c, xx)
    ssr = np.sum((yy - pred)**2)
    sst = np.sum((yy - np.mean(yy))**2)
    r2 = 1.0 - ssr/sst if sst > 0 else np.nan
    return float(c[0]), float(r2)


def plateau_stats(t, q, tmin):
    m = np.isfinite(t) & np.isfinite(q) & (t >= tmin)
    if np.count_nonzero(m) < 2:
        return np.nan, np.nan, np.nan
    a = q[m]
    mean = float(np.mean(a))
    std = float(np.std(a))
    return mean, std, std/abs(mean) if mean != 0 else np.nan


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params()

    files = sorted(
        glob.glob(str(Path(args.run_dir) / "diags" / "diag1*")),
        key=numeric_key,
    )
    if not files:
        raise FileNotFoundError(f"No plotfiles under {args.run_dir}/diags/diag1*")

    rows = []
    print("="*82)
    print("Ishizawa-scale LARGE-SCALE nonlinear saturation diagnostics")
    print(f"Ay x-low-pass: m <= {args.large_scale_max_mode}")
    print("="*82)

    for ik, fn in enumerate(files, 1):
        F = load_fields(fn)
        tci = P["wci"] * F["ds"].current_time.to_value("s")
        if tci < args.tmin:
            continue

        Ay = reconstruct_Ay(F["Bx"], F["Bz"], F["x"], F["z"])
        Ay_lp = lowpass_x(Ay, args.large_scale_max_mode)

        j0, io, ix, orient, AO, AX, psi_SI = choose_system_scale_OX(
            Ay_lp, F["z"]
        )
        psi = psi_SI / (P["B0"] * P["de"])

        w, zdn, zup = separatrix_width(
            Ay_lp[io, :], F["z"], j0, AX, orient
        )
        width = w/P["de"] if np.isfinite(w) else np.nan

        # Independent m=1 harmonic flux estimate from the actual Bz midplane.
        nx = F["Bz"].shape[0]
        dx = F["x"][1] - F["x"][0]
        Lx = nx * dx
        kde1 = 2.0*np.pi*P["de"]/Lx
        ftmid = np.fft.rfft(F["Bz"][:, j0]) / nx
        A1_Bz = 2.0*abs(ftmid[1])/P["B0"] if len(ftmid) > 1 else np.nan
        psi_m1 = 2.0*A1_Bz/kde1 if np.isfinite(A1_Bz) else np.nan

        # Current is deliberately measured on the unfiltered fields.
        jy = corrected_jy(F["Bx"], F["Bz"], F["dx"], F["dz"]) / P["J0"]
        core = np.abs(F["z"]) <= args.current_core_de * P["de"]
        pos = jy[:, core]
        jy_max = float(np.nanmax(pos))
        jy_absmax = float(np.nanmax(np.abs(pos)))
        jy_pct = float(np.nanpercentile(pos, args.current_percentile))

        rows.append([
            step_from_path(fn), tci, psi, psi_m1, width,
            jy_max, jy_absmax, jy_pct,
            F["x"][io]/P["de"], F["x"][ix]/P["de"],
            zdn/P["de"] if np.isfinite(zdn) else np.nan,
            zup/P["de"] if np.isfinite(zup) else np.nan,
            orient, A1_Bz,
        ])

        print(
            f"{ik:3d}/{len(files)} step={int(rows[-1][0]):6d} "
            f"wci*t={tci:7.4f}  Psi_LS={psi:8.3f}  "
            f"Psi_m1={psi_m1:8.3f}  w/de={width:7.3f}  "
            f"Jymax/J0={jy_max:6.3f}"
        )

    a = np.asarray(rows, dtype=float)
    if len(a) == 0:
        raise RuntimeError("No frames satisfy --tmin")

    step,t,psi,psi_m1,width,jymax,jyabs,jypct,xO,xX,zlo,zhi,orient,A1 = a.T

    np.savetxt(
        "ishizawa_saturation_history.txt", a,
        header=(
            "step omega_ci_t Psi_largeScale_B0de Psi_m1_B0de "
            "island_full_width_de Jy_max_J0 absJy_max_J0 "
            f"Jy_p{args.current_percentile:g}_J0 "
            "O_x_de X_x_de separatrix_zlo_de separatrix_zhi_de "
            "O_orientation Bz_m1_midplane_B0"
        ),
    )

    def curve(name, y, ylabel, title, y2=None, y2label=None):
        fig, ax = plt.subplots(figsize=(8.7,5.5))
        ax.plot(t,y,"o-",ms=3.4,label=title if y2 is not None else None)
        if y2 is not None:
            ax.plot(t,y2,"s-",ms=3.0,label=y2label)
            ax.legend()
        ax.axvline(args.plateau_tmin,ls="--",lw=1)
        ax.set_xlabel(r"$\omega_{ci}t$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        if y2 is None:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(name,dpi=210)
        plt.close(fig)

    curve(
        "ishizawa_saturation_psi.png", psi,
        r"$\Psi/(B_0d_e)$",
        "large-scale full-Ay flux",
        psi_m1, r"$m=1$ harmonic estimate",
    )
    curve(
        "ishizawa_saturation_width.png", width,
        r"$w_{\rm island}/d_e$",
        "large-scale island full separatrix width",
    )
    curve(
        "ishizawa_saturation_jymax.png", jymax,
        r"current / $J_0$",
        r"$J_{y,\max}$",
        jypct, fr"$J_y$ p{args.current_percentile:g}",
    )

    fig,axs=plt.subplots(3,1,figsize=(9,10),sharex=True)
    axs[0].plot(t,psi,"o-",ms=3,label="large-scale full-Ay")
    axs[0].plot(t,psi_m1,"s-",ms=3,label="m=1 harmonic")
    axs[0].set_ylabel(r"$\Psi/(B_0d_e)$")
    axs[0].legend(fontsize=8)
    axs[1].plot(t,width,"o-",ms=3)
    axs[1].set_ylabel(r"$w_{\rm island}/d_e$")
    axs[2].plot(t,jymax,"o-",ms=3,label=r"$J_{y,\max}/J_0$")
    axs[2].plot(t,jypct,"s-",ms=3,label=fr"$J_y$ p{args.current_percentile:g}")
    axs[2].set_ylabel(r"current / $J_0$")
    axs[2].set_xlabel(r"$\omega_{ci}t$")
    axs[2].legend(fontsize=8)
    for ax in axs:
        ax.axvline(args.plateau_tmin,ls="--",lw=1)
        ax.grid(alpha=.25)
    fig.suptitle("Ishizawa-scale large-scale saturation diagnostics")
    fig.tight_layout()
    fig.savefig("ishizawa_saturation_three_curves.png",dpi=210)
    plt.close(fig)

    final_t=float(t[-1])
    late_start=max(args.plateau_tmin,final_t-args.slope_window)

    def stats(q):
        mean,std,cv=plateau_stats(t,q,args.plateau_tmin)
        s,r2=linear_slope(t,q,late_start)
        m=(t>=late_start)&np.isfinite(q)
        qbar=np.nanmean(q[m]) if np.any(m) else np.nan
        ns=s/abs(qbar) if np.isfinite(s) and np.isfinite(qbar) and qbar!=0 else np.nan
        return mean,std,cv,s,r2,ns

    sp=stats(psi); sm=stats(psi_m1); sw=stats(width)
    sj=stats(jymax); sjp=stats(jypct)

    with open("ishizawa_saturation_summary.txt","w") as f:
        f.write("Ishizawa-scale LARGE-SCALE nonlinear saturation diagnostics\n")
        f.write("============================================================\n\n")
        f.write(f"frames analysed = {len(t)}\n")
        f.write(f"omega_ci*t range = {t[0]:.8f} .. {t[-1]:.8f}\n")
        f.write(f"Ay large-scale x modes retained = 0..{args.large_scale_max_mode}\n")
        f.write(f"plateau statistics start = {args.plateau_tmin:.8f}\n")
        f.write(f"late slope window = {late_start:.8f} .. {final_t:.8f}\n\n")
        f.write("Final values\n------------\n")
        f.write(f"Psi_largeScale/(B0 de) = {psi[-1]:.8e}\n")
        f.write(f"Psi_m1/(B0 de) = {psi_m1[-1]:.8e}\n")
        f.write(f"island full width/de = {width[-1]:.8e}\n")
        f.write(f"Jy_max/J0 = {jymax[-1]:.8e}\n")
        f.write(f"Jy_p{args.current_percentile:g}/J0 = {jypct[-1]:.8e}\n")
        f.write(f"Bz_m1_midplane/B0 = {A1[-1]:.8e}\n\n")

        f.write("Plateau statistics: mean std CV\n")
        for label,s in [
            ("Psi_largeScale",sp),("Psi_m1",sm),("width",sw),
            ("Jy_max",sj),(f"Jy_p{args.current_percentile:g}",sjp)
        ]:
            f.write(f"{label:16s} {s[0]:.8e} {s[1]:.8e} {s[2]:.8e}\n")

        f.write("\nLate-time slopes: slope R2 normalized_slope\n")
        for label,s in [
            ("Psi_largeScale",sp),("Psi_m1",sm),("width",sw),
            ("Jy_max",sj),(f"Jy_p{args.current_percentile:g}",sjp)
        ]:
            f.write(f"{label:16s} {s[3]:.8e} {s[4]:.6f} {s[5]:.8e}\n")

        f.write("\nInterpretation\n--------------\n")
        f.write("The large-scale Psi and width intentionally track the final system-scale\n")
        f.write("m=1 island rather than the largest local PIC-scale extremum.  A quasi-\n")
        f.write("saturated state is supported when large-scale Psi and width have no\n")
        f.write("sustained secular growth and the current metrics fluctuate around a\n")
        f.write("stationary level.  Compare Psi_largeScale with Psi_m1 as a consistency\n")
        f.write("check; they should approach one another once m=1 dominates strongly.\n")

    print("Saved ishizawa_saturation_history.txt")
    print("Saved ishizawa_saturation_psi.png")
    print("Saved ishizawa_saturation_width.png")
    print("Saved ishizawa_saturation_jymax.png")
    print("Saved ishizawa_saturation_three_curves.png")
    print("Saved ishizawa_saturation_summary.txt")


if __name__ == "__main__":
    main()
