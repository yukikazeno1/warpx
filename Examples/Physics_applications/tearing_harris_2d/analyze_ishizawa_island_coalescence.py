#!/usr/bin/env python3
"""Quantify magnetic-island coalescence in the Ishizawa-scale Harris run.

The 40k run shows a clear nonlinear transition from an early finite-k tearing
band (m~5-7) to fewer, larger islands and stronger low-m modes.  This script
turns that visual impression into quantitative diagnostics:

* O-point and X-point counts on the current-sheet midplane;
* per-island reconnected flux from neighbouring X/O flux differences;
* dominant Fourier mode and spectral centroid <m>;
* characteristic magnetic scale Lx/<m>;
* maximum |E_y| and |J_y| sampled at accepted X points;
* an annotated final topology with the detected X/O locations.

Detection philosophy
--------------------
For a Harris sheet with Bx increasing through z=0, A_y has negative curvature
in z at the midplane.  Consequently, maxima of A_y(x,z=0) are O points and
minima are X points.  At z=0 the equilibrium A_y is independent of x, so the
x-dependent part delta A_y has exactly the same extrema.  To suppress PIC
cell-scale noise before finding extrema, only x Fourier modes m <=
--topology-max-mode are retained.  An island is accepted only if its flux above
both neighbouring X points exceeds a frame-adaptive threshold.

The resulting O/X count is intentionally conservative; it is designed to
measure robust islands, not every tiny local extremum in the PIC field.
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

C = 299792458.0
MU0 = 1.25663706212e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs_ishizawa_scale/ppc49")
    p.add_argument("--max-mode", type=int, default=20,
                   help="maximum m used for spectral centroid")
    p.add_argument("--topology-max-mode", type=int, default=20,
                   help="maximum x mode retained before O/X detection")
    p.add_argument("--core-width-de", type=float, default=2.0)
    p.add_argument("--zoom-de", type=float, default=4.0)
    p.add_argument("--tmin", type=float, default=0.05,
                   help="minimum omega_ci*t included in coalescence plots")
    p.add_argument("--relative-flux-threshold", type=float, default=0.12,
                   help="minimum island flux as fraction of strongest island in each frame")
    p.add_argument("--absolute-flux-threshold", type=float, default=1.0e-3,
                   help="absolute floor on island flux in units B0*d_e")
    p.add_argument("--min-separation-de", type=float, default=2.0,
                   help="minimum periodic separation between accepted O points")
    return p.parse_args()


def numeric_key(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def step_from_path(path):
    s = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    if not s:
        return -1
    q = s.group(1)
    return int(q[1:]) if len(q) > 1 and q.startswith("1") else int(q)


def as_xz(arr, nx, nz):
    a = np.asarray(arr).squeeze()
    if a.shape == (nx, nz):
        return a
    if a.shape == (nz, nx):
        return a.T
    raise RuntimeError(f"Unexpected array shape {a.shape}; expected {(nx,nz)} or {(nz,nx)}")


def load_ey(F):
    """Read E_y from the same plotfile represented by F."""
    ds = F["ds"]
    nx, nz = len(F["x"]), len(F["z"])
    g = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
                         dims=ds.domain_dimensions)
    q = g["boxlib", "Ey"]
    try:
        a = q.to_value("V/m")
    except Exception:
        a = q.to_ndarray()
    return as_xz(a, nx, nz)


def lowpass_periodic(y, max_mode):
    """Periodic x low-pass retaining 0 <= m <= max_mode."""
    y = np.asarray(y, dtype=float)
    ft = np.fft.rfft(y)
    if max_mode + 1 < len(ft):
        ft[max_mode + 1:] = 0.0
    return np.fft.irfft(ft, n=len(y))


def periodic_distance(xa, xb, Lx):
    d = abs(float(xa) - float(xb))
    return min(d, Lx - d)


def periodic_extrema(y):
    """Return indices of strict/plateau-safe periodic maxima and minima."""
    y = np.asarray(y)
    ym = np.roll(y, 1)
    yp = np.roll(y, -1)
    maxima = np.flatnonzero((y >= ym) & (y >= yp) & ((y > ym) | (y > yp)))
    minima = np.flatnonzero((y <= ym) & (y <= yp) & ((y < ym) | (y < yp)))
    return maxima, minima


def previous_next_periodic(indices, i, n):
    """Nearest previous/next member of indices around periodic integer i."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return None, None
    left_delta = (i - indices) % n
    right_delta = (indices - i) % n
    left_delta[left_delta == 0] = n
    right_delta[right_delta == 0] = n
    il = int(indices[np.argmin(left_delta)])
    ir = int(indices[np.argmin(right_delta)])
    return il, ir


def detect_islands(Ay, x, z, B0, de, topology_max_mode=20,
                   relative_threshold=0.12, absolute_threshold=1e-3,
                   min_separation_de=2.0):
    """Detect robust midplane O/X points and return an island catalog.

    Returns a dict with filtered midplane flux, accepted islands and X indices.
    Each island flux is min(A_O-A_Xleft, A_O-A_Xright), normalized by B0*de.
    """
    j0 = int(np.argmin(np.abs(z)))
    amid = Ay[:, j0] - np.mean(Ay[:, j0])
    af = lowpass_periodic(amid, topology_max_mode)
    maxima, minima = periodic_extrema(af)
    nx = len(x)
    Lx = (x[-1] - x[0]) + (x[1] - x[0])

    raw = []
    for io in maxima:
        il, ir = previous_next_periodic(minima, int(io), nx)
        if il is None or ir is None:
            continue
        dl = (af[io] - af[il]) / (B0 * de)
        dr = (af[io] - af[ir]) / (B0 * de)
        psi = min(dl, dr)
        if psi > 0:
            raw.append(dict(io=int(io), il=int(il), ir=int(ir),
                            psi=float(psi), dl=float(dl), dr=float(dr)))

    strongest = max((q["psi"] for q in raw), default=0.0)
    threshold = max(float(absolute_threshold), float(relative_threshold) * strongest)
    candidates = [q for q in raw if q["psi"] >= threshold]

    # Greedy de-duplication protects against tiny split maxima in a broad O point.
    candidates.sort(key=lambda q: q["psi"], reverse=True)
    accepted = []
    min_sep = min_separation_de * de
    for q in candidates:
        xo = x[q["io"]]
        if all(periodic_distance(xo, x[r["io"]], Lx) >= min_sep for r in accepted):
            accepted.append(q)
    accepted.sort(key=lambda q: x[q["io"]])

    xset = sorted({q["il"] for q in accepted} | {q["ir"] for q in accepted})
    return dict(j0=j0, amid=amid, filtered=af, maxima=maxima, minima=minima,
                islands=accepted, x_indices=np.asarray(xset, dtype=int),
                threshold=threshold, strongest=strongest)


def spectrum_core(Bz, z, max_mode, core_half_width, B0):
    core = np.abs(z) <= core_half_width
    ft = np.fft.rfft(Bz, axis=0) / Bz.shape[0]
    amps = np.full(max_mode, np.nan)
    for m in range(1, max_mode + 1):
        if m >= ft.shape[0]:
            break
        a = 2.0 * np.abs(ft[m, :]) / B0
        amps[m - 1] = np.sqrt(np.mean(a[core] ** 2))
    return amps


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params()
    vA = P["B0"] / np.sqrt(MU0 * P["n0"] * P["mi"])

    files = sorted(glob.glob(str(Path(args.run_dir)/"diags"/"diag1*")), key=numeric_key)
    if not files:
        raise FileNotFoundError(f"No diag1 plotfiles under {args.run_dir}/diags")

    rows = []
    catalogs = []
    final_pack = None

    print("="*78)
    print("Ishizawa-scale magnetic-island coalescence analysis")
    print("="*78)

    for k, fn in enumerate(files, 1):
        F = load_fields(fn)
        Ey = load_ey(F)
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
        amps = spectrum_core(
            F["Bz"], F["z"], args.max_mode,
            args.core_width_de * P["de"], P["B0"],
        )
        good = np.isfinite(amps) & (amps > 0)
        if np.any(good):
            modes = np.arange(1, args.max_mode + 1, dtype=float)
            pwr = np.where(good, amps**2, 0.0)
            mdom = int(np.nanargmax(amps) + 1)
            mbar = float(np.sum(modes * pwr) / np.sum(pwr))
        else:
            mdom, mbar = -1, np.nan

        Lxde = ((F["x"][-1]-F["x"][0]) + (F["x"][1]-F["x"][0])) / P["de"]
        lambda_bar = Lxde / mbar if np.isfinite(mbar) and mbar > 0 else np.nan

        j0 = det["j0"]
        jy = corrected_jy(F["Bx"], F["Bz"], F["dx"], F["dz"])
        xids = det["x_indices"]
        if len(xids):
            eyx = np.abs(Ey[xids, j0]) / (vA * P["B0"])
            jyx = np.abs(jy[xids, j0]) / P["J0"]
            max_eyx = float(np.nanmax(eyx))
            mean_eyx = float(np.nanmean(eyx))
            max_jyx = float(np.nanmax(jyx))
        else:
            max_eyx = mean_eyx = max_jyx = np.nan

        psis = np.asarray([q["psi"] for q in det["islands"]], dtype=float)
        psi_max = float(np.max(psis)) if len(psis) else 0.0
        psi_mean = float(np.mean(psis)) if len(psis) else 0.0
        span = (np.max(det["filtered"])-np.min(det["filtered"]))/(P["B0"]*P["de"])

        step = step_from_path(fn)
        row = dict(step=step, tci=tci, NO=len(det["islands"]), NX=len(xids),
                   mdom=mdom, mbar=mbar, lambda_bar_de=lambda_bar,
                   psi_max=psi_max, psi_mean=psi_mean,
                   max_eyx=max_eyx, mean_eyx=mean_eyx, max_jyx=max_jyx,
                   flux_span=float(span), threshold=float(det["threshold"]))
        rows.append(row)

        for q in det["islands"]:
            catalogs.append(dict(step=step, tci=tci,
                                 xO_de=F["x"][q["io"]]/P["de"],
                                 xXL_de=F["x"][q["il"]]/P["de"],
                                 xXR_de=F["x"][q["ir"]]/P["de"],
                                 psi=q["psi"], dl=q["dl"], dr=q["dr"]))

        final_pack = (F, Ay, jy, det)
        print(f"step={step:6d}  wci*t={tci:7.4f}  O={row['NO']:2d} X={row['NX']:2d}  "
              f"m_dom={mdom:2d}  <m>={mbar:6.3f}  psi_max={psi_max:8.4g}")

    if not rows:
        raise RuntimeError("No plotfiles satisfy --tmin")

    # Save machine-readable history.
    with open("ishizawa_island_coalescence_history.txt", "w") as f:
        f.write("# step omega_ci_t N_O N_X m_dom m_bar lambda_bar_de psi_max psi_mean "
                "max_abs_EyX_over_vAB0 mean_abs_EyX_over_vAB0 max_abs_JyX_over_J0 "
                "deltaAy_span threshold\n")
        for r in rows:
            f.write(f"{r['step']:8d} {r['tci']:.10e} {r['NO']:3d} {r['NX']:3d} {r['mdom']:3d} "
                    f"{r['mbar']:.10e} {r['lambda_bar_de']:.10e} {r['psi_max']:.10e} "
                    f"{r['psi_mean']:.10e} {r['max_eyx']:.10e} {r['mean_eyx']:.10e} "
                    f"{r['max_jyx']:.10e} {r['flux_span']:.10e} {r['threshold']:.10e}\n")

    with open("ishizawa_island_catalog.txt", "w") as f:
        f.write("# step omega_ci_t xO_de xXleft_de xXright_de psi_over_B0de psi_left psi_right\n")
        for q in catalogs:
            f.write(f"{q['step']:8d} {q['tci']:.10e} {q['xO_de']:.8e} {q['xXL_de']:.8e} "
                    f"{q['xXR_de']:.8e} {q['psi']:.10e} {q['dl']:.10e} {q['dr']:.10e}\n")

    t = np.asarray([r["tci"] for r in rows])
    NO = np.asarray([r["NO"] for r in rows], dtype=float)
    NX = np.asarray([r["NX"] for r in rows], dtype=float)
    md = np.asarray([r["mdom"] for r in rows], dtype=float)
    mb = np.asarray([r["mbar"] for r in rows], dtype=float)
    lam = np.asarray([r["lambda_bar_de"] for r in rows], dtype=float)
    pmax = np.asarray([r["psi_max"] for r in rows], dtype=float)
    pmean = np.asarray([r["psi_mean"] for r in rows], dtype=float)
    ey = np.asarray([r["max_eyx"] for r in rows], dtype=float)

    # Island count and Fourier-scale comparison.
    fig, ax = plt.subplots(figsize=(9,5.8))
    ax.step(t, NO, where="mid", marker="o", label=r"accepted $N_O$")
    ax.step(t, NX, where="mid", marker="s", label=r"accepted $N_X$")
    ax.plot(t, md, "o-", label=r"dominant mode $m_{dom}$")
    ax.plot(t, mb, "o-", label=r"spectral centroid $\bar m$")
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel("count / mode number")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig("ishizawa_island_count_vs_modes.png", dpi=200)
    plt.close(fig)

    # Characteristic scale growth.
    fig, ax = plt.subplots(figsize=(8.5,5.5))
    ax.plot(t, lam, "o-", label=r"$L_x/\bar m$")
    with np.errstate(divide="ignore", invalid="ignore"):
        island_spacing = Lxde / NO
    ax.plot(t, island_spacing, "s-", label=r"$L_x/N_O$")
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel(r"characteristic scale / $d_e$")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("ishizawa_island_characteristic_scale.png", dpi=200)
    plt.close(fig)

    # Flux growth and X-point electric field.
    fig, ax1 = plt.subplots(figsize=(9,5.8))
    ax1.semilogy(t, np.maximum(pmax,1e-12), "o-", label=r"max island $\Psi/(B_0d_e)$")
    ax1.semilogy(t, np.maximum(pmean,1e-12), "s-", label=r"mean island $\Psi/(B_0d_e)$")
    ax1.set_xlabel(r"$\omega_{ci}t$")
    ax1.set_ylabel(r"island flux / $B_0d_e$")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(t, ey, "^-", label=r"max $|E_y(X)|/(v_AB_0)$")
    ax2.set_ylabel(r"max $|E_y(X)|/(v_AB_0)$")
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper left")
    fig.tight_layout()
    fig.savefig("ishizawa_island_flux_and_xpoint.png", dpi=200)
    plt.close(fig)

    # Final topology annotated with accepted O and X points.
    F, Ay, jy, det = final_pack
    xde = F["x"] / P["de"]
    zde = F["z"] / P["de"]
    zoom = np.abs(zde) <= args.zoom_de
    X, Z = np.meshgrid(xde, zde, indexing="ij")
    jyn = jy / P["J0"]
    alo, ahi = np.nanmin(Ay[:,zoom]), np.nanmax(Ay[:,zoom])

    fig, ax = plt.subplots(figsize=(11,5.8))
    pcm = ax.pcolormesh(X[:,zoom], Z[:,zoom], jyn[:,zoom], shading="auto")
    if ahi > alo:
        ax.contour(X[:,zoom], Z[:,zoom], Ay[:,zoom], levels=np.linspace(alo,ahi,50),
                   colors="k", linewidths=0.45, alpha=0.8)
    j0 = det["j0"]
    if det["islands"]:
        io = [q["io"] for q in det["islands"]]
        ax.scatter(xde[io], np.full(len(io), zde[j0]), s=70, facecolors="none",
                   edgecolors="tab:orange", linewidths=1.8, label="O points")
    if len(det["x_indices"]):
        ax.scatter(xde[det["x_indices"]], np.full(len(det["x_indices"]), zde[j0]),
                   s=70, marker="x", color="tab:cyan", linewidths=2.0, label="X points")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(r"$J_y/J_0$")
    ax.set_xlabel(r"$x/d_e$")
    ax.set_ylabel(r"$z/d_e$")
    ax.set_title(fr"annotated island topology, $\omega_{{ci}}t={rows[-1]['tci']:.3f}$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("ishizawa_island_final_annotated.png", dpi=200)
    plt.close(fig)

    # Compact summary.
    with open("ishizawa_island_coalescence_summary.txt", "w") as f:
        f.write("Ishizawa-scale magnetic-island coalescence summary\n")
        f.write("=================================================\n")
        f.write(f"frames analysed                  = {len(rows)}\n")
        f.write(f"omega_ci*t range                 = {t[0]:.6f} .. {t[-1]:.6f}\n")
        f.write(f"topology max mode                = {args.topology_max_mode}\n")
        f.write(f"relative island flux threshold   = {args.relative_flux_threshold:.6f}\n")
        f.write(f"absolute island flux threshold   = {args.absolute_flux_threshold:.6e} B0 de\n")
        f.write(f"minimum O-point separation       = {args.min_separation_de:.6f} de\n")
        f.write(f"initial accepted N_O             = {int(NO[0])}\n")
        f.write(f"final accepted N_O               = {int(NO[-1])}\n")
        f.write(f"initial dominant mode            = {int(md[0])}\n")
        f.write(f"final dominant mode              = {int(md[-1])}\n")
        f.write(f"initial spectral centroid <m>    = {mb[0]:.6f}\n")
        f.write(f"final spectral centroid <m>      = {mb[-1]:.6f}\n")
        f.write(f"initial characteristic scale     = {lam[0]:.6f} de\n")
        f.write(f"final characteristic scale       = {lam[-1]:.6f} de\n")
        f.write(f"final max island flux            = {pmax[-1]:.6e} B0 de\n")
        f.write(f"final mean island flux           = {pmean[-1]:.6e} B0 de\n")
        f.write(f"final max |Ey(X)|/(vA B0)        = {ey[-1]:.6e}\n")
        f.write("\nInterpret N_O together with m_dom and <m>; transient small islands may\n")
        f.write("cross the detection threshold during nonlinear coalescence.\n")

    print("Saved ishizawa_island_coalescence_history.txt")
    print("Saved ishizawa_island_catalog.txt")
    print("Saved ishizawa_island_count_vs_modes.png")
    print("Saved ishizawa_island_characteristic_scale.png")
    print("Saved ishizawa_island_flux_and_xpoint.png")
    print("Saved ishizawa_island_final_annotated.png")
    print("Saved ishizawa_island_coalescence_summary.txt")


if __name__ == "__main__":
    main()
