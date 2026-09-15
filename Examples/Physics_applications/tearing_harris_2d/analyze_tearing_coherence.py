#!/usr/bin/env python3
"""
Phase-coherence / parity diagnostics for the long-time no-seed Harris runs.

This script complements analyze_tearing_long.py.  The existing A_m diagnostic
is an RMS of |Bz_hat_m(z)| across the current-sheet core.  That is a useful
measure of spectral power, but it does not require the mode to have a coherent
phase in z.  Random PIC fluctuations can therefore give a non-zero A_m.

For every x-Fourier mode m this script additionally computes:

  coherence C_m = |sum_z Bhat_m| / sum_z |Bhat_m|            (0..1)
  even-parity fraction P_even                              (0..1)
  resonant-surface amplitude 2|Bhat_m(z=0)|/B0
  normalized flux amplitude Psi_m = 2|Ahat_y,m(z=0)|/(B0 L)

Using Bz = d A_y / dx,

  Psi_m = 2|Bhat_z,m(z=0)| / (k_m B0 L).

For the symmetric Harris tearing parity, Bz is expected to be approximately
even in z.  A credible coherent tearing mode should therefore show sustained
spectral amplitude together with appreciable phase coherence and even parity.

Expected layout:
    runs_long/ppc16/diags/diag1*
    runs_long/ppc64/diags/diag1*

Outputs:
    tearing_coherence_map_ppc16.png
    tearing_coherence_map_ppc64.png
    tearing_even_parity_map_ppc16.png
    tearing_even_parity_map_ppc64.png
    tearing_flux_history_ppc16.png
    tearing_flux_history_ppc64.png
    tearing_center_history_ppc16.png
    tearing_center_history_ppc64.png
    tearing_coherence_summary.txt
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs_long")
    p.add_argument("--ppc", nargs="+", type=int, default=[16, 64])
    p.add_argument("--max-mode", type=int, default=20)
    p.add_argument("--core-width", type=float, default=1.0)
    p.add_argument("--plot-low-modes", type=int, default=6)
    p.add_argument("--late-tmin", type=float, default=1.0,
                   help="Start time for late-time summary averages, in t/tau_A")
    return p.parse_args()


def numeric_key(path):
    name = os.path.basename(path.rstrip("/"))
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else name


def as_xz(arr, nx, nz):
    a = np.asarray(arr).squeeze()
    if a.shape == (nx, nz):
        return a
    if a.shape == (nz, nx):
        return a.T
    raise RuntimeError(f"Unexpected Bz shape {a.shape}; expected {(nx, nz)} or {(nz, nx)}")


def load_bz(path):
    ds = yt.load(path)
    nx = int(ds.domain_dimensions[0])
    nz = int(ds.domain_dimensions[1])

    xlo = ds.domain_left_edge[0].to_value("m")
    xhi = ds.domain_right_edge[0].to_value("m")
    zlo = ds.domain_left_edge[1].to_value("m")
    zhi = ds.domain_right_edge[1].to_value("m")

    dx = (xhi - xlo) / nx
    dz = (zhi - zlo) / nz
    x = xlo + (np.arange(nx) + 0.5) * dx
    z = zlo + (np.arange(nz) + 0.5) * dz

    grid = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
                            dims=ds.domain_dimensions)
    bz = as_xz(grid["boxlib", "Bz"].to_ndarray(), nx, nz)
    return ds, x, z, bz


def complex_interp_zero(z, h):
    return np.interp(0.0, z, h.real) + 1j*np.interp(0.0, z, h.imag)


def physical_params():
    qe = 1.602176634e-19
    me = 9.1093837139e-31
    eps0 = 8.8541878188e-12
    mu0 = 1.2566370612685e-6
    c = 299792458.0

    n0 = 1.0e19
    mi = 800.0 * me
    uth_e = 0.1

    wpe = np.sqrt(n0 * qe**2 / (eps0 * me))
    de = c / wpe
    L = de
    Te = uth_e**2 * me * c**2
    Ti = 0.1 * Te
    B0 = np.sqrt(2.0 * mu0 * n0 * (Te + Ti))
    vA = B0 / np.sqrt(mu0 * n0 * mi)
    tauA = L / vA
    Lx = 16.0 * L

    return {"B0": B0, "L": L, "Lx": Lx, "tauA": tauA}


def analyze_snapshot(bz, x, z, params, args):
    B0 = params["B0"]
    L = params["L"]
    Lx = params["Lx"]

    fft = np.fft.rfft(bz, axis=0) / bz.shape[0]
    core = np.abs(z) <= args.core_width * L
    zc = z[core]

    amp = np.full(args.max_mode, np.nan)
    coherence = np.full(args.max_mode, np.nan)
    even_fraction = np.full(args.max_mode, np.nan)
    center_amp = np.full(args.max_mode, np.nan)
    flux = np.full(args.max_mode, np.nan)

    for m in range(1, args.max_mode + 1):
        h_all = fft[m, :]
        h = h_all[core]

        amp[m-1] = 2.0*np.sqrt(np.mean(np.abs(h)**2))/B0

        denom = np.sum(np.abs(h))
        coherence[m-1] = np.abs(np.sum(h))/denom if denom > 0.0 else np.nan

        # The grid and core mask are symmetric, so reverse indexing pairs z and -z.
        h_mirror = h[::-1]
        h_even = 0.5*(h + h_mirror)
        h_odd = 0.5*(h - h_mirror)
        ee = np.sum(np.abs(h_even)**2)
        eo = np.sum(np.abs(h_odd)**2)
        even_fraction[m-1] = ee/(ee + eo) if (ee + eo) > 0.0 else np.nan

        h0 = complex_interp_zero(z, h_all)
        center_amp[m-1] = 2.0*np.abs(h0)/B0

        km = 2.0*np.pi*m/Lx
        flux[m-1] = 2.0*np.abs(h0)/(km*B0*L)

    return amp, coherence, even_fraction, center_amp, flux


def analyze_case(plotfiles, params, args):
    t = []
    A = []
    C = []
    P = []
    A0 = []
    PSI = []

    for i, pf in enumerate(plotfiles, 1):
        ds, x, z, bz = load_bz(pf)
        values = analyze_snapshot(bz, x, z, params, args)

        t.append(ds.current_time.to_value("s")/params["tauA"])
        A.append(values[0])
        C.append(values[1])
        P.append(values[2])
        A0.append(values[3])
        PSI.append(values[4])

        if i % 25 == 0 or i == len(plotfiles):
            print(f"  analyzed {i}/{len(plotfiles)}, t/tau_A={t[-1]:.3f}")

    return {
        "t": np.asarray(t),
        "A": np.asarray(A),
        "C": np.asarray(C),
        "P": np.asarray(P),
        "A0": np.asarray(A0),
        "PSI": np.asarray(PSI),
    }


def plot_map(t, data, ppc, ylabel, filename, vmin=0.0, vmax=1.0):
    modes = np.arange(1, data.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.pcolormesh(t, modes, data.T, shading="auto", vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(ylabel)
    ax.set_xlabel(r"$t/\tau_A$")
    ax.set_ylabel("x-Fourier mode m")
    ax.set_yticks(modes)
    ax.set_title(f"{ppc} PPC")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    params = physical_params()
    results = {}

    print("="*78)
    print("Harris tearing phase-coherence / parity analysis")
    print("="*78)

    for ppc in args.ppc:
        pattern = str(Path(args.runs_dir)/f"ppc{ppc}"/"diags"/"diag1*")
        files = sorted(glob.glob(pattern), key=numeric_key)
        if not files:
            raise FileNotFoundError(f"No plotfiles for PPC={ppc}: {pattern}")

        print(f"PPC={ppc}: {len(files)} plotfiles")
        r = analyze_case(files, params, args)
        results[ppc] = r

        plot_map(r["t"], r["C"], ppc, r"phase coherence $C_m$",
                 f"tearing_coherence_map_ppc{ppc}.png")
        plot_map(r["t"], r["P"], ppc, r"even-parity fraction",
                 f"tearing_even_parity_map_ppc{ppc}.png")

        nplot = min(args.plot_low_modes, args.max_mode)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for m in range(1, nplot + 1):
            ax.semilogy(r["t"], r["PSI"][:, m-1], "o-", ms=3, label=f"m={m}")
        ax.set_xlabel(r"$t/\tau_A$")
        ax.set_ylabel(r"$\Psi_m=2|\hat A_{y,m}(0)|/(B_0L)$")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(f"tearing_flux_history_ppc{ppc}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for m in range(1, nplot + 1):
            ax.semilogy(r["t"], r["A0"][:, m-1], "o-", ms=3, label=f"m={m}")
        ax.set_xlabel(r"$t/\tau_A$")
        ax.set_ylabel(r"resonant-surface $2|\hat B_{z,m}(0)|/B_0$")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(f"tearing_center_history_ppc{ppc}.png", dpi=200)
        plt.close(fig)

    with open("tearing_coherence_summary.txt", "w") as f:
        f.write("Harris tearing phase-coherence / parity summary\n")
        f.write("==============================================\n\n")
        f.write(f"late-time averaging begins at t/tau_A = {args.late_tmin:g}\n")
        f.write(f"core = |z| <= {args.core_width:g} L\n\n")

        for ppc in args.ppc:
            r = results[ppc]
            late = r["t"] >= args.late_tmin
            f.write(f"PPC={ppc}\n")
            f.write("mode  <A_rms>       <coherence>  <even_frac>   <A_center>    <Psi>\n")
            for m in range(1, args.max_mode + 1):
                f.write(
                    f"{m:4d}  "
                    f"{np.nanmean(r['A'][late,m-1]):.8e}  "
                    f"{np.nanmean(r['C'][late,m-1]):.6f}  "
                    f"{np.nanmean(r['P'][late,m-1]):.6f}  "
                    f"{np.nanmean(r['A0'][late,m-1]):.8e}  "
                    f"{np.nanmean(r['PSI'][late,m-1]):.8e}\n"
                )
            f.write("\n")

    print("Saved coherence/parity maps, low-mode flux histories, and")
    print("  tearing_coherence_summary.txt")


if __name__ == "__main__":
    main()
