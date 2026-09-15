#!/usr/bin/env python3
"""
Long-time spectral analysis for the 16/64 PPC no-seed Harris tearing runs.

Expected layout:
    runs_long/ppc16/diags/diag1*
    runs_long/ppc64/diags/diag1*

Outputs:
    tearing_long_m1_history.png
    tearing_long_spectrum_map_ppc16.png
    tearing_long_spectrum_map_ppc64.png
    tearing_long_gamma_vs_m.png
    tearing_long_dominant_mode.png
    tearing_long_growth_summary.txt
    tearing_long_mode_history_ppc16.txt
    tearing_long_mode_history_ppc64.txt

The script does not assume m=1 is the fastest-growing mode.  It tracks
m=1..max_mode and screens for sustained positive exponential windows in
ln(A_m) versus t/tau_A.
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
    p.add_argument("--fit-tmin", type=float, default=0.5,
                   help="Do not fit before this t/tau_A")
    p.add_argument("--fit-min-points", type=int, default=12)
    p.add_argument("--fit-min-span", type=float, default=0.35,
                   help="Minimum accepted fit-window span in t/tau_A")
    p.add_argument("--fit-min-r2", type=float, default=0.95)
    p.add_argument("--fit-min-growth", type=float, default=2.0,
                   help="Minimum fitted amplitude growth across a window")
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

    grid = ds.covering_grid(
        level=0,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions,
    )
    bz = as_xz(grid["boxlib", "Bz"].to_ndarray(), nx, nz)
    return ds, x, z, bz


def mode_spectrum_core(bz, x, z, max_mode, L, core_width, B0):
    """Return physical sinusoidal amplitudes A_m/B0 in |z|<=core_width*L."""
    core = np.abs(z) <= core_width * L
    fft = np.fft.rfft(bz, axis=0) / bz.shape[0]

    amps = np.empty(max_mode, dtype=float)
    for m in range(1, max_mode + 1):
        local_amp = 2.0 * np.abs(fft[m, :]) / B0
        amps[m - 1] = np.sqrt(np.mean(local_amp[core] ** 2))
    return amps


def fit_line_r2(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else -np.inf
    return slope, intercept, r2


def find_growth_window(t, amp, args):
    valid = np.isfinite(t) & np.isfinite(amp) & (amp > 0.0) & (t >= args.fit_tmin)
    tv = t[valid]
    av = amp[valid]
    idx = np.nonzero(valid)[0]

    best = None
    n = len(tv)

    for i in range(n):
        for j in range(i + args.fit_min_points, n + 1):
            x = tv[i:j]
            a = av[i:j]
            span = x[-1] - x[0]
            if span < args.fit_min_span:
                continue

            slope, intercept, r2 = fit_line_r2(x, np.log(a))
            if slope <= 0.0:
                continue

            growth = float(np.exp(slope * span))
            if r2 < args.fit_min_r2 or growth < args.fit_min_growth:
                continue

            cand = {
                "i": int(idx[i]),
                "j": int(idx[j - 1]) + 1,
                "t0": float(x[0]),
                "t1": float(x[-1]),
                "span": float(span),
                "gamma_tauA": float(slope),
                "intercept": float(intercept),
                "r2": float(r2),
                "growth": growth,
            }

            if best is None:
                best = cand
            else:
                # Prefer longer accepted windows first, then better R^2.
                score = (cand["span"], cand["r2"])
                best_score = (best["span"], best["r2"])
                if score > best_score:
                    best = cand

    return best


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

    return {
        "B0": B0,
        "L": L,
        "tauA": tauA,
        "wpe": wpe,
    }


def analyze_case(plotfiles, params, args):
    times = []
    spectra = []
    bz_core_rms = []

    for ipf, pf in enumerate(plotfiles, start=1):
        ds, x, z, bz = load_bz(pf)
        core = np.abs(z) <= args.core_width * params["L"]

        tnorm = ds.current_time.to_value("s") / params["tauA"]
        spec = mode_spectrum_core(
            bz, x, z, args.max_mode,
            params["L"], args.core_width, params["B0"]
        )

        times.append(tnorm)
        spectra.append(spec)
        bz_core_rms.append(np.sqrt(np.mean(bz[:, core] ** 2)) / params["B0"])

        if ipf % 25 == 0 or ipf == len(plotfiles):
            print(f"  analyzed {ipf}/{len(plotfiles)} plotfiles, t/tau_A={tnorm:.3f}")

    return {
        "t": np.asarray(times),
        "A": np.asarray(spectra),
        "bz_core_rms": np.asarray(bz_core_rms),
    }


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    params = physical_params()

    results = {}
    fits = {}

    print("=" * 78)
    print("Long-time Harris tearing spectral analysis")
    print("=" * 78)

    for ppc in args.ppc:
        pattern = str(Path(args.runs_dir) / f"ppc{ppc}" / "diags" / "diag1*")
        plotfiles = sorted(glob.glob(pattern), key=numeric_key)
        if not plotfiles:
            raise FileNotFoundError(f"No plotfiles found for PPC={ppc}: {pattern}")

        print(f"PPC={ppc}: {len(plotfiles)} plotfiles")
        r = analyze_case(plotfiles, params, args)
        results[ppc] = r

        case_fits = []
        for imode in range(args.max_mode):
            case_fits.append(find_growth_window(r["t"], r["A"][:, imode], args))
        fits[ppc] = case_fits

        table = np.column_stack([r["t"], r["bz_core_rms"], r["A"]])
        header = "t/tauA Bz_core_rms " + " ".join(
            f"A_m{m}" for m in range(1, args.max_mode + 1)
        )
        np.savetxt(f"tearing_long_mode_history_ppc{ppc}.txt", table, header=header)

    # ------------------------------------------------------------------
    # m=1 comparison
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for ppc in args.ppc:
        r = results[ppc]
        positive = r["A"][:, 0] > 0.0
        ax.semilogy(r["t"][positive], r["A"][positive, 0], "o-", ms=3,
                    label=f"{ppc} PPC")
        fit = fits[ppc][0]
        if fit is not None:
            x = r["t"][fit["i"]:fit["j"]]
            y = np.exp(fit["intercept"] + fit["gamma_tauA"] * x)
            ax.semilogy(x, y, "--", lw=2,
                        label=(f"{ppc} PPC fit: gamma*tauA={fit['gamma_tauA']:.3g}, "
                               f"R2={fit['r2']:.3f}"))
    ax.set_xlabel(r"$t/\tau_A$")
    ax.set_ylabel(r"core $m=1$ amplitude / $B_0$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("tearing_long_m1_history.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # m-t spectrum maps and dominant mode histories
    # ------------------------------------------------------------------
    fig_dom, ax_dom = plt.subplots(figsize=(8, 5))

    for ppc in args.ppc:
        r = results[ppc]
        modes = np.arange(1, args.max_mode + 1)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        floor = max(np.nanmin(r["A"][r["A"] > 0.0]), 1.0e-12)
        img = ax.pcolormesh(
            r["t"], modes,
            np.log10(np.maximum(r["A"].T, floor)),
            shading="auto",
        )
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label(r"$\log_{10}(A_m/B_0)$")
        ax.set_xlabel(r"$t/\tau_A$")
        ax.set_ylabel("x-Fourier mode m")
        ax.set_yticks(modes)
        ax.set_title(f"{ppc} PPC")
        fig.tight_layout()
        fig.savefig(f"tearing_long_spectrum_map_ppc{ppc}.png", dpi=200)
        plt.close(fig)

        dominant = np.argmax(r["A"], axis=1) + 1
        ax_dom.plot(r["t"], dominant, "o-", ms=3, label=f"{ppc} PPC")

    ax_dom.set_xlabel(r"$t/\tau_A$")
    ax_dom.set_ylabel("instantaneous dominant x-Fourier mode")
    ax_dom.set_ylim(0.5, args.max_mode + 0.5)
    ax_dom.set_yticks(np.arange(1, args.max_mode + 1))
    ax_dom.grid(alpha=0.25)
    ax_dom.legend()
    fig_dom.tight_layout()
    fig_dom.savefig("tearing_long_dominant_mode.png", dpi=200)
    plt.close(fig_dom)

    # ------------------------------------------------------------------
    # gamma(m) comparison
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    modes = np.arange(1, args.max_mode + 1)
    for ppc in args.ppc:
        gamma = np.full(args.max_mode, np.nan)
        for i, fit in enumerate(fits[ppc]):
            if fit is not None:
                gamma[i] = fit["gamma_tauA"]
        ax.plot(modes, gamma, "o-", label=f"{ppc} PPC")
    ax.axhline(0.0, lw=0.8)
    ax.set_xlabel("x-Fourier mode m")
    ax.set_ylabel(r"accepted $\gamma_m\tau_A$")
    ax.set_xticks(modes)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("tearing_long_gamma_vs_m.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------
    with open("tearing_long_growth_summary.txt", "w") as f:
        f.write("Long-time Harris tearing growth screening\n")
        f.write("========================================\n\n")
        f.write(f"fit tmin      = {args.fit_tmin:g} tau_A\n")
        f.write(f"min points    = {args.fit_min_points}\n")
        f.write(f"min span      = {args.fit_min_span:g} tau_A\n")
        f.write(f"min R2        = {args.fit_min_r2:g}\n")
        f.write(f"min growth    = {args.fit_min_growth:g}\n\n")

        for ppc in args.ppc:
            f.write(f"PPC={ppc}\n")
            f.write("mode   gamma*tauA       R2       growth     t0       t1\n")
            for m, fit in enumerate(fits[ppc], start=1):
                if fit is None:
                    f.write(f"{m:4d}   no accepted window\n")
                else:
                    f.write(
                        f"{m:4d}   {fit['gamma_tauA']:.8e}  {fit['r2']:.6f}  "
                        f"{fit['growth']:.5f}  {fit['t0']:.5f}  {fit['t1']:.5f}\n"
                    )
            f.write("\n")

    print()
    print("Saved:")
    print("  tearing_long_m1_history.png")
    for ppc in args.ppc:
        print(f"  tearing_long_spectrum_map_ppc{ppc}.png")
    print("  tearing_long_gamma_vs_m.png")
    print("  tearing_long_dominant_mode.png")
    print("  tearing_long_growth_summary.txt")


if __name__ == "__main__":
    main()
