#!/usr/bin/env python3
"""
Analyze the no-seed 4/16/64 PPC Harris-sheet tearing convergence runs.

The main purpose is to separate finite-particle PIC noise from genuine
noise-seeded m=1 tearing-mode selection.

Expected directory layout (created by run_ppc_convergence.sh):

    runs/ppc4/diags/diag1*
    runs/ppc16/diags/diag1*
    runs/ppc64/diags/diag1*

Outputs:
    ppc_convergence_summary.txt
    ppc_convergence_summary.csv
    ppc_convergence_noise_scaling.png
    ppc_convergence_m1_history.png
    ppc_convergence_m1_fraction.png
    ppc_convergence_spectrum_early.png
    ppc_convergence_spectrum_final.png

Interpretation guide:
  * Pure PIC shot noise should approximately scale as Nppc^(-1/2).
  * If higher-PPC cases begin from lower noise floors but later exhibit the
    same slope in ln(A_m1) versus time (with a delayed onset), that is evidence
    for physical noise-seeded tearing.
  * If all magnetic fluctuations keep scaling with Nppc^(-1/2) and no common
    exponential m=1 slope develops, the signal is still noise-dominated.
"""

import argparse
import csv
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--ppc", nargs="+", type=int, default=[4, 16, 64])
    p.add_argument("--noise-tmin", type=float, default=0.02,
                   help="Early-noise window lower bound in t/tau_A")
    p.add_argument("--noise-tmax", type=float, default=0.10,
                   help="Early-noise window upper bound in t/tau_A")
    p.add_argument("--core-width", type=float, default=1.0,
                   help="Core half-width in L: |z| <= core_width*L")
    p.add_argument("--max-mode", type=int, default=20)
    p.add_argument("--early-spectrum-time", type=float, default=0.05,
                   help="Target early spectrum time in t/tau_A")
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


def m_mode_core_amplitude(bz, x, z, m, Lx, L, core_width, B0):
    k = 2.0 * np.pi * m / Lx
    phase = np.exp(-1j * k * x[:, None])
    hat_z = np.mean(bz * phase, axis=0)
    local_amp = 2.0 * np.abs(hat_z) / B0
    core = np.abs(z) <= core_width * L
    return np.sqrt(np.mean(local_amp[core] ** 2))


def analyze_case(plotfiles, params, core_width, max_mode):
    B0 = params["B0"]
    L = params["L"]
    Lx = params["Lx"]
    tauA = params["tauA"]

    times = []
    tnorm = []
    bz_rms = []
    bz_core_rms = []
    m1_core = []
    m1_fraction = []

    snapshots = []

    for pf in plotfiles:
        ds, x, z, bz = load_bz(pf)
        core = np.abs(z) <= core_width * L

        brms = np.sqrt(np.mean(bz**2)) / B0
        bcore = np.sqrt(np.mean(bz[:, core]**2)) / B0
        a1 = m_mode_core_amplitude(bz, x, z, 1, Lx, L, core_width, B0)

        m1_rms = a1 / np.sqrt(2.0)
        frac = np.nan if bcore == 0.0 else np.clip((m1_rms / bcore)**2, 0.0, 1.0)

        t = ds.current_time.to_value("s")
        times.append(t)
        tnorm.append(t / tauA)
        bz_rms.append(brms)
        bz_core_rms.append(bcore)
        m1_core.append(a1)
        m1_fraction.append(frac)
        snapshots.append((t / tauA, x, z, bz))

    return {
        "times": np.asarray(times),
        "tnorm": np.asarray(tnorm),
        "bz_rms": np.asarray(bz_rms),
        "bz_core_rms": np.asarray(bz_core_rms),
        "m1_core": np.asarray(m1_core),
        "m1_fraction": np.asarray(m1_fraction),
        "snapshots": snapshots,
    }


def spectrum_from_snapshot(snapshot, params, core_width, max_mode):
    _, x, z, bz = snapshot
    amps = []
    for m in range(1, max_mode + 1):
        amps.append(
            m_mode_core_amplitude(
                bz, x, z, m,
                params["Lx"], params["L"], core_width, params["B0"]
            )
        )
    return np.arange(1, max_mode + 1), np.asarray(amps)


def nearest_snapshot(snapshots, target_tnorm):
    idx = int(np.argmin([abs(s[0] - target_tnorm) for s in snapshots]))
    return snapshots[idx]


def fit_power_law(ppc, noise):
    ppc = np.asarray(ppc, dtype=float)
    noise = np.asarray(noise, dtype=float)
    valid = np.isfinite(noise) & (noise > 0)
    if np.count_nonzero(valid) < 2:
        return np.nan, np.nan

    x = np.log(ppc[valid])
    y = np.log(noise[valid])
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)

    # Same physical normalization as the input deck.
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

    params = {"B0": B0, "L": L, "Lx": Lx, "tauA": tauA}

    results = {}
    summary_rows = []

    print("=" * 78)
    print("PPC convergence analysis")
    print("=" * 78)
    print(f"Early-noise window: {args.noise_tmin:g} <= t/tau_A <= {args.noise_tmax:g}")
    print(f"Core: |z| <= {args.core_width:g} L")

    for ppc in args.ppc:
        pattern = str(Path(args.runs_dir) / f"ppc{ppc}" / "diags" / "diag1*")
        plotfiles = sorted(glob.glob(pattern), key=numeric_key)
        if not plotfiles:
            raise FileNotFoundError(f"No plotfiles found for PPC={ppc}: {pattern}")

        r = analyze_case(plotfiles, params, args.core_width, args.max_mode)
        results[ppc] = r

        early = (
            (r["tnorm"] >= args.noise_tmin)
            & (r["tnorm"] <= args.noise_tmax)
        )
        if not np.any(early):
            raise RuntimeError(
                f"PPC={ppc}: no diagnostic samples in requested early-noise window"
            )

        noise_box = float(np.median(r["bz_rms"][early]))
        noise_core = float(np.median(r["bz_core_rms"][early]))
        m1_early = float(np.median(r["m1_core"][early]))

        summary_rows.append({
            "ppc": ppc,
            "noise_box": noise_box,
            "noise_core": noise_core,
            "m1_early": m1_early,
            "final_t_tauA": float(r["tnorm"][-1]),
            "final_bz_rms": float(r["bz_rms"][-1]),
            "final_bz_core_rms": float(r["bz_core_rms"][-1]),
            "final_m1_core": float(r["m1_core"][-1]),
            "final_m1_fraction": float(r["m1_fraction"][-1]),
        })

        print(
            f"PPC={ppc:3d}: early core Bz_rms/B0={noise_core:.6e}, "
            f"early m1={m1_early:.6e}, final m1={r['m1_core'][-1]:.6e}"
        )

    ppc_arr = np.asarray([row["ppc"] for row in summary_rows], dtype=float)
    noise_core_arr = np.asarray([row["noise_core"] for row in summary_rows])
    noise_box_arr = np.asarray([row["noise_box"] for row in summary_rows])

    alpha_core, intercept_core = fit_power_law(ppc_arr, noise_core_arr)
    alpha_box, intercept_box = fit_power_law(ppc_arr, noise_box_arr)

    print()
    print(f"Core-noise power-law exponent alpha = {alpha_core:.6f}")
    print(f"Box-noise  power-law exponent alpha = {alpha_box:.6f}")
    print("Pure finite-particle shot-noise expectation: alpha = -0.5")

    # ------------------------------------------------------------------
    # Summary text / CSV
    # ------------------------------------------------------------------
    with open("ppc_convergence_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open("ppc_convergence_summary.txt", "w") as f:
        f.write("Harris tearing PPC convergence summary\n")
        f.write("=====================================\n\n")
        f.write(
            f"Noise window: {args.noise_tmin:g} <= t/tau_A <= {args.noise_tmax:g}\n"
        )
        f.write(f"Core: |z| <= {args.core_width:g} L\n\n")
        f.write("PPC   early_Bz_rms   early_core_Bz_rms   early_m1   final_m1   final_m1_fraction\n")
        for row in summary_rows:
            f.write(
                f"{row['ppc']:3d}  {row['noise_box']:.8e}  {row['noise_core']:.8e}  "
                f"{row['m1_early']:.8e}  {row['final_m1_core']:.8e}  "
                f"{row['final_m1_fraction']:.8e}\n"
            )
        f.write("\n")
        f.write(f"core noise fit: noise ~ PPC^({alpha_core:.8f})\n")
        f.write(f"box  noise fit: noise ~ PPC^({alpha_box:.8f})\n")
        f.write("shot-noise reference exponent: -0.5\n")

    # ------------------------------------------------------------------
    # Plot: early noise scaling
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ppc_arr, noise_core_arr, "o-", label=r"measured core $B_{z,\rm rms}$")
    ax.loglog(ppc_arr, noise_box_arr, "s-", label=r"measured box $B_{z,\rm rms}$")

    ref = noise_core_arr[0] * np.sqrt(ppc_arr[0] / ppc_arr)
    ax.loglog(ppc_arr, ref, "--", label=r"$N_{\rm ppc}^{-1/2}$ reference")

    if np.isfinite(alpha_core):
        fitline = np.exp(intercept_core) * ppc_arr**alpha_core
        ax.loglog(ppc_arr, fitline, ":", label=rf"core fit: $\alpha={alpha_core:.3f}$")

    ax.set_xlabel("PPC per species")
    ax.set_ylabel(r"early $B_z/B_0$ noise level")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("ppc_convergence_noise_scaling.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot: m=1 history
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for ppc in args.ppc:
        r = results[ppc]
        # Avoid log(0) by plotting only positive samples.
        positive = r["m1_core"] > 0.0
        ax.semilogy(
            r["tnorm"][positive],
            r["m1_core"][positive],
            marker="o",
            ms=3,
            label=f"{ppc} PPC",
        )
    ax.set_xlabel(r"$t/\tau_A$")
    ax.set_ylabel(r"core $m=1$ amplitude")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("ppc_convergence_m1_history.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot: m=1 magnetic-energy fraction
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for ppc in args.ppc:
        r = results[ppc]
        ax.plot(
            r["tnorm"],
            r["m1_fraction"],
            marker="o",
            ms=3,
            label=f"{ppc} PPC",
        )
    ax.set_xlabel(r"$t/\tau_A$")
    ax.set_ylabel(r"core $m=1$ fraction of $B_z$ perturbation energy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("ppc_convergence_m1_fraction.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Spectra: early and final
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for ppc in args.ppc:
        snap = nearest_snapshot(results[ppc]["snapshots"], args.early_spectrum_time)
        modes, amp = spectrum_from_snapshot(snap, params, args.core_width, args.max_mode)
        ax.semilogy(modes, amp, marker="o", ms=3, label=f"{ppc} PPC, t/tauA={snap[0]:.3f}")
    ax.set_xlabel("x-Fourier mode m")
    ax.set_ylabel(r"core mode amplitude / $B_0$")
    ax.set_xticks(np.arange(1, args.max_mode + 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("ppc_convergence_spectrum_early.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for ppc in args.ppc:
        snap = results[ppc]["snapshots"][-1]
        modes, amp = spectrum_from_snapshot(snap, params, args.core_width, args.max_mode)
        ax.semilogy(modes, amp, marker="o", ms=3, label=f"{ppc} PPC, t/tauA={snap[0]:.3f}")
    ax.set_xlabel("x-Fourier mode m")
    ax.set_ylabel(r"core mode amplitude / $B_0$")
    ax.set_xticks(np.arange(1, args.max_mode + 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("ppc_convergence_spectrum_final.png", dpi=200)
    plt.close(fig)

    print()
    print("Saved:")
    print("  ppc_convergence_summary.txt")
    print("  ppc_convergence_summary.csv")
    print("  ppc_convergence_noise_scaling.png")
    print("  ppc_convergence_m1_history.png")
    print("  ppc_convergence_m1_fraction.png")
    print("  ppc_convergence_spectrum_early.png")
    print("  ppc_convergence_spectrum_final.png")


if __name__ == "__main__":
    main()
