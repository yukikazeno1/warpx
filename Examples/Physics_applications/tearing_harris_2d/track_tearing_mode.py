#!/usr/bin/env python3
"""
Single-run tearing-mode diagnostics for the 2D Harris-sheet pilot.

This script is phase-independent in x and also computes a pure x-Fourier m=1
measure that does not assume a fixed tearing eigenfunction in z.
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import yt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="diags/diag1*")
    p.add_argument("--seed-eps", type=float, default=0.0,
                   help="Initial imposed deltaB/B0, used only for the analytic reference curve")
    p.add_argument("--core-width", type=float, default=1.0)
    p.add_argument("--out-prefix", default="tearing_mode")
    return p.parse_args()


def numeric_key(path):
    name = os.path.basename(path.rstrip("/"))
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else name


def as_xz(a, nx, nz):
    a = np.asarray(a).squeeze()
    if a.shape == (nx, nz):
        return a
    if a.shape == (nz, nx):
        return a.T
    raise RuntimeError(f"Unexpected field shape {a.shape}")


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)

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
    Lz = 8.0 * L
    kx = 2.0 * np.pi / Lx
    kz = np.pi / Lz

    files = sorted(glob.glob(args.pattern), key=numeric_key)
    if not files:
        raise FileNotFoundError(f"No plotfiles matched {args.pattern}")

    times = []
    As_hist = []
    Ac_hist = []
    template_amp = []
    phase_hist = []
    center_amp = []
    core_amp = []
    bz_rms = []
    bz_core_rms = []
    residual_rms = []
    m1_fraction = []

    first_profile = None
    last_profile = None
    z_first = None
    z_last = None

    for i, fn in enumerate(files):
        ds = yt.load(fn)
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

        grid = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
        bz = as_xz(grid["boxlib", "Bz"].to_ndarray(), nx, nz)
        X, Z = np.meshgrid(x, z, indexing="ij")

        # Original seed template, with both x quadratures.
        ts = -np.sin(kx * X) * np.cos(kz * Z)
        tc =  np.cos(kx * X) * np.cos(kz * Z)
        As = np.sum(bz * ts) / np.sum(ts**2) / B0
        Ac = np.sum(bz * tc) / np.sum(tc**2) / B0
        Atemp = np.sqrt(As**2 + Ac**2)
        phase = np.arctan2(Ac, As)
        best_template = B0 * (As * ts + Ac * tc)
        resid = np.sqrt(np.mean((bz - best_template)**2)) / B0

        # Pure x Fourier m=1, no z-eigenfunction assumption.
        pf = np.exp(-1j * kx * x[:, None])
        hat1 = np.mean(bz * pf, axis=0)
        A1z = 2.0 * np.abs(hat1) / B0

        # Interpolate complex coefficient to z=0.
        center_hat = np.interp(0.0, z, np.real(hat1)) + 1j*np.interp(0.0, z, np.imag(hat1))
        Acenter = 2.0 * np.abs(center_hat) / B0

        core = np.abs(z) <= args.core_width * L
        Acore = np.sqrt(np.mean(A1z[core]**2))
        Brms = np.sqrt(np.mean(bz**2)) / B0
        Bcore = np.sqrt(np.mean(bz[:, core]**2)) / B0
        frac = np.nan if Bcore == 0.0 else np.clip(((Acore/np.sqrt(2.0))/Bcore)**2, 0.0, 1.0)

        times.append(ds.current_time.to_value("s"))
        As_hist.append(As)
        Ac_hist.append(Ac)
        template_amp.append(Atemp)
        phase_hist.append(phase)
        center_amp.append(Acenter)
        core_amp.append(Acore)
        bz_rms.append(Brms)
        bz_core_rms.append(Bcore)
        residual_rms.append(resid)
        m1_fraction.append(frac)

        if i == 0:
            z_first = z.copy()
            first_profile = A1z.copy()
        if i == len(files) - 1:
            z_last = z.copy()
            last_profile = A1z.copy()

    times = np.asarray(times)
    tnorm = times / tauA
    As_hist = np.asarray(As_hist)
    Ac_hist = np.asarray(Ac_hist)
    template_amp = np.asarray(template_amp)
    phase_hist = np.unwrap(np.asarray(phase_hist))
    center_amp = np.asarray(center_amp)
    core_amp = np.asarray(core_amp)
    bz_rms = np.asarray(bz_rms)
    bz_core_rms = np.asarray(bz_core_rms)
    residual_rms = np.asarray(residual_rms)
    m1_fraction = np.asarray(m1_fraction)

    np.savetxt(
        f"{args.out_prefix}_history.txt",
        np.column_stack([
            times, wpe*times, tnorm, As_hist, Ac_hist, template_amp,
            phase_hist, center_amp, core_amp, bz_rms, bz_core_rms,
            residual_rms, m1_fraction,
        ]),
        header=(
            "t[s] wpe*t t/tauA As Ac A_template phase_unwrapped "
            "A_m1_center A_m1_core Bz_rms Bz_core_rms residual_rms m1_fraction"
        ),
    )

    print("=" * 78)
    print("Initial/final diagnostic summary")
    print("=" * 78)
    print(f"A_template(0)       = {template_amp[0]:.6e}")
    print(f"A_template(final)   = {template_amp[-1]:.6e}")
    print(f"A_m1_center(0)      = {center_amp[0]:.6e}")
    print(f"A_m1_center(final)  = {center_amp[-1]:.6e}")
    print(f"A_m1_core(0)        = {core_amp[0]:.6e}")
    print(f"A_m1_core(final)    = {core_amp[-1]:.6e}")
    print(f"Bz_rms(0)           = {bz_rms[0]:.6e}")
    print(f"Bz_rms(final)       = {bz_rms[-1]:.6e}")
    print(f"Bz_core_rms(0)      = {bz_core_rms[0]:.6e}")
    print(f"Bz_core_rms(final)  = {bz_core_rms[-1]:.6e}")
    print(f"m1 core energy frac = {m1_fraction[0]} -> {m1_fraction[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for y, label, marker in [
        (template_amp, r"template $m=1$", "o"),
        (core_amp, rf"x-Fourier $m=1$, $|z|\leq{args.core_width:g}L$", "s"),
        (center_amp, r"x-Fourier $m=1$ at $z=0$", "^"),
    ]:
        good = y > 0.0
        ax.semilogy(tnorm[good], y[good], marker=marker, ms=3, label=label)
    good = bz_rms > 0.0
    ax.semilogy(tnorm[good], bz_rms[good], "--", label=r"$B_{z,\rm rms}$")
    good = residual_rms > 0.0
    ax.semilogy(tnorm[good], residual_rms[good], ":", label="non-template RMS")
    ax.set_xlabel(r"$t/\tau_A$")
    ax.set_ylabel("normalized magnetic amplitude")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_history.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(z_first/L, first_profile, label="initial")
    ax.plot(z_last/L, last_profile, label="final")
    seed_ref = args.seed_eps * np.abs(np.cos(kz * z_first))
    ax.plot(z_first/L, seed_ref, "--", label=rf"analytic seed $\epsilon_B={args.seed_eps:g}$")
    ax.axvline(-args.core_width, ls=":", lw=0.8)
    ax.axvline(+args.core_width, ls=":", lw=0.8)
    ax.set_xlabel(r"$z/L$")
    ax.set_ylabel(r"$2|\hat B_{z,m=1}(z)|/B_0$")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_m1_z_profile.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tnorm, m1_fraction, "o-", ms=3)
    ax.set_xlabel(r"$t/\tau_A$")
    ax.set_ylabel(r"core $m=1$ fraction of $B_z$ perturbation energy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_m1_energy_fraction.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
