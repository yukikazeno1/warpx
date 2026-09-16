#!/usr/bin/env python3
"""Diagnostics for the 40,000-step Ishizawa-scale Harris pilot.

This script complements analyze_ishizawa_scale_early.py after the run reaches
omega_ci*t ~= 1.  It focuses on the transition from linear mode growth to
nonlinear island formation:

* complex Fourier coefficients and phase locking for selected x modes;
* spatial phase coherence across the sheet core;
* perturbation flux delta A_y = A_y - <A_y>_x;
* zoomed final topology near the current sheet;
* a midplane perturbation-flux-span history.

It deliberately does not declare reconnection from a local extremum alone.
Use the full-A_y topology and the perturbation topology together.
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

QE = 1.602176634e-19
ME = 9.1093837139e-31
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
C = 299792458.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs_ishizawa_scale/ppc49")
    p.add_argument("--modes", default="4,5,6,7,8,9",
                   help="comma-separated x-Fourier mode numbers")
    p.add_argument("--core-width-de", type=float, default=2.0)
    p.add_argument("--phase-tmin", type=float, default=0.25,
                   help="minimum omega_ci*t for phase-lock statistics")
    p.add_argument("--zoom-de", type=float, default=4.0,
                   help="half-height of topology zoom in d_e")
    return p.parse_args()


def numeric_key(path):
    name = os.path.basename(path.rstrip("/"))
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


def as_xz(arr, nx, nz):
    a = np.asarray(arr).squeeze()
    if a.shape == (nx, nz):
        return a
    if a.shape == (nz, nx):
        return a.T
    raise RuntimeError(f"Unexpected field shape {a.shape}")


def params():
    n0 = 1.0e19
    mi = 800.0 * ME
    wpe = np.sqrt(n0 * QE**2 / (EPS0 * ME))
    de = C / wpe
    wce = wpe / 3.5
    wci = wce / 800.0
    B0 = ME * wce / QE
    J0 = B0 / (MU0 * de)
    return dict(n0=n0, mi=mi, wpe=wpe, de=de, wce=wce, wci=wci,
                B0=B0, J0=J0)


def load_fields(path):
    ds = yt.load(path)
    nx = int(ds.domain_dimensions[0])
    nz = int(ds.domain_dimensions[1])
    xlo = ds.domain_left_edge[0].to_value("m")
    xhi = ds.domain_right_edge[0].to_value("m")
    zlo = ds.domain_left_edge[1].to_value("m")
    zhi = ds.domain_right_edge[1].to_value("m")
    dx = (xhi-xlo)/nx
    dz = (zhi-zlo)/nz
    x = xlo + (np.arange(nx)+0.5)*dx
    z = zlo + (np.arange(nz)+0.5)*dz
    g = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
                         dims=ds.domain_dimensions)

    def fld(name, unit):
        q = g["boxlib", name]
        try:
            a = q.to_value(unit)
        except Exception:
            a = q.to_ndarray()
        return as_xz(a, nx, nz)

    return dict(ds=ds, x=x, z=z, dx=dx, dz=dz,
                Bx=fld("Bx", "T"), By=fld("By", "T"), Bz=fld("Bz", "T"))


def reconstruct_Ay(Bx, Bz, x, z):
    """Reconstruct A_y with Bz=dAy/dx and Bx=-dAy/dz."""
    nx, nz = Bx.shape
    dx = x[1]-x[0]
    dz = z[1]-z[0]
    j0 = int(np.argmin(np.abs(z)))
    ay_mid = np.zeros(nx)
    for i in range(1, nx):
        ay_mid[i] = ay_mid[i-1] + 0.5*(Bz[i-1,j0]+Bz[i,j0])*dx
    # Remove the tiny non-periodic integration drift only.
    ay_mid -= np.linspace(0.0, ay_mid[-1]-ay_mid[0], nx)
    ay_mid -= np.mean(ay_mid)

    Ay = np.zeros_like(Bx)
    Ay[:,j0] = ay_mid
    for j in range(j0+1, nz):
        Ay[:,j] = Ay[:,j-1] - 0.5*(Bx[:,j-1]+Bx[:,j])*dz
    for j in range(j0-1, -1, -1):
        Ay[:,j] = Ay[:,j+1] + 0.5*(Bx[:,j+1]+Bx[:,j])*dz
    return Ay


def corrected_jy(Bx, Bz, dx, dz):
    # (curl B)_y = dBx/dz - dBz/dx in x-z geometry.
    dBx_dz = np.gradient(Bx, dz, axis=1, edge_order=2)
    dBz_dx = np.gradient(Bz, dx, axis=0, edge_order=2)
    return (dBx_dz-dBz_dx)/MU0


def circular_resultant(phi):
    if len(phi) == 0:
        return np.nan
    return float(np.abs(np.mean(np.exp(1j*phi))))


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params()
    modes = [int(x) for x in args.modes.split(",") if x.strip()]

    files = sorted(glob.glob(str(Path(args.run_dir)/"diags"/"diag1*")), key=numeric_key)
    if not files:
        raise FileNotFoundError("No diag1 plotfiles found")

    tci = []
    coeff = {m: [] for m in modes}
    spatial_coh = {m: [] for m in modes}
    flux_span = []
    final = None

    for k, fn in enumerate(files, 1):
        F = load_fields(fn)
        x, z = F["x"], F["z"]
        t = F["ds"].current_time.to_value("s")
        tci.append(P["wci"]*t)
        core = np.abs(z) <= args.core_width_de*P["de"]

        ft = np.fft.rfft(F["Bz"], axis=0)/F["Bz"].shape[0]
        for m in modes:
            if m >= ft.shape[0]:
                coeff[m].append(np.nan+1j*np.nan)
                spatial_coh[m].append(np.nan)
                continue
            fm = ft[m,core]
            # Complex coherent coefficient; factor 2 converts one-sided FFT
            # coefficient to sinusoidal amplitude.
            cm = 2.0*np.mean(fm)/P["B0"]
            coeff[m].append(cm)
            denom = np.mean(np.abs(fm))
            spatial_coh[m].append(float(np.abs(np.mean(fm))/denom) if denom > 0 else np.nan)

        Ay = reconstruct_Ay(F["Bx"], F["Bz"], x, z)
        dAy = Ay - np.mean(Ay, axis=0, keepdims=True)
        j0 = int(np.argmin(np.abs(z)))
        flux_span.append((np.max(dAy[:,j0])-np.min(dAy[:,j0]))/(P["B0"]*P["de"]))
        final = (F, Ay, dAy)
        print(f"{k:3d}/{len(files)} omega_ci*t={tci[-1]:.5f}  deltaAy_span={flux_span[-1]:.4e}")

    tci = np.asarray(tci)
    flux_span = np.asarray(flux_span)
    for m in modes:
        coeff[m] = np.asarray(coeff[m], dtype=complex)
        spatial_coh[m] = np.asarray(spatial_coh[m], dtype=float)

    # Phase/amplitude history.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9,10), sharex=True)
    for m in modes:
        c = coeff[m]
        ax1.semilogy(tci, np.abs(c), "o-", ms=3, label=f"m={m}")
        ax2.plot(tci, np.unwrap(np.angle(c)), "o-", ms=3, label=f"m={m}")
        ax3.plot(tci, spatial_coh[m], "o-", ms=3, label=f"m={m}")
    ax1.set_ylabel(r"coherent $|\hat B_{z,m}|/B_0$")
    ax2.set_ylabel("unwrapped phase [rad]")
    ax3.set_ylabel("z-phase coherence")
    ax3.set_xlabel(r"$\omega_{ci}t$")
    ax3.set_ylim(0,1.05)
    for ax in (ax1,ax2,ax3):
        ax.grid(alpha=0.25)
    ax1.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig("ishizawa_scale_phase_locking.png", dpi=200)
    plt.close(fig)

    # Flux-span history.
    fig, ax = plt.subplots(figsize=(8,5.5))
    ax.semilogy(tci, np.maximum(flux_span,1e-16), "o-")
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel(r"midplane $\Delta(\delta A_y)/(B_0d_e)$")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("ishizawa_scale_deltaAy_span.png", dpi=200)
    plt.close(fig)

    # Final perturbation topology near sheet.
    F, Ay, dAy = final
    xde = F["x"]/P["de"]
    zde = F["z"]/P["de"]
    zoom = np.abs(zde) <= args.zoom_de
    X, Z = np.meshgrid(xde, zde, indexing="ij")
    dnorm = dAy/(P["B0"]*P["de"])
    jy = corrected_jy(F["Bx"],F["Bz"],F["dx"],F["dz"])/P["J0"]
    vmax = np.nanmax(np.abs(dnorm[:,zoom]))
    levels = np.linspace(-vmax, vmax, 31) if vmax > 0 else 31

    fig, ax = plt.subplots(figsize=(11,5.5))
    pcm = ax.pcolormesh(X[:,zoom], Z[:,zoom], dnorm[:,zoom], shading="auto")
    if vmax > 0:
        ax.contour(X[:,zoom], Z[:,zoom], dnorm[:,zoom], levels=levels, colors="k", linewidths=0.45, alpha=0.6)
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(r"$\delta A_y/(B_0d_e)$")
    ax.set_xlabel(r"$x/d_e$")
    ax.set_ylabel(r"$z/d_e$")
    ax.set_title(fr"perturbation-flux topology, $\omega_{{ci}}t={tci[-1]:.3f}$")
    fig.tight_layout()
    fig.savefig("ishizawa_scale_deltaAy_topology.png", dpi=200)
    plt.close(fig)

    # Corrected current + full-Ay zoom for topology cross-check.
    fig, ax = plt.subplots(figsize=(11,5.5))
    pcm = ax.pcolormesh(X[:,zoom], Z[:,zoom], jy[:,zoom], shading="auto")
    alo, ahi = np.nanmin(Ay[:,zoom]), np.nanmax(Ay[:,zoom])
    if ahi > alo:
        ax.contour(X[:,zoom], Z[:,zoom], Ay[:,zoom], levels=np.linspace(alo,ahi,45), colors="k", linewidths=0.45)
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(r"$J_y/J_0$")
    ax.set_xlabel(r"$x/d_e$")
    ax.set_ylabel(r"$z/d_e$")
    ax.set_title(fr"full topology zoom, $\omega_{{ci}}t={tci[-1]:.3f}$")
    fig.tight_layout()
    fig.savefig("ishizawa_scale_full_topology_zoom.png", dpi=200)
    plt.close(fig)

    # Text summary.
    mask = tci >= args.phase_tmin
    with open("ishizawa_scale_phase_summary.txt","w") as f:
        f.write("Ishizawa-scale 40k phase/topology diagnostics\n")
        f.write("=============================================\n")
        f.write(f"final omega_ci*t = {tci[-1]:.8f}\n")
        f.write(f"phase statistics use omega_ci*t >= {args.phase_tmin:g}\n")
        f.write("mode final_coherent_amp mean_spatial_coherence phase_resultant phase_unwrapped_span_rad\n")
        for m in modes:
            c = coeff[m]
            good = mask & np.isfinite(c.real) & np.isfinite(c.imag) & (np.abs(c)>0)
            phi = np.angle(c[good])
            up = np.unwrap(phi)
            span = float(np.max(up)-np.min(up)) if len(up) else np.nan
            f.write(f"{m:3d} {abs(c[-1]):.8e} {np.nanmean(spatial_coh[m][good]):.6f} "
                    f"{circular_resultant(phi):.6f} {span:.6f}\n")
        f.write(f"\nfinal deltaAy midplane span/(B0 de) = {flux_span[-1]:.8e}\n")

    print("Saved ishizawa_scale_phase_locking.png")
    print("Saved ishizawa_scale_deltaAy_span.png")
    print("Saved ishizawa_scale_deltaAy_topology.png")
    print("Saved ishizawa_scale_full_topology_zoom.png")
    print("Saved ishizawa_scale_phase_summary.txt")


if __name__ == "__main__":
    main()
