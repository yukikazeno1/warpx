#!/usr/bin/env python3
"""
Early field diagnostics for the 512x512 Ishizawa-scale Harris pilot.

The script is intentionally field-only and is designed for the first
10,000-step size-isolation run.  Time is normalized with omega_ci t.

Outputs
-------
ishizawa_scale_early_summary.txt
ishizawa_scale_mode_history.txt
ishizawa_scale_low_modes.png
ishizawa_scale_spectrum_map.png
ishizawa_scale_field_history.png
ishizawa_scale_final_equilibrium.png
ishizawa_scale_final_current.png
ishizawa_scale_final_topology.png

The topology diagnostic reconstructs A_y from B_z along the midplane and B_x
away from the midplane.  It is meant for qualitative X/O/island inspection;
it does not declare reconnection solely from a local extremum of A_y.
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
    p.add_argument("--max-mode", type=int, default=20)
    p.add_argument("--core-width-de", type=float, default=2.0,
                   help="half-width of the sheet core used for Fourier/RMS diagnostics")
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
    raise RuntimeError(f"Unexpected array shape {a.shape}; expected {(nx, nz)} or {(nz, nx)}")


def params():
    n0 = 1.0e19
    mi = 800.0 * ME
    wpe = np.sqrt(n0 * QE**2 / (EPS0 * ME))
    de = C / wpe
    di = np.sqrt(mi / ME) * de
    wce = wpe / 3.5
    wci = wce / 800.0
    B0 = ME * wce / QE
    Te = B0**2 / (4.0 * MU0 * n0)
    Ti = Te
    lambda_D = np.sqrt(EPS0 * Te / (n0 * QE**2))
    L = de
    J0 = B0 / (MU0 * L)
    dt = 0.02 / wce
    return dict(n0=n0, mi=mi, wpe=wpe, de=de, di=di, wce=wce, wci=wci,
                B0=B0, Te=Te, Ti=Ti, lambda_D=lambda_D, L=L, J0=J0, dt=dt)


def load_fields(path):
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

    g = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
                         dims=ds.domain_dimensions)

    def fld(name, unit=None):
        q = g["boxlib", name]
        if unit is None:
            arr = q.to_ndarray()
        else:
            try:
                arr = q.to_value(unit)
            except Exception:
                # 2D AMReX plotfiles sometimes carry incomplete current-density
                # metadata.  Native WarpX values are SI; preserve them here.
                arr = q.to_ndarray()
        return as_xz(arr, nx, nz)

    F = {
        "ds": ds, "x": x, "z": z, "dx": dx, "dz": dz,
        "Bx": fld("Bx", "T"), "By": fld("By", "T"), "Bz": fld("Bz", "T"),
        "Ex": fld("Ex", "V/m"), "Ey": fld("Ey", "V/m"), "Ez": fld("Ez", "V/m"),
        "jx": fld("jx"), "jy": fld("jy"), "jz": fld("jz"),
    }
    return F


def spectrum_core(bz, z, max_mode, core_half_width, B0):
    core = np.abs(z) <= core_half_width
    fft = np.fft.rfft(bz, axis=0) / bz.shape[0]
    out = np.full(max_mode, np.nan)
    for m in range(1, max_mode + 1):
        if m >= fft.shape[0]:
            break
        local = 2.0 * np.abs(fft[m, :]) / B0
        out[m - 1] = np.sqrt(np.mean(local[core] ** 2))
    return out


def curl_jy(Bx, Bz, dx, dz):
    dBz_dx = np.gradient(Bz, dx, axis=0, edge_order=2)
    dBx_dz = np.gradient(Bx, dz, axis=1, edge_order=2)
    return (dBz_dx - dBx_dz) / MU0


def fwhm(z, y):
    y = np.asarray(y)
    if not np.any(np.isfinite(y)):
        return np.nan
    ymax = np.nanmax(y)
    if ymax <= 0:
        return np.nan
    mask = y >= 0.5 * ymax
    ids = np.flatnonzero(mask)
    if ids.size < 2:
        return np.nan
    return z[ids[-1]] - z[ids[0]]


def reconstruct_Ay(Bx, Bz, x, z):
    """Reconstruct A_y using Bz=dAy/dx at z~0 and Bx=-dAy/dz away from it."""
    nx, nz = Bx.shape
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    j0 = int(np.argmin(np.abs(z)))

    ay_mid = np.zeros(nx)
    for i in range(1, nx):
        ay_mid[i] = ay_mid[i-1] + 0.5 * (Bz[i-1, j0] + Bz[i, j0]) * dx
    # remove any tiny linear non-periodic drift from numerical noise
    ay_mid -= np.linspace(0.0, ay_mid[-1] - ay_mid[0], nx)
    ay_mid -= np.mean(ay_mid)

    Ay = np.zeros_like(Bx)
    Ay[:, j0] = ay_mid
    for j in range(j0 + 1, nz):
        Ay[:, j] = Ay[:, j-1] - 0.5 * (Bx[:, j-1] + Bx[:, j]) * dz
    for j in range(j0 - 1, -1, -1):
        Ay[:, j] = Ay[:, j+1] + 0.5 * (Bx[:, j+1] + Bx[:, j]) * dz
    return Ay


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params()

    pattern = str(Path(args.run_dir) / "diags" / "diag1*")
    files = sorted(glob.glob(pattern), key=numeric_key)
    if not files:
        raise FileNotFoundError(f"No plotfiles found: {pattern}")

    print("=" * 78)
    print("Ishizawa-scale early field analysis")
    print("=" * 78)
    print(f"plotfiles       : {len(files)}")
    print(f"d_i/d_e         : {P['di']/P['de']:.6f}")
    print(f"lambda_D/d_e    : {P['lambda_D']/P['de']:.6f}")
    print(f"omega_ci*dt     : {P['wci']*P['dt']:.8e}")

    tci, spectra, bz_rms, by_rms, ey_rms, bx_err, jy_err, widths, flux_span = ([] for _ in range(9))
    final = None

    core_half = args.core_width_de * P["de"]

    for i, fn in enumerate(files, 1):
        F = load_fields(fn)
        x, z = F["x"], F["z"]
        core = np.abs(z) <= core_half
        t = F["ds"].current_time.to_value("s")
        tci.append(P["wci"] * t)
        spectra.append(spectrum_core(F["Bz"], z, args.max_mode, core_half, P["B0"]))
        bz_rms.append(np.sqrt(np.mean(F["Bz"][:, core]**2)) / P["B0"])
        by_rms.append(np.sqrt(np.mean(F["By"][:, core]**2)) / P["B0"])
        ey_rms.append(np.sqrt(np.mean(F["Ey"][:, core]**2)) / (C * P["B0"]))

        bx_mean = np.mean(F["Bx"], axis=0)
        bx_ref = P["B0"] * np.tanh(z / P["L"])
        bx_err.append(np.sqrt(np.mean((bx_mean - bx_ref)**2)) / P["B0"])

        jy_curl = curl_jy(F["Bx"], F["Bz"], F["dx"], F["dz"])
        jy_mean = np.mean(jy_curl, axis=0)
        jy_ref = P["J0"] / np.cosh(z / P["L"])**2
        sheet = np.abs(z) <= 4.0 * P["L"]
        jy_err.append(np.sqrt(np.mean((jy_mean[sheet] - jy_ref[sheet])**2)) / P["J0"])
        widths.append(fwhm(z, jy_mean) / P["L"])

        Ay = reconstruct_Ay(F["Bx"], F["Bz"], x, z)
        jmid = int(np.argmin(np.abs(z)))
        flux_span.append((np.max(Ay[:, jmid]) - np.min(Ay[:, jmid])) / (P["B0"] * P["L"]))

        final = (F, jy_curl, Ay)
        print(f"  {i:3d}/{len(files)}  omega_ci*t={tci[-1]:.5f}  "
              f"Bz_core_rms/B0={bz_rms[-1]:.3e}")

    tci = np.asarray(tci)
    A = np.asarray(spectra)
    bz_rms = np.asarray(bz_rms)
    by_rms = np.asarray(by_rms)
    ey_rms = np.asarray(ey_rms)
    bx_err = np.asarray(bx_err)
    jy_err = np.asarray(jy_err)
    widths = np.asarray(widths)
    flux_span = np.asarray(flux_span)

    # numerical history table
    table = np.column_stack([tci, bz_rms, by_rms, ey_rms, bx_err, jy_err, widths, flux_span, A])
    header = ("omega_ci_t Bz_core_rms_B0 By_core_rms_B0 Ey_core_rms_cB0 "
              "Bx_profile_rms_B0 Jy_profile_rms_J0 Jy_FWHM_L Ay_mid_span_B0L " +
              " ".join(f"A_m{m}" for m in range(1, args.max_mode + 1)))
    np.savetxt("ishizawa_scale_mode_history.txt", table, header=header)

    # low modes
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in range(1, min(10, args.max_mode) + 1):
        ax.semilogy(tci, A[:, m-1], "o-", ms=4, label=f"m={m}")
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel(r"core $B_z$ Fourier amplitude / $B_0$")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig("ishizawa_scale_low_modes.png", dpi=200)
    plt.close(fig)

    # m-t spectrum with physical k labels available in summary
    modes = np.arange(1, args.max_mode + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    positive = A[np.isfinite(A) & (A > 0)]
    floor = max(np.min(positive) if positive.size else 1e-12, 1e-12)
    im = ax.pcolormesh(tci, modes, np.log10(np.maximum(A.T, floor)), shading="auto")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$\log_{10}(A_m/B_0)$")
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel("x-Fourier mode m")
    ax.set_yticks(modes)
    fig.tight_layout()
    fig.savefig("ishizawa_scale_spectrum_map.png", dpi=200)
    plt.close(fig)

    # field/equilibrium history
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogy(tci, bz_rms, "o-", label=r"$B_{z,\rm rms}/B_0$")
    ax.semilogy(tci, by_rms, "o-", label=r"$B_{y,\rm rms}/B_0$")
    ax.semilogy(tci, ey_rms, "o-", label=r"$E_{y,\rm rms}/(cB_0)$")
    ax.semilogy(tci, bx_err, "o-", label=r"$B_x$ profile RMS error")
    ax.semilogy(tci, jy_err, "o-", label=r"$J_y$ profile RMS error")
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel("normalized amplitude/error")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("ishizawa_scale_field_history.png", dpi=200)
    plt.close(fig)

    F, jy_curl, Ay = final
    x, z = F["x"], F["z"]
    bx_mean = np.mean(F["Bx"], axis=0)
    jy_mean = np.mean(jy_curl, axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(z/P["L"], bx_mean/P["B0"], label="final x-avg")
    ax.plot(z/P["L"], np.tanh(z/P["L"]), "--", label="initial analytic tanh")
    ax.set_xlim(-8, 8)
    ax.set_xlabel(r"$z/d_e$")
    ax.set_ylabel(r"$\langle B_x\rangle_x/B_0$")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("ishizawa_scale_final_equilibrium.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(z/P["L"], jy_mean/P["J0"], label=r"final $(\nabla\times B)_y/(\mu_0J_0)$")
    ax.plot(z/P["L"], 1.0/np.cosh(z/P["L"])**2, "--", label=r"initial $\mathrm{sech}^2$")
    ax.axvline(P["di"]/P["L"], ls=":", lw=1, label=r"$+d_i$")
    ax.axvline(-P["di"]/P["L"], ls=":", lw=1, label=r"$-d_i$")
    ax.set_xlim(-32, 32)
    ax.set_xlabel(r"$z/d_e$")
    ax.set_ylabel(r"$\langle J_y\rangle_x/J_0$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("ishizawa_scale_final_current.png", dpi=200)
    plt.close(fig)

    # final topology: current background + Ay contours
    fig, ax = plt.subplots(figsize=(10, 5.8))
    pcm = ax.pcolormesh(x/P["de"], z/P["de"], (jy_curl/P["J0"]).T, shading="auto")
    fig.colorbar(pcm, ax=ax, label=r"$J_y/J_0$")
    levels = np.linspace(np.percentile(Ay, 3), np.percentile(Ay, 97), 35)
    ax.contour(x/P["de"], z/P["de"], Ay.T, levels=levels, colors="k", linewidths=0.45, alpha=0.65)
    ax.axhline(P["di"]/P["de"], ls=":", lw=1)
    ax.axhline(-P["di"]/P["de"], ls=":", lw=1)
    ax.set_xlabel(r"$x/d_e$")
    ax.set_ylabel(r"$z/d_e$")
    ax.set_title(fr"final topology, $\omega_{{ci}}t={tci[-1]:.3f}$")
    fig.tight_layout()
    fig.savefig("ishizawa_scale_final_topology.png", dpi=200)
    plt.close(fig)

    # text summary
    lx = x[-1] - x[0] + (x[1]-x[0])
    with open("ishizawa_scale_early_summary.txt", "w") as f:
        f.write("Ishizawa-scale early size-isolation diagnostics\n")
        f.write("===============================================\n\n")
        f.write(f"plotfiles              = {len(files)}\n")
        f.write(f"final omega_ci*t       = {tci[-1]:.8f}\n")
        f.write(f"Lx/de, Lz/de           = {lx/P['de']:.6f}, {(z[-1]-z[0]+(z[1]-z[0]))/P['de']:.6f}\n")
        f.write(f"di/de                  = {P['di']/P['de']:.8f}\n")
        f.write(f"lambda_D/de            = {P['lambda_D']/P['de']:.8f}\n")
        f.write(f"final Bz core rms/B0   = {bz_rms[-1]:.8e}\n")
        f.write(f"final By core rms/B0   = {by_rms[-1]:.8e}\n")
        f.write(f"final Ey core rms/cB0  = {ey_rms[-1]:.8e}\n")
        f.write(f"final Bx profile error = {bx_err[-1]:.8e}\n")
        f.write(f"final Jy profile error = {jy_err[-1]:.8e}\n")
        f.write(f"final Jy FWHM/de       = {widths[-1]:.8f}\n")
        f.write(f"final Ay mid span/B0de = {flux_span[-1]:.8e}\n\n")
        f.write("mode   k*de        k*di        DeltaPrime*de    final A_m/B0\n")
        for m in range(1, args.max_mode + 1):
            kde = 2.0*np.pi*m*P['de']/lx
            kdi = kde * P['di']/P['de']
            dpL = 2.0*(1.0/kde - kde) if kde > 0 else np.nan
            f.write(f"{m:4d}  {kde:10.6f}  {kdi:10.6f}  {dpL:14.6f}  {A[-1,m-1]:.8e}\n")

    print("Saved Ishizawa-scale early diagnostics:")
    print("  ishizawa_scale_early_summary.txt")
    print("  ishizawa_scale_mode_history.txt")
    print("  ishizawa_scale_low_modes.png")
    print("  ishizawa_scale_spectrum_map.png")
    print("  ishizawa_scale_field_history.png")
    print("  ishizawa_scale_final_equilibrium.png")
    print("  ishizawa_scale_final_current.png")
    print("  ishizawa_scale_final_topology.png")


if __name__ == "__main__":
    main()
