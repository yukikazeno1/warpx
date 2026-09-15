#!/usr/bin/env python3
"""
Field-only reconnection/topology diagnostics for the long-time Harris-sheet run.

This script is intentionally compatible with the existing long-run plotfiles,
which contain mesh fields but no particle dumps.  It is inspired by the field-
level diagnostics in Ishizawa & Horiuchi, PRL 95, 045003 (2005), but it does NOT
claim to reproduce their steady driven-reconnection setup or their full pressure-
tensor decomposition.

Geometry mapping relative to Ishizawa (paper x-y plane, current z):
    paper x (outflow)       -> WarpX x
    paper y (inflow)        -> WarpX z
    paper z (out of plane)  -> WarpX y
Thus paper J_z and E_z correspond to WarpX J_y and E_y.

Available diagnostics from current mesh output:
  * reconstruct A_y(x,z) from B_x and B_z
  * identify a dominant midplane X/O pair
  * reconnected flux |A_y(O)-A_y(X)|/(B0 L)
  * reconnection electric field E_y(X)/(v_A B0)
  * current-sheet profile J_y(z) through the X point
  * SI Hall electric term [(J x B)_y/(e n_i)]/(v_A B0)
  * final topology plot and inflow-line profiles
  * initial estimates of electron/ion meandering scales from the Harris state

Full Ishizawa Fig. 3-type generalized-Ohm-law diagnostics additionally require
species flow velocities and pressure tensors.  Those are not present in the
existing mesh-only plotfiles.
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
    p.add_argument("--ppc", type=int, default=64)
    p.add_argument("--late-tmin", type=float, default=2.2,
                   help="Late-time averaging lower bound in t/tau_A")
    return p.parse_args()


def numeric_key(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else path


def as_xz(a, nx, nz):
    a = np.asarray(a).squeeze()
    if a.shape == (nx, nz):
        return a
    if a.shape == (nz, nx):
        return a.T
    raise RuntimeError(f"unexpected field shape {a.shape}, expected {(nx,nz)} or {(nz,nx)}")


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
    di = np.sqrt(mi / me) * de

    vthe = np.sqrt(Te / me)
    vthi = np.sqrt(Ti / mi)
    rhoe0 = me * vthe / (qe * B0)
    rhoi0 = mi * vthi / (qe * B0)

    # For B=B0*tanh(z/L): rho_s(z)=rho_s0/tanh(s),
    # L_B/L = sinh(s)cosh(s); rho_s=L_B -> sinh^2(s)=rho_s0/L.
    lme = L * np.arcsinh(np.sqrt(rhoe0 / L))
    lmi = L * np.arcsinh(np.sqrt(rhoi0 / L))

    return dict(qe=qe, me=me, mi=mi, mu0=mu0, c=c, n0=n0,
                de=de, di=di, L=L, Te=Te, Ti=Ti, B0=B0,
                vA=vA, tauA=tauA, lme=lme, lmi=lmi)


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

    def fld(name, unit=None):
        q = g["boxlib", name]
        arr = q.to_value(unit) if unit else q.to_ndarray()
        return as_xz(arr, nx, nz)

    out = {
        "ds": ds, "x": x, "z": z, "dx": dx, "dz": dz,
        "Bx": fld("Bx", "T"), "Bz": fld("Bz", "T"),
        "Ey": fld("Ey", "V/m"),
        "jx": fld("jx", "A/m**2"), "jy": fld("jy", "A/m**2"),
        "jz": fld("jz", "A/m**2"),
        "rho_i": fld("rho_ions", "C/m**3"),
    }
    return out


def reconstruct_Ay(Bx, Bz, x, z):
    """Reconstruct A_y with Bz=dAy/dx and Bx=-dAy/dz, up to a gauge."""
    nx, nz = Bx.shape
    dx = x[1]-x[0]
    dz = z[1]-z[0]

    # Nonzero-kx part from Bz.
    bzh = np.fft.rfft(Bz, axis=0)
    k = 2.0*np.pi*np.fft.rfftfreq(nx, d=dx)
    ayh = np.zeros_like(bzh, dtype=complex)
    ayh[1:, :] = bzh[1:, :] / (1j*k[1:, None])
    ay_fluc = np.fft.irfft(ayh, n=nx, axis=0)

    # kx=0 equilibrium/background part from <Bx>_x = -dAy0/dz.
    bx0 = np.mean(Bx, axis=0)
    ay0 = np.zeros(nz)
    for j in range(1, nz):
        ay0[j] = ay0[j-1] - 0.5*(bx0[j-1]+bx0[j])*dz
    iz0 = int(np.argmin(np.abs(z)))
    ay0 -= ay0[iz0]

    return ay_fluc + ay0[None, :]


def midplane_xo(Ay, jy, x, z):
    """Find a dominant X/O pair from midplane extrema and Hessian parity."""
    iz = int(np.argmin(np.abs(z)))
    line = Ay[:, iz]
    nx = len(x)
    dx = x[1]-x[0]
    dz = z[1]-z[0]

    left = np.roll(line, 1)
    right = np.roll(line, -1)
    extrema = ((line >= left) & (line >= right)) | ((line <= left) & (line <= right))

    d2x = (right - 2.0*line + left) / dx**2
    if 0 < iz < Ay.shape[1]-1:
        d2z = (Ay[:, iz+1] - 2.0*Ay[:, iz] + Ay[:, iz-1]) / dz**2
    else:
        d2z = np.zeros(nx)

    det = d2x*d2z
    xs = np.where(extrema & (det < 0.0))[0]
    os_ = np.where(extrema & (det > 0.0))[0]

    # Fallback: use all extrema if Hessian classification is too noisy.
    if len(xs) == 0 or len(os_) == 0:
        ext = np.where(extrema)[0]
        if len(ext) < 2:
            ix = int(np.argmax(np.abs(jy[:, iz])))
            io = (ix + nx//2) % nx
            return ix, io, iz
        # Split by curvature sign along x as a fallback.
        xs = ext[d2x[ext] > 0]
        os_ = ext[d2x[ext] < 0]
        if len(xs) == 0 or len(os_) == 0:
            ix = int(ext[np.argmax(np.abs(jy[ext, iz]))])
            io = int(ext[np.argmax(np.abs(line[ext]-line[ix]))])
            return ix, io, iz

    # Choose the X/O pair with largest flux contrast, using periodic x distance
    # only as a weak tie-breaker.
    best = None
    for ix in xs:
        for io in os_:
            dpsi = abs(line[io]-line[ix])
            dd = min((io-ix) % nx, (ix-io) % nx)
            score = (dpsi, -dd)
            if best is None or score > best[0]:
                best = (score, int(ix), int(io))
    return best[1], best[2], iz


def hall_electric_y(F, p):
    # (J x B)_y = J_z B_x - J_x B_z in the WarpX x-y-z basis.
    jxb_y = F["jz"]*F["Bx"] - F["jx"]*F["Bz"]
    ni = F["rho_i"] / p["qe"]
    # Avoid division by tiny density in the dilute edges.
    floor = 1.0e-6 * p["n0"]
    out = np.full_like(jxb_y, np.nan)
    mask = ni > floor
    out[mask] = jxb_y[mask] / (p["qe"]*ni[mask])
    return out


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    p = physical_params()

    pattern = str(Path(args.runs_dir) / f"ppc{args.ppc}" / "diags" / "diag1*")
    files = sorted(glob.glob(pattern), key=numeric_key)
    if not files:
        raise FileNotFoundError(f"no plotfiles found: {pattern}")

    print("="*78)
    print("Ishizawa-inspired field/topology diagnostics")
    print("="*78)
    print(f"PPC                  : {args.ppc}")
    print(f"plotfiles            : {len(files)}")
    print(f"d_i/d_e              : {p['di']/p['de']:.6f}")
    print(f"initial l_me/d_e     : {p['lme']/p['de']:.6f}")
    print(f"initial l_mi/d_e     : {p['lmi']/p['de']:.6f}")
    print(f"box half-height/d_e  : 4.0")
    print("NOTE: d_i lies far outside the present box; Ishizawa's full l_mi<z<d_i")
    print("      gyroviscous-cancellation region cannot be represented here.")

    rows = []
    last = None
    for ipf, fn in enumerate(files, start=1):
        F = load_fields(fn)
        Ay = reconstruct_Ay(F["Bx"], F["Bz"], F["x"], F["z"])
        ix, io, iz = midplane_xo(Ay, F["jy"], F["x"], F["z"])
        Hall = hall_electric_y(F, p)

        t = F["ds"].current_time.to_value("s") / p["tauA"]
        psi = abs(Ay[io, iz] - Ay[ix, iz]) / (p["B0"]*p["L"])
        eyx = F["Ey"][ix, iz] / (p["vA"]*p["B0"])
        hallx = Hall[ix, iz] / (p["vA"]*p["B0"])
        jyx = F["jy"][ix, iz] / (p["B0"]/(p["mu0"]*p["L"]))

        rows.append((t, psi, F["x"][ix]/p["L"], F["x"][io]/p["L"],
                     eyx, hallx, jyx))
        last = (F, Ay, Hall, ix, io, iz, t)

        if ipf % 25 == 0 or ipf == len(files):
            print(f"  analyzed {ipf}/{len(files)}, t/tau_A={t:.3f}, Psi_rec={psi:.4e}")

    rows = np.asarray(rows)
    np.savetxt(
        f"reconnection_ishizawa_history_ppc{args.ppc}.txt", rows,
        header="t/tauA Psi_rec/(B0L) x_X/L x_O/L Ey_X/(vAB0) Hall_X/(vAB0) Jy_X/J0"
    )

    # History plot.
    fig, ax = plt.subplots(figsize=(8,5))
    ax.semilogy(rows[:,0], np.maximum(rows[:,1], 1e-12), "o-", ms=3)
    ax.set_xlabel(r"$t/\tau_A$")
    ax.set_ylabel(r"$|A_y(O)-A_y(X)|/(B_0L)$")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"reconnection_flux_history_ppc{args.ppc}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(rows[:,0], rows[:,4], "o-", ms=3, label=r"$E_y(X)/(v_AB_0)$")
    ax.plot(rows[:,0], rows[:,5], "s-", ms=3, label=r"$E_{Hall,y}(X)/(v_AB_0)$")
    ax.set_xlabel(r"$t/\tau_A$")
    ax.set_ylabel("normalized out-of-plane electric field")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"reconnection_xpoint_fields_ppc{args.ppc}.png", dpi=200)
    plt.close(fig)

    F, Ay, Hall, ix, io, iz, tf = last
    X, Z = np.meshgrid(F["x"]/p["L"], F["z"]/p["L"], indexing="ij")
    J0 = p["B0"]/(p["mu0"]*p["L"])

    # Topology/current plot analogous in spirit to Ishizawa's current-layer view.
    fig, ax = plt.subplots(figsize=(9,5))
    pcm = ax.pcolormesh(X, Z, F["jy"]/J0, shading="auto")
    levels = np.linspace(np.nanmin(Ay/(p["B0"]*p["L"])),
                         np.nanmax(Ay/(p["B0"]*p["L"])), 25)
    ax.contour(X, Z, Ay/(p["B0"]*p["L"]), levels=levels,
               linewidths=0.5, colors="k", alpha=0.45)
    ax.plot(F["x"][ix]/p["L"], F["z"][iz]/p["L"], "x", ms=9, mew=2, label="X candidate")
    ax.plot(F["x"][io]/p["L"], F["z"][iz]/p["L"], "o", ms=6, fillstyle="none", label="O candidate")
    fig.colorbar(pcm, ax=ax, label=r"$J_y/J_0$")
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel(r"$z/L$")
    ax.set_title(rf"{args.ppc} PPC, $t/\tau_A={tf:.3f}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"reconnection_final_topology_ppc{args.ppc}.png", dpi=200)
    plt.close(fig)

    # Inflow current profile through X point, with initial meandering-scale estimates.
    zz = F["z"]/p["L"]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(zz, F["jy"][ix,:]/J0, label=r"$J_y/J_0$")
    for s in (-1,1):
        ax.axvline(s*p["lme"]/p["L"], ls=":", lw=1)
        ax.axvline(s*p["lmi"]/p["L"], ls="--", lw=1)
    ax.set_xlabel(r"$z/L$")
    ax.set_ylabel(r"$J_y/J_0$")
    ax.set_title("inflow profile through X candidate; dotted=l_me, dashed=l_mi (initial estimates)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"reconnection_final_current_profile_ppc{args.ppc}.png", dpi=200)
    plt.close(fig)

    # Field/Hall profile: this is only a partial analogue of Ishizawa Fig. 3,
    # because species flows and pressure tensors are not in the present output.
    E0 = p["vA"]*p["B0"]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(zz, F["Ey"][ix,:]/E0, label=r"$E_y$")
    ax.plot(zz, Hall[ix,:]/E0, label=r"$(J\times B)_y/(en_i)$")
    ax.axhline(0.0, lw=0.8)
    for s in (-1,1):
        ax.axvline(s*p["lmi"]/p["L"], ls="--", lw=1)
    ax.set_xlabel(r"$z/L$")
    ax.set_ylabel(r"normalized by $v_AB_0$")
    ax.set_title("partial Ishizawa-style inflow profile")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"reconnection_final_hall_profile_ppc{args.ppc}.png", dpi=200)
    plt.close(fig)

    late = rows[:,0] >= args.late_tmin
    with open(f"reconnection_ishizawa_summary_ppc{args.ppc}.txt", "w") as f:
        f.write("Ishizawa-inspired field/topology summary\n")
        f.write("=======================================\n\n")
        f.write(f"final t/tau_A = {rows[-1,0]:.8f}\n")
        f.write(f"final Psi_rec/(B0 L) = {rows[-1,1]:.8e}\n")
        f.write(f"final Ey_X/(vA B0) = {rows[-1,4]:.8e}\n")
        f.write(f"final Hall_X/(vA B0) = {rows[-1,5]:.8e}\n")
        f.write(f"final Jy_X/J0 = {rows[-1,6]:.8e}\n\n")
        if np.any(late):
            f.write(f"late averaging t/tau_A >= {args.late_tmin:g}\n")
            f.write(f"<Psi_rec> = {np.mean(rows[late,1]):.8e}\n")
            f.write(f"std(Psi_rec) = {np.std(rows[late,1]):.8e}\n")
            f.write(f"<|Ey_X|> = {np.mean(np.abs(rows[late,4])):.8e}\n")
        f.write("\nScale comparison (initial Harris estimates)\n")
        f.write(f"L=d_e\n")
        f.write(f"d_i/d_e = {p['di']/p['de']:.8f}\n")
        f.write(f"l_me/d_e = {p['lme']/p['de']:.8f}\n")
        f.write(f"l_mi/d_e = {p['lmi']/p['de']:.8f}\n")
        f.write("box z/d_e = [-4,4]\n")
        f.write("Therefore the present box does not contain the d_i scale.\n")
        f.write("A full Ishizawa gyroviscous-cancellation test requires a larger box\n")
        f.write("plus species velocity and pressure-tensor moments.\n")

    print("Saved:")
    print(f"  reconnection_ishizawa_history_ppc{args.ppc}.txt")
    print(f"  reconnection_ishizawa_summary_ppc{args.ppc}.txt")
    print(f"  reconnection_flux_history_ppc{args.ppc}.png")
    print(f"  reconnection_xpoint_fields_ppc{args.ppc}.png")
    print(f"  reconnection_final_topology_ppc{args.ppc}.png")
    print(f"  reconnection_final_current_profile_ppc{args.ppc}.png")
    print(f"  reconnection_final_hall_profile_ppc{args.ppc}.png")


if __name__ == "__main__":
    main()
