#!/usr/bin/env python3
"""Ishizawa-inspired field/topology diagnostics for the long-time Harris run.

This version is deliberately robust to AMReX/yt metadata issues in 2D WarpX
plotfiles.  In the current plotfiles, jx/jy/jz may be labelled as ``A`` and
rho_ions may be labelled dimensionless even though the native stored numerical
values are physical mesh quantities.  We therefore read J and rho in native
form, infer one fixed conversion scale from the first state, and independently
cross-check current against curl(B)/mu0.

Geometry mapping to Ishizawa & Horiuchi (2005):
  paper x (outflow)      -> WarpX x
  paper y (inflow)       -> WarpX z
  paper z (out of plane) -> WarpX y
Thus paper J_z and E_z correspond to WarpX J_y and E_y.

The existing mesh-only output supports topology, X/O points, reconnecting flux,
E_y at the X point, current-sheet profiles, B_y Hall structure, and a partial
Hall-term estimate.  Full pressure-tensor/inertia decomposition requires
particle/species moments and is not claimed here.
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
    p.add_argument("--late-tmin", type=float, default=2.2)
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
    mi = 800.0*me
    uth_e = 0.1

    wpe = np.sqrt(n0*qe**2/(eps0*me))
    de = c/wpe
    L = de
    Te = uth_e**2*me*c**2
    Ti = 0.1*Te
    B0 = np.sqrt(2.0*mu0*n0*(Te+Ti))
    vA = B0/np.sqrt(mu0*n0*mi)
    tauA = L/vA
    di = np.sqrt(mi/me)*de

    vthe = np.sqrt(Te/me)
    vthi = np.sqrt(Ti/mi)
    rhoe0 = me*vthe/(qe*B0)
    rhoi0 = mi*vthi/(qe*B0)
    lme = L*np.arcsinh(np.sqrt(rhoe0/L))
    lmi = L*np.arcsinh(np.sqrt(rhoi0/L))

    return dict(qe=qe, me=me, mi=mi, eps0=eps0, mu0=mu0, c=c, n0=n0,
                de=de, di=di, L=L, Te=Te, Ti=Ti, B0=B0, vA=vA,
                tauA=tauA, lme=lme, lmi=lmi)


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

    def fld_si(name, unit):
        return as_xz(g["boxlib", name].to_value(unit), nx, nz)

    def fld_native(name):
        q = g["boxlib", name]
        return as_xz(q.to_ndarray(), nx, nz), str(q.units)

    jx, ujx = fld_native("jx")
    jy, ujy = fld_native("jy")
    jz, ujz = fld_native("jz")
    rhoi, urhoi = fld_native("rho_ions")

    return {
        "ds": ds, "x": x, "z": z, "dx": dx, "dz": dz,
        "Bx": fld_si("Bx", "T"), "By": fld_si("By", "T"),
        "Bz": fld_si("Bz", "T"), "Ey": fld_si("Ey", "V/m"),
        "jx_raw": jx, "jy_raw": jy, "jz_raw": jz,
        "j_units": (ujx, ujy, ujz),
        "rho_i_raw": rhoi, "rho_i_units": urhoi,
    }


def curl_current(F, mu0):
    dx, dz = F["dx"], F["dz"]
    dBy_dz = np.gradient(F["By"], dz, axis=1, edge_order=2)
    dBy_dx = np.gradient(F["By"], dx, axis=0, edge_order=2)
    dBx_dz = np.gradient(F["Bx"], dz, axis=1, edge_order=2)
    dBz_dx = np.gradient(F["Bz"], dx, axis=0, edge_order=2)
    return (-dBy_dz/mu0,
            (dBx_dz-dBz_dx)/mu0,
            dBy_dx/mu0)


def infer_rho_scale(raw, p):
    """Infer whether native rho_ions values are charge density or number density."""
    peak = float(np.nanpercentile(np.abs(raw), 99.5))
    charge_ref = p["qe"]*p["n0"]
    number_ref = p["n0"]

    if peak > 0 and 1.0e-4 <= peak/charge_ref <= 1.0e4:
        return 1.0, "native values consistent with C/m^3"
    if peak > 0 and 1.0e-4 <= peak/number_ref <= 1.0e4:
        return p["qe"], "native values look like number density; multiplied by e"

    # Last-resort normalization: make the 99.5th percentile equal to e*n0.
    # The script reports this prominently because it is an inference.
    if peak > 0:
        return charge_ref/peak, "fallback scale chosen from peak ~ e*n0"
    return 1.0, "rho field is zero; no scale inference possible"


def infer_current_scale(jraw, jcurl, z, L):
    """Infer one native-J scale from the first state using the Harris core.

    This is a metadata repair, not a physical assertion that curl(B)/mu0 equals
    deposited J exactly.  In an unsteady EM system the displacement current can
    produce a real difference.  We therefore use only one fixed scale and keep
    reporting the residual afterwards.
    """
    core = np.abs(z) <= 2.0*L
    mask = np.broadcast_to(core[None, :], jraw.shape)
    mask &= np.isfinite(jraw) & np.isfinite(jcurl)
    den = np.sum(jraw[mask]**2)
    if den <= 0:
        return 1.0
    return float(np.sum(jraw[mask]*jcurl[mask])/den)


def reconstruct_Ay(Bx, Bz, x, z):
    nx, nz = Bx.shape
    dx = x[1]-x[0]
    dz = z[1]-z[0]

    bzh = np.fft.rfft(Bz, axis=0)
    k = 2.0*np.pi*np.fft.rfftfreq(nx, d=dx)
    ayh = np.zeros_like(bzh, dtype=complex)
    ayh[1:, :] = bzh[1:, :] / (1j*k[1:, None])
    ay_fluc = np.fft.irfft(ayh, n=nx, axis=0)

    bx0 = np.mean(Bx, axis=0)
    ay0 = np.zeros(nz)
    for j in range(1, nz):
        ay0[j] = ay0[j-1] - 0.5*(bx0[j-1]+bx0[j])*dz
    iz0 = int(np.argmin(np.abs(z)))
    ay0 -= ay0[iz0]
    return ay_fluc + ay0[None, :]


def midplane_xo(Ay, jy, x, z):
    iz = int(np.argmin(np.abs(z)))
    line = Ay[:, iz]
    nx = len(x)
    dx = x[1]-x[0]
    dz = z[1]-z[0]

    left = np.roll(line, 1)
    right = np.roll(line, -1)
    extrema = ((line >= left) & (line >= right)) | ((line <= left) & (line <= right))
    d2x = (right-2.0*line+left)/dx**2
    if 0 < iz < Ay.shape[1]-1:
        d2z = (Ay[:, iz+1]-2.0*Ay[:, iz]+Ay[:, iz-1])/dz**2
    else:
        d2z = np.zeros(nx)

    det = d2x*d2z
    xs = np.where(extrema & (det < 0.0))[0]
    os_ = np.where(extrema & (det > 0.0))[0]

    if len(xs) == 0 or len(os_) == 0:
        ext = np.where(extrema)[0]
        if len(ext) < 2:
            ix = int(np.argmax(np.abs(jy[:, iz])))
            return ix, (ix+nx//2) % nx, iz
        xs = ext[d2x[ext] > 0]
        os_ = ext[d2x[ext] < 0]
        if len(xs) == 0 or len(os_) == 0:
            ix = int(ext[np.argmax(np.abs(jy[ext, iz]))])
            io = int(ext[np.argmax(np.abs(line[ext]-line[ix]))])
            return ix, io, iz

    best = None
    for ix in xs:
        for io in os_:
            dpsi = abs(line[io]-line[ix])
            dd = min((io-ix) % nx, (ix-io) % nx)
            score = (dpsi, -dd)
            if best is None or score > best[0]:
                best = (score, int(ix), int(io))
    return best[1], best[2], iz


def hall_electric_y(jx, jz, Bx, Bz, rho_i, p):
    jxb_y = jz*Bx - jx*Bz
    ni = rho_i/p["qe"]
    out = np.full_like(jxb_y, np.nan)
    mask = ni > 1.0e-6*p["n0"]
    out[mask] = jxb_y[mask]/(p["qe"]*ni[mask])
    return out


def relative_rms(a, b, mask):
    mask = mask & np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return np.nan
    den = np.sqrt(np.mean(b[mask]**2))
    return np.sqrt(np.mean((a[mask]-b[mask])**2))/den if den > 0 else np.nan


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    p = physical_params()

    pattern = str(Path(args.runs_dir)/f"ppc{args.ppc}"/"diags"/"diag1*")
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
    print("box half-height/d_e  : 4.0")
    print("NOTE: d_i lies far outside the present box; the full l_mi<z<d_i")
    print("      gyroviscous-cancellation region cannot be represented here.")

    # Infer unit-repair scales from the first state only, then keep them fixed.
    F0 = load_fields(files[0])
    J0cx, J0cy, J0cz = curl_current(F0, p["mu0"])
    rho_scale, rho_note = infer_rho_scale(F0["rho_i_raw"], p)
    j_scale = infer_current_scale(F0["jy_raw"], J0cy, F0["z"], p["L"])

    print(f"yt current metadata units : {F0['j_units']}")
    print(f"yt rho_ions metadata unit : {F0['rho_i_units']}")
    print(f"native rho scale -> C/m^3 : {rho_scale:.8e} ({rho_note})")
    print(f"native J scale -> A/m^2   : {j_scale:.8e} (inferred once from initial Jy vs curlB)")

    rows = []
    last = None

    for ipf, fn in enumerate(files, start=1):
        F = load_fields(fn)
        F["rho_i"] = rho_scale*F["rho_i_raw"]
        F["jx"] = j_scale*F["jx_raw"]
        F["jy"] = j_scale*F["jy_raw"]
        F["jz"] = j_scale*F["jz_raw"]

        Jcx, Jcy, Jcz = curl_current(F, p["mu0"])
        Ay = reconstruct_Ay(F["Bx"], F["Bz"], F["x"], F["z"])
        ix, io, iz = midplane_xo(Ay, F["jy"], F["x"], F["z"])

        Hall_dep = hall_electric_y(F["jx"], F["jz"], F["Bx"], F["Bz"], F["rho_i"], p)
        Hall_curl = hall_electric_y(Jcx, Jcz, F["Bx"], F["Bz"], F["rho_i"], p)

        core = np.abs(F["z"]) <= p["L"]
        mask2d = np.broadcast_to(core[None, :], F["jy"].shape)
        jy_curl_err = relative_rms(F["jy"], Jcy, mask2d)

        t = F["ds"].current_time.to_value("s")/p["tauA"]
        psi = abs(Ay[io, iz]-Ay[ix, iz])/(p["B0"]*p["L"])
        E0 = p["vA"]*p["B0"]
        Jnorm = p["B0"]/(p["mu0"]*p["L"])
        eyx = F["Ey"][ix, iz]/E0
        hallx = Hall_dep[ix, iz]/E0
        hallcurlx = Hall_curl[ix, iz]/E0
        jyx = F["jy"][ix, iz]/Jnorm

        rows.append((t, psi, F["x"][ix]/p["L"], F["x"][io]/p["L"],
                     eyx, hallx, hallcurlx, jyx, jy_curl_err))
        last = (F, Ay, Hall_dep, Hall_curl, ix, io, iz, t, Jcy)

        if ipf % 25 == 0 or ipf == len(files):
            print(f"  analyzed {ipf}/{len(files)}, t/tau_A={t:.3f}, "
                  f"Psi_rec={psi:.4e}, Jy-vs-curl relRMS={jy_curl_err:.3e}")

    rows = np.asarray(rows)
    np.savetxt(f"reconnection_ishizawa_history_ppc{args.ppc}.txt", rows,
               header=("t/tauA Psi_rec/(B0L) x_X/L x_O/L Ey_X/(vAB0) "
                       "Hall_dep_X/(vAB0) Hall_curl_X/(vAB0) Jy_X/J0 JyCurl_relRMS"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(rows[:,0], np.maximum(rows[:,1], 1e-12), "o-", ms=3)
    ax.set_xlabel(r"$t/\tau_A$")
    ax.set_ylabel(r"$|A_y(O)-A_y(X)|/(B_0L)$")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"reconnection_flux_history_ppc{args.ppc}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rows[:,0], rows[:,4], "o-", ms=3, label=r"$E_y(X)$")
    ax.plot(rows[:,0], rows[:,5], "s-", ms=3, label="Hall from deposited J")
    ax.plot(rows[:,0], rows[:,6], "^-", ms=3, label=r"Hall from $\nabla\times B/\mu_0$")
    ax.set_xlabel(r"$t/\tau_A$")
    ax.set_ylabel(r"normalized by $v_AB_0$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"reconnection_xpoint_fields_ppc{args.ppc}.png", dpi=200)
    plt.close(fig)

    F, Ay, Hall_dep, Hall_curl, ix, io, iz, tf, Jcy = last
    X, Z = np.meshgrid(F["x"]/p["L"], F["z"]/p["L"], indexing="ij")
    Jnorm = p["B0"]/(p["mu0"]*p["L"])
    E0 = p["vA"]*p["B0"]

    fig, ax = plt.subplots(figsize=(9, 5))
    pcm = ax.pcolormesh(X, Z, F["jy"]/Jnorm, shading="auto")
    An = Ay/(p["B0"]*p["L"])
    levels = np.linspace(np.nanmin(An), np.nanmax(An), 25)
    ax.contour(X, Z, An, levels=levels, linewidths=0.5, colors="k", alpha=0.45)
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

    zz = F["z"]/p["L"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(zz, F["jy"][ix,:]/Jnorm, label="deposited $J_y$")
    ax.plot(zz, Jcy[ix,:]/Jnorm, "--", label=r"$(\nabla\times B)_y/\mu_0$")
    for s in (-1, 1):
        ax.axvline(s*p["lme"]/p["L"], ls=":", lw=1)
        ax.axvline(s*p["lmi"]/p["L"], ls="--", lw=1)
    ax.set_xlabel(r"$z/L$")
    ax.set_ylabel(r"$J_y/J_0$")
    ax.set_title("inflow profile through X; dotted=l_me, dashed=l_mi")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"reconnection_final_current_profile_ppc{args.ppc}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(zz, F["Ey"][ix,:]/E0, label=r"$E_y$")
    ax.plot(zz, Hall_dep[ix,:]/E0, label="Hall: deposited J")
    ax.plot(zz, Hall_curl[ix,:]/E0, "--", label=r"Hall: $\nabla\times B/\mu_0$")
    ax.axhline(0.0, lw=0.8)
    for s in (-1, 1):
        ax.axvline(s*p["lmi"]/p["L"], ls="--", lw=1)
    ax.set_xlabel(r"$z/L$")
    ax.set_ylabel(r"normalized by $v_AB_0$")
    ax.set_title("partial Ishizawa-style inflow profile")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"reconnection_final_hall_profile_ppc{args.ppc}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    pcm = ax.pcolormesh(X, Z, F["By"]/p["B0"], shading="auto")
    fig.colorbar(pcm, ax=ax, label=r"$B_y/B_0$")
    ax.plot(F["x"][ix]/p["L"], F["z"][iz]/p["L"], "kx", ms=8, mew=2)
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel(r"$z/L$")
    ax.set_title("out-of-plane Hall magnetic field")
    fig.tight_layout()
    fig.savefig(f"reconnection_final_hall_By_ppc{args.ppc}.png", dpi=200)
    plt.close(fig)

    late = rows[:,0] >= args.late_tmin
    with open(f"reconnection_ishizawa_summary_ppc{args.ppc}.txt", "w") as f:
        f.write("Ishizawa-inspired field/topology summary\n")
        f.write("=======================================\n\n")
        f.write(f"yt current metadata units = {F0['j_units']}\n")
        f.write(f"yt rho_ions metadata unit = {F0['rho_i_units']}\n")
        f.write(f"rho native scale to C/m^3 = {rho_scale:.8e} ({rho_note})\n")
        f.write(f"J native scale to A/m^2 = {j_scale:.8e}\n")
        f.write(f"final t/tau_A = {rows[-1,0]:.8f}\n")
        f.write(f"final Psi_rec/(B0 L) = {rows[-1,1]:.8e}\n")
        f.write(f"final Ey_X/(vA B0) = {rows[-1,4]:.8e}\n")
        f.write(f"final Hall_dep_X/(vA B0) = {rows[-1,5]:.8e}\n")
        f.write(f"final Hall_curl_X/(vA B0) = {rows[-1,6]:.8e}\n")
        f.write(f"final Jy_X/J0 = {rows[-1,7]:.8e}\n")
        f.write(f"final Jy-vs-curl relative RMS in |z|<=L = {rows[-1,8]:.8e}\n\n")
        if np.any(late):
            f.write(f"late averaging t/tau_A >= {args.late_tmin:g}\n")
            f.write(f"<Psi_rec> = {np.mean(rows[late,1]):.8e}\n")
            f.write(f"std(Psi_rec) = {np.std(rows[late,1]):.8e}\n")
            f.write(f"<|Ey_X|> = {np.mean(np.abs(rows[late,4])):.8e}\n")
        f.write("\nScale comparison (initial Harris estimates)\n")
        f.write("L=d_e\n")
        f.write(f"d_i/d_e = {p['di']/p['de']:.8f}\n")
        f.write(f"l_me/d_e = {p['lme']/p['de']:.8f}\n")
        f.write(f"l_mi/d_e = {p['lmi']/p['de']:.8f}\n")
        f.write("box z/d_e = [-4,4]\n")
        f.write("The present box does not contain d_i. A full Ishizawa gyroviscous-\n")
        f.write("cancellation test needs a larger box plus species velocity/pressure moments.\n")

    print("Saved Ishizawa-inspired field diagnostics.")


if __name__ == "__main__":
    main()
