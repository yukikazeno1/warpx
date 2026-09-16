#!/usr/bin/env python3
"""
Ishizawa-style particle-moment / generalized-Ohm diagnostics for the
particle-rich Harris-sheet snapshots.

This script is intended for the four (or more) plotfiles written by
run_ishizawa_particle_snapshot.sh.  It reconstructs coarse-grained species
velocity moments and a non-relativistic pressure tensor from the actual
particles, then evaluates the out-of-plane (WarpX y) species momentum balance
along an automatically located X-candidate line.

Geometry mapping:
  simulation plane : x-z
  out of plane     : y
  paper Ishizawa x : WarpX x
  paper Ishizawa y : WarpX z
  paper Ishizawa z : WarpX y

For species s, the diagnostic evaluates

  E_y + (u_s x B)_y
    = (1/(q_s n_s)) (div P_s)_y
      + (m_s/q_s) [du_sy/dt + u_s . grad u_sy]

with
  (u x B)_y = u_z B_x - u_x B_z
  (div P)_y = dP_yx/dx + dP_yz/dz.

The pressure tensor used here is the same non-relativistic fluid moment form
used in Ishizawa-style momentum balances,

  P_ij = m n <(v_i-u_i)(v_j-u_j)>.

Particles are mildly relativistic in the present pilot; velocities themselves
are recovered exactly from relativistic momenta p/(gamma m), while the fluid
pressure decomposition is kept in the conventional non-relativistic form for
comparison with Ishizawa & Horiuchi (2005).

Important limitations:
  * This pilot box does not contain d_i, so it cannot test the full
    l_mi < |z| < d_i gyroviscous-cancellation region.
  * The X point is only an X *candidate* unless the magnetic topology actually
    contains a clear separatrix/island.
  * Pressure gradients amplify PIC noise.  The default coarsening factor 4
    gives ~1024 particles/species/coarse-cell for a 64-PPC fine-grid run.

Outputs:
  ishizawa_particle_moments_summary.txt
  ishizawa_ion_ohm_profile.png
  ishizawa_electron_ohm_profile.png
  ishizawa_frozen_in_compare.png
  ishizawa_pressure_terms.png
  ishizawa_species_flow_profile.png
  ishizawa_moment_closure.txt
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
    p.add_argument("--runs-dir", default="runs_ishizawa")
    p.add_argument("--ppc", type=int, default=64)
    p.add_argument("--coarsen", type=int, default=4,
                   help="coarse cells per fine-cell block in x and z")
    p.add_argument("--density-cut", type=float, default=0.02,
                   help="only trust normalized profiles where n/n0 exceeds this")
    return p.parse_args()


def numeric_key(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else path


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
    lme = L * np.arcsinh(np.sqrt(rhoe0 / L))
    lmi = L * np.arcsinh(np.sqrt(rhoi0 / L))

    return dict(qe=qe, me=me, mi=mi, eps0=eps0, mu0=mu0, c=c, n0=n0,
                de=de, di=di, L=L, Te=Te, Ti=Ti, B0=B0, vA=vA,
                tauA=tauA, lme=lme, lmi=lmi)


def as_xz(a, nx, nz):
    a = np.asarray(a).squeeze()
    if a.shape == (nx, nz):
        return a
    if a.shape == (nz, nx):
        return a.T
    raise RuntimeError(f"unexpected field shape {a.shape}; expected {(nx,nz)} or {(nz,nx)}")


def block_average(a, c):
    nx, nz = a.shape
    if nx % c or nz % c:
        raise ValueError(f"grid {(nx,nz)} is not divisible by coarsen={c}")
    return a.reshape(nx//c, c, nz//c, c).mean(axis=(1,3))


def native_mesh(g, name, nx, nz):
    q = g["boxlib", name]
    return as_xz(q.to_ndarray(), nx, nz), str(q.units)


def si_mesh(g, name, unit, nx, nz):
    q = g["boxlib", name]
    try:
        return as_xz(q.to_value(unit), nx, nz)
    except Exception:
        return as_xz(q.to_ndarray(), nx, nz)


def particle_array(ad, species, name, unit=None):
    q = ad[species, name]
    if unit is not None:
        try:
            return np.asarray(q.to_value(unit))
        except Exception:
            pass
    return np.asarray(q.to_ndarray())


def load_mesh(path, coarsen):
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

    fields = {}
    for name, unit in [("Ex","V/m"),("Ey","V/m"),("Ez","V/m"),
                       ("Bx","T"),("By","T"),("Bz","T")]:
        fields[name] = block_average(si_mesh(g, name, unit, nx, nz), coarsen)

    rhoi, urhoi = native_mesh(g, "rho_ions", nx, nz)
    rhoe, urhoe = native_mesh(g, "rho_electrons", nx, nz)
    fields["rho_i"] = block_average(rhoi, coarsen)
    fields["rho_e"] = block_average(rhoe, coarsen)
    fields["rho_units"] = (urhoi, urhoe)

    xc = x.reshape(nx//coarsen, coarsen).mean(axis=1)
    zc = z.reshape(nz//coarsen, coarsen).mean(axis=1)
    fields.update(ds=ds, x=xc, z=zc, xlo=xlo, xhi=xhi, zlo=zlo, zhi=zhi,
                  dx=(xhi-xlo)/(nx//coarsen), dz=(zhi-zlo)/(nz//coarsen),
                  nx=nx//coarsen, nz=nz//coarsen,
                  t=ds.current_time.to_value("s"))
    return fields


def deposit_species(path, species, mass, mesh, p):
    ds = yt.load(path)
    ad = ds.all_data()

    # WarpX 2D XZ plotfiles expose physical z as particle_position_y.
    xp = particle_array(ad, species, "particle_position_x", "m")
    zp = particle_array(ad, species, "particle_position_y", "m")
    px = particle_array(ad, species, "particle_momentum_x", "kg*m/s")
    py = particle_array(ad, species, "particle_momentum_y", "kg*m/s")
    pz = particle_array(ad, species, "particle_momentum_z", "kg*m/s")
    w = particle_array(ad, species, "particle_weight")

    mc = mass * p["c"]
    gamma = np.sqrt(1.0 + (px*px + py*py + pz*pz)/(mc*mc))
    vx = px/(gamma*mass)
    vy = py/(gamma*mass)
    vz = pz/(gamma*mass)

    nx, nz = mesh["nx"], mesh["nz"]
    ix = np.floor((xp-mesh["xlo"])/mesh["dx"]).astype(np.int64)
    iz = np.floor((zp-mesh["zlo"])/mesh["dz"]).astype(np.int64)
    ix = np.clip(ix, 0, nx-1)
    iz = np.clip(iz, 0, nz-1)
    cell = ix*nz + iz
    ncell = nx*nz

    def bc(weight):
        return np.bincount(cell, weights=weight, minlength=ncell).reshape(nx,nz)

    sw = bc(w)
    tiny = np.finfo(float).tiny
    inv = 1.0/np.maximum(sw, tiny)

    ux = bc(w*vx)*inv
    uy = bc(w*vy)*inv
    uz = bc(w*vz)*inv

    # Central velocity covariance.  Ratios are independent of the unknown 2D
    # macro-particle volume normalization; absolute density comes from rho_s.
    cxx = bc(w*vx*vx)*inv - ux*ux
    cyy = bc(w*vy*vy)*inv - uy*uy
    czz = bc(w*vz*vz)*inv - uz*uz
    cxy = bc(w*vx*vy)*inv - ux*uy
    cxz = bc(w*vx*vz)*inv - ux*uz
    cyz = bc(w*vy*vz)*inv - uy*uz

    for a in (cxx,cyy,czz):
        a[a < 0] = np.maximum(a[a < 0], 0.0)

    return dict(Nmacro=len(w), sw=sw, ux=ux, uy=uy, uz=uz,
                cxx=cxx, cyy=cyy, czz=czz, cxy=cxy, cxz=cxz, cyz=cyz)


def reconstruct_Ay(Bx, Bz, x, z):
    nx, nz = Bx.shape
    dx = x[1]-x[0]
    dz = z[1]-z[0]
    bzh = np.fft.rfft(Bz, axis=0)
    k = 2.0*np.pi*np.fft.rfftfreq(nx, d=dx)
    ayh = np.zeros_like(bzh, dtype=complex)
    ayh[1:,:] = bzh[1:,:]/(1j*k[1:,None])
    ay = np.fft.irfft(ayh, n=nx, axis=0)
    bx0 = np.mean(Bx, axis=0)
    ay0 = np.zeros(nz)
    for j in range(1,nz):
        ay0[j] = ay0[j-1] - 0.5*(bx0[j-1]+bx0[j])*dz
    iz0 = int(np.argmin(np.abs(z)))
    ay0 -= ay0[iz0]
    return ay + ay0[None,:]


def x_candidate(Ay, x, z):
    iz = int(np.argmin(np.abs(z)))
    line = Ay[:,iz]
    dx = x[1]-x[0]
    dz = z[1]-z[0]
    left = np.roll(line,1)
    right = np.roll(line,-1)
    extrema = ((line>=left)&(line>=right)) | ((line<=left)&(line<=right))
    d2x = (right-2*line+left)/dx**2
    d2z = (Ay[:,iz+1]-2*Ay[:,iz]+Ay[:,iz-1])/dz**2
    cand = np.where(extrema & (d2x*d2z < 0))[0]
    if len(cand):
        # Choose the strongest saddle curvature, not the largest flux excursion.
        score = np.abs(d2x[cand]*d2z[cand])
        return int(cand[np.argmax(score)]), iz
    return int(np.argmax(np.abs(d2x))), iz


def time_slope(stack, times):
    """Least-squares d/dt across all particle-rich states, cell by cell."""
    t = np.asarray(times, dtype=float)
    tc = t - np.mean(t)
    den = np.sum(tc*tc)
    return np.sum(tc[:,None,None]*stack, axis=0)/den


def grad_xz(a, dx, dz):
    return (np.gradient(a, dx, axis=0, edge_order=2),
            np.gradient(a, dz, axis=1, edge_order=2))


def species_balance(mesh, mom, n, mass, charge, duy_dt):
    ux,uy,uz = mom["ux"],mom["uy"],mom["uz"]
    Bx,Bz,Ey = mesh["Bx"],mesh["Bz"],mesh["Ey"]
    lhs = Ey + uz*Bx - ux*Bz

    Pxy = mass*n*mom["cxy"]
    Pyz = mass*n*mom["cyz"]
    dPxy_dx = np.gradient(Pxy, mesh["dx"], axis=0, edge_order=2)
    dPyz_dz = np.gradient(Pyz, mesh["dz"], axis=1, edge_order=2)
    divPy = dPxy_dx + dPyz_dz

    duy_dx, duy_dz = grad_xz(uy, mesh["dx"], mesh["dz"])
    conv_acc = ux*duy_dx + uz*duy_dz

    qn = charge*n
    pressure = np.full_like(lhs, np.nan)
    temporal = np.full_like(lhs, np.nan)
    convective = np.full_like(lhs, np.nan)
    good = n > 0
    pressure[good] = divPy[good]/qn[good]
    temporal[good] = (mass/charge)*duy_dt[good]
    convective[good] = (mass/charge)*conv_acc[good]
    rhs = pressure + temporal + convective
    residual = lhs-rhs

    return dict(lhs=lhs, pressure=pressure, temporal=temporal,
                convective=convective, rhs=rhs, residual=residual,
                Pxy=Pxy, Pyz=Pyz)


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    p = physical_params()

    pattern = str(Path(args.runs_dir)/f"ppc{args.ppc}"/"diags"/"diag1*")
    files = sorted(glob.glob(pattern), key=numeric_key)
    if len(files) < 3:
        raise RuntimeError(f"need at least 3 particle-rich plotfiles; found {len(files)} at {pattern}")

    print("="*78)
    print("Ishizawa particle-moment / generalized-Ohm analysis")
    print("="*78)
    print(f"plotfiles : {len(files)}")
    print(f"coarsen   : {args.coarsen}")
    print(f"PPC       : {args.ppc}")

    meshes=[]
    emoms=[]
    imoms=[]
    times=[]

    for i,fn in enumerate(files,1):
        M=load_mesh(fn,args.coarsen)
        E=deposit_species(fn,"electrons",p["me"],M,p)
        I=deposit_species(fn,"ions",p["mi"],M,p)
        meshes.append(M); emoms.append(E); imoms.append(I); times.append(M["t"])
        print(f"  {i}/{len(files)} {os.path.basename(fn)}: Ne={E['Nmacro']}, Ni={I['Nmacro']}, t={M['t']:.6e} s")

    # Native rho arrays have already been cross-validated in the field-only
    # diagnostic: their numerical values are physical C/m^3 despite yt's
    # dimensionless label.
    n_i_stack=np.stack([np.abs(M["rho_i"])/p["qe"] for M in meshes])
    n_e_stack=np.stack([np.abs(M["rho_e"])/p["qe"] for M in meshes])
    n_i=np.mean(n_i_stack,axis=0)
    n_e=np.mean(n_e_stack,axis=0)

    def avg_mom(moms,key): return np.mean(np.stack([m[key] for m in moms]),axis=0)
    Eavg={k:avg_mom(emoms,k) for k in ["ux","uy","uz","cxx","cyy","czz","cxy","cxz","cyz"]}
    Iavg={k:avg_mom(imoms,k) for k in ["ux","uy","uz","cxx","cyy","czz","cxy","cxz","cyz"]}
    Mavg={k:np.mean(np.stack([M[k] for M in meshes]),axis=0) for k in ["Ex","Ey","Ez","Bx","By","Bz"]}
    Mavg.update(x=meshes[0]["x"],z=meshes[0]["z"],dx=meshes[0]["dx"],dz=meshes[0]["dz"],
                nx=meshes[0]["nx"],nz=meshes[0]["nz"])

    due_dt=time_slope(np.stack([m["uy"] for m in emoms]),times)
    dui_dt=time_slope(np.stack([m["uy"] for m in imoms]),times)

    Be=reconstruct_Ay(Mavg["Bx"],Mavg["Bz"],Mavg["x"],Mavg["z"])
    ix,iz=x_candidate(Be,Mavg["x"],Mavg["z"])
    xX=Mavg["x"][ix]/p["L"]

    EB=species_balance(Mavg,Eavg,n_e,p["me"],-p["qe"],due_dt)
    IB=species_balance(Mavg,Iavg,n_i,p["mi"],+p["qe"],dui_dt)

    E0=p["vA"]*p["B0"]
    z=Mavg["z"]/p["L"]
    density_mask=(n_i[ix,:]/p["n0"] >= args.density_cut) & (n_e[ix,:]/p["n0"] >= args.density_cut)

    def masked(a):
        y=np.asarray(a[ix,:],float).copy()/E0
        y[~density_mask]=np.nan
        return y

    # Closure metrics only in the trusted-density core.
    core=density_mask & (np.abs(z)<=2.0)
    def relerr(B):
        a=B["lhs"][ix,:][core]
        b=B["rhs"][ix,:][core]
        den=np.sqrt(np.mean(a*a))
        return np.sqrt(np.mean((a-b)**2))/den if den>0 else np.nan

    eerr=relerr(EB); ierr=relerr(IB)

    with open("ishizawa_particle_moments_summary.txt","w") as f:
        f.write("Ishizawa particle-moment / generalized-Ohm summary\n")
        f.write("==================================================\n\n")
        f.write(f"particle-rich states = {len(files)}\n")
        f.write(f"coarsening factor = {args.coarsen}\n")
        f.write(f"X-candidate x/L = {xX:.8f}\n")
        f.write(f"density trust threshold n/n0 = {args.density_cut:.6f}\n")
        f.write(f"electron closure relative RMS, |z|<=2L = {eerr:.8e}\n")
        f.write(f"ion closure relative RMS, |z|<=2L = {ierr:.8e}\n")
        f.write(f"l_me/L = {p['lme']/p['L']:.8f}\n")
        f.write(f"l_mi/L = {p['lmi']/p['L']:.8f}\n")
        f.write(f"d_i/L = {p['di']/p['L']:.8f}\n")
        f.write("NOTE: d_i is outside the present box; this can only test the inner kinetic region.\n")

    def plot_balance(B,title,out):
        fig,ax=plt.subplots(figsize=(8,5.5))
        ax.plot(z,masked(B["lhs"]),label=r"$E_y+(u\times B)_y$")
        ax.plot(z,masked(B["pressure"]),label=r"$(\nabla\cdot P)_y/(qn)$")
        ax.plot(z,masked(B["convective"]),label="convective inertia")
        ax.plot(z,masked(B["temporal"]),label="temporal inertia")
        ax.plot(z,masked(B["rhs"]),"--",lw=2,label="RHS sum")
        ax.axhline(0,lw=0.8)
        for s in (-1,1):
            ax.axvline(s*p["lme"]/p["L"],ls=":",lw=1)
            ax.axvline(s*p["lmi"]/p["L"],ls="--",lw=1)
        ax.set_xlabel(r"$z/L$")
        ax.set_ylabel(r"normalized by $v_A B_0$")
        ax.set_title(title+rf"; X candidate $x/L={xX:.3f}$")
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out,dpi=200); plt.close(fig)

    plot_balance(IB,"ion momentum balance","ishizawa_ion_ohm_profile.png")
    plot_balance(EB,"electron momentum balance","ishizawa_electron_ohm_profile.png")

    fig,ax=plt.subplots(figsize=(8,5.5))
    ax.plot(z,masked(IB["lhs"]),label="ions")
    ax.plot(z,masked(EB["lhs"]),label="electrons")
    ax.plot(z,Mavg["Ey"][ix,:]/E0,label=r"$E_y$",alpha=0.8)
    for s in (-1,1):
        ax.axvline(s*p["lmi"]/p["L"],ls="--",lw=1)
    ax.set_xlabel(r"$z/L$"); ax.set_ylabel(r"frozen-in violation / $v_AB_0$")
    ax.set_title("species frozen-in violation")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig("ishizawa_frozen_in_compare.png",dpi=200); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5.5))
    ax.plot(z,masked(IB["pressure"]),label="ion pressure tensor")
    ax.plot(z,masked(EB["pressure"]),label="electron pressure tensor")
    ax.axhline(0,lw=0.8)
    for s in (-1,1):
        ax.axvline(s*p["lme"]/p["L"],ls=":",lw=1)
        ax.axvline(s*p["lmi"]/p["L"],ls="--",lw=1)
    ax.set_xlabel(r"$z/L$"); ax.set_ylabel(r"$(\nabla\cdot P)_y/(qn)/(v_AB_0)$")
    ax.set_title("nongyrotropic pressure-tensor contribution")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig("ishizawa_pressure_terms.png",dpi=200); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5.5))
    ax.plot(z,Eavg["uy"][ix,:]/p["vA"],label=r"$u_{ey}/v_A$")
    ax.plot(z,Iavg["uy"][ix,:]/p["vA"],label=r"$u_{iy}/v_A$")
    ax2=ax.twinx()
    ax2.plot(z,n_e[ix,:]/p["n0"],"--",alpha=0.6,label=r"$n_e/n_0$")
    ax2.plot(z,n_i[ix,:]/p["n0"],":",alpha=0.6,label=r"$n_i/n_0$")
    ax.set_xlabel(r"$z/L$"); ax.set_ylabel(r"out-of-plane flow / $v_A$")
    ax2.set_ylabel(r"density / $n_0$")
    ax.grid(alpha=0.25)
    lines=ax.get_lines()+ax2.get_lines(); labels=[l.get_label() for l in lines]
    ax.legend(lines,labels,fontsize=8,loc="best")
    fig.tight_layout(); fig.savefig("ishizawa_species_flow_profile.png",dpi=200); plt.close(fig)

    # Detailed closure table along the X-candidate line.
    cols=np.column_stack([
        z,n_e[ix,:]/p["n0"],n_i[ix,:]/p["n0"],
        EB["lhs"][ix,:]/E0,EB["pressure"][ix,:]/E0,
        EB["convective"][ix,:]/E0,EB["temporal"][ix,:]/E0,EB["rhs"][ix,:]/E0,
        IB["lhs"][ix,:]/E0,IB["pressure"][ix,:]/E0,
        IB["convective"][ix,:]/E0,IB["temporal"][ix,:]/E0,IB["rhs"][ix,:]/E0])
    np.savetxt("ishizawa_moment_closure.txt",cols,
               header="z/L ne/n0 ni/n0 e_LHS e_P e_conv e_dt e_RHS i_LHS i_P i_conv i_dt i_RHS")

    print("="*78)
    print(f"X candidate x/L = {xX:.6f}")
    print(f"electron closure relRMS = {eerr:.6e}")
    print(f"ion closure relRMS      = {ierr:.6e}")
    print("Saved Ishizawa particle-moment diagnostics.")


if __name__ == "__main__":
    main()
