#!/usr/bin/env python3
"""Phase-resolved field/energy diagnostics for the nonlinear m=1 breathing mode.

This script uses the late-time large-scale saturation history to select one
complete breathing cycle and four representative phases:

  P0  expanded / Psi maximum
  P1  contracting (midway from maximum to minimum)
  P2  contracted / Psi minimum
  P3  expanding (midway from minimum to next maximum)

For each phase it loads the corresponding WarpX plotfile, aligns the system-
scale m=1 O point to x=0 using the complex m=1 phase of A_y, and diagnoses:

  * corrected curl-B J_y/J0
  * magnetic energy density B^2/B0^2
  * reconnection electric field E_y/(c B0)
  * ion/electron number density / n0
  * electromagnetic work J dot E /(J0 c B0)
  * vertical magnetic-pressure and magnetic-tension forces
  * integrated/averaged scalar phase metrics

The purpose is to test whether the observed island breathing is accompanied by
coherent magnetic compression, current-sheet strengthening/weakening and
field-to-particle energy exchange.

Outputs
-------
ishizawa_breathing_phase_table.txt
ishizawa_breathing_phase_overview.png
ishizawa_breathing_phase_density.png
ishizawa_breathing_phase_jdote.png
ishizawa_breathing_phase_force_z.png
ishizawa_breathing_expanded_minus_contracted.png
ishizawa_breathing_phase_summary.txt
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import yt

from analyze_ishizawa_scale_40000 import (
    QE, EPS0, MU0, C,
    params, as_xz, reconstruct_Ay, corrected_jy,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs_ishizawa_scale/ppc49")
    p.add_argument("--history", required=True,
                   help="ishizawa_saturation_history.txt from the same run")
    p.add_argument("--cycle-tmin", type=float, default=2.8,
                   help="choose the latest complete Psi cycle after this time")
    p.add_argument("--cycle-tmax", type=float, default=4.0)
    p.add_argument("--period-min", type=float, default=0.45)
    p.add_argument("--period-max", type=float, default=0.85)
    p.add_argument("--prominence-frac", type=float, default=0.18)
    p.add_argument("--zoom-z-de", type=float, default=12.0)
    p.add_argument("--metric-core-z-de", type=float, default=12.0)
    p.add_argument("--force-x-halfwidth-de", type=float, default=8.0)
    return p.parse_args()


def numeric_key(path):
    name = os.path.basename(path.rstrip("/"))
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


def step_from_plotfile(path):
    name = os.path.basename(path.rstrip("/"))
    if not name.startswith("diag1"):
        return -1
    s = name[len("diag1"):]
    return int(s) if s.isdigit() else -1


def load_history(path):
    a = np.loadtxt(path)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[1] < 8:
        raise RuntimeError("Expected v2 saturation history with >=8 columns")
    return dict(
        step=a[:,0].astype(int),
        t=a[:,1],
        psi=a[:,2],
        psi_m1=a[:,3],
        width=a[:,4],
        jymax=a[:,5],
        jyabs=a[:,6],
        jyp=a[:,7],
    )


def select_cycle(H, tmin, tmax, period_min, period_max, prominence_frac):
    m = (
        np.isfinite(H["t"]) & np.isfinite(H["psi"]) &
        (H["t"] >= tmin) & (H["t"] <= tmax)
    )
    t = H["t"][m]
    y = H["psi"][m]
    idx_global = np.flatnonzero(m)
    if len(t) < 10:
        raise RuntimeError("Too few points in requested cycle-search interval")

    dt = float(np.median(np.diff(t)))
    distance = max(1, int(round(period_min/dt)))
    prom = prominence_frac*np.std(y)
    peaks, _ = find_peaks(y, distance=distance, prominence=prom)
    troughs, _ = find_peaks(-y, distance=max(1,distance//2),
                            prominence=0.5*prom)

    candidates = []
    for ia, ib in zip(peaks[:-1], peaks[1:]):
        T = t[ib]-t[ia]
        if not (period_min <= T <= period_max):
            continue
        q = troughs[(troughs > ia) & (troughs < ib)]
        if len(q) == 0:
            continue
        iq = q[np.argmin(y[q])]
        candidates.append((ia, iq, ib))

    if not candidates:
        raise RuntimeError(
            "No complete peak-trough-peak cycle found; adjust --cycle-tmin, "
            "--cycle-tmax, --period-min/max or prominence."
        )

    # Latest complete cycle.
    ia, iq, ib = candidates[-1]

    def nearest(local_target):
        j = int(np.argmin(np.abs(t-local_target)))
        return idx_global[j]

    ga = idx_global[ia]
    gq = idx_global[iq]
    gb = idx_global[ib]
    gcontract = nearest(0.5*(H["t"][ga]+H["t"][gq]))
    gexpand = nearest(0.5*(H["t"][gq]+H["t"][gb]))

    # Four non-duplicate representative phases.
    inds = [ga, gcontract, gq, gexpand]
    labels = ["expanded", "contracting", "contracted", "expanding"]
    return labels, inds, gb


def load_full_fields(path):
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
    g = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )

    def fld(name, unit):
        q = g["boxlib", name]
        try:
            a = q.to_value(unit)
        except Exception:
            a = q.to_ndarray()
        return as_xz(a, nx, nz)

    return dict(
        ds=ds, x=x, z=z, dx=dx, dz=dz,
        Ex=fld("Ex","V/m"), Ey=fld("Ey","V/m"), Ez=fld("Ez","V/m"),
        Bx=fld("Bx","T"), By=fld("By","T"), Bz=fld("Bz","T"),
        jx=fld("jx","A/m**2"), jy=fld("jy","A/m**2"),
        jz=fld("jz","A/m**2"),
        rho=fld("rho","C/m**3"),
        rho_e=fld("rho_electrons","C/m**3"),
        rho_i=fld("rho_ions","C/m**3"),
    )


def m1_O_index(Ay, x):
    j0 = Ay.shape[1]//2
    c = np.fft.rfft(Ay[:,j0])/Ay.shape[0]
    phi = np.angle(c[1])
    Lx = Ay.shape[0]*(x[1]-x[0])
    k1 = 2*np.pi/Lx
    xmin = x[0]
    xO = xmin - phi/k1
    xO = xmin + np.mod(xO-xmin, Lx)
    iO = int(np.argmin(np.abs(x-xO)))
    return iO, xO


def align_x(a, iO):
    """Roll x so the m=1 O point is at the central x index."""
    n = a.shape[0]
    shift = n//2 - iO
    return np.roll(a, shift, axis=0), shift


def sym_limit(a, quant=99.0):
    q = np.nanpercentile(np.abs(a), quant)
    return max(float(q), 1e-30)


def process_phase(F, P):
    Ay = reconstruct_Ay(F["Bx"], F["Bz"], F["x"], F["z"])
    iO, xO = m1_O_index(Ay, F["x"])

    aligned = {}
    for name in [
        "Ex","Ey","Ez","Bx","By","Bz","jx","jy","jz","rho","rho_e","rho_i"
    ]:
        aligned[name], shift = align_x(F[name], iO)
    AyA, _ = align_x(Ay, iO)

    xrel = (
        np.arange(len(F["x"])) - len(F["x"])//2
    )*F["dx"]

    Bx,By,Bz = aligned["Bx"],aligned["By"],aligned["Bz"]
    Ex,Ey,Ez = aligned["Ex"],aligned["Ey"],aligned["Ez"]
    jx,jy,jz = aligned["jx"],aligned["jy"],aligned["jz"]

    jy_curl = corrected_jy(Bx,Bz,F["dx"],F["dz"])
    B2 = Bx*Bx + By*By + Bz*Bz
    JdotE = jx*Ex + jy*Ey + jz*Ez

    ni = aligned["rho_i"]/QE
    ne = -aligned["rho_e"]/QE

    # Magnetic-force decomposition in x-z plane:
    # (JxB)_z = -d_z(B^2/2mu0) + (B dot grad)B_z/mu0.
    pmag = B2/(2*MU0)
    Fp_z = -np.gradient(pmag,F["dz"],axis=1,edge_order=2)
    dBz_dx = np.gradient(Bz,F["dx"],axis=0,edge_order=2)
    dBz_dz = np.gradient(Bz,F["dz"],axis=1,edge_order=2)
    Ft_z = (Bx*dBz_dx + Bz*dBz_dz)/MU0
    FL_z = jx*By - jy*Bx

    return dict(
        Ay=AyA, x=xrel, z=F["z"], xO=xO,
        Bx=Bx,By=By,Bz=Bz,Ex=Ex,Ey=Ey,Ez=Ez,
        jx=jx,jy=jy,jz=jz,jy_curl=jy_curl,
        B2=B2,JdotE=JdotE,ni=ni,ne=ne,
        Fp_z=Fp_z,Ft_z=Ft_z,FL_z=FL_z,
    )


def phase_metrics(D, F, P, core_z_de, force_x_halfwidth_de):
    x,z=D["x"],D["z"]
    core = np.abs(z) <= core_z_de*P["de"]
    xforce = np.abs(x) <= force_x_halfwidth_de*P["de"]

    dA = F["dx"]*F["dz"]
    normUB = (P["B0"]**2/(2*MU0))*P["de"]**2
    normUE = (EPS0*(C*P["B0"])**2/2)*P["de"]**2
    normJE = P["J0"]*C*P["B0"]*P["de"]**2
    normF = P["J0"]*P["B0"]

    UBz = np.sum(D["Bz"][:,core]**2/(2*MU0))*dA/normUB
    UBtot = np.sum(D["B2"][:,core]/(2*MU0))*dA/normUB
    E2 = D["Ex"]**2+D["Ey"]**2+D["Ez"]**2
    UE = np.sum(E2[:,core]*EPS0/2)*dA/normUE
    PJE = np.sum(D["JdotE"][:,core])*dA/normJE

    jyp = np.nanpercentile(D["jy_curl"][:,core]/P["J0"],99.5)
    nim = np.nanmean(D["ni"][:,core]/P["n0"])
    nem = np.nanmean(D["ne"][:,core]/P["n0"])

    # Mean absolute force levels near O point, normalized to J0 B0.
    box = np.ix_(xforce,core)
    Ft = np.nanmean(np.abs(D["Ft_z"][box]))/normF
    Fp = np.nanmean(np.abs(D["Fp_z"][box]))/normF
    FL = np.nanmean(np.abs(D["FL_z"][box]))/normF

    return dict(
        UBz=UBz,UBtot=UBtot,UE=UE,PJE=PJE,jyp=jyp,
        ni=nim,ne=nem,Ft=Ft,Fp=Fp,FL=FL,
    )


def panel4(phases, key, zmask, P, title, cbar_label, symmetric=False,
           transform=lambda x:x):
    vals=[transform(D[key][:,zmask]) for D in phases]
    if symmetric:
        vmax=max(sym_limit(v) for v in vals)
        vmin=-vmax
    else:
        lo=min(np.nanpercentile(v,1) for v in vals)
        hi=max(np.nanpercentile(v,99) for v in vals)
        vmin,vmax=lo,hi

    fig,axs=plt.subplots(1,4,figsize=(16,4.2),sharex=True,sharey=True)
    for ax,D,v in zip(axs,phases,vals):
        X,Z=np.meshgrid(D["x"]/P["de"],D["z"][zmask]/P["de"],indexing="ij")
        p=ax.pcolormesh(X,Z,v,shading="auto",vmin=vmin,vmax=vmax)
        ax.set_title(D["label"]+fr"\n$\omega_{{ci}}t={D['t']:.3f}$")
        ax.set_xlabel(r"$x/d_e$")
        ax.grid(alpha=.12)
    axs[0].set_ylabel(r"$z/d_e$")
    fig.colorbar(p,ax=axs.ravel().tolist(),shrink=.88,label=cbar_label)
    fig.suptitle(title)
    fig.subplots_adjust(left=.06,right=.94,bottom=.13,top=.82,wspace=.08)
    return fig


def main():
    args=parse_args()
    yt.funcs.mylog.setLevel(40)
    P=params()
    H=load_history(args.history)

    labels,inds,next_peak = select_cycle(
        H,args.cycle_tmin,args.cycle_tmax,
        args.period_min,args.period_max,args.prominence_frac
    )

    plotfiles=sorted(
        glob.glob(str(Path(args.run_dir)/"diags"/"diag1*")),
        key=numeric_key
    )
    bystep={step_from_plotfile(p):p for p in plotfiles}

    phases=[]
    rows=[]
    for label,ih in zip(labels,inds):
        step=int(H["step"][ih])
        if step not in bystep:
            # Robust fallback to nearest available diagnostic step.
            avail=np.asarray(sorted(bystep))
            step=int(avail[np.argmin(np.abs(avail-step))])
        fn=bystep[step]
        F=load_full_fields(fn)
        D=process_phase(F,P)
        D["label"]=label
        D["step"]=step
        D["t"]=P["wci"]*F["ds"].current_time.to_value("s")
        D["psi"]=float(H["psi"][ih])
        D["width"]=float(H["width"][ih])
        M=phase_metrics(D,F,P,args.metric_core_z_de,args.force_x_halfwidth_de)
        D.update(M)
        phases.append(D)
        rows.append([
            step,D["t"],D["psi"],D["width"],D["xO"]/P["de"],
            M["UBz"],M["UBtot"],M["UE"],M["PJE"],M["jyp"],
            M["ni"],M["ne"],M["Ft"],M["Fp"],M["FL"],
        ])

    np.savetxt(
        "ishizawa_breathing_phase_table.txt",np.asarray(rows),
        header=(
            "step omega_ci_t Psi_B0de width_de xO_de "
            "UBz_norm UBtotal_norm UE_norm int_JdotE_norm Jy_p99p5_J0 "
            "mean_ni_n0 mean_ne_n0 "
            "mean_abs_tension_z_J0B0 mean_abs_magpressure_z_J0B0 "
            "mean_abs_Lorentz_z_J0B0"
        )
    )

    zmask=np.abs(phases[0]["z"])<=args.zoom_z_de*P["de"]

    fig=panel4(
        phases,"jy_curl",zmask,P,
        "Breathing phases: corrected current density",
        r"$J_y/J_0$",True,lambda a:a/P["J0"]
    )
    fig.savefig("ishizawa_breathing_phase_Jy.png",dpi=210)
    plt.close(fig)

    fig=panel4(
        phases,"B2",zmask,P,
        "Breathing phases: magnetic energy-density proxy",
        r"$B^2/B_0^2$",False,lambda a:a/P["B0"]**2
    )
    fig.savefig("ishizawa_breathing_phase_Benergy.png",dpi=210)
    plt.close(fig)

    fig=panel4(
        phases,"Ey",zmask,P,
        "Breathing phases: reconnection electric field",
        r"$E_y/(cB_0)$",True,lambda a:a/(C*P["B0"])
    )
    fig.savefig("ishizawa_breathing_phase_Ey.png",dpi=210)
    plt.close(fig)

    # Density: two rows (ions/electrons), four phases.
    fig,axs=plt.subplots(2,4,figsize=(16,7.2),sharex=True,sharey=True)
    vals_i=[D["ni"][:,zmask]/P["n0"] for D in phases]
    vals_e=[D["ne"][:,zmask]/P["n0"] for D in phases]
    lo=min(np.nanpercentile(v,1) for v in vals_i+vals_e)
    hi=max(np.nanpercentile(v,99) for v in vals_i+vals_e)
    for col,D in enumerate(phases):
        X,Z=np.meshgrid(D["x"]/P["de"],D["z"][zmask]/P["de"],indexing="ij")
        p0=axs[0,col].pcolormesh(X,Z,vals_i[col],shading="auto",vmin=lo,vmax=hi)
        axs[1,col].pcolormesh(X,Z,vals_e[col],shading="auto",vmin=lo,vmax=hi)
        axs[0,col].set_title(D["label"]+fr"\n$\omega_{{ci}}t={D['t']:.3f}$")
        axs[1,col].set_xlabel(r"$x/d_e$")
    axs[0,0].set_ylabel(r"ions: $z/d_e$")
    axs[1,0].set_ylabel(r"electrons: $z/d_e$")
    fig.colorbar(p0,ax=axs.ravel().tolist(),shrink=.88,label=r"$n_s/n_0$")
    fig.suptitle("Breathing phases: species number densities")
    fig.subplots_adjust(left=.06,right=.94,bottom=.10,top=.87,wspace=.08,hspace=.10)
    fig.savefig("ishizawa_breathing_phase_density.png",dpi=210)
    plt.close(fig)

    fig=panel4(
        phases,"JdotE",zmask,P,
        r"Breathing phases: electromagnetic work $\mathbf{J}\cdot\mathbf{E}$",
        r"$\mathbf{J}\cdot\mathbf{E}/(J_0cB_0)$",True,
        lambda a:a/(P["J0"]*C*P["B0"])
    )
    fig.savefig("ishizawa_breathing_phase_jdote.png",dpi=210)
    plt.close(fig)

    # Force-z comparison along an x-window around O.
    fig,axs=plt.subplots(1,4,figsize=(16,4.4),sharex=True,sharey=True)
    for ax,D in zip(axs,phases):
        xm=np.abs(D["x"])<=args.force_x_halfwidth_de*P["de"]
        z=D["z"]/P["de"]
        norm=P["J0"]*P["B0"]
        ax.plot(z,np.mean(D["Ft_z"][xm,:],axis=0)/norm,label="tension")
        ax.plot(z,np.mean(D["Fp_z"][xm,:],axis=0)/norm,label="-grad Pmag")
        ax.plot(z,np.mean(D["FL_z"][xm,:],axis=0)/norm,label="JxB")
        ax.axhline(0,lw=.8)
        ax.set_xlim(-args.zoom_z_de,args.zoom_z_de)
        ax.set_title(D["label"]+fr"\n$\omega_{{ci}}t={D['t']:.3f}$")
        ax.set_xlabel(r"$z/d_e$")
        ax.grid(alpha=.2)
    axs[0].set_ylabel(r"vertical magnetic force / $(J_0B_0)$")
    axs[-1].legend(fontsize=8)
    fig.suptitle("Vertical magnetic-pressure/tension force through the island")
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_phase_force_z.png",dpi=210)
    plt.close(fig)

    # Compact overview: Jy, B2, Ey, JdotE.
    fig,axs=plt.subplots(4,4,figsize=(16,13),sharex=True,sharey=True)
    specs=[
        ("jy_curl",lambda a:a/P["J0"],r"$J_y/J_0$",True),
        ("B2",lambda a:a/P["B0"]**2,r"$B^2/B_0^2$",False),
        ("Ey",lambda a:a/(C*P["B0"]),r"$E_y/(cB_0)$",True),
        ("JdotE",lambda a:a/(P["J0"]*C*P["B0"]),
         r"$J\cdot E/(J_0cB_0)$",True),
    ]
    for r,(key,trans,label,sym) in enumerate(specs):
        arr=[trans(D[key][:,zmask]) for D in phases]
        if sym:
            vmax=max(sym_limit(v) for v in arr); vmin=-vmax
        else:
            vmin=min(np.nanpercentile(v,1) for v in arr)
            vmax=max(np.nanpercentile(v,99) for v in arr)
        for c,(D,v) in enumerate(zip(phases,arr)):
            X,Z=np.meshgrid(D["x"]/P["de"],D["z"][zmask]/P["de"],indexing="ij")
            p=axs[r,c].pcolormesh(X,Z,v,shading="auto",vmin=vmin,vmax=vmax)
            if r==0:
                axs[r,c].set_title(D["label"]+fr"\n$\omega_{{ci}}t={D['t']:.3f}$")
            if r==3:
                axs[r,c].set_xlabel(r"$x/d_e$")
            if c==0:
                axs[r,c].set_ylabel(label+"\n"+r"$z/d_e$")
        fig.colorbar(p,ax=axs[r,:].tolist(),shrink=.78)
    fig.suptitle("Four-phase structure of the nonlinear island breathing cycle")
    fig.subplots_adjust(left=.07,right=.94,bottom=.07,top=.91,wspace=.08,hspace=.10)
    fig.savefig("ishizawa_breathing_phase_overview.png",dpi=210)
    plt.close(fig)

    # Expanded - contracted direct differences.
    A=phases[0]; Cc=phases[2]
    fig,axs=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    diffs=[
        ((A["B2"]-Cc["B2"])/P["B0"]**2,r"$\Delta(B^2/B_0^2)$"),
        ((A["jy_curl"]-Cc["jy_curl"])/P["J0"],r"$\Delta(J_y/J_0)$"),
        ((A["ni"]-Cc["ni"])/P["n0"],r"$\Delta(n_i/n_0)$"),
        ((A["JdotE"]-Cc["JdotE"])/(P["J0"]*C*P["B0"]),
         r"$\Delta[J\cdot E/(J_0cB_0)]$"),
    ]
    X,Z=np.meshgrid(A["x"]/P["de"],A["z"][zmask]/P["de"],indexing="ij")
    for ax,(v,label) in zip(axs.ravel(),diffs):
        vv=v[:,zmask]
        lim=sym_limit(vv)
        p=ax.pcolormesh(X,Z,vv,shading="auto",vmin=-lim,vmax=lim)
        ax.set_title(label+"  (expanded - contracted)")
        ax.set_xlabel(r"$x/d_e$"); ax.set_ylabel(r"$z/d_e$")
        fig.colorbar(p,ax=ax,shrink=.82)
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_expanded_minus_contracted.png",dpi=210)
    plt.close(fig)

    with open("ishizawa_breathing_phase_summary.txt","w") as f:
        f.write("Ishizawa-scale four-phase breathing diagnostics\n")
        f.write("================================================\n\n")
        f.write("Selected late complete cycle\n")
        f.write("----------------------------\n")
        f.write(
            f"peak1 t={H['t'][inds[0]]:.8f}, "
            f"trough t={H['t'][inds[2]]:.8f}, "
            f"next peak t={H['t'][next_peak]:.8f}\n"
        )
        f.write(
            f"cycle period Delta(omega_ci*t) = "
            f"{H['t'][next_peak]-H['t'][inds[0]]:.8f}\n\n"
        )
        f.write(
            "phase step omega_ci_t Psi width_de xO_de UBz UBtotal UE "
            "intJdotE Jy_p99.5 ni ne |Tz| |Pmag_z| |JxB_z|\n"
        )
        for label,row in zip(labels,rows):
            f.write(label+" "+" ".join(f"{x:.8e}" for x in row)+"\n")

        f.write("\nInterpretation guide\n")
        f.write("--------------------\n")
        f.write("Compare expanded versus contracted states first.  If UBz/B2, current,\n")
        f.write("density and magnetic-force structure change coherently with Psi and width,\n")
        f.write("the breathing is a real field/plasma compression mode rather than a purely\n")
        f.write("geometric contour effect.  The sign and phase of integrated J.E indicate\n")
        f.write("whether electromagnetic energy is transferred to particles (J.E>0) or\n")
        f.write("returned to the fields (J.E<0) over the breathing cycle.\n")

    print("Selected phases:")
    for D in phases:
        print(
            f"  {D['label']:11s} step={D['step']:6d} "
            f"omega_ci*t={D['t']:.4f} Psi={D['psi']:.4f} "
            f"w/de={D['width']:.3f}"
        )
    print("Saved ishizawa_breathing_phase_table.txt")
    print("Saved ishizawa_breathing_phase_overview.png")
    print("Saved ishizawa_breathing_phase_Jy.png")
    print("Saved ishizawa_breathing_phase_Benergy.png")
    print("Saved ishizawa_breathing_phase_Ey.png")
    print("Saved ishizawa_breathing_phase_density.png")
    print("Saved ishizawa_breathing_phase_jdote.png")
    print("Saved ishizawa_breathing_phase_force_z.png")
    print("Saved ishizawa_breathing_expanded_minus_contracted.png")
    print("Saved ishizawa_breathing_phase_summary.txt")


if __name__=="__main__":
    main()
