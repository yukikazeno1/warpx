#!/usr/bin/env python3
"""Thermodynamic decomposition of the dominant non-affine Pxx restoring force.

The previous non-affine diagnostic shows that the restoring pressure response
is dominated by Pxx.  This script asks *why* Pxx changes.

Using the two near-zero-q crossings as the reference state,

    Pxx = m n cxx,

with cxx=< (v_x-u_x)^2 >, the perturbation is split exactly as

    delta Pxx =
        m cxx0 delta n
      + m n0 delta cxx
      + m delta n delta cxx.

These are labelled density/compression, thermal-variance, and nonlinear-cross
contributions.  Each term is projected onto the same reconstructed non-affine
breathing mode through

    Q = int delta Pxx * d_x xi_x dA.

A fixed reference modal inertia is used for this linear-response decomposition
so the channel fractions are not contaminated by the weak phase variation of
the instantaneous normalization.

Outputs
-------
ishizawa_pxx_thermodynamics.txt
ishizawa_pxx_thermodynamics_summary.txt
ishizawa_pxx_thermodynamics_time.png
ishizawa_pxx_thermodynamics_vs_q.png
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import QE, ME, C, params
from analyze_ishizawa_particle_moments import load_mesh, deposit_species
from analyze_ishizawa_nonaffine_breathing_mode import (
    step_from_parent, harmonic_fit, smooth2, weighted_mean, linfit
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--phase-root",default="runs_ishizawa_particle_phases/ppc49")
    p.add_argument("--em-history",required=True)
    p.add_argument("--coarsen",type=int,default=4)
    p.add_argument("--period",type=float,default=0.63)
    p.add_argument("--core-z-de",type=float,default=12.0)
    p.add_argument("--mode-density-cut",type=float,default=0.02)
    p.add_argument("--smooth-passes",type=int,default=2)
    return p.parse_args()


def slope_stats(q,y):
    s,b,r2=linfit(q,y)
    corr=float(np.corrcoef(q,y)[0,1]) if len(q)>2 else np.nan
    return s,b,r2,corr


def main():
    args=parse_args()
    yt.funcs.mylog.setLevel(40)
    P=params()
    p=dict(qe=QE,me=ME,mi=P["mi"],c=C,n0=P["n0"],
           de=P["de"],wci=P["wci"],B0=P["B0"])

    files=sorted(
        glob.glob(str(Path(args.phase_root)/"step*"/"diags"/"diag1*")),
        key=step_from_parent
    )
    if len(files)<4:
        raise RuntimeError(f"need four particle-rich phase plotfiles; found {len(files)}")

    eh=np.loadtxt(args.em_history)
    if eh.ndim==1: eh=eh[None,:]
    ht=eh[:,1]; hq=eh[:,4]
    qfit,hqdot,hqddot,Rq,cq=harmonic_fit(ht,hq,args.period)

    states=[]
    for fn in files:
        M=load_mesh(fn,args.coarsen)
        I=deposit_species(fn,"ions",P["mi"],M,p)
        E=deposit_species(fn,"electrons",ME,M,p)
        tci=P["wci"]*M["t"]
        ni=np.abs(M["rho_i"])/QE
        ne=np.abs(M["rho_e"])/QE
        states.append(dict(
            step=step_from_parent(fn),M=M,I=I,E=E,ni=ni,ne=ne,t=tci,
            q=float(np.interp(tci,ht,hq)),
            qdot=float(np.interp(tci,ht,hqdot)),
            qdd=float(np.interp(tci,ht,hqddot))
        ))

    # Same crossing-based non-affine mode reconstruction as the previous tests.
    pos=[s for s in states if s["qdot"]>0]
    neg=[s for s in states if s["qdot"]<0]
    if not pos or not neg:
        raise RuntimeError("need opposite-sign qdot crossings")
    cneg=min(neg,key=lambda s:abs(s["q"]))
    cpos=min(pos,key=lambda s:abs(s["q"]))
    crossings=[cneg,cpos]

    xis=[]
    for s in crossings:
        z=s["M"]["z"]
        core=(np.abs(z)[None,:] <= args.core_z_de*P["de"])
        w=s["ni"]*core
        ux0=weighted_mean(s["I"]["ux"],w)
        uz0=weighted_mean(s["I"]["uz"],w)
        qdot_phys=P["wci"]*s["qdot"]
        xis.append(((s["I"]["ux"]-ux0)/qdot_phys,
                    (s["I"]["uz"]-uz0)/qdot_phys))

    xix=smooth2(0.5*(xis[0][0]+xis[1][0]),args.smooth_passes)
    xiz=smooth2(0.5*(xis[0][1]+xis[1][1]),args.smooth_passes)

    nref_i=0.5*(crossings[0]["ni"]+crossings[1]["ni"])
    nref_e=0.5*(crossings[0]["ne"]+crossings[1]["ne"])
    cxxref_i=0.5*(crossings[0]["I"]["cxx"]+crossings[1]["I"]["cxx"])
    cxxref_e=0.5*(crossings[0]["E"]["cxx"]+crossings[1]["E"]["cxx"])

    z=crossings[0]["M"]["z"]
    zmax=args.core_z_de*P["de"]
    nmean=nref_i
    dens_env=nmean/(nmean+args.mode_density_cut*P["n0"])
    zenv=np.zeros_like(z)
    inside=np.abs(z)<zmax
    zenv[inside]=0.5*(1+np.cos(np.pi*z[inside]/zmax))
    env=dens_env*zenv[None,:]
    xix*=env; xiz*=env

    M0=crossings[0]["M"]
    dx,dz=M0["dx"],M0["dz"]
    dxx=np.gradient(xix,dx,axis=0,edge_order=2)

    # Fixed reference modal inertia for linear-response channel fractions.
    rhoref=P["mi"]*nref_i + ME*nref_e
    Mref=float(np.sum(rhoref*(xix*xix+xiz*xiz))*dx*dz)
    fac=1.0/(Mref*P["wci"]**2)

    # Useful weight for scalar core thermodynamic summaries.
    wcore=np.abs(dxx)*env
    sw=max(np.sum(wcore),1e-300)

    rows=[]
    for s in states:
        ni,ne=s["ni"],s["ne"]
        ci=s["I"]["cxx"]; ce=s["E"]["cxx"]

        dni=ni-nref_i; dne=ne-nref_e
        dci=ci-cxxref_i; dce=ce-cxxref_e

        # Exact algebraic decomposition of delta Pxx.
        Pi_n=P["mi"]*cxxref_i*dni
        Pi_T=P["mi"]*nref_i*dci
        Pi_X=P["mi"]*dni*dci
        Pe_n=ME*cxxref_e*dne
        Pe_T=ME*nref_e*dce
        Pe_X=ME*dne*dce

        def proj(A):
            return float(np.sum(A*dxx)*dx*dz)*fac

        ai_n,ai_T,ai_X=proj(Pi_n),proj(Pi_T),proj(Pi_X)
        ae_n,ae_T,ae_X=proj(Pe_n),proj(Pe_T),proj(Pe_X)
        ai=ai_n+ai_T+ai_X
        ae=ae_n+ae_T+ae_X

        # Modal-weighted scalar summaries: density and x-temperature variance.
        ni_bar=float(np.sum(wcore*ni)/sw)
        ne_bar=float(np.sum(wcore*ne)/sw)
        ci_bar=float(np.sum(wcore*ci)/sw)
        ce_bar=float(np.sum(wcore*ce)/sw)

        rows.append([
            s["step"],s["t"],s["q"],s["qdot"],s["qdd"],
            ai_n,ai_T,ai_X,ai,
            ae_n,ae_T,ae_X,ae,
            ai+ae,
            ni_bar/P["n0"],ne_bar/P["n0"],
            ci_bar/cxxref_i.mean(),ce_bar/cxxref_e.mean()
        ])

    a=np.asarray(rows,float)
    a=a[np.argsort(a[:,1])]
    np.savetxt(
        "ishizawa_pxx_thermodynamics.txt",a,
        header=(
            "step omega_ci_t q dq_dtau measured_ddq "
            "a_i_density a_i_thermal a_i_cross a_i_total "
            "a_e_density a_e_thermal a_e_cross a_e_total a_all_total "
            "weighted_ni_over_n0 weighted_ne_over_n0 "
            "weighted_cxx_i_over_refmean weighted_cxx_e_over_refmean"
        )
    )

    t=a[:,1]; q=a[:,2]
    channels=[
        ("ion density/compression",5),
        ("ion thermal variance",6),
        ("ion nonlinear cross",7),
        ("ion Pxx total",8),
        ("electron density/compression",9),
        ("electron thermal variance",10),
        ("electron nonlinear cross",11),
        ("electron Pxx total",12),
        ("all-species Pxx total",13),
    ]

    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for name,col in channels[:4]:
        axs[0].plot(t,a[:,col],"o-",label=name)
    for name,col in channels[4:8]:
        axs[1].plot(t,a[:,col],"o-",label=name)
    for ax in axs:
        ax.axhline(0,lw=.7); ax.grid(alpha=.25); ax.legend(fontsize=8)
        ax.set_ylabel("modal acceleration")
    axs[1].set_xlabel(r"$\omega_{ci}t$")
    fig.suptitle(r"Thermodynamic decomposition of the dominant $P_{xx}$ restoring channel")
    fig.tight_layout()
    fig.savefig("ishizawa_pxx_thermodynamics_time.png",dpi=210)
    plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for name,col in channels[:4]:
        axs[0].plot(q,a[:,col],"o-",label=name)
    for name,col in channels[4:8]:
        axs[1].plot(q,a[:,col],"o-",label=name)
    for ax,title in zip(axs,["ions","electrons"]):
        ax.axhline(0,lw=.7); ax.axvline(0,lw=.7)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
        ax.set_xlabel("q"); ax.set_ylabel("modal acceleration")
        ax.set_title(title)
    fig.suptitle(r"$P_{xx}$ force-displacement decomposition")
    fig.tight_layout()
    fig.savefig("ishizawa_pxx_thermodynamics_vs_q.png",dpi=210)
    plt.close(fig)

    stats={}
    for name,col in channels:
        stats[name]=slope_stats(q,a[:,col])

    with open("ishizawa_pxx_thermodynamics_summary.txt","w") as f:
        f.write("Thermodynamic decomposition of non-affine Pxx restoring force\n")
        f.write("=============================================================\n\n")
        f.write(f"period omega_ci*T = {args.period:.8f}\n")
        f.write(f"q harmonic R2 = {Rq:.8f}\n")
        f.write(f"reference modal inertia = {Mref:.8e}\n")
        f.write(
            f"reference crossings: step {cneg['step']} (qdot<0), "
            f"step {cpos['step']} (qdot>0)\n\n"
        )
        f.write("force-displacement slopes\n")
        f.write("-------------------------\n")
        f.write("channel                         slope_vs_q       R2       corr(q,channel)\n")
        for name,(s,b,r2,corr) in stats.items():
            f.write(f"{name:31s} {s:+14.6e} {r2:9.5f} {corr:+12.6f}\n")

        si=stats["ion Pxx total"][0]
        se=stats["electron Pxx total"][0]
        f.write("\nion Pxx restoring-slope fractions\n")
        f.write("----------------------------------\n")
        if abs(si)>0:
            f.write(f"density/compression fraction = {stats['ion density/compression'][0]/si:.8f}\n")
            f.write(f"thermal-variance fraction = {stats['ion thermal variance'][0]/si:.8f}\n")
            f.write(f"nonlinear-cross fraction = {stats['ion nonlinear cross'][0]/si:.8f}\n")
        f.write("\nelectron Pxx restoring-slope fractions\n")
        f.write("---------------------------------------\n")
        if abs(se)>0:
            f.write(f"density/compression fraction = {stats['electron density/compression'][0]/se:.8f}\n")
            f.write(f"thermal-variance fraction = {stats['electron thermal variance'][0]/se:.8f}\n")
            f.write(f"nonlinear-cross fraction = {stats['electron nonlinear cross'][0]/se:.8f}\n")

        f.write("\nInterpretation\n")
        f.write("--------------\n")
        f.write(
            "A negative slope is restoring.  If the density/compression term "
            "dominates ion Pxx, the global mode is primarily an elastic "
            "longitudinal compression/rarefaction of the trapped ion "
            "population.  If the thermal-variance term dominates, the restoring "
            "response is instead controlled by phase-dependent ion heating/"
            "anisotropy.  A large nonlinear-cross term signals a strongly "
            "nonlinear thermodynamic response.  With four phases this remains "
            "a mechanism screen; dense particle-rich sampling is required for "
            "precision phase-lag and stiffness measurements.\n"
        )

    print("Saved ishizawa_pxx_thermodynamics.txt")
    print("Saved ishizawa_pxx_thermodynamics_summary.txt")
    print("Saved ishizawa_pxx_thermodynamics_time.png")
    print("Saved ishizawa_pxx_thermodynamics_vs_q.png")


if __name__=="__main__":
    main()
