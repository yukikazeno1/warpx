#!/usr/bin/env python3
"""Decompose the dominant ion random-xx restoring channel into density-like and
velocity-dispersion-like responses.

For each coarse cell and time,

    R_xx = sum W (p_x-<p_x>)(v_x-<v_x>)

is the random/pressure-like xx momentum flux used by the validated weak-form
budget.  With N = sum W and S_xx = R_xx/N, write exactly

    R_xx = Nbar*Sbar
         + Sbar*dN
         + Nbar*dS
         + dN*dS.

The four terms are:
- baseline: time-independent;
- occupancy/density-like: Sbar*dN;
- specific-stress/velocity-dispersion-like: Nbar*dS;
- nonlinear covariance: dN*dS.

Contracting each term with the same fixed d_x xi_x used by the q-matched weak
form gives a direct decomposition of the dominant restoring channel.  In the
nonrelativistic limit, S_xx is proportional to the local x thermal/velocity
variance per ion.

The script reports both global results and a conservative common deep-core
mask that lies inside chi<0.25 at both q~0 crossings.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import params
from analyze_ishizawa_ion_kinetic_decomposition import (
    build_fixed_mode, ion_terms, phasor_map
)
from analyze_ishizawa_modal_particle_quadrature import fit_phasor
from analyze_ishizawa_ion_restoring_flux_shells import (
    topology, island_flux_coordinate, find_plotfile
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--phase-root",required=True)
    p.add_argument("--em-history",required=True)
    p.add_argument("--period",type=float,required=True)
    p.add_argument("--tmin",type=float,default=4.05)
    p.add_argument("--tmax",type=float,default=5.90)
    p.add_argument("--q-tmin",type=float,default=4.05)
    p.add_argument("--q-tmax",type=float,default=5.95)
    p.add_argument("--mode-coarsen",type=int,default=4)
    p.add_argument("--core-z-de",type=float,default=12.0)
    p.add_argument("--mode-density-cut",type=float,default=0.02)
    p.add_argument("--smooth-passes",type=int,default=2)
    p.add_argument("--x-edge-taper-de",type=float,default=0.0)
    p.add_argument("--periodic-x-gradient",action="store_true")
    p.add_argument("--large-scale-max-mode",type=int,default=3)
    p.add_argument("--out-prefix",default="ishizawa_ion_randxx_mechanism")
    return p.parse_args()


def channel_stats(t,q,y,T,Zref):
    Z,R2,_,_=fit_phasor(t,y,T)
    Zq,_,_,_=fit_phasor(t,q,T)
    phase=float(np.angle(Z/Zq)) if abs(Zq)>0 else np.nan
    return dict(
        Z=Z,R2=R2,
        amp_over_ref=float(abs(Z)/max(abs(Zref),1e-300)),
        phase_q=phase,
        align_rest=float(np.cos(np.angle(Z/(-Zq)))) if abs(Zq)>0 else np.nan,
        align_damp=float(np.cos(np.angle(Z/(-1j*Zq)))) if abs(Zq)>0 else np.nan,
    )


def main():
    args=parse_args()
    yt.funcs.mylog.setLevel(40)
    P=params()

    states,Mmode,xix,xiz,grads,crossings,qkind,Rq,q_tmin,q_tmax=build_fixed_mode(args,P)
    Gxx=grads[0]

    Wt=[]
    Rxt=[]
    t=[]
    q=[]
    for k,s in enumerate(states,1):
        R=ion_terms(s["fn"],P,Mmode,grads)
        Wt.append(R["weight_map"])
        Rxt.append(R["rand_xx_flux_map"])
        t.append(s["t"])
        q.append(s["q"])
        print(f"[{k:02d}/{len(states):02d}] step={s['step']} tau={s['t']:.6f} Nion={R['N']}")

    Wt=np.asarray(Wt,float)
    Rxt=np.asarray(Rxt,float)
    t=np.asarray(t,float)
    q=np.asarray(q,float)

    S=np.zeros_like(Rxt)
    m=Wt>0
    S[m]=Rxt[m]/Wt[m]

    W0=np.mean(Wt,axis=0)
    S0=np.mean(S,axis=0)
    dW=Wt-W0[None,:,:]
    dS=S-S0[None,:,:]

    Rbase=W0*S0
    Rocc=S0[None,:,:]*dW
    Rspec=W0[None,:,:]*dS
    Rcross=dW*dS

    # Exact cellwise identity check.
    recon=Rbase[None,:,:]+Rocc+Rspec+Rcross
    rel=np.sqrt(np.mean((recon-Rxt)**2))/max(np.sqrt(np.mean(Rxt**2)),1e-300)

    # Contract each mechanism with the same fixed weak-form kernel.
    # Divide by omega_ci to match the existing kinetic-budget normalization.
    def Qof(Rmap,mask=None):
        if mask is None:
            return np.sum(Rmap*Gxx[None,:,:],axis=(1,2))/P["wci"]
        return np.sum(Rmap[:,mask]*Gxx[mask][None,:],axis=1)/P["wci"]

    Qfull=Qof(Rxt)
    Qocc=Qof(Rocc)
    Qspec=Qof(Rspec)
    Qcross=Qof(Rcross)

    # Build a conservative fixed deep-core mask: chi<0.25 and inside island at
    # both q~0 crossings.
    masks=[]
    for s in crossings:
        Ts=topology(
            find_plotfile(args.phase_root,s["step"]),
            args.mode_coarsen,args.large_scale_max_mode,P
        )
        chi,isl=island_flux_coordinate(Ts)
        masks.append(isl & np.isfinite(chi) & (chi>=0.0) & (chi<0.25))
    core=masks[0] & masks[1]

    Qfull_core=Qof(Rxt,core)
    Qocc_core=Qof(Rocc,core)
    Qspec_core=Qof(Rspec,core)
    Qcross_core=Qof(Rcross,core)

    Zref,_,_,_=fit_phasor(t,Qfull,args.period)
    stats={}
    for name,y in [
        ("rand_xx_total",Qfull),
        ("occupancy_density_like",Qocc),
        ("specific_stress_like",Qspec),
        ("nonlinear_cross",Qcross),
        ("core_rand_xx_total",Qfull_core),
        ("core_occupancy_density_like",Qocc_core),
        ("core_specific_stress_like",Qspec_core),
        ("core_nonlinear_cross",Qcross_core),
    ]:
        stats[name]=channel_stats(t,q,y,args.period,Zref)

    # Spatial phasors and restoring projections.
    Zq,_,_,_=fit_phasor(t,q,args.period)
    eq=Zq/max(abs(Zq),1e-300)
    Zocc=phasor_map(t,Rocc*Gxx[None,:,:]/P["wci"],args.period)
    Zspec=phasor_map(t,Rspec*Gxx[None,:,:]/P["wci"],args.period)
    Zcross=phasor_map(t,Rcross*Gxx[None,:,:]/P["wci"],args.period)

    def restoring(Z):
        return np.real(Z*np.conj(-eq))/max(abs(Zref),1e-300)

    Mocc=restoring(Zocc)
    Mspec=restoring(Zspec)
    Mcross=restoring(Zcross)

    with open(args.out_prefix+"_summary.txt","w") as f:
        f.write("Ion random-xx restoring mechanism decomposition\n")
        f.write("=============================================\n\n")
        f.write(f"points = {len(t)}\n")
        f.write(f"window omega_ci*t = {t.min():.8f} .. {t.max():.8f}\n")
        f.write(f"period omega_ci*T = {args.period:.8f}\n")
        f.write(f"mode crossings = {crossings[0]['step']} {crossings[1]['step']}\n")
        f.write(f"q normalization window = {q_tmin:.8f} .. {q_tmax:.8f}\n")
        f.write(f"cellwise Rxx reconstruction relRMS = {rel:.8e}\n")
        f.write(f"common chi<0.25 core cells = {np.count_nonzero(core)} / {core.size}\n\n")

        f.write("Global breathing-frequency decomposition\n")
        f.write("----------------------------------------\n")
        for name in [
            "rand_xx_total","occupancy_density_like",
            "specific_stress_like","nonlinear_cross"
        ]:
            s=stats[name]; Z=s["Z"]
            f.write(
                f"{name:30s} Re={Z.real:+.8e} Im={Z.imag:+.8e} "
                f"|Z|/|Zrandxx|={s['amp_over_ref']:.8f} "
                f"R2={s['R2']:.8f} phase(Q/q)={s['phase_q']:+.8f} "
                f"align(-q)={s['align_rest']:+.8f} "
                f"align(-qdot)={s['align_damp']:+.8f}\n"
            )

        f.write("\nCommon deep-core (chi<0.25 at both crossings)\n")
        f.write("------------------------------------------------\n")
        for name in [
            "core_rand_xx_total","core_occupancy_density_like",
            "core_specific_stress_like","core_nonlinear_cross"
        ]:
            s=stats[name]; Z=s["Z"]
            f.write(
                f"{name:30s} Re={Z.real:+.8e} Im={Z.imag:+.8e} "
                f"|Z|/|Zrandxx_global|={s['amp_over_ref']:.8f} "
                f"R2={s['R2']:.8f} phase(Q/q)={s['phase_q']:+.8f} "
                f"align(-q)={s['align_rest']:+.8f}\n"
            )

        f.write("\nSpatial restoring sums / |Zrandxx|\n")
        f.write(f"occupancy/density-like = {np.sum(Mocc):+.10e}\n")
        f.write(f"specific-stress-like = {np.sum(Mspec):+.10e}\n")
        f.write(f"nonlinear cross = {np.sum(Mcross):+.10e}\n")
        f.write(
            "\nInterpretation: occupancy/density-like measures modulation of "
            "particle number at fixed mean specific random xx stress; "
            "specific-stress-like measures modulation of random xx stress per "
            "particle at fixed mean occupancy; nonlinear_cross is their product.\n"
        )

    np.savez_compressed(
        args.out_prefix+"_spatial.npz",
        x_de=Mmode["x"]/P["de"],z_de=Mmode["z"]/P["de"],
        core_mask=core,
        occupancy_restoring=Mocc,
        specific_stress_restoring=Mspec,
        nonlinear_cross_restoring=Mcross,
    )

    # Time-harmonic comparison.
    om=2*np.pi/args.period
    tt=t-t[0]
    def hs(Z):
        return np.real(Z*np.exp(1j*om*tt))/max(abs(Zref),1e-300)

    fig,ax=plt.subplots(figsize=(9,5.7))
    for name,label in [
        ("rand_xx_total","rand xx total"),
        ("occupancy_density_like","density/occupancy-like"),
        ("specific_stress_like","specific-stress-like"),
        ("nonlinear_cross","nonlinear cross")
    ]:
        ax.plot(t,hs(stats[name]["Z"]),"o-",label=label)
    ax.axhline(0,lw=.7)
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel(r"fundamental / $|Z_{rand,xx}|$")
    ax.set_title("Ion random-xx restoring mechanism")
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_prefix+"_time.png",dpi=210)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,6.5))
    for name,label in [
        ("occupancy_density_like","density"),
        ("specific_stress_like","specific stress"),
        ("nonlinear_cross","cross"),
        ("rand_xx_total","rand xx total")
    ]:
        Z=stats[name]["Z"]/max(abs(Zref),1e-300)
        ax.arrow(0,0,Z.real,Z.imag,length_includes_head=True,
                 head_width=.025,head_length=.04)
        ax.text(Z.real,Z.imag,label)
    ax.axhline(0,lw=.7); ax.axvline(0,lw=.7)
    ax.set_xlabel(r"Re$(Z)/|Z_{rand,xx}|$")
    ax.set_ylabel(r"Im$(Z)/|Z_{rand,xx}|$")
    ax.set_title("Ion random-xx mechanism phasors")
    ax.grid(alpha=.25); ax.set_aspect("equal",adjustable="datalim")
    fig.tight_layout()
    fig.savefig(args.out_prefix+"_complex.png",dpi=210)
    plt.close(fig)

    xde=Mmode["x"]/P["de"]; zde=Mmode["z"]/P["de"]
    extent=[xde.min(),xde.max(),zde.min(),zde.max()]
    def save_map(a,name,title):
        fig,ax=plt.subplots(figsize=(8.5,5.8))
        vmax=np.nanpercentile(np.abs(a),99.5)
        im=ax.imshow(a.T,origin="lower",extent=extent,aspect="auto",
                     vmin=-vmax,vmax=vmax,cmap="RdBu_r")
        ax.contour(xde,zde,core.T.astype(float),levels=[0.5],
                   colors="k",linewidths=1.0)
        ax.set_xlabel(r"$x/d_e$"); ax.set_ylabel(r"$z/d_e$")
        ax.set_title(title+"; black = common chi<0.25 core")
        fig.colorbar(im,ax=ax,label=r"restoring contribution / $|Z_{rand,xx}|$")
        fig.tight_layout(); fig.savefig(name,dpi=210); plt.close(fig)

    save_map(Mocc,args.out_prefix+"_occupancy_restoring_map.png",
             "Ion random-xx density/occupancy-like restoring")
    save_map(Mspec,args.out_prefix+"_specific_stress_restoring_map.png",
             "Ion random-xx specific-stress restoring")
    save_map(Mcross,args.out_prefix+"_nonlinear_cross_restoring_map.png",
             "Ion random-xx nonlinear cross restoring")

    print("Saved",args.out_prefix+"_summary.txt")
    print("Saved",args.out_prefix+"_time.png")
    print("Saved",args.out_prefix+"_complex.png")
    print("Saved spatial maps and NPZ.")


if __name__=="__main__":
    main()
