#!/usr/bin/env python3
"""Generalized electromagnetic-force test for the saturated island breathing.

This extends analyze_ishizawa_restoring_oscillator.py in two ways:

1. add the electric force density rho_c E_z to JxB;
2. diagnose the *incremental* (mean-subtracted) force-displacement relation
   separately for magnetic pressure, magnetic tension, JxB, electric force,
   and total electromagnetic force.

For the width coordinate q = w/<w>-1 and affine vertical displacement
delta z = z delta q,

    Q_k = integral z f_{k,z} dA
    I   = integral rho_i z^2 dA

and the dimensionless generalized acceleration is

    a_k = Q_k/(I omega_ci^2).

A restoring contribution should satisfy a_k ~= -Omega_k^2 q, i.e. it should
have a negative force-displacement slope and be approximately pi out of phase
with q.

Outputs
-------
ishizawa_generalized_em_force_history.txt
ishizawa_generalized_em_force.png
ishizawa_generalized_em_force_vs_q.png
ishizawa_generalized_em_force_summary.txt
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import QE, params
from analyze_ishizawa_breathing_phases import load_full_fields, process_phase


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",default="runs_ishizawa_scale/ppc49")
    p.add_argument("--history",required=True)
    p.add_argument("--tmin",type=float,default=2.8)
    p.add_argument("--tmax",type=float,default=4.0)
    p.add_argument("--period",type=float,default=0.63)
    p.add_argument("--core-z-de",type=float,default=12.0)
    p.add_argument("--density-cut",type=float,default=1e-3)
    return p.parse_args()


def numeric_key(path):
    m=re.search(r"(\d+)$",os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def step_from_plotfile(path):
    s=os.path.basename(path.rstrip("/"))
    if not s.startswith("diag1"):
        return -1
    t=s[len("diag1"):]
    return int(t) if t.isdigit() else -1


def load_history(path):
    a=np.loadtxt(path)
    if a.ndim==1:
        a=a[None,:]
    return dict(step=a[:,0].astype(int),t=a[:,1],psi=a[:,2],width=a[:,4])


def harmonic_fit(t,y,T):
    u=t-t[0]
    om=2*np.pi/T
    M=np.column_stack([np.ones_like(u),u,np.sin(om*u),np.cos(om*u)])
    c,*_=np.linalg.lstsq(M,y,rcond=None)
    pred=M@c
    ssr=np.sum((y-pred)**2)
    sst=np.sum((y-np.mean(y))**2)
    r2=1-ssr/sst if sst>0 else np.nan
    amp=float(np.hypot(c[2],c[3]))
    phi=float(np.arctan2(c[3],c[2]))
    return pred,amp,phi,float(r2),c


def wrap(x):
    return float(np.arctan2(np.sin(x),np.cos(x)))


def linfit(x,y):
    m=np.isfinite(x)&np.isfinite(y)
    x=x[m]; y=y[m]
    A=np.column_stack([x,np.ones_like(x)])
    c,*_=np.linalg.lstsq(A,y,rcond=None)
    pred=A@c
    ssr=np.sum((y-pred)**2)
    sst=np.sum((y-np.mean(y))**2)
    r2=1-ssr/sst if sst>0 else np.nan
    return float(c[0]),float(c[1]),float(r2)


def main():
    args=parse_args()
    yt.funcs.mylog.setLevel(40)
    P=params()
    H=load_history(args.history)

    files=sorted(glob.glob(str(Path(args.run_dir)/"diags"/"diag1*")),
                 key=numeric_key)
    bystep={step_from_plotfile(p):p for p in files}

    rows=[]
    for i,(step,t) in enumerate(zip(H["step"],H["t"])):
        if not (args.tmin <= t <= args.tmax):
            continue
        fn=bystep.get(int(step))
        if fn is None:
            continue

        F=load_full_fields(fn)
        D=process_phase(F,P)

        X,Z=np.meshgrid(D["x"],D["z"],indexing="ij")
        dA=F["dx"]*F["dz"]

        ni=np.maximum(D["ni"],0.0)
        ne=np.maximum(D["ne"],0.0)
        rho_m=P["mi"]*ni
        rho_c=QE*(ni-ne)

        zmask=np.abs(D["z"])<=args.core_z_de*P["de"]
        mask=np.broadcast_to(zmask[None,:],ni.shape)
        pmask=mask & (ni>=args.density_cut*P["n0"])

        I=np.sum((rho_m*Z*Z)[pmask])*dA

        Fe_z=rho_c*D["Ez"]
        Fem_z=D["FL_z"]+Fe_z

        def Q(f):
            return np.sum((Z*f)[pmask])*dA

        Qp=Q(D["Fp_z"])
        Qt=Q(D["Ft_z"])
        Ql=Q(D["FL_z"])
        Qe=Q(Fe_z)
        Qem=Q(Fem_z)

        fac=1.0/(I*P["wci"]**2) if I>0 else np.nan
        rows.append([
            int(step),t,H["psi"][i],H["width"][i],I,
            Qp,Qt,Ql,Qe,Qem,
            Qp*fac,Qt*fac,Ql*fac,Qe*fac,Qem*fac
        ])

    a=np.asarray(rows,float)
    if len(a)<10:
        raise RuntimeError("Too few matched field frames")

    step,t,psi,width,I,Qp,Qt,Ql,Qe,Qem,ap,at,al,ae,aem=a.T
    q=width/np.mean(width)-1.0

    # Measured acceleration from the known breathing harmonic.
    qfit,Aq,phiq,Rq,cq=harmonic_fit(t,q,args.period)
    om=2*np.pi/args.period
    u=t-t[0]
    ddq_fit=-(om**2)*(cq[2]*np.sin(om*u)+cq[3]*np.cos(om*u))

    series={
        "Pmag":ap,
        "tension":at,
        "JxB":al,
        "rhoE":ae,
        "EM":aem,
    }

    stats={}
    fits={}
    harmonics={}
    for name,y in series.items():
        pred,A,phi,R,c=harmonic_fit(t,y,args.period)
        harmonics[name]=c[2]*np.sin(om*u)+c[3]*np.cos(om*u)
        slope,intercept,Rlin=linfit(q,y)
        # residual phase relative to the ideal restoring relation y ~ -q
        dphi=wrap(phi-phiq-np.pi)
        T_signed=2*np.pi/np.sqrt(-slope) if slope<0 else np.nan
        stats[name]=dict(A=A,phi=phi,R=R,slope=slope,
                         intercept=intercept,Rlin=Rlin,
                         dphi=dphi,T_signed=T_signed)
        fits[name]=pred

    em_h=harmonics["EM"]
    corr_em=float(np.corrcoef(ddq_fit,em_h)[0,1])
    rms_ref=float(np.sqrt(np.mean(ddq_fit**2)))
    mismatch=float(np.sqrt(np.mean((ddq_fit-em_h)**2))/rms_ref)

    # The force that would still be required to reproduce the observed
    # breathing acceleration after subtracting the measured EM contribution.
    # This is a diagnostic residual, not yet an identified physical force.
    areq=ddq_fit-em_h
    areq_fit,Areq,phireq,Rreq,creq=harmonic_fit(t,areq,args.period)
    req_slope,req_intercept,req_Rlin=linfit(q,areq)
    req_phase_resid=wrap(phireq-phiq-np.pi)
    req_T=2*np.pi/np.sqrt(-req_slope) if req_slope<0 else np.nan
    required_total_slope=-(2*np.pi/args.period)**2

    np.savetxt(
        "ishizawa_generalized_em_force_history.txt",
        np.column_stack([
            step,t,psi,width,q,I,Qp,Qt,Ql,Qe,Qem,ap,at,al,ae,aem,ddq_fit,areq
        ]),
        header=(
            "step omega_ci_t Psi width_de q I Q_Pmag Q_tension Q_JxB Q_rhoE "
            "Q_EM a_Pmag a_tension a_JxB a_rhoE a_EM measured_ddq_harmonic "
            "required_nonEM_acceleration"
        )
    )

    # Time-history plot: mean-subtracted generalized accelerations.
    fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True)
    axs[0].plot(t,q,label="q")
    axs[0].plot(t,qfit,"--",label="T=0.63 fit")
    axs[0].set_ylabel("width displacement")
    axs[0].legend(fontsize=8)

    axs[1].plot(t,ddq_fit,lw=2,label="measured harmonic acceleration")
    for name in ["JxB","rhoE","EM"]:
        axs[1].plot(t,harmonics[name],label=name)
    axs[1].plot(t,areq,"--",lw=2,label="required non-EM residual")
    axs[1].axhline(0,lw=.7)
    axs[1].set_ylabel("demeaned acceleration")
    axs[1].legend(fontsize=8)

    for name in ["Pmag","tension","JxB"]:
        axs[2].plot(t,harmonics[name],label=name)
    axs[2].axhline(0,lw=.7)
    axs[2].set_ylabel("magnetic components")
    axs[2].set_xlabel(r"$\omega_{ci}t$")
    axs[2].legend(fontsize=8)
    for ax in axs: ax.grid(alpha=.25)
    fig.suptitle("Incremental generalized electromagnetic-force test")
    fig.tight_layout()
    fig.savefig("ishizawa_generalized_em_force.png",dpi=210)
    plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(14,4.5),sharex=True)
    groups=[("Pmag","tension"),("JxB","rhoE"),("EM",)]
    for ax,names in zip(axs,groups):
        for name in names:
            y=series[name]
            st=stats[name]
            ax.scatter(q,y-np.mean(y),s=22,label=name)
            xx=np.linspace(q.min(),q.max(),200)
            ax.plot(xx,st["slope"]*xx,label=f'{name}: slope={st["slope"]:.1f}')
        ax.axhline(0,lw=.7); ax.axvline(0,lw=.7)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
        ax.set_xlabel("q")
    # Overlay the dynamically required non-EM residual on the final panel.
    axs[2].scatter(q,areq,s=22,label="required non-EM residual")
    xx=np.linspace(q.min(),q.max(),200)
    axs[2].plot(xx,req_slope*xx+req_intercept,"--",
                label=f"required residual: slope={req_slope:.1f}")
    axs[2].legend(fontsize=8)
    axs[0].set_ylabel("demeaned generalized acceleration")
    fig.suptitle("Force-displacement slopes: restoring requires negative slope")
    fig.tight_layout()
    fig.savefig("ishizawa_generalized_em_force_vs_q.png",dpi=210)
    plt.close(fig)

    with open("ishizawa_generalized_em_force_summary.txt","w") as f:
        f.write("Ishizawa-scale generalized electromagnetic-force test\n")
        f.write("=====================================================\n\n")
        f.write(f"interval omega_ci*t = {t[0]:.8f} .. {t[-1]:.8f}\n")
        f.write(f"measured period = {args.period:.8f}\n")
        f.write(f"samples = {len(t)}\n")
        f.write(f"core |z|/de <= {args.core_z_de:.6f}\n")
        f.write(f"density cut n_i/n0 >= {args.density_cut:.3e}\n")
        f.write(f"width harmonic R2 = {Rq:.8f}\n\n")
        f.write("component          amp        R2harm    phase_resid(rad)    slope(a/q)    R2lin    T_from_negative_slope\n")
        f.write("------------------------------------------------------------------------------------------------------\n")
        for name in ["Pmag","tension","JxB","rhoE","EM"]:
            s=stats[name]
            f.write(
                f"{name:10s} {s['A']:11.4e} {s['R']:10.6f} "
                f"{s['dphi']:17.8f} {s['slope']:13.5e} "
                f"{s['Rlin']:9.6f} {s['T_signed']:21.8f}\n"
            )
        f.write("\n")
        f.write(f"corr(measured harmonic acceleration, total EM harmonic) = {corr_em:.8f}\n")
        f.write(f"relative RMS pure-harmonic total-EM acceleration mismatch = {mismatch:.8e}\n\n")
        f.write("Required non-EM residual (measured acceleration - EM harmonic)\n")
        f.write("--------------------------------------------------------------\n")
        f.write(f"ideal total restoring slope from measured period = {required_total_slope:.8e}\n")
        f.write(f"residual harmonic amplitude = {Areq:.8e}\n")
        f.write(f"residual harmonic R2 = {Rreq:.8f}\n")
        f.write(f"residual phase relative to ideal restoring = {req_phase_resid:.8f} rad\n")
        f.write(f"residual slope versus q = {req_slope:.8e}\n")
        f.write(f"residual slope R2 = {req_Rlin:.8f}\n")
        f.write(f"standalone period from negative residual slope = {req_T:.8f}\n\n")
        f.write("Interpretation:\n")
        f.write("  restoring => negative slope and phase_resid close to 0.\n")
        f.write("  positive slope => anti-restoring contribution for this width coordinate.\n")
        f.write("  If JxB and total EM both fail while pressure/tension nearly cancel,\n")
        f.write("  the breathing coordinate is not closed by electromagnetic force alone;\n")
        f.write("  particle pressure/kinetic stress or a non-affine coordinate is required.\n")

    print("Saved ishizawa_generalized_em_force_summary.txt")
    print("Saved ishizawa_generalized_em_force_history.txt")
    print("Saved ishizawa_generalized_em_force.png")
    print("Saved ishizawa_generalized_em_force_vs_q.png")


if __name__=="__main__":
    main()
