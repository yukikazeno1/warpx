#!/usr/bin/env python3
"""Time-resolved mechanism diagnostics for nonlinear m=1 island breathing.

Consumes the late-time plotfiles and the large-scale saturation history.
It quantifies, across many breathing cycles:

  1) electromagnetic energy and Poynting-theorem balance in a fixed core,
  2) phase relation of J.E and boundary Poynting flux to Psi_LS,
  3) density-shape moments (sigma_x, sigma_z, aspect ratio, area proxy)
     after m=1 phase alignment.

This is intended to distinguish:
  - simple compression,
  - shape/ellipticity breathing,
  - field-particle energy exchange,
  - Poynting-flux redistribution.

Outputs
-------
ishizawa_breathing_mechanism_history.txt
ishizawa_breathing_poynting_balance.png
ishizawa_breathing_shape_moments.png
ishizawa_breathing_mechanism_phase.png
ishizawa_breathing_mechanism_summary.txt
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import MU0, EPS0, C, params
from analyze_ishizawa_breathing_phases import (
    load_full_fields, process_phase
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs_ishizawa_scale/ppc49")
    p.add_argument("--history", required=True)
    p.add_argument("--tmin", type=float, default=2.8)
    p.add_argument("--tmax", type=float, default=4.0)
    p.add_argument("--core-z-de", type=float, default=12.0)
    p.add_argument("--period", type=float, default=0.63)
    return p.parse_args()


def numeric_key(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def step_from_plotfile(path):
    name = os.path.basename(path.rstrip("/"))
    if not name.startswith("diag1"):
        return -1
    s = name[len("diag1"):]
    return int(s) if s.isdigit() else -1


def load_history(path):
    a=np.loadtxt(path)
    if a.ndim==1:
        a=a[None,:]
    return dict(
        step=a[:,0].astype(int), t=a[:,1], psi=a[:,2],
        psi_m1=a[:,3], width=a[:,4]
    )


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


def wrap_phase(x):
    return float(np.arctan2(np.sin(x),np.cos(x)))


def weighted_shape(n,x,z):
    """Density-weighted second moments after O-point alignment."""
    w=np.clip(np.asarray(n,float),0,None)
    W=np.sum(w)
    if not np.isfinite(W) or W<=0:
        return (np.nan,)*6
    X,Z=np.meshgrid(x,z,indexing="ij")
    xb=float(np.sum(w*X)/W)
    zb=float(np.sum(w*Z)/W)
    sx=float(np.sqrt(np.sum(w*(X-xb)**2)/W))
    sz=float(np.sqrt(np.sum(w*(Z-zb)**2)/W))
    aspect=sz/sx if sx>0 else np.nan
    area=sx*sz
    return xb,zb,sx,sz,aspect,area


def main():
    args=parse_args()
    yt.funcs.mylog.setLevel(40)
    P=params()
    H=load_history(args.history)

    files=sorted(
        glob.glob(str(Path(args.run_dir)/"diags"/"diag1*")),
        key=numeric_key
    )
    bystep={step_from_plotfile(p):p for p in files}

    rows=[]
    for i,(step,t) in enumerate(zip(H["step"],H["t"])):
        if t<args.tmin or t>args.tmax:
            continue
        if int(step) not in bystep:
            continue

        F=load_full_fields(bystep[int(step)])
        D=process_phase(F,P)

        core=np.abs(D["z"])<=args.core_z_de*P["de"]
        dA=F["dx"]*F["dz"]

        # EM energy in fixed core.
        B2=D["Bx"]**2+D["By"]**2+D["Bz"]**2
        E2=D["Ex"]**2+D["Ey"]**2+D["Ez"]**2
        uB=B2/(2*MU0)
        uE=EPS0*E2/2
        UB=np.sum(uB[:,core])*dA
        UE=np.sum(uE[:,core])*dA
        UEM=UB+UE
        UBz=np.sum(D["Bz"][:,core]**2/(2*MU0))*dA

        # Integrated J.E in same fixed core.
        JdotE=D["jx"]*D["Ex"]+D["jy"]*D["Ey"]+D["jz"]*D["Ez"]
        WJE=np.sum(JdotE[:,core])*dA

        # Poynting flux through z=+/- core boundary.
        Sz=(D["Ex"]*D["By"]-D["Ey"]*D["Bx"])/MU0
        ids=np.flatnonzero(core)
        jlo=int(ids[0]); jhi=int(ids[-1])
        Pout=np.sum(Sz[:,jhi]-Sz[:,jlo])*F["dx"]

        # Ion-density shape moments in aligned frame. Restrict to fixed core.
        ni=np.clip(D["ni"][:,core],0,None)
        xb,zb,sx,sz,aspect,area=weighted_shape(
            ni,D["x"],D["z"][core]
        )
        Ncore=np.sum(ni)*dA

        rows.append([
            int(step),t,H["psi"][i],H["width"][i],
            UB,UE,UEM,UBz,WJE,Pout,
            Ncore,xb/P["de"],zb/P["de"],
            sx/P["de"],sz/P["de"],aspect,area/P["de"]**2
        ])
        print(
            f"step={int(step):6d} tci={t:.4f} "
            f"Psi={H['psi'][i]:.3f} w={H['width'][i]:.3f} "
            f"aspect={aspect:.3f}"
        )

    a=np.asarray(rows,float)
    if len(a)<8:
        raise RuntimeError("Too few mechanism samples")

    step,t,psi,width,UB,UE,UEM,UBz,WJE,Pout,Ncore,xb,zb,sx,sz,aspect,area=a.T

    # Common normalization for Poynting theorem.
    normU=(P["B0"]**2/(2*MU0))*P["de"]**2
    Uhat=UEM/normU
    UBhat=UB/normU
    UEhat=UE/normU
    UBzhat=UBz/normU
    normP=normU*P["wci"]
    What=WJE/normP
    Fhat=Pout/normP

    dUdtau=np.gradient(Uhat,t,edge_order=2)
    residual=dUdtau+Fhat+What

    out=np.column_stack([
        step,t,psi,width,UBhat,UEhat,Uhat,UBzhat,
        What,Fhat,dUdtau,residual,Ncore,
        xb,zb,sx,sz,aspect,area
    ])
    np.savetxt(
        "ishizawa_breathing_mechanism_history.txt",out,
        header=(
            "step omega_ci_t Psi_B0de width_de "
            "UB_norm UE_norm UEM_norm UBz_norm "
            "intJdotE_norm PoyntingOut_norm dUEM_dtau_norm poynting_residual "
            "Nion_core xbar_de zbar_de sigma_x_de sigma_z_de aspect_z_over_x "
            "sigma_x_sigma_z_de2"
        )
    )

    # Harmonic phase relations.
    _,Apsi,phipsi,Rpsi,_=harmonic_fit(t,psi,args.period)
    series={
        "width":width,
        "UBz":UBzhat,
        "UEM":Uhat,
        "JdotE":What,
        "Pout":Fhat,
        "aspect":aspect,
        "sigx":sx,
        "sigz":sz,
        "area":area,
    }
    fits={}
    for k,y in series.items():
        pred,A,phi,R,c=harmonic_fit(t,y,args.period)
        fits[k]=(pred,A,phi,R,c,wrap_phase(phi-phipsi))

    # Figure: Poynting theorem.
    fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True)
    axs[0].plot(t,Uhat,label="U_EM")
    axs[0].plot(t,UBhat,label="U_B")
    axs[0].plot(t,UEhat,label="U_E")
    axs[0].plot(t,UBzhat,label="U_Bz")
    axs[0].set_ylabel("energy / U0"); axs[0].legend(ncol=4,fontsize=8)
    axs[1].plot(t,dUdtau,label=r"$dU_{EM}/d(\omega_{ci}t)$")
    axs[1].plot(t,Fhat,label="Poynting out")
    axs[1].plot(t,What,label=r"$\int J\cdot E$")
    axs[1].set_ylabel("power / (U0 omega_ci)")
    axs[1].legend(fontsize=8)
    axs[2].plot(t,residual,label="closure residual")
    axs[2].axhline(0,lw=.8)
    axs[2].set_ylabel("balance residual")
    axs[2].set_xlabel(r"$\omega_{ci}t$")
    for ax in axs: ax.grid(alpha=.25)
    fig.suptitle("Breathing-cycle electromagnetic energy and Poynting balance")
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_poynting_balance.png",dpi=210)
    plt.close(fig)

    # Figure: density-shape moments.
    fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True)
    axs[0].plot(t,psi,label=r"$\Psi$")
    ax2=axs[0].twinx()
    ax2.plot(t,width,ls="--",label="island width")
    axs[0].set_ylabel(r"$\Psi/(B_0d_e)$")
    ax2.set_ylabel(r"$w/d_e$")
    axs[1].plot(t,sx,label=r"$\sigma_x$")
    axs[1].plot(t,sz,label=r"$\sigma_z$")
    axs[1].set_ylabel(r"density RMS size / $d_e$")
    axs[1].legend()
    axs[2].plot(t,aspect,label=r"$\sigma_z/\sigma_x$")
    axs[2].plot(t,area/np.mean(area),label=r"$(\sigma_x\sigma_z)/\langle\cdot\rangle$")
    axs[2].plot(t,Ncore/np.mean(Ncore),label=r"$N_i/\langle N_i\rangle$")
    axs[2].set_ylabel("shape / normalized amount")
    axs[2].set_xlabel(r"$\omega_{ci}t$")
    axs[2].legend()
    for ax in axs: ax.grid(alpha=.25)
    fig.suptitle("Ion-density shape moments during magnetic-island breathing")
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_shape_moments.png",dpi=210)
    plt.close(fig)

    # Figure: normalized detrended phase comparison.
    def detr(y):
        c=np.polyfit(t,y,1)
        z=y-np.polyval(c,t)
        s=np.std(z)
        return z/s if s>0 else z

    fig,ax=plt.subplots(figsize=(9,5.8))
    for label,y in [
        (r"$\Psi$",psi),
        (r"$U_{Bz}$",UBzhat),
        (r"$J\cdot E$",What),
        ("Poynting out",Fhat),
        ("density aspect",aspect),
    ]:
        ax.plot(t,detr(y),label=label)
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel("normalized detrended fluctuation")
    ax.grid(alpha=.25); ax.legend(ncol=2,fontsize=8)
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_mechanism_phase.png",dpi=210)
    plt.close(fig)

    with open("ishizawa_breathing_mechanism_summary.txt","w") as f:
        f.write("Ishizawa-scale breathing mechanism diagnostics\n")
        f.write("=============================================\n\n")
        f.write(f"interval omega_ci*t = {t[0]:.8f} .. {t[-1]:.8f}\n")
        f.write(f"samples = {len(t)}\n")
        f.write(f"reference breathing period = {args.period:.8f}\n")
        f.write(f"core |z|/de <= {args.core_z_de:.4f}\n\n")

        f.write("Poynting-balance statistics\n")
        f.write("---------------------------\n")
        f.write(f"rms dU/dtau = {np.sqrt(np.mean(dUdtau**2)):.8e}\n")
        f.write(f"rms PoyntingOut = {np.sqrt(np.mean(Fhat**2)):.8e}\n")
        f.write(f"rms intJdotE = {np.sqrt(np.mean(What**2)):.8e}\n")
        f.write(f"rms closure residual = {np.sqrt(np.mean(residual**2)):.8e}\n")
        denom=np.sqrt(np.mean((dUdtau**2+Fhat**2+What**2)))
        f.write(f"relative closure residual = {np.sqrt(np.mean(residual**2))/denom:.8e}\n\n")

        f.write("Density-shape variation\n")
        f.write("-----------------------\n")
        for name,y in [
            ("sigma_x/de",sx),("sigma_z/de",sz),("aspect_z_over_x",aspect),
            ("area_proxy_de2",area),("Nion_core",Ncore)
        ]:
            f.write(
                f"{name:18s} mean={np.mean(y):.8e} "
                f"std={np.std(y):.8e} CV={np.std(y)/abs(np.mean(y)):.8e}\n"
            )

        f.write("\nFixed-period harmonic phase relative to Psi\n")
        f.write("-------------------------------------------\n")
        f.write(f"Psi amplitude={Apsi:.8e} R2={Rpsi:.6f}\n")
        for k,(pred,A,phi,R,c,dphi) in fits.items():
            f.write(
                f"{k:10s} amp={A:.8e} R2={R:.6f} "
                f"phase_minus_Psi={dphi:.8f} rad "
                f"lag={dphi/(2*np.pi)*args.period:.8f}\n"
            )

        f.write("\nInterpretation guide\n")
        f.write("--------------------\n")
        f.write("If sigma_z rises while sigma_x falls with nearly constant Nion and area\n")
        f.write("proxy, the breathing is primarily a shape/ellipticity oscillation rather\n")
        f.write("than bulk compression.  Poynting closure distinguishes local J.E exchange\n")
        f.write("from electromagnetic energy transported through the z boundaries.\n")

    print("Saved ishizawa_breathing_mechanism_history.txt")
    print("Saved ishizawa_breathing_poynting_balance.png")
    print("Saved ishizawa_breathing_shape_moments.png")
    print("Saved ishizawa_breathing_mechanism_phase.png")
    print("Saved ishizawa_breathing_mechanism_summary.txt")


if __name__=="__main__":
    main()
