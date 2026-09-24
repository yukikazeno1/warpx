#!/usr/bin/env python3
"""Test whether the late breathing mode is oscillatory reconnection.

For the system-scale low-pass magnetic flux
    Psi = s [A_y(O)-A_y(X)] / (B0 de),
with s the O/X orientation used by analyze_ishizawa_saturation.py, 2-D
Faraday's law gives
    dPsi/d(omega_ci t)
      = s [E_y(X)-E_y(O)] / (B0 de omega_ci).

The script therefore compares the measured derivative of Psi_LS with the
low-mode O-X electric-field difference.  A good match, together with periodic
sign reversal, is direct evidence that the nonlinear breathing is an
oscillatory reconnection / re-reconnection cycle rather than only a geometric
motion of an already-formed island.

Outputs
-------
ishizawa_oscillatory_reconnection_history.txt
ishizawa_oscillatory_reconnection.png
ishizawa_oscillatory_reconnection_summary.txt
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import params, reconstruct_Ay
from analyze_ishizawa_breathing_phases import load_full_fields
from analyze_ishizawa_saturation import lowpass_x, choose_system_scale_OX


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",default="runs_ishizawa_scale/ppc49")
    p.add_argument("--history",required=True)
    p.add_argument("--tmin",type=float,default=2.8)
    p.add_argument("--tmax",type=float,default=4.0)
    p.add_argument("--period",type=float,default=0.63)
    p.add_argument("--max-mode",type=int,default=3)
    p.add_argument("--z-average-de",type=float,default=0.0,
                   help="optional half-width for Ey averaging around z=0")
    return p.parse_args()


def numeric_key(path):
    m=re.search(r"(\d+)$",os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def step_from_plotfile(path):
    name=os.path.basename(path.rstrip("/"))
    if not name.startswith("diag1"):
        return -1
    s=name[len("diag1"):]
    return int(s) if s.isdigit() else -1


def load_history(path):
    a=np.loadtxt(path)
    if a.ndim==1:
        a=a[None,:]
    return dict(step=a[:,0].astype(int),t=a[:,1],psi=a[:,2])


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


def linreg(x,y):
    m=np.isfinite(x)&np.isfinite(y)
    x=x[m]; y=y[m]
    A=np.column_stack([x,np.ones_like(x)])
    c,*_=np.linalg.lstsq(A,y,rcond=None)
    pred=A@c
    ssr=np.sum((y-pred)**2)
    sst=np.sum((y-np.mean(y))**2)
    r2=1-ssr/sst if sst>0 else np.nan
    return float(c[0]),float(c[1]),float(r2),pred


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
        if t<args.tmin or t>args.tmax or int(step) not in bystep:
            continue
        F=load_full_fields(bystep[int(step)])
        Ay=reconstruct_Ay(F["Bx"],F["Bz"],F["x"],F["z"])
        Ay_lp=lowpass_x(Ay,args.max_mode)
        Ey_lp=lowpass_x(F["Ey"],args.max_mode)

        j0,io,ix,orient,AO,AX,psi_si=choose_system_scale_OX(Ay_lp,F["z"])

        if args.z_average_de>0:
            zm=np.abs(F["z"])<=args.z_average_de*P["de"]
            EyO=float(np.mean(Ey_lp[io,zm]))
            EyX=float(np.mean(Ey_lp[ix,zm]))
        else:
            EyO=float(Ey_lp[io,j0])
            EyX=float(Ey_lp[ix,j0])

        psi_from_fields=psi_si/(P["B0"]*P["de"])
        dpsi_faraday=orient*(EyX-EyO)/(P["B0"]*P["de"]*P["wci"])

        rows.append([
            int(step),t,H["psi"][i],psi_from_fields,
            io,ix,orient,
            F["x"][io]/P["de"],F["x"][ix]/P["de"],
            EyO/(P["B0"]*299792458.0),
            EyX/(P["B0"]*299792458.0),
            dpsi_faraday
        ])

    a=np.asarray(rows,float)
    if len(a)<8:
        raise RuntimeError("Too few matched frames")

    (step,t,psi_hist,psi_field,io,ix,orient,xO,xX,
     EyO_cB,EyX_cB,dpsi_E)=a.T

    dpsi_num=np.gradient(psi_hist,t,edge_order=2)

    # Relation between the two reconnection-rate estimates.
    corr=float(np.corrcoef(dpsi_num,dpsi_E)[0,1])
    slope,intercept,r2,pred=linreg(dpsi_E,dpsi_num)
    rms_num=float(np.sqrt(np.mean(dpsi_num**2)))
    rms_err=float(np.sqrt(np.mean((dpsi_num-dpsi_E)**2)))
    rel_rms=rms_err/rms_num if rms_num>0 else np.nan

    # Reconstruct Psi by integrating the measured O-X electric-field difference.
    psi_int=np.empty_like(psi_hist)
    psi_int[0]=psi_hist[0]
    for k in range(1,len(t)):
        dt=t[k]-t[k-1]
        psi_int[k]=psi_int[k-1]+0.5*(dpsi_E[k]+dpsi_E[k-1])*dt
    # Remove one constant offset only (already anchored at first point).
    integ_rms=float(np.sqrt(np.mean((psi_int-psi_hist)**2)))

    # Fixed breathing harmonic phases.
    _,Apsi,phipsi,Rpsi,cpsi=harmonic_fit(t,psi_hist,args.period)
    _,Adn,phidn,Rdn,cdn=harmonic_fit(t,dpsi_num,args.period)
    _,Ade,phide,Rde,cde=harmonic_fit(t,dpsi_E,args.period)
    dphi_E_psi=wrap_phase(phide-phipsi)
    dphi_E_dpsi=wrap_phase(phide-phidn)

    np.savetxt(
        "ishizawa_oscillatory_reconnection_history.txt",
        np.column_stack([
            step,t,psi_hist,psi_field,dpsi_num,dpsi_E,psi_int,
            xO,xX,EyO_cB,EyX_cB,orient
        ]),
        header=(
            "step omega_ci_t Psi_history Psi_from_fields "
            "dPsi_dtau_numeric dPsi_dtau_from_Ey Psi_integrated_from_Ey "
            "xO_de xX_de EyO_cB0 EyX_cB0 orientation"
        )
    )

    fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True)
    axs[0].plot(t,psi_hist,"o-",ms=3,label=r"$\Psi_{LS}$")
    axs[0].plot(t,psi_int,"--",label=r"$\int (E_y^X-E_y^O)dt$")
    axs[0].set_ylabel(r"$\Psi/(B_0d_e)$")
    axs[0].legend(fontsize=8)

    axs[1].plot(t,dpsi_num,label=r"$d\Psi/d(\omega_{ci}t)$ numeric")
    axs[1].plot(t,dpsi_E,label=r"$s(E_y^X-E_y^O)/(B_0d_e\omega_{ci})$")
    axs[1].axhline(0,lw=.8)
    axs[1].set_ylabel("reconnection rate")
    axs[1].legend(fontsize=8)

    axs[2].plot(t,EyX_cB-EyO_cB,label=r"$(E_y^X-E_y^O)/(cB_0)$")
    axs[2].axhline(0,lw=.8)
    axs[2].set_ylabel(r"$\Delta E_y/(cB_0)$")
    axs[2].set_xlabel(r"$\omega_{ci}t$")
    axs[2].legend(fontsize=8)
    for ax in axs: ax.grid(alpha=.25)
    fig.suptitle("System-scale oscillatory reconnection test")
    fig.tight_layout()
    fig.savefig("ishizawa_oscillatory_reconnection.png",dpi=210)
    plt.close(fig)

    # Scatter relation.
    fig,ax=plt.subplots(figsize=(6.2,5.5))
    ax.scatter(dpsi_E,dpsi_num,s=24)
    xx=np.linspace(np.min(dpsi_E),np.max(dpsi_E),200)
    ax.plot(xx,xx,"--",label="1:1")
    ax.plot(xx,slope*xx+intercept,label=f"fit slope={slope:.3f}")
    ax.set_xlabel("Faraday O-X rate")
    ax.set_ylabel("numeric dPsi/dtau")
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    fig.savefig("ishizawa_oscillatory_reconnection_scatter.png",dpi=210)
    plt.close(fig)

    with open("ishizawa_oscillatory_reconnection_summary.txt","w") as f:
        f.write("Ishizawa-scale oscillatory reconnection diagnostic\n")
        f.write("=================================================\n\n")
        f.write(f"interval omega_ci*t = {t[0]:.8f} .. {t[-1]:.8f}\n")
        f.write(f"samples = {len(t)}\n")
        f.write(f"x low-pass max mode = {args.max_mode}\n")
        f.write(f"reference period = {args.period:.8f}\n")
        f.write(f"orientation values = {sorted(set(orient.astype(int)))}\n\n")
        f.write("Faraday-law comparison\n")
        f.write("----------------------\n")
        f.write(f"corr(dPsi_numeric, dPsi_from_Ey) = {corr:.8f}\n")
        f.write(f"linear-fit slope numeric vs Ey = {slope:.8f}\n")
        f.write(f"linear-fit intercept = {intercept:.8e}\n")
        f.write(f"linear-fit R2 = {r2:.8f}\n")
        f.write(f"relative RMS rate mismatch = {rel_rms:.8e}\n")
        f.write(f"RMS Psi reconstruction mismatch = {integ_rms:.8e}\n\n")
        f.write("Fixed-period harmonic relation\n")
        f.write("------------------------------\n")
        f.write(f"Psi amp={Apsi:.8e} R2={Rpsi:.6f}\n")
        f.write(f"dPsi numeric amp={Adn:.8e} R2={Rdn:.6f}\n")
        f.write(f"dPsi from Ey amp={Ade:.8e} R2={Rde:.6f}\n")
        f.write(f"phase(Ey-rate)-phase(Psi) = {dphi_E_psi:.8f} rad\n")
        f.write(f"phase(Ey-rate)-phase(dPsi numeric) = {dphi_E_dpsi:.8f} rad\n\n")
        f.write("Interpretation\n")
        f.write("--------------\n")
        f.write("A periodic sign reversal of the O-X Ey difference, with a high correlation\n")
        f.write("to dPsi/dtau and the expected ~pi/2 phase relative to Psi, demonstrates\n")
        f.write("oscillatory reconnection/re-reconnection of the system-scale island.\n")

    print("Saved ishizawa_oscillatory_reconnection_history.txt")
    print("Saved ishizawa_oscillatory_reconnection.png")
    print("Saved ishizawa_oscillatory_reconnection_scatter.png")
    print("Saved ishizawa_oscillatory_reconnection_summary.txt")


if __name__=="__main__":
    main()
