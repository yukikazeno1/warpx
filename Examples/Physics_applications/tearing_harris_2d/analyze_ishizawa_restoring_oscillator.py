#!/usr/bin/env python3
"""Generalized Lorentz-restoring-force test for the saturated island breathing.

Treat the late m=1 island width as a global breathing coordinate

    q(t) = w(t)/<w> - 1 .

For an approximately affine vertical displacement delta z = z delta q, the
generalized force conjugate to q is

    Q = integral z f_z dA,

and the corresponding ion inertial coefficient is

    I = integral rho_i z^2 dA.

Hence a purely magnetic restoring oscillator would satisfy

    q_ddot = Q_L / I,

with
    Q_L = integral z (J x B)_z dA.

The magnetic force is also decomposed into
    -grad(B^2/2mu0)
and
    (B.grad)B_z/mu0.

The script compares the measured d2q/d(omega_ci t)^2 with Q/I/omega_ci^2,
fits a magnetic stiffness
    Q/I/omega_ci^2 ~= -Omega_mag^2 q + const,
and reports the implied period
    T_mag omega_ci = 2 pi / Omega_mag.

This is a field-level test of a global magnetic restoring mode; it does not
assume that the breathing period equals an Alfvén transit time on a particular
flux surface.

Outputs
-------
ishizawa_restoring_oscillator_history.txt
ishizawa_restoring_oscillator.png
ishizawa_restoring_force_vs_displacement.png
ishizawa_restoring_oscillator_summary.txt
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import QE, MU0, params
from analyze_ishizawa_breathing_phases import load_full_fields, process_phase


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",default="runs_ishizawa_scale/ppc49")
    p.add_argument("--history",required=True)
    p.add_argument("--tmin",type=float,default=2.8)
    p.add_argument("--tmax",type=float,default=4.0)
    p.add_argument("--period",type=float,default=0.63)
    p.add_argument("--core-z-de",type=float,default=12.0)
    p.add_argument("--density-cut",type=float,default=1.0e-4,
                   help="minimum aligned ion density n_i/n0 included in inertia")
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
        D=process_phase(F,P)

        X,Z=np.meshgrid(D["x"],D["z"],indexing="ij")
        dA=F["dx"]*F["dz"]

        ni=np.maximum(D["ni"],0.0)
        zmask=np.abs(D["z"])<=args.core_z_de*P["de"]
        mask=np.broadcast_to(zmask[None,:],ni.shape)
        imask=mask & (ni>=args.density_cut*P["n0"])

        rho_m=P["mi"]*ni

        # Generalized inertial coefficient for affine vertical breathing.
        I=np.sum((rho_m*Z*Z)[imask])*dA

        # Generalized magnetic forces.
        QL=np.sum((Z*D["FL_z"])[mask])*dA
        QP=np.sum((Z*D["Fp_z"])[mask])*dA
        QT=np.sum((Z*D["Ft_z"])[mask])*dA

        amag=QL/I if I>0 else np.nan
        ap=QP/I if I>0 else np.nan
        at=QT/I if I>0 else np.nan

        rows.append([
            int(step),t,H["psi"][i],H["width"][i],
            I,QL,QP,QT,
            amag/P["wci"]**2,ap/P["wci"]**2,at/P["wci"]**2,
            np.sum(rho_m[imask])*dA,
        ])

    a=np.asarray(rows,float)
    if len(a)<10:
        raise RuntimeError("Too few matched frames")

    step,t,psi,width,I,QL,QP,QT,amag,ap,at,mass=a.T

    # Dimensionless global breathing coordinate.
    wmean=np.mean(width)
    q=width/wmean-1.0

    # Numerical acceleration in tau=omega_ci*t.
    dq=np.gradient(q,t,edge_order=2)
    ddq=np.gradient(dq,t,edge_order=2)

    # Smooth measured acceleration at the known breathing fundamental.
    qfit,Aq,phiq,Rq,cq=harmonic_fit(t,q,args.period)
    om=2*np.pi/args.period
    u=t-t[0]
    ddq_fit=-(om**2)*(cq[2]*np.sin(om*u)+cq[3]*np.cos(om*u))

    # Harmonic magnetic acceleration and components.
    amag_fit,Aam,pham,Ram,cam=harmonic_fit(t,amag,args.period)
    ap_fit,Aap,phap,Rap,cap=harmonic_fit(t,ap,args.period)
    at_fit,Aat,phat,Rat,cat=harmonic_fit(t,at,args.period)

    # Linear stiffness from raw force-displacement loop.
    slope,intercept,Rstiff,pred=linreg(q,amag)
    omega2_mag=max(0.0,-slope)
    Tmag=2*np.pi/np.sqrt(omega2_mag) if omega2_mag>0 else np.nan

    # Same using harmonic amplitudes: if amag ~= -Omega^2 q.
    omega2_amp=Aam/Aq if Aq>0 else np.nan
    Tamp=2*np.pi/np.sqrt(omega2_amp) if omega2_amp>0 else np.nan
    phase_restore=wrap(pham-phiq-np.pi)

    # Compare magnetic acceleration to measured harmonic acceleration.
    corr_raw=float(np.corrcoef(ddq,amag)[0,1])
    corr_harm=float(np.corrcoef(ddq_fit,amag_fit)[0,1])
    rms_ref=float(np.sqrt(np.mean(ddq_fit**2)))
    rel_harm=float(np.sqrt(np.mean((ddq_fit-amag_fit)**2))/rms_ref)

    np.savetxt(
        "ishizawa_restoring_oscillator_history.txt",
        np.column_stack([
            step,t,psi,width,q,dq,ddq,ddq_fit,
            I,QL,QP,QT,amag,ap,at,mass
        ]),
        header=(
            "step omega_ci_t Psi width_de q dq_dtau ddq_dtau2 "
            "ddq_harmonic I Q_L Q_Pmag Q_tension "
            "a_L_over_wci2 a_P_over_wci2 a_T_over_wci2 ion_mass_core"
        )
    )

    fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True)
    axs[0].plot(t,q,label="width displacement q")
    axs[0].plot(t,qfit,"--",label="T=0.63 harmonic")
    axs[0].set_ylabel("q")
    axs[0].legend(fontsize=8)

    axs[1].plot(t,ddq,label=r"numeric $d^2q/d\tau^2$",alpha=.55)
    axs[1].plot(t,ddq_fit,lw=2,label="measured harmonic acceleration")
    axs[1].plot(t,amag_fit,lw=2,label=r"magnetic $Q_L/I/\omega_{ci}^2$")
    axs[1].axhline(0,lw=.7)
    axs[1].set_ylabel("dimensionless acceleration")
    axs[1].legend(fontsize=8)

    axs[2].plot(t,ap_fit,label="magnetic-pressure contribution")
    axs[2].plot(t,at_fit,label="tension contribution")
    axs[2].plot(t,amag_fit,lw=2,label="sum Lorentz")
    axs[2].axhline(0,lw=.7)
    axs[2].set_xlabel(r"$\omega_{ci}t$")
    axs[2].set_ylabel(r"$Q/I/\omega_{ci}^2$")
    axs[2].legend(fontsize=8)
    for ax in axs: ax.grid(alpha=.25)
    fig.suptitle("Generalized magnetic restoring-force test")
    fig.tight_layout()
    fig.savefig("ishizawa_restoring_oscillator.png",dpi=210)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(6.5,5.5))
    ax.scatter(q,amag,s=28,label="raw frames")
    xx=np.linspace(np.min(q),np.max(q),200)
    ax.plot(xx,slope*xx+intercept,label=f"fit: slope={slope:.3g}")
    ax.axhline(0,lw=.7); ax.axvline(0,lw=.7)
    ax.set_xlabel("width displacement q")
    ax.set_ylabel(r"$Q_L/(I\omega_{ci}^2)$")
    ax.grid(alpha=.25); ax.legend()
    ax.set_title(f"magnetic stiffness fit, R2={Rstiff:.3f}")
    fig.tight_layout()
    fig.savefig("ishizawa_restoring_force_vs_displacement.png",dpi=210)
    plt.close(fig)

    with open("ishizawa_restoring_oscillator_summary.txt","w") as f:
        f.write("Ishizawa-scale generalized magnetic restoring-oscillator test\n")
        f.write("===========================================================\n\n")
        f.write(f"interval omega_ci*t = {t[0]:.8f} .. {t[-1]:.8f}\n")
        f.write(f"measured breathing period = {args.period:.8f}\n")
        f.write(f"samples = {len(t)}\n")
        f.write(f"core |z|/de <= {args.core_z_de:.6f}\n")
        f.write(f"inertia density cut n_i/n0 >= {args.density_cut:.3e}\n\n")

        f.write("Width harmonic\n")
        f.write("--------------\n")
        f.write(f"A_q = {Aq:.8e}\n")
        f.write(f"R2_q = {Rq:.8f}\n\n")

        f.write("Magnetic restoring acceleration harmonic\n")
        f.write("----------------------------------------\n")
        f.write(f"A_mag = {Aam:.8e}\n")
        f.write(f"R2_mag = {Ram:.8f}\n")
        f.write(f"phase_mag_minus_q_minus_pi = {phase_restore:.8f} rad\n")
        f.write(f"pressure amp = {Aap:.8e}, R2={Rap:.8f}\n")
        f.write(f"tension amp = {Aat:.8e}, R2={Rat:.8f}\n\n")

        f.write("Effective stiffness / predicted period\n")
        f.write("--------------------------------------\n")
        f.write(f"raw stiffness slope d(a_mag)/dq = {slope:.8e}\n")
        f.write(f"raw stiffness R2 = {Rstiff:.8f}\n")
        f.write(f"T_mag*omega_ci from raw stiffness = {Tmag:.8f}\n")
        f.write(f"Omega_mag^2 from harmonic amp ratio = {omega2_amp:.8e}\n")
        f.write(f"T_mag*omega_ci from harmonic amp ratio = {Tamp:.8f}\n\n")

        f.write("Acceleration closure\n")
        f.write("--------------------\n")
        f.write(f"corr(raw ddq, raw magnetic acceleration) = {corr_raw:.8f}\n")
        f.write(f"corr(harmonic ddq, harmonic magnetic acceleration) = {corr_harm:.8f}\n")
        f.write(f"relative RMS harmonic acceleration mismatch = {rel_harm:.8e}\n\n")

        f.write("Interpretation guide\n")
        f.write("--------------------\n")
        f.write(
            "A magnetic-restoring interpretation is supported if the generalized "
            "Lorentz acceleration is approximately pi out of phase with q, "
            "the force-displacement relation has a stable negative slope, and "
            "the implied T_mag is close to the measured breathing period.  "
            "Poor acceleration closure means that pressure/kinetic stresses are "
            "dynamically important even if the Lorentz force is restoring.\n"
        )

    print("Saved ishizawa_restoring_oscillator_history.txt")
    print("Saved ishizawa_restoring_oscillator.png")
    print("Saved ishizawa_restoring_force_vs_displacement.png")
    print("Saved ishizawa_restoring_oscillator_summary.txt")


if __name__=="__main__":
    main()
