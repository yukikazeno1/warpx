#!/usr/bin/env python3
"""Component-resolved energy-flow diagnostics for the late m=1 breathing mode.

Reads the same plotfiles/history as analyze_ishizawa_breathing_mechanism.py and
decomposes:

  magnetic energy: UBx, UBy, UBz
  electric energy: UEx, UEy, UEz
  field-particle work: JxEx, JyEy, JzEz
  z-boundary Poynting flux:
      S_z = (E_x B_y - E_y B_x)/mu0
          = S_z^(ExBy) + S_z^(-EyBx)

It also evaluates the Poynting theorem at the *breathing fundamental* rather
than over the full noisy time series.  This is useful because the slow
breathing signal is much cleaner than the raw cell-centered/staggered energy
balance.

Outputs
-------
ishizawa_breathing_energy_components_history.txt
ishizawa_breathing_magnetic_components.png
ishizawa_breathing_work_flux_components.png
ishizawa_breathing_fundamental_balance.png
ishizawa_breathing_energy_components_summary.txt
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import MU0, EPS0, params
from analyze_ishizawa_breathing_phases import load_full_fields, process_phase


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
    a = np.loadtxt(path)
    if a.ndim == 1:
        a = a[None, :]
    return dict(
        step=a[:,0].astype(int), t=a[:,1], psi=a[:,2],
        psi_m1=a[:,3], width=a[:,4],
    )


def harmonic_fit(t, y, T):
    u = t-t[0]
    om = 2*np.pi/T
    M = np.column_stack([
        np.ones_like(u), u, np.sin(om*u), np.cos(om*u)
    ])
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    pred = M@c
    ssr = np.sum((y-pred)**2)
    sst = np.sum((y-np.mean(y))**2)
    r2 = 1-ssr/sst if sst > 0 else np.nan
    amp = float(np.hypot(c[2], c[3]))
    phi = float(np.arctan2(c[3], c[2]))
    return pred, amp, phi, float(r2), c


def wrap_phase(x):
    return float(np.arctan2(np.sin(x), np.cos(x)))


def detr(y, t):
    c = np.polyfit(t, y, 1)
    z = y-np.polyval(c,t)
    s = np.std(z)
    return z/s if s > 0 else z


def fundamental_derivative_coeff(c, T):
    """Return sin/cos coefficients of d/dtau of fitted harmonic."""
    om = 2*np.pi/T
    # y = ... + a sin(om u) + b cos(om u)
    # dy/dtau = ... - b om sin + a om cos
    return np.array([-c[3]*om, c[2]*om], float)


def harmonic_vec(c):
    """Sin/cos coefficient vector."""
    return np.array([c[2], c[3]], float)


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params()
    H = load_history(args.history)

    files = sorted(
        glob.glob(str(Path(args.run_dir)/"diags"/"diag1*")),
        key=numeric_key
    )
    bystep = {step_from_plotfile(p):p for p in files}

    rows = []
    for i,(step,t) in enumerate(zip(H["step"],H["t"])):
        if t < args.tmin or t > args.tmax:
            continue
        if int(step) not in bystep:
            continue

        F = load_full_fields(bystep[int(step)])
        D = process_phase(F,P)
        core = np.abs(D["z"]) <= args.core_z_de*P["de"]
        ids = np.flatnonzero(core)
        jlo, jhi = int(ids[0]), int(ids[-1])
        dA = F["dx"]*F["dz"]

        def eint(A, pref):
            return np.sum(pref*A[:,core]**2)*dA

        UBx = eint(D["Bx"], 1/(2*MU0))
        UBy = eint(D["By"], 1/(2*MU0))
        UBz = eint(D["Bz"], 1/(2*MU0))
        UEx = eint(D["Ex"], EPS0/2)
        UEy = eint(D["Ey"], EPS0/2)
        UEz = eint(D["Ez"], EPS0/2)

        WJxEx = np.sum((D["jx"]*D["Ex"])[:,core])*dA
        WJyEy = np.sum((D["jy"]*D["Ey"])[:,core])*dA
        WJzEz = np.sum((D["jz"]*D["Ez"])[:,core])*dA

        Sz1 = D["Ex"]*D["By"]/MU0
        Sz2 = -D["Ey"]*D["Bx"]/MU0
        P1 = np.sum(Sz1[:,jhi]-Sz1[:,jlo])*F["dx"]
        P2 = np.sum(Sz2[:,jhi]-Sz2[:,jlo])*F["dx"]

        rows.append([
            int(step), t, H["psi"][i], H["width"][i],
            UBx,UBy,UBz,UEx,UEy,UEz,
            WJxEx,WJyEy,WJzEz,P1,P2
        ])

    a = np.asarray(rows,float)
    if len(a) < 8:
        raise RuntimeError("Too few samples")
    (step,t,psi,width,UBx,UBy,UBz,UEx,UEy,UEz,
     WJxEx,WJyEy,WJzEz,P1,P2) = a.T

    normU=(P["B0"]**2/(2*MU0))*P["de"]**2
    normP=normU*P["wci"]

    UBx/=normU; UBy/=normU; UBz/=normU
    UEx/=normU; UEy/=normU; UEz/=normU
    WJxEx/=normP; WJyEy/=normP; WJzEz/=normP
    P1/=normP; P2/=normP

    UB=UBx+UBy+UBz
    UE=UEx+UEy+UEz
    UEM=UB+UE
    WJE=WJxEx+WJyEy+WJzEz
    Pout=P1+P2

    out=np.column_stack([
        step,t,psi,width,
        UBx,UBy,UBz,UEx,UEy,UEz,
        WJxEx,WJyEy,WJzEz,P1,P2,UB,UE,UEM,WJE,Pout
    ])
    np.savetxt(
        "ishizawa_breathing_energy_components_history.txt",out,
        header=(
            "step omega_ci_t Psi width "
            "UBx UBy UBz UEx UEy UEz "
            "JxEx JyEy JzEz Pout_ExBy Pout_minusEyBx "
            "UB UE UEM JdotE Pout"
        )
    )

    _,Apsi,phipsi,Rpsi,cpsi=harmonic_fit(t,psi,args.period)

    names = [
        "UBx","UBy","UBz","UEx","UEy","UEz",
        "JxEx","JyEy","JzEz","Pout_ExBy","Pout_minusEyBx",
        "UB","UE","UEM","JdotE","Pout"
    ]
    ys = [
        UBx,UBy,UBz,UEx,UEy,UEz,
        WJxEx,WJyEy,WJzEz,P1,P2,
        UB,UE,UEM,WJE,Pout
    ]
    fits={}
    for n,y in zip(names,ys):
        pred,A,phi,R,c=harmonic_fit(t,y,args.period)
        fits[n]=(pred,A,phi,R,c,wrap_phase(phi-phipsi))

    # Fundamental Poynting closure:
    # d U_EM / dtau + Pout + J.E = 0
    cU=fits["UEM"][4]
    cJ=fits["JdotE"][4]
    cP=fits["Pout"][4]
    v_dU=fundamental_derivative_coeff(cU,args.period)
    v_J=harmonic_vec(cJ)
    v_P=harmonic_vec(cP)
    v_res=v_dU+v_J+v_P
    A_dU=float(np.linalg.norm(v_dU))
    A_J=float(np.linalg.norm(v_J))
    A_P=float(np.linalg.norm(v_P))
    A_res=float(np.linalg.norm(v_res))
    rel_sum=A_res/(A_dU+A_J+A_P)
    rel_rss=A_res/np.sqrt(A_dU*A_dU+A_J*A_J+A_P*A_P)

    # Magnetic component plot.
    fig,axs=plt.subplots(2,1,figsize=(9,7.5),sharex=True)
    axs[0].plot(t,UBx,label=r"$U_{Bx}$")
    axs[0].plot(t,UBy,label=r"$U_{By}$")
    axs[0].plot(t,UBz,label=r"$U_{Bz}$")
    axs[0].plot(t,UB,label=r"$U_B$",lw=2)
    axs[0].set_ylabel("magnetic energy / U0")
    axs[0].legend(ncol=4,fontsize=8)
    axs[1].plot(t,detr(psi,t),label=r"$\Psi$")
    axs[1].plot(t,detr(UBx,t),label=r"$U_{Bx}$")
    axs[1].plot(t,detr(UBy,t),label=r"$U_{By}$")
    axs[1].plot(t,detr(UBz,t),label=r"$U_{Bz}$")
    axs[1].set_ylabel("normalized detrended")
    axs[1].set_xlabel(r"$\omega_{ci}t$")
    axs[1].legend(ncol=4,fontsize=8)
    for ax in axs: ax.grid(alpha=.25)
    fig.suptitle("Magnetic-energy component exchange during breathing")
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_magnetic_components.png",dpi=210)
    plt.close(fig)

    # Work/Poynting decomposition.
    fig,axs=plt.subplots(2,1,figsize=(9,7.5),sharex=True)
    axs[0].plot(t,WJxEx,label=r"$\int J_xE_x$")
    axs[0].plot(t,WJyEy,label=r"$\int J_yE_y$")
    axs[0].plot(t,WJzEz,label=r"$\int J_zE_z$")
    axs[0].plot(t,WJE,label=r"$\int J\cdot E$",lw=2)
    axs[0].set_ylabel("power / (U0 omega_ci)")
    axs[0].legend(ncol=4,fontsize=8)
    axs[1].plot(t,P1,label=r"$P_{out}^{E_xB_y}$")
    axs[1].plot(t,P2,label=r"$P_{out}^{-E_yB_x}$")
    axs[1].plot(t,Pout,label=r"$P_{out}$",lw=2)
    axs[1].set_ylabel("power / (U0 omega_ci)")
    axs[1].set_xlabel(r"$\omega_{ci}t$")
    axs[1].legend(fontsize=8)
    for ax in axs: ax.grid(alpha=.25)
    fig.suptitle("Field-particle work and Poynting-flux channels")
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_work_flux_components.png",dpi=210)
    plt.close(fig)

    # Fundamental-only balance reconstructed from harmonic fits.
    predU=fits["UEM"][0]
    cU=fits["UEM"][4]
    u=t-t[0]; om=2*np.pi/args.period
    dUpred=cU[1]-cU[3]*om*np.sin(om*u)+cU[2]*om*np.cos(om*u)
    predJ=fits["JdotE"][0]
    predP=fits["Pout"][0]
    # Remove fitted linear trend from J/P for fundamental-only comparison.
    predJ_h=predJ-(cJ[0]+cJ[1]*u)
    predP_h=predP-(cP[0]+cP[1]*u)
    dU_h=dUpred-cU[1]
    balance=dU_h+predJ_h+predP_h

    fig,ax=plt.subplots(figsize=(9,5.7))
    ax.plot(t,dU_h,label=r"$dU_{EM}/d\tau$ fundamental")
    ax.plot(t,predJ_h,label=r"$J\cdot E$ fundamental")
    ax.plot(t,predP_h,label="Poynting-out fundamental")
    ax.plot(t,balance,label="fundamental residual",lw=2)
    ax.axhline(0,lw=.8)
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel("fundamental power / (U0 omega_ci)")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    ax.set_title(
        fr"Breathing-frequency Poynting closure: residual/(sum amps)={rel_sum:.3f}"
    )
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_fundamental_balance.png",dpi=210)
    plt.close(fig)

    with open("ishizawa_breathing_energy_components_summary.txt","w") as f:
        f.write("Ishizawa-scale breathing energy-component diagnostics\n")
        f.write("=====================================================\n\n")
        f.write(f"interval omega_ci*t = {t[0]:.8f} .. {t[-1]:.8f}\n")
        f.write(f"samples = {len(t)}\n")
        f.write(f"reference period = {args.period:.8f}\n\n")
        f.write("Fixed-period harmonic components relative to Psi\n")
        f.write("------------------------------------------------\n")
        f.write(f"Psi amp={Apsi:.8e} R2={Rpsi:.6f}\n")
        for n in names:
            pred,A,phi,R,c,dphi=fits[n]
            f.write(
                f"{n:18s} amp={A:.8e} R2={R:.6f} "
                f"phase_minus_Psi={dphi:.8f} rad "
                f"lag={dphi/(2*np.pi)*args.period:.8f}\n"
            )
        f.write("\nBreathing-fundamental Poynting closure\n")
        f.write("--------------------------------------\n")
        f.write(f"|dU/dtau| harmonic amplitude = {A_dU:.8e}\n")
        f.write(f"|J.E| harmonic amplitude = {A_J:.8e}\n")
        f.write(f"|Pout| harmonic amplitude = {A_P:.8e}\n")
        f.write(f"|closure residual| harmonic amplitude = {A_res:.8e}\n")
        f.write(f"residual / sum(amplitudes) = {rel_sum:.8e}\n")
        f.write(f"residual / rss(amplitudes) = {rel_rss:.8e}\n")
        f.write(
            f"phase(Pout)-phase(J.E) = "
            f"{wrap_phase(fits['Pout'][2]-fits['JdotE'][2]):.8f} rad\n"
        )
        f.write("\nInterpretation guide\n")
        f.write("--------------------\n")
        f.write("UBx/UBz phases show whether breathing mainly redistributes magnetic energy\n")
        f.write("between reconnecting and reconnected components.  JyEy dominance would\n")
        f.write("identify the reconnection-electric-field channel of J.E.  The -EyBx term\n")
        f.write("is the expected dominant z-directed Poynting channel for a weak-guide-field\n")
        f.write("Harris configuration.  Fundamental closure is more meaningful for the slow\n")
        f.write("breathing mode than the raw broadband closure from cell-centered outputs.\n")

    print("Saved ishizawa_breathing_energy_components_history.txt")
    print("Saved ishizawa_breathing_magnetic_components.png")
    print("Saved ishizawa_breathing_work_flux_components.png")
    print("Saved ishizawa_breathing_fundamental_balance.png")
    print("Saved ishizawa_breathing_energy_components_summary.txt")


if __name__=="__main__":
    main()
