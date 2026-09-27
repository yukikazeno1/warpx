#!/usr/bin/env python3
"""Close a reduced breathing-mode oscillator with dense Pxx + EM diagnostics.

The Pxx thermodynamic diagnostic uses exactly the same width coordinate q that
is stored in ishizawa_generalized_em_force_history.txt.  This script fits

    a_j = b_j + K_j q + C_j dq/dtau,   tau = omega_ci t,

for the electromagnetic channels on the dense field history and combines them
with the phase-resolved Pxx response.

For a harmonic breathing mode with period T,

    q'' = -omega_b^2 q,  omega_b = 2*pi/T,

so the required leading-order coefficients are

    K_required = -omega_b^2,   C_required = 0.

The reduced closure tested here is

    q'' ~= a_EM + a_Pxx.

K<0 is restoring.  C<0 is damping and C>0 is antidamping.

This is deliberately a reduced closure: P_xz/P_zz, bulk stress, and other
particle channels are not included unless separately shown to be negligible.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--em-history",required=True)
    p.add_argument("--pxx-history",required=True)
    p.add_argument("--period",type=float,default=0.63)
    p.add_argument("--tmin",type=float,default=None)
    p.add_argument("--tmax",type=float,default=None)
    return p.parse_args()


def harmonic_qdot(t,q,T):
    om=2*np.pi/T
    u=t-t[0]
    M=np.column_stack([np.ones_like(u),u,np.sin(om*u),np.cos(om*u)])
    c,*_=np.linalg.lstsq(M,q,rcond=None)
    qfit=M@c
    qdot=c[1] + om*(c[2]*np.cos(om*u)-c[3]*np.sin(om*u))
    ssr=np.sum((q-qfit)**2)
    sst=np.sum((q-q.mean())**2)
    r2=1-ssr/sst if sst>0 else np.nan
    return qfit,qdot,r2,c


def harmonic_complex_fit(t,y,T):
    """Return trend+fundamental fit and complex fundamental amplitude.

    Convention:
        y_h = s sin(omega t) + c cos(omega t)
        Z_y = c - i s
    so y_h = Re[Z_y exp(i omega t)].
    """
    om=2*np.pi/T
    M=np.column_stack([np.ones_like(t),t,np.sin(om*t),np.cos(om*t)])
    coef,*_=np.linalg.lstsq(M,y,rcond=None)
    pred=M@coef
    fundamental=coef[2]*np.sin(om*t)+coef[3]*np.cos(om*t)
    z=complex(coef[3],-coef[2])
    ssr=np.sum((y-pred)**2)
    sst=np.sum((y-y.mean())**2)
    r2=1-ssr/sst if sst>0 else np.nan
    return dict(coef=coef,pred=pred,fundamental=fundamental,Z=z,R2=float(r2))


def direct_transfer(t,q,y,T):
    """Fundamental transfer H = Z_y/Z_q = K + i omega C."""
    fq=harmonic_complex_fit(t,q,T)
    fy=harmonic_complex_fit(t,y,T)
    H=fy["Z"]/fq["Z"]
    om=2*np.pi/T
    return dict(
        K=float(np.real(H)),
        C=float(np.imag(H)/om),
        H=H,
        qfit=fq,
        yfit=fy,
    )


def fit_kc(q,qd,y):
    X=np.column_stack([np.ones_like(q),q,qd])
    beta,*_=np.linalg.lstsq(X,y,rcond=None)
    yp=X@beta
    ssr=np.sum((y-yp)**2)
    sst=np.sum((y-y.mean())**2)
    r2=1-ssr/sst if sst>0 else np.nan
    return dict(b=float(beta[0]),K=float(beta[1]),C=float(beta[2]),
                R2=float(r2),pred=yp)


def response_metrics(K,C,omega):
    H=K+1j*omega*C
    phase_rest=np.angle(H/(-1.0+0j))
    return abs(H),float(phase_rest)


def main():
    args=parse_args()
    em=np.loadtxt(args.em_history)
    px=np.loadtxt(args.pxx_history)
    if em.ndim==1: em=em[None,:]
    if px.ndim==1: px=px[None,:]

    # generalized EM history columns:
    # 0 step,1 t,2 Psi,3 width,4 q,5 I,6..10 Qs,
    # 11 a_Pmag,12 a_tension,13 a_JxB,14 a_rhoE,15 a_EM,
    # 16 measured_ddq_harmonic,17 required_nonEM_acceleration
    te=em[:,1]; qe=em[:,4]
    tp=px[:,1]; qp=px[:,2]; qdp=px[:,3]

    tmin=max(te.min(),tp.min()) if args.tmin is None else args.tmin
    tmax=min(te.max(),tp.max()) if args.tmax is None else args.tmax
    me=(te>=tmin)&(te<=tmax)
    mp=(tp>=tmin)&(tp<=tmax)
    te=te[me]; qe=qe[me]
    if len(te)<8 or np.count_nonzero(mp)<6:
        raise RuntimeError("too few overlapping EM/Pxx samples")

    qefit,qde,Rq,_=harmonic_qdot(te,qe,args.period)
    omega=2*np.pi/args.period

    em_channels={
        "magnetic pressure":em[me,11],
        "magnetic tension":em[me,12],
        "JxB":em[me,13],
        "rhoE":em[me,14],
        "EM total":em[me,15],
    }
    emfit={k:fit_kc(qe,qde,v) for k,v in em_channels.items()}

    # Pxx columns from thermodynamic history:
    # 5 i-density,6 i-thermal,7 i-cross,8 i-total,
    # 9 e-density,10 e-thermal,11 e-cross,12 e-total,13 all-total
    pxx_channels={
        "ion density":px[mp,5],
        "ion thermal":px[mp,6],
        "ion cross":px[mp,7],
        "ion Pxx":px[mp,8],
        "electron density":px[mp,9],
        "electron thermal":px[mp,10],
        "electron cross":px[mp,11],
        "electron Pxx":px[mp,12],
        "Pxx total":px[mp,13],
    }
    pfit={k:fit_kc(qp[mp],qdp[mp],v) for k,v in pxx_channels.items()}

    # Direct breathing-frequency transfer functions.  These retain only the
    # fundamental at the observed breathing frequency and are therefore much
    # less sensitive to DC offsets, unrelated harmonics, and the poor raw
    # time-domain R2 of a strongly cancelling EM total.
    emharm={k:direct_transfer(te,qe,v,args.period) for k,v in em_channels.items()}
    pxharm={k:direct_transfer(tp[mp],qp[mp],v,args.period) for k,v in pxx_channels.items()}

    Kreq=-omega**2
    Creq=0.0
    Kem=emfit["EM total"]["K"]; Cem=emfit["EM total"]["C"]
    Kp=pfit["Pxx total"]["K"]; Cp=pfit["Pxx total"]["C"]
    Knet=Kem+Kp
    Cnet=Cem+Cp
    Kres=Kreq-Knet
    Cres=Creq-Cnet

    Tpred=2*np.pi/np.sqrt(-Knet) if Knet<0 else np.nan
    zeta=(-Cnet/(2*np.sqrt(-Knet))) if Knet<0 else np.nan
    growth_rate=Cnet/2.0

    amp_req,phase_req=response_metrics(Kreq,Creq,omega)
    amp_em,phase_em=response_metrics(Kem,Cem,omega)
    amp_p,phase_p=response_metrics(Kp,Cp,omega)
    amp_net,phase_net=response_metrics(Knet,Cnet,omega)

    # Fundamental-only complex closure.
    Hem=emharm["EM total"]["H"]
    Hp=pxharm["Pxx total"]["H"]
    Hnet=Hem+Hp
    Hreq=complex(Kreq,0.0)
    Hmiss=Hreq-Hnet
    Kem_h=float(np.real(Hem)); Cem_h=float(np.imag(Hem)/omega)
    Kp_h=float(np.real(Hp)); Cp_h=float(np.imag(Hp)/omega)
    Knet_h=float(np.real(Hnet)); Cnet_h=float(np.imag(Hnet)/omega)
    Kmiss_h=float(np.real(Hmiss)); Cmiss_h=float(np.imag(Hmiss)/omega)
    rel_complex_error=float(abs(Hmiss)/abs(Hreq))
    Tpred_h=2*np.pi/np.sqrt(-Knet_h) if Knet_h<0 else np.nan

    with open("ishizawa_reduced_oscillator_closure_summary.txt","w") as f:
        f.write("Reduced breathing-mode oscillator closure\n")
        f.write("========================================\n\n")
        f.write(f"overlap omega_ci*t = {tmin:.8f} .. {tmax:.8f}\n")
        f.write(f"period omega_ci*T = {args.period:.8f}\n")
        f.write(f"omega_b/omega_ci = {omega:.8f}\n")
        f.write(f"EM q harmonic R2 = {Rq:.8f}\n")
        f.write(f"EM samples = {len(te)}\n")
        f.write(f"Pxx samples = {np.count_nonzero(mp)}\n\n")

        f.write("Fit model: a = b + K q + C dq/dtau\n")
        f.write("K<0 restoring; C<0 damping.\n\n")

        f.write("EM channels\n")
        f.write("-----------\n")
        f.write("channel                  K              C            R2\n")
        for name,s in emfit.items():
            f.write(f"{name:20s} {s['K']:+13.6e} {s['C']:+13.6e} {s['R2']:10.6f}\n")

        f.write("\nPxx channels\n")
        f.write("------------\n")
        f.write("channel                  K              C            R2\n")
        for name,s in pfit.items():
            f.write(f"{name:20s} {s['K']:+13.6e} {s['C']:+13.6e} {s['R2']:10.6f}\n")

        f.write("\nReduced closure\n")
        f.write("---------------\n")
        f.write(f"required K = -omega_b^2 = {Kreq:+.8e}\n")
        f.write(f"required C = {Creq:+.8e}\n")
        f.write(f"K_EM = {Kem:+.8e}\n")
        f.write(f"C_EM = {Cem:+.8e}\n")
        f.write(f"K_Pxx = {Kp:+.8e}\n")
        f.write(f"C_Pxx = {Cp:+.8e}\n")
        f.write(f"K_EM+Pxx = {Knet:+.8e}\n")
        f.write(f"C_EM+Pxx = {Cnet:+.8e}\n")
        f.write(f"missing K = required - closure = {Kres:+.8e}\n")
        f.write(f"missing C = required - closure = {Cres:+.8e}\n")
        f.write(f"relative stiffness closure error = {abs(Kres)/abs(Kreq):.8e}\n")
        f.write(f"period predicted from K_EM+Pxx = {Tpred:.8f}\n")
        f.write(f"effective damping ratio from closure = {zeta:.8e}\n")
        f.write(f"amplitude growth/damping exponent C_net/2 = {growth_rate:+.8e} per unit omega_ci*t\n")

        f.write("\nRegression-derived complex response at observed breathing frequency\n")
        f.write("---------------------------------------------------------------\n")
        f.write("channel        |K+i omega C|      phase relative to restoring [rad]\n")
        f.write(f"required       {amp_req:14.6e} {phase_req:+16.8f}\n")
        f.write(f"EM             {amp_em:14.6e} {phase_em:+16.8f}\n")
        f.write(f"Pxx            {amp_p:14.6e} {phase_p:+16.8f}\n")
        f.write(f"EM+Pxx         {amp_net:14.6e} {phase_net:+16.8f}\n")

        f.write("\nDirect fundamental-only complex closure\n")
        f.write("---------------------------------------\n")
        f.write("The transfer is H = Z_force/Z_q = K + i omega C.\n")
        f.write(f"EM fundamental-fit R2 = {emharm['EM total']['yfit']['R2']:.8f}\n")
        f.write(f"Pxx fundamental-fit R2 = {pxharm['Pxx total']['yfit']['R2']:.8f}\n")
        f.write(f"K_EM,fund = {Kem_h:+.8e}\n")
        f.write(f"C_EM,fund = {Cem_h:+.8e}\n")
        f.write(f"K_Pxx,fund = {Kp_h:+.8e}\n")
        f.write(f"C_Pxx,fund = {Cp_h:+.8e}\n")
        f.write(f"K_net,fund = {Knet_h:+.8e}\n")
        f.write(f"C_net,fund = {Cnet_h:+.8e}\n")
        f.write(f"K_missing,fund = {Kmiss_h:+.8e}\n")
        f.write(f"C_missing,fund = {Cmiss_h:+.8e}\n")
        f.write(f"relative complex closure error = {rel_complex_error:.8e}\n")
        f.write(f"period from fundamental net stiffness = {Tpred_h:.8f}\n")
        f.write(f"|H_required| = {abs(Hreq):.8e}\n")
        f.write(f"|H_EM| = {abs(Hem):.8e}\n")
        f.write(f"|H_Pxx| = {abs(Hp):.8e}\n")
        f.write(f"|H_net| = {abs(Hnet):.8e}\n")
        f.write(f"phase(H_net/restoring) = {np.angle(Hnet/(-1.0+0j)):+.8f} rad\n")

        f.write("\nCaution:\n")
        f.write("This is a reduced closure.  A non-zero residual can contain P_xz/P_zz,\n")
        f.write("bulk stress, mode-shape evolution, and normalization differences.\n")

    # Plot coefficient closure.
    labels=["required","EM","Pxx","EM+Pxx","missing"]
    Ks=[Kreq,Kem,Kp,Knet,Kres]
    Cs=[Creq,Cem,Cp,Cnet,Cres]
    x=np.arange(len(labels))
    fig,axs=plt.subplots(2,1,figsize=(8,7),sharex=True)
    axs[0].bar(x,Ks)
    axs[0].axhline(0,lw=.8)
    axs[0].set_ylabel("K")
    axs[0].set_title("Reduced breathing-mode stiffness closure")
    axs[1].bar(x,Cs)
    axs[1].axhline(0,lw=.8)
    axs[1].set_ylabel("C")
    axs[1].set_xticks(x,labels,rotation=20)
    for ax in axs: ax.grid(axis="y",alpha=.25)
    fig.tight_layout()
    fig.savefig("ishizawa_reduced_oscillator_closure_coefficients.png",dpi=210)
    plt.close(fig)

    # Plot only breathing-frequency perturbations at the particle-rich phases.
    # Do NOT compare raw generalized forces to -omega^2 q: the raw forces carry
    # large equilibrium/DC offsets that are absorbed by the fitted intercept b.
    tpp=tp[mp]
    qfund_p=pxharm["Pxx total"]["qfit"]["fundamental"]
    areq_h=Kreq*qfund_p
    aem_h=np.interp(tpp,te,emharm["EM total"]["yfit"]["fundamental"])
    apxx_h=pxharm["Pxx total"]["yfit"]["fundamental"]
    acl_h=aem_h+apxx_h
    resid_h=areq_h-acl_h

    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
    axs[0].plot(tpp,areq_h,"o-",label="required fundamental")
    axs[0].plot(tpp,aem_h,"o-",label="EM fundamental")
    axs[0].plot(tpp,apxx_h,"o-",label="Pxx fundamental")
    axs[0].plot(tpp,acl_h,"o-",label="EM + Pxx fundamental")
    axs[0].axhline(0,lw=.7); axs[0].grid(alpha=.25); axs[0].legend()
    axs[0].set_ylabel("fundamental modal acceleration")

    axs[1].plot(tpp,resid_h,"o-",label="required - (EM+Pxx), fundamental")
    axs[1].axhline(0,lw=.7); axs[1].grid(alpha=.25); axs[1].legend()
    axs[1].set_ylabel("fundamental closure residual")
    axs[1].set_xlabel(r"$\omega_{ci}t$")
    fig.suptitle("Breathing-frequency reduced closure at particle-rich phases")
    fig.tight_layout()
    fig.savefig("ishizawa_reduced_oscillator_closure_time.png",dpi=210)
    plt.close(fig)

    # Complex-plane closure diagram.
    fig,ax=plt.subplots(figsize=(6.5,6))
    vecs=[("required",Hreq),("EM",Hem),("Pxx",Hp),("EM+Pxx",Hnet),("missing",Hmiss)]
    for name,z in vecs:
        ax.arrow(0,0,np.real(z),np.imag(z),length_includes_head=True,
                 head_width=3.0,head_length=4.0,alpha=.8)
        ax.text(np.real(z)*1.03,np.imag(z)*1.03,name,fontsize=9)
    ax.axhline(0,lw=.7); ax.axvline(0,lw=.7)
    ax.set_xlabel("Re(H) = K")
    ax.set_ylabel(r"Im(H) = $\omega C$")
    ax.set_title("Breathing-frequency complex-force closure")
    ax.grid(alpha=.25)
    ax.set_aspect("equal",adjustable="datalim")
    fig.tight_layout()
    fig.savefig("ishizawa_reduced_oscillator_closure_complex.png",dpi=210)
    plt.close(fig)

    print("Saved ishizawa_reduced_oscillator_closure_summary.txt")
    print("Saved ishizawa_reduced_oscillator_closure_coefficients.png")
    print("Saved ishizawa_reduced_oscillator_closure_time.png")
    print("Saved ishizawa_reduced_oscillator_closure_complex.png")


if __name__=="__main__":
    main()
