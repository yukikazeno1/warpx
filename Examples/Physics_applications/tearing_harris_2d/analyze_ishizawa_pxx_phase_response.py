#!/usr/bin/env python3
"""Phase-resolved closure test for the dominant non-affine Pxx channel.

Reads ishizawa_pxx_thermodynamics.txt produced by
analyze_ishizawa_pxx_thermodynamics.py and fits every projected force channel
to

    a_j(tau) = b_j + K_j q(tau) + C_j dq/dtau,

where tau = omega_ci t.

K_j is the in-phase force-displacement stiffness.  Negative K is restoring.
C_j is the quadrature/velocity coefficient.  Since
< a_j dq/dtau > = C_j <(dq/dtau)^2> over a cycle, C<0 is damping and C>0
is antidamping.

The same file also contains modal-weighted density and cxx.  We fit the
effective one-dimensional closure

    cxx ~ n^(Gamma_x - 1)

for ions and electrons.  This is a global/modal-weighted effective exponent,
not a local thermodynamic equation of state.

Dense phase sampling (>=8 points per breathing cycle) is strongly recommended.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


CHANNELS = [
    ("ion density/compression", 5),
    ("ion thermal variance", 6),
    ("ion nonlinear cross", 7),
    ("ion Pxx total", 8),
    ("electron density/compression", 9),
    ("electron thermal variance", 10),
    ("electron nonlinear cross", 11),
    ("electron Pxx total", 12),
    ("all-species Pxx total", 13),
]


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--history",default="ishizawa_pxx_thermodynamics.txt")
    p.add_argument("--period",type=float,default=0.63)
    p.add_argument("--min-points",type=int,default=8)
    return p.parse_args()


def fit_kc(q,qd,y):
    X=np.column_stack([np.ones_like(q),q,qd])
    beta,*_=np.linalg.lstsq(X,y,rcond=None)
    yp=X@beta
    ssr=np.sum((y-yp)**2)
    sst=np.sum((y-y.mean())**2)
    r2=1.0-ssr/sst if sst>0 else np.nan
    return beta[0],beta[1],beta[2],r2,yp


def fit_log_closure(n,c):
    good=(n>0)&(c>0)&np.isfinite(n)&np.isfinite(c)
    x=np.log(n[good]); y=np.log(c[good])
    if len(x)<3:
        return np.nan,np.nan,np.nan
    A=np.column_stack([np.ones_like(x),x])
    b,*_=np.linalg.lstsq(A,y,rcond=None)
    yp=A@b
    ssr=np.sum((y-yp)**2); sst=np.sum((y-y.mean())**2)
    r2=1.0-ssr/sst if sst>0 else np.nan
    return 1.0+b[1],r2,b[1]


def main():
    args=parse_args()
    a=np.loadtxt(args.history)
    if a.ndim==1: a=a[None,:]
    a=a[np.argsort(a[:,1])]
    if len(a)<args.min_points:
        print(f"WARNING: only {len(a)} phase points; >= {args.min_points} strongly recommended.")

    tau=a[:,1]; q=a[:,2]; qd=a[:,3]
    omega=2*np.pi/args.period

    fits={}
    for name,col in CHANNELS:
        fits[name]=fit_kc(q,qd,a[:,col])

    gamma_i,r2gi,_=fit_log_closure(a[:,14],a[:,16])
    gamma_e,r2ge,_=fit_log_closure(a[:,15],a[:,17])

    out=Path("ishizawa_pxx_phase_response_summary.txt")
    with out.open("w") as f:
        f.write("Phase-resolved response of the non-affine Pxx channel\n")
        f.write("=====================================================\n\n")
        f.write(f"points = {len(a)}\n")
        f.write(f"period omega_ci*T = {args.period:.8f}\n")
        f.write(f"angular frequency in tau units = {omega:.8f}\n\n")
        f.write("Fit model: a = b + K q + C dq/dtau\n")
        f.write("Negative K: restoring.  Negative C: damping; positive C: antidamping.\n\n")
        f.write("channel                         K              C          omega*C/|K|      R2\n")
        for name,(b,K,C,r2,yp) in fits.items():
            ratio=omega*C/max(abs(K),1e-300)
            f.write(f"{name:31s} {K:+13.6e} {C:+13.6e} {ratio:+13.6e} {r2:9.5f}\n")

        f.write("\nEffective modal-weighted x-polytropic closure\n")
        f.write("---------------------------------------------\n")
        f.write("Definition: cxx ~ n^(Gamma_x-1), so Pxx ~ n^Gamma_x.\n")
        f.write(f"ion Gamma_x = {gamma_i:.8f}, R2(log closure) = {r2gi:.8f}\n")
        f.write(f"electron Gamma_x = {gamma_e:.8f}, R2(log closure) = {r2ge:.8f}\n")
        f.write("\nCaution: Gamma_x is a global/modal-weighted effective exponent, not a local EOS.\n")

    # Time-domain channel plot.
    fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True)
    axs[0].plot(tau,q,"o-",label="q")
    qdn=qd/max(np.max(np.abs(qd)),1e-300)*max(np.max(np.abs(q)),1e-300)
    axs[0].plot(tau,qdn,"s--",label="scaled dq/dtau")
    axs[0].axhline(0,lw=.7); axs[0].grid(alpha=.25); axs[0].legend()
    axs[0].set_ylabel("displacement / phase")

    for name,col in CHANNELS[:4]:
        axs[1].plot(tau,a[:,col],"o-",label=name)
    axs[1].axhline(0,lw=.7); axs[1].grid(alpha=.25); axs[1].legend(fontsize=8)
    axs[1].set_ylabel("ion modal accel.")

    for name,col in CHANNELS[4:8]:
        axs[2].plot(tau,a[:,col],"o-",label=name)
    axs[2].axhline(0,lw=.7); axs[2].grid(alpha=.25); axs[2].legend(fontsize=8)
    axs[2].set_ylabel("electron modal accel.")
    axs[2].set_xlabel(r"$\omega_{ci}t$")
    fig.suptitle("Phase-resolved thermodynamic response of the non-affine Pxx channel")
    fig.tight_layout()
    fig.savefig("ishizawa_pxx_phase_response_time.png",dpi=210)
    plt.close(fig)

    # Hysteresis / force-displacement loops with fitted elastic+quadrature response.
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,subset,title in [
        (axs[0],CHANNELS[:4],"ions"),
        (axs[1],CHANNELS[4:8],"electrons"),
    ]:
        for name,col in subset:
            b,K,C,r2,yp=fits[name]
            ax.plot(q,a[:,col],"o-",label=f"{name} data")
            order=np.argsort(q)
            ax.plot(q[order],yp[order],"--",alpha=.75,label=f"{name} K+C fit")
        ax.axhline(0,lw=.7); ax.axvline(0,lw=.7); ax.grid(alpha=.25)
        ax.set_xlabel("q"); ax.set_ylabel("modal acceleration"); ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("Pxx force-displacement loops: elastic vs quadrature response")
    fig.tight_layout()
    fig.savefig("ishizawa_pxx_phase_response_loops.png",dpi=210)
    plt.close(fig)

    # Effective closure plot.
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    for ax,n,c,g,r2,title in [
        (axs[0],a[:,14],a[:,16],gamma_i,r2gi,"ions"),
        (axs[1],a[:,15],a[:,17],gamma_e,r2ge,"electrons"),
    ]:
        ax.scatter(n,c)
        good=(n>0)&(c>0)
        if np.count_nonzero(good)>=3 and np.isfinite(g):
            nn=np.linspace(n[good].min(),n[good].max(),200)
            # Fit intercept again for plotting.
            x=np.log(n[good]); y=np.log(c[good])
            A=np.column_stack([np.ones_like(x),x])
            b,*_=np.linalg.lstsq(A,y,rcond=None)
            cc=np.exp(b[0])*nn**b[1]
            ax.plot(nn,cc,label=fr"$\Gamma_x={g:.2f}$, $R^2={r2:.2f}$")
        ax.set_xlabel("modal-weighted n / n0")
        ax.set_ylabel("modal-weighted cxx / reference")
        ax.grid(alpha=.25); ax.legend(); ax.set_title(title)
    fig.suptitle(r"Effective modal-weighted closure: $c_{xx}\propto n^{\Gamma_x-1}$")
    fig.tight_layout()
    fig.savefig("ishizawa_pxx_effective_polytrope.png",dpi=210)
    plt.close(fig)

    print(f"Saved {out}")
    print("Saved ishizawa_pxx_phase_response_time.png")
    print("Saved ishizawa_pxx_phase_response_loops.png")
    print("Saved ishizawa_pxx_effective_polytrope.png")


if __name__=="__main__":
    main()
