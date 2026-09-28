#!/usr/bin/env python3
"""Time-integrated weak-form closure for Ishizawa particle quadrature.

This diagnostic avoids differentiating P_xi and avoids assuming that each
channel is a single sinusoid.  Starting from the history produced by
analyze_ishizawa_modal_particle_quadrature.py, it tests

    P_xi(t)-P_xi(t0) = integral_{t0}^t RHS(tau) d tau

for both the volume-only RHS and, when available, the particle-strip
surface-corrected RHS.

The integrated identity is the preferred cheap follow-up when the phasor R^2
of P_xi / EM / ion kinetic is low.

Outputs
-------
ishizawa_particle_quadrature_integral_summary.txt
ishizawa_particle_quadrature_integral_history.txt
ishizawa_particle_quadrature_integral.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("history",
                   help="ishizawa_particle_quadrature_closure_history.txt")
    p.add_argument("--out-prefix",
                   default="ishizawa_particle_quadrature_integral")
    return p.parse_args()


def cumtrapz_nonuniform(t, y):
    t=np.asarray(t,float); y=np.asarray(y,float)
    out=np.zeros_like(y)
    if len(y)>1:
        out[1:]=np.cumsum(0.5*(y[1:]+y[:-1])*(t[1:]-t[:-1]))
    return out


def metrics(obs, pred):
    obs=np.asarray(obs,float); pred=np.asarray(pred,float)
    res=obs-pred
    rms_obs=np.sqrt(np.mean(obs*obs))
    rms_res=np.sqrt(np.mean(res*res))
    rel=rms_res/max(rms_obs,1e-300)
    corr=np.corrcoef(obs,pred)[0,1] if len(obs)>1 else np.nan
    amp=np.sqrt(np.mean(pred*pred))/max(rms_obs,1e-300)
    end=abs(res[-1])/max(np.max(np.abs(obs)),1e-300)
    return dict(rel=float(rel),corr=float(corr),amp=float(amp),
                endpoint=float(end),res=res)


def main():
    args=parse_args()
    a=np.loadtxt(args.history)
    if a.ndim==1:
        a=a[None,:]
    if a.shape[1] < 8:
        raise RuntimeError("History needs at least 8 columns.")

    # Sort defensively in physical time.
    a=a[np.argsort(a[:,1])]
    t=a[:,1]
    P=a[:,3]
    rhs_vol=a[:,7]
    rhs_corr=a[:,20] if a.shape[1] >= 21 else None

    good=np.isfinite(t)&np.isfinite(P)&np.isfinite(rhs_vol)
    if rhs_corr is not None:
        good &= np.isfinite(rhs_corr)
    t=t[good]; P=P[good]; rhs_vol=rhs_vol[good]
    if rhs_corr is not None:
        rhs_corr=rhs_corr[good]

    if len(t)<4:
        raise RuntimeError("Too few finite samples for integrated closure.")

    dP=P-P[0]
    Ivol=cumtrapz_nonuniform(t,rhs_vol)
    Mvol=metrics(dP,Ivol)

    if rhs_corr is not None:
        Icorr=cumtrapz_nonuniform(t,rhs_corr)
        Mcorr=metrics(dP,Icorr)
    else:
        Icorr=np.full_like(dP,np.nan)
        Mcorr=None

    out=np.column_stack([t,P,dP,rhs_vol,Ivol,dP-Ivol,Icorr,dP-Icorr])
    np.savetxt(
        args.out_prefix+"_history.txt",out,
        header=(
            "omega_ci_t Pxi DeltaPxi RHS_volume Integral_RHS_volume "
            "residual_volume Integral_RHS_corrected residual_corrected"
        )
    )

    with open(args.out_prefix+"_summary.txt","w") as f:
        f.write("Particle-quadrature time-integrated weak-form closure\n")
        f.write("====================================================\n\n")
        f.write("Identity: Delta P_xi = integral RHS d(omega_ci*t)\n")
        f.write(f"points = {len(t)}\n")
        f.write(f"window = {t[0]:.8f} .. {t[-1]:.8f}\n")
        f.write(f"dt median = {np.median(np.diff(t)):.8f}\n\n")
        f.write("Volume-only RHS\n")
        f.write("----------------\n")
        f.write(f"integrated relative RMS error = {Mvol['rel']:.8e}\n")
        f.write(f"corr(DeltaP, integral RHS) = {Mvol['corr']:.8f}\n")
        f.write(f"RMS(integral RHS)/RMS(DeltaP) = {Mvol['amp']:.8f}\n")
        f.write(f"endpoint residual / max|DeltaP| = {Mvol['endpoint']:.8e}\n\n")
        if Mcorr is not None:
            f.write("Surface-corrected RHS\n")
            f.write("---------------------\n")
            f.write(f"integrated relative RMS error = {Mcorr['rel']:.8e}\n")
            f.write(f"corr(DeltaP, integral RHS) = {Mcorr['corr']:.8f}\n")
            f.write(f"RMS(integral RHS)/RMS(DeltaP) = {Mcorr['amp']:.8f}\n")
            f.write(f"endpoint residual / max|DeltaP| = {Mcorr['endpoint']:.8e}\n")

    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
    axs[0].plot(t,dP,"o-",label=r"$\Delta P_\xi$")
    axs[0].plot(t,Ivol,"-",lw=2,label="integral volume RHS")
    if Mcorr is not None:
        axs[0].plot(t,Icorr,"--",lw=2,label="integral corrected RHS")
    axs[0].set_ylabel("cumulative modal momentum")
    axs[0].grid(alpha=.25); axs[0].legend()

    axs[1].plot(t,dP-Ivol,"-",label="volume residual")
    if Mcorr is not None:
        axs[1].plot(t,dP-Icorr,"--",label="corrected residual")
    axs[1].axhline(0,lw=.8)
    axs[1].set_xlabel(r"$\omega_{ci}t$")
    axs[1].set_ylabel("integrated residual")
    axs[1].grid(alpha=.25); axs[1].legend()
    fig.tight_layout()
    fig.savefig(args.out_prefix+".png",dpi=210)
    plt.close(fig)

    print("Saved",args.out_prefix+"_summary.txt")
    print("Saved",args.out_prefix+"_history.txt")
    print("Saved",args.out_prefix+".png")


if __name__=="__main__":
    main()
