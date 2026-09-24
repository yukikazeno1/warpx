#!/usr/bin/env python3
"""Long-baseline breathing-mode diagnostics for the Ishizawa-scale Harris run.

Designed for the >= 160000-step (omega_ci*t ~= 4) data set.  It consumes the
history written by analyze_ishizawa_saturation.py and quantifies:

  * cycle-to-cycle period stability;
  * FFT/ACF/shared-period estimates on a multi-cycle interval;
  * breathing-amplitude evolution and exponential damping/growth rate;
  * phase relation of Psi_LS, island width, and robust Jy percentile;
  * O-point drift, to distinguish breathing from simple island sloshing.

Time is always x = omega_ci*t.  Damping rate is therefore reported in units
gamma_d/omega_ci from A ~ exp(-gamma_d * x).

Outputs
-------
ishizawa_breathing_long_summary.txt
ishizawa_breathing_long_cycles.txt
ishizawa_breathing_long_timeseries.png
ishizawa_breathing_long_spectrum.png
ishizawa_breathing_long_envelope.png
ishizawa_breathing_long_phase.png
ishizawa_breathing_long_OX_motion.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True)
    p.add_argument("--tmin", type=float, default=1.25)
    p.add_argument("--tmax", type=float, default=None)
    p.add_argument("--period-min", type=float, default=0.45)
    p.add_argument("--period-max", type=float, default=0.95)
    p.add_argument("--min-peak-distance", type=float, default=0.45)
    p.add_argument("--peak-prominence-frac", type=float, default=0.20)
    p.add_argument("--Lx-de", type=float, default=64.0)
    return p.parse_args()


def load_history(path):
    a = np.loadtxt(path)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[1] < 14:
        raise RuntimeError(
            "Expected v2 saturation history with 14 columns. "
            "Re-run analyze_ishizawa_saturation.py first."
        )
    return dict(
        step=a[:,0], t=a[:,1], psi=a[:,2], psi_m1=a[:,3],
        width=a[:,4], jymax=a[:,5], jyabs=a[:,6], jyp=a[:,7],
        xO=a[:,8], xX=a[:,9], zlo=a[:,10], zhi=a[:,11],
        orient=a[:,12], bz_m1=a[:,13],
    )


def detrend_linear(t, y):
    m = np.isfinite(t) & np.isfinite(y)
    c = np.polyfit(t[m], y[m], 1)
    tr = np.polyval(c, t)
    return y-tr, c


def acf_unbiased(y):
    z = y-np.mean(y)
    n = len(z)
    a = np.correlate(z,z,mode="full")[n-1:]
    a /= np.arange(n,0,-1)
    if a[0] != 0:
        a /= a[0]
    return a


def acf_period(t, yd, pmin, pmax):
    dt = np.median(np.diff(t))
    ac = acf_unbiased(yd)
    lag = np.arange(len(ac))*dt
    pk,_ = find_peaks(ac, prominence=0.05)
    v = pk[(lag[pk]>=pmin)&(lag[pk]<=pmax)]
    if len(v)==0:
        return np.nan,lag,ac
    i = v[np.argmax(ac[v])]
    return float(lag[i]),lag,ac


def fft_spectrum(t, yd):
    dt=float(np.median(np.diff(t)))
    n=len(yd)
    win=np.hanning(n)
    z=(yd-np.mean(yd))*win
    nfft=max(8192,2**int(np.ceil(np.log2(max(16*n,16)))))
    f=np.fft.rfftfreq(nfft,d=dt)
    p=np.abs(np.fft.rfft(z,n=nfft))**2
    return f,p


def fft_period(t, yd, pmin, pmax):
    f,p=fft_spectrum(t,yd)
    valid=(f>0)&(1/f>=pmin)&(1/f<=pmax)
    if not np.any(valid):
        return np.nan,f,p
    ids=np.flatnonzero(valid)
    i=ids[np.argmax(p[valid])]
    return float(1/f[i]),f,p


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
    return dict(c=c,pred=pred,r2=r2,amp=amp,phi=phi)


def joint_period(t, series, pmin, pmax, ngrid=6000):
    Ts=np.linspace(pmin,pmax,ngrid)
    u=t-t[0]
    variances=[np.var(y) if np.var(y)>0 else 1 for y in series]
    score=np.empty_like(Ts)
    for i,T in enumerate(Ts):
        om=2*np.pi/T
        M=np.column_stack([np.ones_like(u),u,np.sin(om*u),np.cos(om*u)])
        sc=0.0
        for y,v in zip(series,variances):
            c,*_=np.linalg.lstsq(M,y,rcond=None)
            r=y-M@c
            sc += np.mean(r*r)/v
        score[i]=sc
    ib=np.argmin(score)
    return float(Ts[ib]),Ts,score


def periodic_unwrap(x,L):
    """Unwrap a coordinate defined periodically on a box of length L."""
    phase=2*np.pi*np.asarray(x)/L
    return np.unwrap(phase)*L/(2*np.pi)


def find_cycle_extrema(t,y,min_dist,prom_frac):
    dt=float(np.median(np.diff(t)))
    distance=max(1,int(round(min_dist/dt)))
    prom=prom_frac*np.std(y)
    p,_=find_peaks(y,distance=distance,prominence=prom)
    q,_=find_peaks(-y,distance=distance,prominence=prom)
    return p,q


def cycle_table(t,y,p,q):
    """One row per adjacent peak pair: peak times, period, enclosed trough, amplitude."""
    rows=[]
    for a,b in zip(p[:-1],p[1:]):
        inside=q[(q>a)&(q<b)]
        if len(inside)==0:
            continue
        iq=inside[np.argmin(y[inside])]
        pmean=0.5*(y[a]+y[b])
        amp=0.5*(pmean-y[iq])
        tc=0.5*(t[a]+t[b])
        rows.append([tc,t[a],t[b],t[b]-t[a],t[iq],y[a],y[b],y[iq],amp])
    return np.asarray(rows,float) if rows else np.empty((0,9))


def exp_envelope_fit(t,a):
    m=np.isfinite(t)&np.isfinite(a)&(a>0)
    if np.count_nonzero(m)<2:
        return np.nan,np.nan,np.nan
    x=t[m]; z=np.log(a[m])
    c=np.polyfit(x,z,1)
    pred=np.polyval(c,x)
    ssr=np.sum((z-pred)**2); sst=np.sum((z-np.mean(z))**2)
    r2=1-ssr/sst if sst>0 else np.nan
    # ln A = const - gamma_d t
    return float(-c[0]),float(np.exp(c[1])),float(r2)


def wrap_phase(d):
    return float(np.arctan2(np.sin(d),np.cos(d)))


def main():
    args=parse_args()
    H=load_history(args.history)
    tmax=H["t"][-1] if args.tmax is None else args.tmax
    m=(H["t"]>=args.tmin)&(H["t"]<=tmax)
    keys=["t","psi","psi_m1","width","jymax","jyp","xO","xX"]
    D={k:np.asarray(H[k])[m] for k in keys}
    t=D["t"]
    good=np.isfinite(t)&np.isfinite(D["psi"])&np.isfinite(D["width"])&np.isfinite(D["jyp"])
    for k in D:
        D[k]=D[k][good]
    t=D["t"]
    if len(t)<40:
        raise RuntimeError("Too few late-time samples.")

    psid,_=detrend_linear(t,D["psi"])
    wd,_=detrend_linear(t,D["width"])
    jd,_=detrend_linear(t,D["jyp"])

    ppsi,qpsi=find_cycle_extrema(
        t,D["psi"],args.min_peak_distance,args.peak_prominence_frac
    )
    cycles=cycle_table(t,D["psi"],ppsi,qpsi)
    np.savetxt(
        "ishizawa_breathing_long_cycles.txt",cycles,
        header="cycle_center peak1_t peak2_t period trough_t peak1 peak2 trough amplitude"
    )

    peak_period=np.median(np.diff(t[ppsi])) if len(ppsi)>=2 else np.nan
    acT,lag,ac=acf_period(t,psid,args.period_min,args.period_max)
    fftT,f,P=fft_period(t,psid,args.period_min,args.period_max)
    Tjoint,Ts,score=joint_period(
        t,[D["psi"],D["width"]],args.period_min,args.period_max
    )

    Fpsi=harmonic_fit(t,D["psi"],Tjoint)
    Fw=harmonic_fit(t,D["width"],Tjoint)
    Fj=harmonic_fit(t,D["jyp"],Tjoint)
    Fp1=harmonic_fit(t,D["psi_m1"],Tjoint)

    dphi_w=wrap_phase(Fw["phi"]-Fpsi["phi"])
    dphi_j=wrap_phase(Fj["phi"]-Fpsi["phi"])
    lag_w=dphi_w/(2*np.pi)*Tjoint
    lag_j=dphi_j/(2*np.pi)*Tjoint

    corr_pw=float(np.corrcoef(psid,wd)[0,1])
    corr_pj=float(np.corrcoef(psid,jd)[0,1])

    gamma_cycle=A0_cycle=r2_cycle=np.nan
    if len(cycles)>=2:
        gamma_cycle,A0_cycle,r2_cycle=exp_envelope_fit(cycles[:,0],cycles[:,8])

    # Hilbert amplitude gives an independent envelope estimate. Exclude one
    # half-period at both ends to suppress edge artifacts.
    z=hilbert(psid)
    env=np.abs(z)
    edge=0.5*Tjoint
    mh=(t>=t[0]+edge)&(t<=t[-1]-edge)
    gamma_h,A0_h,r2_h=exp_envelope_fit(t[mh],env[mh])

    # O/X motion: unwrap periodic coordinates, then remove mean/linear drift.
    xO=periodic_unwrap(D["xO"],args.Lx_de)
    xX=periodic_unwrap(D["xX"],args.Lx_de)
    xOd,cO=detrend_linear(t,xO)
    xXd,cX=detrend_linear(t,xX)
    rms_xO=float(np.sqrt(np.mean(xOd*xOd)))
    rms_xX=float(np.sqrt(np.mean(xXd*xXd)))

    # Figures.
    fig,axs=plt.subplots(3,1,figsize=(9.2,10),sharex=True)
    axs[0].plot(t,D["psi"],"o-",ms=3,label=r"$\Psi_{LS}$")
    axs[0].plot(t,Fpsi["pred"],"--",lw=2,label=fr"$T\omega_{{ci}}={Tjoint:.4f}$")
    axs[0].plot(t[ppsi],D["psi"][ppsi],"s",fillstyle="none",ms=7,label="peaks")
    axs[0].set_ylabel(r"$\Psi/(B_0d_e)$"); axs[0].legend(fontsize=8)
    axs[1].plot(t,D["width"],"o-",ms=3,label="width")
    axs[1].plot(t,Fw["pred"],"--",lw=2)
    axs[1].set_ylabel(r"$w/d_e$")
    axs[2].plot(t,D["jyp"],"o-",ms=3,label=r"$J_y$ p99.5")
    axs[2].plot(t,Fj["pred"],"--",lw=2,label=fr"$\Delta\phi_{{J,\Psi}}={dphi_j:.2f}$ rad")
    axs[2].set_ylabel(r"$J/J_0$"); axs[2].set_xlabel(r"$\omega_{ci}t$")
    axs[2].legend(fontsize=8)
    for ax in axs: ax.grid(alpha=.25)
    fig.suptitle("Long-baseline island breathing and current response")
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_long_timeseries.png",dpi=210)
    plt.close(fig)

    fig,axs=plt.subplots(2,1,figsize=(9,7.5),sharex=True)
    if len(cycles):
        axs[0].plot(cycles[:,0],cycles[:,3],"o-")
        axs[0].axhline(Tjoint,ls="--",label="joint period")
        axs[0].set_ylabel(r"cycle $T\omega_{ci}$")
        axs[0].legend()
        axs[1].plot(cycles[:,0],cycles[:,8],"o-",label="peak-trough amplitude")
        if np.isfinite(gamma_cycle):
            pred=A0_cycle*np.exp(-gamma_cycle*cycles[:,0])
            axs[1].plot(cycles[:,0],pred,"--",
                label=fr"$\gamma_d/\omega_{{ci}}={gamma_cycle:.3g}$")
        axs[1].set_ylabel(r"$A_\Psi/(B_0d_e)$")
        axs[1].legend()
    axs[1].set_xlabel(r"$\omega_{ci}t$")
    for ax in axs: ax.grid(alpha=.25)
    fig.suptitle("Cycle stability and breathing-amplitude envelope")
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_long_envelope.png",dpi=210)
    plt.close(fig)

    fW,PW=fft_spectrum(t,wd)
    fJ,PJ=fft_spectrum(t,jd)
    fig,ax=plt.subplots(figsize=(8.8,5.5))
    def norm(P,f):
        v=f>0
        return P/np.max(P[v])
    ax.plot(f,norm(P,f),label=r"$\Psi$")
    ax.plot(fW,norm(PW,fW),label=r"$w$")
    ax.plot(fJ,norm(PJ,fJ),label=r"$J_y$ p99.5")
    ax.axvline(1/Tjoint,ls="--",label=fr"$1/T={1/Tjoint:.3f}$")
    ax.set_xlim(0,4)
    ax.set_xlabel(r"cycles per unit $\omega_{ci}t$")
    ax.set_ylabel("normalized FFT power")
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_long_spectrum.png",dpi=210)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5.5))
    zpsi=psid/np.std(psid); zw=wd/np.std(wd); zj=jd/np.std(jd)
    ax.plot(t,zpsi,label=r"$\Psi$")
    ax.plot(t,zw,label=r"$w$")
    ax.plot(t,zj,label=r"$J_y$ p99.5")
    ax.set_xlabel(r"$\omega_{ci}t$"); ax.set_ylabel("normalized detrended fluctuation")
    ax.set_title(fr"$corr(\Psi,w)={corr_pw:.3f}$, $corr(\Psi,J)={corr_pj:.3f}$")
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_long_phase.png",dpi=210)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5.5))
    ax.plot(t,xO,label=r"$x_O$ unwrapped")
    ax.plot(t,xX,label=r"$x_X$ unwrapped")
    ax.plot(t,np.polyval(cO,t),"--",alpha=.7)
    ax.plot(t,np.polyval(cX,t),"--",alpha=.7)
    ax.set_xlabel(r"$\omega_{ci}t$"); ax.set_ylabel(r"$x/d_e$")
    ax.set_title(fr"O/X drift: rms detrended O={rms_xO:.3f} $d_e$, X={rms_xX:.3f} $d_e$")
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_long_OX_motion.png",dpi=210)
    plt.close(fig)

    with open("ishizawa_breathing_long_summary.txt","w") as s:
        s.write("Ishizawa-scale LONG-baseline breathing diagnostics\n")
        s.write("=================================================\n\n")
        s.write(f"interval omega_ci*t = {t[0]:.8f} .. {t[-1]:.8f}\n")
        s.write(f"samples = {len(t)}, dt_hat = {np.median(np.diff(t)):.8f}\n")
        s.write(f"duration = {t[-1]-t[0]:.8f}\n")
        s.write(f"approx cycles covered = {(t[-1]-t[0])/Tjoint:.6f}\n\n")
        s.write("Period stability\n----------------\n")
        s.write(f"peak-spacing median = {peak_period:.8f}\n")
        s.write(f"ACF period = {acT:.8f}\n")
        s.write(f"FFT period = {fftT:.8f}\n")
        s.write(f"joint Psi-width period = {Tjoint:.8f}\n")
        if len(cycles):
            s.write(f"cycle periods mean/std = {np.mean(cycles[:,3]):.8f} {np.std(cycles[:,3]):.8f}\n")
            s.write(f"cycle period CV = {np.std(cycles[:,3])/np.mean(cycles[:,3]):.8e}\n")
        s.write(f"omega_breath/omega_ci = {2*np.pi/Tjoint:.8f}\n\n")
        s.write("Harmonic-fit quality and phase\n------------------------------\n")
        s.write(f"Psi R2={Fpsi['r2']:.6f}, amp={Fpsi['amp']:.8e}\n")
        s.write(f"width R2={Fw['r2']:.6f}, amp={Fw['amp']:.8e}\n")
        s.write(f"Psi_m1 R2={Fp1['r2']:.6f}, amp={Fp1['amp']:.8e}\n")
        s.write(f"Jp99.5 R2={Fj['r2']:.6f}, amp={Fj['amp']:.8e}\n")
        s.write(f"phase width-Psi [rad] = {dphi_w:.8f}, lag={lag_w:.8f}\n")
        s.write(f"phase J-Psi [rad] = {dphi_j:.8f}, lag={lag_j:.8f}\n")
        s.write(f"detrended corr(Psi,width) = {corr_pw:.8f}\n")
        s.write(f"detrended corr(Psi,Jp99.5) = {corr_pj:.8f}\n\n")
        s.write("Amplitude-envelope evolution\n----------------------------\n")
        s.write(f"cycle-envelope gamma_d/omega_ci = {gamma_cycle:.8e}, R2={r2_cycle:.6f}\n")
        s.write(f"Hilbert-envelope gamma_d/omega_ci = {gamma_h:.8e}, R2={r2_h:.6f}\n")
        s.write("Positive gamma_d means damping; negative means a growing envelope.\n\n")
        s.write("O/X positional motion\n---------------------\n")
        s.write(f"O detrended rms displacement/de = {rms_xO:.8e}\n")
        s.write(f"X detrended rms displacement/de = {rms_xX:.8e}\n")
        s.write(f"O linear drift dx/d(omega_ci*t) [de] = {cO[0]:.8e}\n")
        s.write(f"X linear drift dx/d(omega_ci*t) [de] = {cX[0]:.8e}\n\n")
        s.write("Interpretation guide\n--------------------\n")
        s.write("Stable cycle periods with small period CV support a persistent nonlinear mode.\n")
        s.write("gamma_d ~ 0 supports a persistent breathing oscillation; gamma_d > 0 supports damping.\n")
        s.write("A small O-point positional rms compared with island width supports breathing rather than sloshing.\n")
        s.write("A fixed J-Psi phase relation supports a coupled magnetic-current eigenoscillation.\n")

    print("Saved ishizawa_breathing_long_summary.txt")
    print("Saved ishizawa_breathing_long_cycles.txt")
    print("Saved ishizawa_breathing_long_timeseries.png")
    print("Saved ishizawa_breathing_long_spectrum.png")
    print("Saved ishizawa_breathing_long_envelope.png")
    print("Saved ishizawa_breathing_long_phase.png")
    print("Saved ishizawa_breathing_long_OX_motion.png")


if __name__=="__main__":
    main()
