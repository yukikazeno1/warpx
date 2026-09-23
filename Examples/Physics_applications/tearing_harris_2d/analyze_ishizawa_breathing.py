#!/usr/bin/env python3
"""Breathing-mode diagnostics for the nonlinear Ishizawa-scale m=1 island.

Input is the history produced by analyze_ishizawa_saturation.py.

Diagnostics:
  * detrended large-scale reconnection flux Psi_LS(t)
  * detrended island width w(t)
  * peak-to-peak period
  * FFT period (with Hann window; zero padding only for smooth display)
  * unbiased autocorrelation period
  * sinusoid + linear-trend fits
  * shared-period joint fit of Psi and width
  * Psi-width correlation and fitted phase lag
  * comparison with reference Alfven crossing times

Time is x = omega_ci * t throughout.

Important:
  The available 1.25--2.5 interval contains only about two oscillation cycles.
  Therefore FFT frequency resolution is intrinsically limited. Peak spacing,
  autocorrelation and the shared-period fit should be considered together.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend, find_peaks
from scipy.optimize import least_squares


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True)
    p.add_argument("--tmin", type=float, default=1.25)
    p.add_argument("--tmax", type=float, default=None)
    p.add_argument("--period-min", type=float, default=0.35)
    p.add_argument("--period-max", type=float, default=1.20)
    p.add_argument("--peak-prominence-frac", type=float, default=0.25)
    p.add_argument("--min-peak-distance", type=float, default=0.35)
    p.add_argument("--mi-me", type=float, default=800.0)
    p.add_argument("--wpe-wce", type=float, default=3.5)
    p.add_argument("--Lx-de", type=float, default=64.0)
    return p.parse_args()


def load_history(path):
    a = np.loadtxt(path)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[1] < 8:
        raise RuntimeError(
            "Expected the v2 saturation history with >=8 columns: "
            "step, omega_ci_t, Psi_LS, Psi_m1, width, Jymax, |Jy|max, Jy_p99.5 ..."
        )
    return {
        "step": a[:, 0],
        "t": a[:, 1],
        "psi": a[:, 2],
        "psi_m1": a[:, 3],
        "width": a[:, 4],
        "jymax": a[:, 5],
        "jyabs": a[:, 6],
        "jyp995": a[:, 7],
    }


def linfit_remove(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[good], y[good]
    c = np.polyfit(xx, yy, 1)
    trend = np.polyval(c, xx)
    return xx, yy, yy - trend, c


def fft_period(x, yd):
    dt = float(np.median(np.diff(x)))
    n = len(yd)
    if n < 5:
        return np.nan, np.array([]), np.array([])
    win = np.hanning(n)
    z = (yd - np.mean(yd)) * win
    # Zero padding smooths the spectral curve but does NOT improve the true
    # frequency resolution, which remains about 1/(x[-1]-x[0]).
    nfft = max(4096, 2 ** int(np.ceil(np.log2(max(8*n, 16)))))
    f = np.fft.rfftfreq(nfft, d=dt)
    p = np.abs(np.fft.rfft(z, n=nfft))**2
    valid = f > 0
    if not np.any(valid):
        return np.nan, f, p
    i = np.flatnonzero(valid)[np.argmax(p[valid])]
    return 1.0/f[i], f, p


def unbiased_autocorr(y):
    z = np.asarray(y, float) - np.nanmean(y)
    n = len(z)
    raw = np.correlate(z, z, mode="full")[n-1:]
    raw = raw / np.arange(n, 0, -1)
    if raw[0] != 0:
        raw = raw/raw[0]
    return raw


def autocorr_period(x, yd, pmin, pmax):
    dt = float(np.median(np.diff(x)))
    ac = unbiased_autocorr(yd)
    lag = np.arange(len(ac))*dt
    pk, _ = find_peaks(ac, prominence=0.08)
    valid = pk[(lag[pk] >= pmin) & (lag[pk] <= pmax)]
    if len(valid) == 0:
        return np.nan, lag, ac
    # Highest positive autocorrelation peak in the requested period range.
    i = valid[np.argmax(ac[valid])]
    return float(lag[i]), lag, ac


def peak_period(x, y, prominence_frac, min_distance):
    dt = float(np.median(np.diff(x)))
    prom = prominence_frac*np.nanstd(y)
    distance = max(1, int(round(min_distance/dt)))
    pk, _ = find_peaks(y, prominence=prom, distance=distance)
    if len(pk) < 2:
        return np.nan, pk
    return float(np.median(np.diff(x[pk]))), pk


def sine_fit(x, y, T0, Tmin, Tmax):
    x0 = x[0]
    u = x - x0

    def residual(p):
        c, s, a, b, T = p
        pred = c + s*u + a*np.sin(2*np.pi*u/T) + b*np.cos(2*np.pi*u/T)
        return pred-y

    amp0 = 0.5*(np.nanmax(y)-np.nanmin(y))
    p0 = [float(np.nanmean(y)), 0.0, amp0, 0.0, T0]
    lo = [-np.inf, -np.inf, -np.inf, -np.inf, Tmin]
    hi = [ np.inf,  np.inf,  np.inf,  np.inf, Tmax]
    r = least_squares(residual, p0, bounds=(lo, hi), max_nfev=50000)
    c,s,a,b,T = r.x
    pred = y + r.fun
    ssr = np.sum(r.fun**2)
    sst = np.sum((y-np.mean(y))**2)
    r2 = 1.0-ssr/sst if sst > 0 else np.nan
    amp = float(np.hypot(a,b))
    # a sin(theta)+b cos(theta)=A sin(theta+phi)
    phi = float(np.arctan2(b,a))
    return dict(c=c,s=s,a=a,b=b,T=T,amp=amp,phi=phi,r2=r2,pred=pred)


def joint_period_scan(x, y1, y2, Tmin, Tmax, ngrid=5000):
    """Find common period by linear LS fits at each trial T.

    Each series is independently allowed a constant, linear trend, sine and
    cosine amplitude. Residuals are normalized by each series variance.
    """
    u = x-x[0]
    Ts = np.linspace(Tmin, Tmax, ngrid)
    v1 = np.var(y1)
    v2 = np.var(y2)
    v1 = v1 if v1 > 0 else 1.0
    v2 = v2 if v2 > 0 else 1.0
    score = np.empty_like(Ts)
    best = None

    for i,T in enumerate(Ts):
        om = 2*np.pi/T
        M = np.column_stack([
            np.ones_like(u), u, np.sin(om*u), np.cos(om*u)
        ])
        c1, *_ = np.linalg.lstsq(M, y1, rcond=None)
        c2, *_ = np.linalg.lstsq(M, y2, rcond=None)
        r1 = y1-M@c1
        r2 = y2-M@c2
        score[i] = np.mean(r1*r1)/v1 + np.mean(r2*r2)/v2
        if best is None or score[i] < best[0]:
            best = (score[i], T, c1, c2, M)

    _,T,c1,c2,M = best
    pred1=M@c1
    pred2=M@c2
    phi1=np.arctan2(c1[3], c1[2])
    phi2=np.arctan2(c2[3], c2[2])
    dphi=np.arctan2(np.sin(phi2-phi1), np.cos(phi2-phi1))
    lag=dphi/(2*np.pi)*T
    return dict(
        T=float(T), score=score, Ts=Ts,
        pred1=pred1, pred2=pred2,
        phi1=float(phi1), phi2=float(phi2),
        dphi=float(dphi), lag=float(lag)
    )


def window_sensitivity(t, psi, width, Tmin, Tmax):
    starts = []
    Tjoint = []
    upper = min(1.55, t[-1]-0.75)
    for t0 in np.arange(max(t[0], 1.20), upper+1e-9, 0.05):
        m = t >= t0
        if np.count_nonzero(m) < 20:
            continue
        j = joint_period_scan(t[m], psi[m], width[m], Tmin, Tmax, ngrid=1800)
        starts.append(t0)
        Tjoint.append(j["T"])
    return np.asarray(starts), np.asarray(Tjoint)


def main():
    args = parse_args()
    H = load_history(args.history)
    t = H["t"]
    tmax = float(t[-1]) if args.tmax is None else args.tmax
    m = (
        np.isfinite(t) & np.isfinite(H["psi"]) & np.isfinite(H["width"]) &
        (t >= args.tmin) & (t <= tmax)
    )
    tt = t[m]
    psi = H["psi"][m]
    psi1 = H["psi_m1"][m]
    width = H["width"][m]

    if len(tt) < 20:
        raise RuntimeError("Too few points in requested breathing interval.")

    _,_,psid,_ = linfit_remove(tt, psi)
    _,_,wd,_ = linfit_remove(tt, width)
    _,_,psi1d,_ = linfit_remove(tt, psi1)

    ppk_psi, pkpsi = peak_period(
        tt, psi, args.peak_prominence_frac, args.min_peak_distance
    )
    ppk_w, pkw = peak_period(
        tt, width, args.peak_prominence_frac, args.min_peak_distance
    )

    pfft_psi, fpsi, Ppsi = fft_period(tt, psid)
    pfft_w, fw, Pw = fft_period(tt, wd)

    pac_psi, lagpsi, acpsi = autocorr_period(
        tt, psid, args.period_min, args.period_max
    )
    pac_w, lagw, acw = autocorr_period(
        tt, wd, args.period_min, args.period_max
    )

    seeds = [q for q in [ppk_psi, ppk_w, pac_psi, pac_w] if np.isfinite(q)]
    T0 = float(np.median(seeds)) if seeds else 0.68

    fitpsi = sine_fit(tt, psi, T0, args.period_min, args.period_max)
    fitw = sine_fit(tt, width, T0, args.period_min, args.period_max)
    fitpsi1 = sine_fit(tt, psi1, T0, args.period_min, args.period_max)
    joint = joint_period_scan(
        tt, psi, width, args.period_min, args.period_max
    )

    # Detrended linear correlation and zero-lag relation.
    corr_pw = float(np.corrcoef(psid, wd)[0,1])
    corr_p1 = float(np.corrcoef(psid, psi1d)[0,1])

    # Window-sensitivity range is a useful empirical uncertainty estimate when
    # only ~2 cycles are available.
    ws, WT = window_sensitivity(
        tt, psi, width, args.period_min, args.period_max
    )
    Tmed = float(np.median(WT)) if len(WT) else np.nan
    Tmin_emp = float(np.min(WT)) if len(WT) else np.nan
    Tmax_emp = float(np.max(WT)) if len(WT) else np.nan

    # Characteristic reference scales.
    sqrt_m = np.sqrt(args.mi_me)
    vA_over_c = 1.0/(args.wpe_wce*sqrt_m)
    # In these normalizations vA/(de*omega_ci) = sqrt(mi/me).
    width_mean = float(np.mean(width))
    tauA_width = width_mean/sqrt_m
    tauA_halfwidth = 0.5*width_mean/sqrt_m
    tauA_Lx = args.Lx_de/sqrt_m
    Tgyro = 2*np.pi

    Tbest = joint["T"]
    fhat = 1.0/Tbest                 # cycles per unit omega_ci*t
    omega_over_wci = 2*np.pi/Tbest  # angular frequency / omega_ci

    # Figure 1: time series and shared-period fits.
    fig, axs = plt.subplots(2,1,figsize=(9,8),sharex=True)
    axs[0].plot(tt, psi, "o-", ms=3, label=r"$\Psi_{LS}$")
    axs[0].plot(tt, joint["pred1"], "--", lw=2,
                label=fr"shared-period fit, $T\omega_{{ci}}={Tbest:.3f}$")
    axs[0].plot(tt[pkpsi], psi[pkpsi], "s", ms=7, fillstyle="none",
                label="detected peaks")
    axs[0].set_ylabel(r"$\Psi/(B_0d_e)$")
    axs[0].legend(fontsize=8)
    axs[1].plot(tt, width, "o-", ms=3, label=r"$w_{island}$")
    axs[1].plot(tt, joint["pred2"], "--", lw=2,
                label="shared-period fit")
    axs[1].plot(tt[pkw], width[pkw], "s", ms=7, fillstyle="none")
    axs[1].set_ylabel(r"$w/d_e$")
    axs[1].set_xlabel(r"$\omega_{ci}t$")
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=.25)
    fig.suptitle("System-scale magnetic-island breathing oscillation")
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_timeseries.png",dpi=210)
    plt.close(fig)

    # Figure 2: normalized FFT power.
    fig, ax = plt.subplots(figsize=(8.7,5.5))
    ok1 = fpsi > 0
    ok2 = fw > 0
    p1 = Ppsi/np.max(Ppsi[ok1]) if np.any(ok1) else Ppsi
    p2 = Pw/np.max(Pw[ok2]) if np.any(ok2) else Pw
    ax.plot(fpsi[ok1], p1[ok1], label=r"$\Psi_{LS}$")
    ax.plot(fw[ok2], p2[ok2], label=r"$w$")
    ax.axvline(1/Tbest, ls="--",
               label=fr"joint fit: $f/\omega_{{ci}}={1/Tbest:.3f}$ cycles")
    ax.set_xlim(0, min(5.0, np.max(fpsi)))
    ax.set_xlabel(r"cycles per unit $\omega_{ci}t$")
    ax.set_ylabel("normalized FFT power")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_fft.png",dpi=210)
    plt.close(fig)

    # Figure 3: autocorrelation.
    fig, ax = plt.subplots(figsize=(8.7,5.5))
    ax.plot(lagpsi, acpsi, label=r"$\Psi_{LS}$")
    ax.plot(lagw, acw, label=r"$w$")
    ax.axvline(Tbest, ls="--",
               label=fr"joint $T\omega_{{ci}}={Tbest:.3f}$")
    ax.set_xlim(0, min(args.period_max*1.6, lagpsi[-1]))
    ax.set_xlabel(r"lag $\Delta(\omega_{ci}t)$")
    ax.set_ylabel("unbiased autocorrelation")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_autocorr.png",dpi=210)
    plt.close(fig)

    # Figure 4: detrended normalized Psi and width to expose phase relation.
    zpsi = psid/np.std(psid)
    zw = wd/np.std(wd)
    fig, ax = plt.subplots(figsize=(9,5.5))
    ax.plot(tt,zpsi,"o-",ms=3,label=r"detrended $\Psi_{LS}$")
    ax.plot(tt,zw,"s-",ms=3,label=r"detrended $w$")
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel("normalized fluctuation")
    ax.set_title(
        fr"$corr(\Psi,w)={corr_pw:.3f}$, "
        fr"fit phase lag $={joint['dphi']:.3f}$ rad"
    )
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("ishizawa_breathing_phase_relation.png",dpi=210)
    plt.close(fig)

    with open("ishizawa_breathing_summary.txt","w") as f:
        f.write("Ishizawa-scale magnetic-island breathing diagnostics\n")
        f.write("====================================================\n\n")
        f.write(f"analysis interval omega_ci*t = {tt[0]:.8f} .. {tt[-1]:.8f}\n")
        f.write(f"samples = {len(tt)}, dt_hat = {np.median(np.diff(tt)):.8f}\n")
        f.write(f"observation duration = {tt[-1]-tt[0]:.8f}\n\n")

        f.write("Period estimates in Delta(omega_ci*t)\n")
        f.write("--------------------------------------\n")
        f.write(f"peak spacing Psi = {ppk_psi:.8f}\n")
        f.write(f"peak spacing width = {ppk_w:.8f}\n")
        f.write(f"autocorrelation Psi = {pac_psi:.8f}\n")
        f.write(f"autocorrelation width = {pac_w:.8f}\n")
        f.write(f"FFT Psi = {pfft_psi:.8f}\n")
        f.write(f"FFT width = {pfft_w:.8f}\n")
        f.write(f"sinusoid fit Psi = {fitpsi['T']:.8f}, R2={fitpsi['r2']:.6f}\n")
        f.write(f"sinusoid fit width = {fitw['T']:.8f}, R2={fitw['r2']:.6f}\n")
        f.write(f"sinusoid fit Psi_m1 = {fitpsi1['T']:.8f}, R2={fitpsi1['r2']:.6f}\n")
        f.write(f"JOINT shared-period fit = {Tbest:.8f}\n")
        f.write(
            f"window-sensitivity joint T median/min/max = "
            f"{Tmed:.8f} {Tmin_emp:.8f} {Tmax_emp:.8f}\n\n"
        )

        f.write("Phase/coherence between Psi and width\n")
        f.write("-------------------------------------\n")
        f.write(f"detrended Pearson corr(Psi,width) = {corr_pw:.8f}\n")
        f.write(f"detrended Pearson corr(Psi,Psi_m1) = {corr_p1:.8f}\n")
        f.write(f"joint fitted phase difference width-Psi [rad] = {joint['dphi']:.8f}\n")
        f.write(f"equivalent fitted time lag Delta(omega_ci*t) = {joint['lag']:.8f}\n\n")

        f.write("Breathing frequency\n")
        f.write("-------------------\n")
        f.write(f"cycles per unit omega_ci*t = {fhat:.8f}\n")
        f.write(f"omega_breath/omega_ci = {omega_over_wci:.8f}\n\n")

        f.write("Reference characteristic times\n")
        f.write("------------------------------\n")
        f.write(f"mi/me = {args.mi_me:.8f}\n")
        f.write(f"omega_pe/omega_ce = {args.wpe_wce:.8f}\n")
        f.write(f"reference vA/c = {vA_over_c:.8e}\n")
        f.write(f"mean island full width/de = {width_mean:.8f}\n")
        f.write(f"Alfven crossing across full mean width: omega_ci*tau_A = {tauA_width:.8f}\n")
        f.write(f"Alfven crossing across half mean width: omega_ci*tau_A = {tauA_halfwidth:.8f}\n")
        f.write(f"Alfven crossing across Lx={args.Lx_de:g} de: omega_ci*tau_A = {tauA_Lx:.8f}\n")
        f.write(f"full ion gyroperiod: omega_ci*T_ci = {Tgyro:.8f}\n\n")

        f.write("Caution\n")
        f.write("-------\n")
        f.write("Only about two breathing cycles are currently available, so the raw FFT\n")
        f.write("frequency resolution is poor. Peak spacing, autocorrelation and the shared\n")
        f.write("sinusoid fit are more informative at this stage.  Also, this pure Harris\n")
        f.write("case has no uniform upstream background density, so the vA value above is\n")
        f.write("a reference scale based on n0, not a uniquely defined upstream Alfven speed.\n")
        f.write("A quantitative ion-bounce comparison requires ion orbit/phase-space data;\n")
        f.write("it should not be inferred from the field history alone.\n")

    print("Saved ishizawa_breathing_timeseries.png")
    print("Saved ishizawa_breathing_fft.png")
    print("Saved ishizawa_breathing_autocorr.png")
    print("Saved ishizawa_breathing_phase_relation.png")
    print("Saved ishizawa_breathing_summary.txt")


if __name__ == "__main__":
    main()
