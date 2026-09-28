#!/usr/bin/env python3
"""Robustness tests for the particle-quadrature breathing-frequency closure.

Reads one or more ishizawa_particle_quadrature_closure_history.txt files and
quantifies whether the reported complex closure error is stable against:
  1) breathing-period variation,
  2) leave-one-out temporal sampling,
  3) first/last-point removal.

No particle data are re-read, so this is cheap.

Expected history columns (current diagnostic):
 step tau q Pxi_total QEM Qkin_i Qkin_e Qrhs_volume Pxi_i Pxi_e
 QEM_i QEM_e Ni Ne Qsurf_i Qsurf_e Qsurf_x_i Qsurf_x_e
 Qsurf_z_i Qsurf_z_e Qrhs_corrected [face columns may follow]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("histories", nargs="+",
                   help="one or more particle-quadrature history files")
    p.add_argument("--period", type=float, default=0.63)
    p.add_argument("--period-min", type=float, default=0.56)
    p.add_argument("--period-max", type=float, default=0.70)
    p.add_argument("--period-n", type=int, default=141)
    p.add_argument("--out-prefix", default="ishizawa_particle_quadrature_robustness")
    p.add_argument("--window-width", type=float, default=1.26,
                   help="sliding-window width in omega_ci*t; default is 2*T for T=0.63")
    p.add_argument("--window-step", type=float, default=0.20,
                   help="spacing between sliding-window starts")
    p.add_argument("--min-window-points", type=int, default=12,
                   help="minimum samples required in each sliding window")
    return p.parse_args()


def fit_phasor(t, y, T):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    u = t - t[0]
    om = 2.0*np.pi/T
    M = np.column_stack([np.ones_like(u), u, np.cos(om*u), np.sin(om*u)])
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    pred = M @ c
    ssr = np.sum((y-pred)**2)
    sst = np.sum((y-y.mean())**2)
    r2 = 1.0-ssr/sst if sst > 0 else np.nan
    return complex(c[2], -c[3]), float(r2)


def closure_for_rows(a, T):
    t = a[:,1]
    Zp, Rp = fit_phasor(t, a[:,3], T)
    om = 2.0*np.pi/T
    Zlhs = 1j*om*Zp

    Zem, Re = fit_phasor(t, a[:,4], T)
    Zki, Ri = fit_phasor(t, a[:,5], T)
    Zke, Ree = fit_phasor(t, a[:,6], T)
    Zvol = Zem + Zki + Zke

    if a.shape[1] >= 16:
        Zsi, _ = fit_phasor(t, a[:,14], T)
        Zse, _ = fit_phasor(t, a[:,15], T)
        Zsurf = Zsi + Zse
    else:
        Zsurf = 0j

    Zcorr = Zvol - Zsurf
    den = max(abs(Zlhs), 1e-300)
    return dict(
        lhs=Zlhs, vol=Zvol, surf=Zsurf, corr=Zcorr,
        err_vol=abs(Zlhs-Zvol)/den,
        err_corr=abs(Zlhs-Zcorr)/den,
        amp_vol=abs(Zvol)/den,
        amp_corr=abs(Zcorr)/den,
        phase_vol=np.angle(Zvol/Zlhs),
        phase_corr=np.angle(Zcorr/Zlhs),
        surf_frac=abs(Zsurf)/den,
        R_P=Rp, R_EM=Re, R_i=Ri, R_e=Ree,
    )


def label_from_path(p):
    parent = Path(p).parent.name
    name = parent if parent else Path(p).stem
    return name.replace("analysis_ishizawa_particle_boundary_strip", "h=")


def main():
    args = parse_args()
    periods = np.linspace(args.period_min, args.period_max, args.period_n)

    all_results = []
    sliding_results = []
    lines = []
    lines.append("Particle-quadrature closure robustness\n")
    lines.append("======================================\n\n")

    fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for path in args.histories:
        a = np.loadtxt(path)
        if a.ndim == 1:
            a = a[None,:]
        if len(a) < 6:
            raise RuntimeError(f"{path}: need >= 6 rows, got {len(a)}")
        lab = label_from_path(path)

        base = closure_for_rows(a, args.period)

        ev=[]; ec=[]
        for T in periods:
            r=closure_for_rows(a,T)
            ev.append(r["err_vol"])
            ec.append(r["err_corr"])
        ev=np.asarray(ev); ec=np.asarray(ec)

        loo_v=[]; loo_c=[]
        for k in range(len(a)):
            b=np.delete(a,k,axis=0)
            r=closure_for_rows(b,args.period)
            loo_v.append(r["err_vol"])
            loo_c.append(r["err_corr"])
        loo_v=np.asarray(loo_v); loo_c=np.asarray(loo_c)

        edge=[]
        if len(a) >= 7:
            for sl in (slice(1,None), slice(None,-1), slice(1,-1)):
                r=closure_for_rows(a[sl],args.period)
                edge.append((r["err_vol"],r["err_corr"]))

        all_results.append((lab,base,loo_v,loo_c,ev,ec))

        # Sliding-window robustness.  This is the key diagnostic for a dense
        # particle-rich continuation: a physical closure should give nearly
        # the same complex error and phase in overlapping windows spanning
        # roughly two breathing periods.
        sw=[]
        t=a[:,1]
        width=float(args.window_width)
        step=float(args.window_step)
        if width > 0.0 and step > 0.0 and t.max()-t.min() >= width:
            starts=np.arange(t.min(),t.max()-width+0.5*step,step)
            for t0 in starts:
                mask=(t >= t0-1e-12) & (t <= t0+width+1e-12)
                if np.count_nonzero(mask) < args.min_window_points:
                    continue
                rr=closure_for_rows(a[mask],args.period)
                sw.append((
                    0.5*(t0+t0+width), np.count_nonzero(mask),
                    rr["err_vol"],rr["err_corr"],rr["phase_corr"],
                    rr["amp_corr"],rr["surf_frac"]
                ))
        sliding_results.append((lab,np.asarray(sw,float) if sw else np.empty((0,7))))

        lines.append(f"[{lab}] {path}\n")
        lines.append(f"points = {len(a)}\n")
        lines.append(f"base T = {args.period:.8f}\n")
        lines.append(f"base volume error = {base['err_vol']:.8e}\n")
        lines.append(f"base corrected error = {base['err_corr']:.8e}\n")
        lines.append(f"surface |Z|/|LHS| = {base['surf_frac']:.8e}\n")
        lines.append(f"base corrected phase [rad] = {base['phase_corr']:+.8e}\n")
        lines.append(f"LOO volume error min/median/max = "
                     f"{loo_v.min():.8e} {np.median(loo_v):.8e} {loo_v.max():.8e}\n")
        lines.append(f"LOO corrected error min/median/max = "
                     f"{loo_c.min():.8e} {np.median(loo_c):.8e} {loo_c.max():.8e}\n")
        lines.append(f"period-scan volume error min/max = {ev.min():.8e} {ev.max():.8e}\n")
        lines.append(f"period at min volume error = {periods[np.argmin(ev)]:.8f}\n")
        lines.append(f"period-scan corrected error min/max = {ec.min():.8e} {ec.max():.8e}\n")
        lines.append(f"period at min corrected error = {periods[np.argmin(ec)]:.8f}\n")
        if edge:
            names=["drop first","drop last","drop both"]
            for nm,(x,y) in zip(names,edge):
                lines.append(f"{nm}: volume={x:.8e} corrected={y:.8e}\n")
        if sw:
            swa=np.asarray(sw,float)
            lines.append(
                f"sliding windows: width={args.window_width:.8f} "
                f"step={args.window_step:.8f} count={len(swa)}\n"
            )
            lines.append(
                "sliding volume error min/median/max = "
                f"{swa[:,2].min():.8e} {np.median(swa[:,2]):.8e} {swa[:,2].max():.8e}\n"
            )
            lines.append(
                "sliding corrected error min/median/max = "
                f"{swa[:,3].min():.8e} {np.median(swa[:,3]):.8e} {swa[:,3].max():.8e}\n"
            )
            lines.append(
                "sliding corrected phase min/median/max [rad] = "
                f"{swa[:,4].min():+.8e} {np.median(swa[:,4]):+.8e} {swa[:,4].max():+.8e}\n"
            )
        else:
            lines.append(
                f"sliding windows: none (need span >= {args.window_width:.8f} "
                f"and >= {args.min_window_points} points/window)\n"
            )
        lines.append("\n")

        axs[0].plot(periods, ev, label=f"{lab} volume")
        axs[0].plot(periods, ec, "--", label=f"{lab} corrected")
        axs[1].plot(np.arange(len(a)), loo_c, "o-", label=lab)

    axs[0].axvline(args.period, ls=":", lw=1)
    axs[0].set_ylabel("complex closure error")
    axs[0].set_title("Period sensitivity of particle-quadrature closure")
    axs[0].grid(alpha=.25)
    axs[0].legend(fontsize=8, ncol=2)

    axs[1].set_xlabel("left-out temporal sample index")
    axs[1].set_ylabel("surface-corrected complex error")
    axs[1].set_title("Leave-one-out temporal robustness at fixed period")
    axs[1].grid(alpha=.25)
    axs[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out_prefix + ".png", dpi=210)
    plt.close(fig)

    # Compact comparison plot at the nominal period.
    labels=[r[0] for r in all_results]
    vol=np.array([r[1]["err_vol"] for r in all_results])
    cor=np.array([r[1]["err_corr"] for r in all_results])
    sf =np.array([r[1]["surf_frac"] for r in all_results])

    x=np.arange(len(labels))
    fig,ax=plt.subplots(figsize=(8,5.5))
    w=.34
    ax.bar(x-w/2,vol,w,label="volume-only error")
    ax.bar(x+w/2,cor,w,label="surface-corrected error")
    ax.plot(x,sf,"o-",label=r"$|Z_{surf}|/|Z_{LHS}|$")
    ax.set_xticks(x,labels,rotation=20)
    ax.set_ylabel("relative amplitude")
    ax.set_title("Boundary-strip comparison at nominal breathing period")
    ax.grid(axis="y",alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_prefix + "_strip_compare.png",dpi=210)
    plt.close(fig)

    # Sliding-window figure is written only when at least one input history
    # contains enough temporal coverage.
    if any(len(a)>0 for _,a in sliding_results):
        fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
        for lab,sw in sliding_results:
            if len(sw)==0:
                continue
            axs[0].plot(sw[:,0],sw[:,2],"o-",label=f"{lab} volume error")
            axs[0].plot(sw[:,0],sw[:,3],"o--",label=f"{lab} corrected error")
            axs[1].plot(sw[:,0],sw[:,4],"o-",label=f"{lab} corrected phase")
        axs[0].set_ylabel("complex closure error")
        axs[0].set_title(
            f"Sliding-window closure robustness: width={args.window_width:g}"
        )
        axs[0].grid(alpha=.25); axs[0].legend(fontsize=8)
        axs[1].axhline(0,lw=.8)
        axs[1].set_xlabel(r"window center $\omega_{ci}t$")
        axs[1].set_ylabel("phase(RHS/LHS) [rad]")
        axs[1].grid(alpha=.25); axs[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out_prefix + "_sliding.png",dpi=210)
        plt.close(fig)

    with open(args.out_prefix + "_summary.txt","w") as f:
        f.writelines(lines)

    print("Saved", args.out_prefix + "_summary.txt")
    print("Saved", args.out_prefix + ".png")
    print("Saved", args.out_prefix + "_strip_compare.png")
    if any(len(a)>0 for _,a in sliding_results):
        print("Saved", args.out_prefix + "_sliding.png")


if __name__ == "__main__":
    main()
