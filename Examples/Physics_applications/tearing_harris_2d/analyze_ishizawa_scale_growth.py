#!/usr/bin/env python3
"""Growth-rate screening for the Ishizawa-scale Harris pilot.

Reads ishizawa_scale_mode_history.txt produced by
analyze_ishizawa_scale_early.py.  Time is x = omega_ci*t, so a fitted slope
of ln(A_m) versus x is gamma/omega_ci.

The script reports both a common fixed-window slope and an automatic robust
window screen.  It is intended to distinguish sustained linear growth from
initial PIC/noise transients.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("history", nargs="?", default="ishizawa_scale_mode_history.txt")
    p.add_argument("--max-mode", type=int, default=20)
    p.add_argument("--fixed-t0", type=float, default=0.05)
    p.add_argument("--fixed-t1", type=float, default=0.25)
    p.add_argument("--auto-tmin", type=float, default=0.05)
    p.add_argument("--min-points", type=int, default=6)
    p.add_argument("--min-span", type=float, default=0.10)
    p.add_argument("--min-r2", type=float, default=0.90)
    p.add_argument("--min-growth", type=float, default=2.0)
    return p.parse_args()


def fit(x, a):
    y = np.log(a)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope*x + intercept
    ss_res = np.sum((y-pred)**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2


def best_window(t, a, args):
    valid = np.isfinite(t) & np.isfinite(a) & (a > 0) & (t >= args.auto_tmin)
    tv, av = t[valid], a[valid]
    if len(tv) < args.min_points:
        return None
    best = None
    n = len(tv)
    for i in range(n):
        for j in range(i + args.min_points, n + 1):
            x, y = tv[i:j], av[i:j]
            span = x[-1] - x[0]
            if span < args.min_span:
                continue
            slope, intercept, r2 = fit(x, y)
            if slope <= 0 or not np.isfinite(r2):
                continue
            growth = float(np.exp(slope*span))
            if r2 < args.min_r2 or growth < args.min_growth:
                continue
            cand = dict(t0=float(x[0]), t1=float(x[-1]), slope=float(slope),
                        intercept=float(intercept), r2=float(r2), growth=growth,
                        span=float(span))
            if best is None or (cand['span'], cand['r2']) > (best['span'], best['r2']):
                best = cand
    return best


def main():
    args = parse_args()
    d = np.loadtxt(args.history)
    t = d[:,0]
    # columns: t, Bz, By, Ey, Bxerr, Jyerr, width, flux, then A_m1...
    A = d[:,8:8+args.max_mode]
    modes = np.arange(1, A.shape[1]+1)

    fixed = []
    accepted = []
    msk = (t >= args.fixed_t0) & (t <= args.fixed_t1)
    for i,m in enumerate(modes):
        am = A[:,i]
        vf = msk & np.isfinite(am) & (am > 0)
        if np.count_nonzero(vf) >= 3:
            slope, intercept, r2 = fit(t[vf], am[vf])
            growth = float(np.exp(slope*(t[vf][-1]-t[vf][0])))
            fixed.append((slope, r2, growth, intercept))
        else:
            fixed.append((np.nan,np.nan,np.nan,np.nan))
        accepted.append(best_window(t, am, args))

    with open("ishizawa_scale_growth_summary.txt","w") as f:
        f.write("Ishizawa-scale growth-rate screening\n")
        f.write("====================================\n\n")
        f.write("Time coordinate is omega_ci*t, therefore slope = gamma/omega_ci.\n")
        f.write(f"fixed window = [{args.fixed_t0:g}, {args.fixed_t1:g}]\n")
        f.write(f"auto criteria: t>={args.auto_tmin:g}, min points={args.min_points}, ")
        f.write(f"min span={args.min_span:g}, R2>={args.min_r2:g}, growth>={args.min_growth:g}\n\n")
        f.write("mode  fixed_gamma/wci  fixed_R2  fixed_growth   accepted_gamma/wci  accepted_R2  accepted_growth  t0  t1\n")
        for i,m in enumerate(modes):
            s,r,g,_ = fixed[i]
            b = accepted[i]
            if b is None:
                f.write(f"{m:3d}  {s: .8e}  {r: .5f}  {g: .4f}   no accepted window\n")
            else:
                f.write(f"{m:3d}  {s: .8e}  {r: .5f}  {g: .4f}   {b['slope']: .8e}  {b['r2']:.5f}  {b['growth']:.4f}  {b['t0']:.4f}  {b['t1']:.4f}\n")
        f.write("\nConversion: gamma/omega_ce = (gamma/omega_ci)/800; gamma/omega_pe = (gamma/omega_ci)/2800.\n")

    # all low modes + accepted fits
    fig, ax = plt.subplots(figsize=(9,6))
    for i,m in enumerate(modes[:10]):
        ax.semilogy(t, A[:,i], "o-", ms=4, label=f"m={m}")
        b = accepted[i]
        if b is not None:
            xx = np.linspace(b['t0'], b['t1'], 100)
            yy = np.exp(b['intercept'] + b['slope']*xx)
            ax.semilogy(xx, yy, "--", lw=2)
    ax.axvspan(args.fixed_t0, args.fixed_t1, alpha=0.08)
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel(r"core $B_z$ Fourier amplitude / $B_0$")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig("ishizawa_scale_growth_fits.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,5.5))
    fg = np.array([x[0] for x in fixed])
    ax.plot(modes, fg, "o-", label="fixed-window slope")
    ag = np.array([b['slope'] if b is not None else np.nan for b in accepted])
    ax.plot(modes, ag, "s-", label="accepted robust window")
    ax.axhline(0, lw=0.8)
    ax.set_xlabel("x-Fourier mode m")
    ax.set_ylabel(r"$\gamma/\omega_{ci}$")
    ax.set_xticks(modes)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("ishizawa_scale_gamma_vs_mode.png", dpi=200)
    plt.close(fig)

    print("Saved ishizawa_scale_growth_summary.txt")
    print("Saved ishizawa_scale_growth_fits.png")
    print("Saved ishizawa_scale_gamma_vs_mode.png")


if __name__ == "__main__":
    main()
