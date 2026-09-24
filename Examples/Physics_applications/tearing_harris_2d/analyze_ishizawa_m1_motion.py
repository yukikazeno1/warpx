#!/usr/bin/env python3
"""Continuous system-scale m=1 island-position diagnostic.

The late Ishizawa-scale run is dominated by m=1.  Frame-by-frame global
extremum picking can jump between equivalent/local O/X extrema and therefore
overestimate apparent island translation.  This script instead derives the
system-scale O-point position from the continuously unwrapped complex phase
of the m=1 Fourier coefficient of full A_y at z~0.

For
    A_y,1(x) = 2 Re[C1 exp(i k1 (x-xmin))],
the m=1 maximum is at
    x_O = xmin - arg(C1)/k1  (mod Lx).
Unwrapping arg(C1) produces a continuous O-point trajectory.  The associated
m=1 X point is x_X = x_O + Lx/2.

Outputs:
  ishizawa_m1_motion_history.txt
  ishizawa_m1_motion.png
  ishizawa_m1_motion_summary.txt
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import params, load_fields, reconstruct_Ay


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs_ishizawa_scale/ppc49")
    p.add_argument("--tmin", type=float, default=1.25)
    p.add_argument("--tmax", type=float, default=None)
    p.add_argument("--Lx-de", type=float, default=64.0)
    p.add_argument("--reference-period", type=float, default=0.63)
    return p.parse_args()


def numeric_key(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def step_from_path(path):
    name = os.path.basename(path.rstrip("/"))
    m = re.search(r"(\d+)$", name)
    if not m:
        return -1
    s = m.group(1)
    if name.startswith("diag1") and s.startswith("1") and len(s) > 1:
        return int(s[1:])
    return int(s)


def detrend_linear(t, y):
    c = np.polyfit(t, y, 1)
    return y - np.polyval(c, t), c


def harmonic_fit(t, y, T):
    u = t - t[0]
    om = 2*np.pi/T
    M = np.column_stack([
        np.ones_like(u), u, np.sin(om*u), np.cos(om*u)
    ])
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    pred = M @ c
    ssr = np.sum((y-pred)**2)
    sst = np.sum((y-np.mean(y))**2)
    r2 = 1-ssr/sst if sst > 0 else np.nan
    amp = float(np.hypot(c[2], c[3]))
    phase = float(np.arctan2(c[3], c[2]))
    return pred, c, amp, phase, float(r2)


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params()

    files = sorted(
        glob.glob(str(Path(args.run_dir)/"diags"/"diag1*")),
        key=numeric_key,
    )
    if not files:
        raise FileNotFoundError("No diag1* plotfiles found")

    rows = []
    raw_phase = []

    for fn in files:
        F = load_fields(fn)
        tci = P["wci"] * F["ds"].current_time.to_value("s")
        if tci < args.tmin:
            continue
        if args.tmax is not None and tci > args.tmax:
            continue

        Ay = reconstruct_Ay(F["Bx"], F["Bz"], F["x"], F["z"])
        j0 = int(np.argmin(np.abs(F["z"])))
        amid = Ay[:, j0]
        nx = len(F["x"])
        dx = float(F["x"][1]-F["x"][0])
        Lx = nx*dx
        xmin = float(F["x"][0])

        c1 = np.fft.rfft(amid)/nx
        C = c1[1]
        phase = float(np.angle(C))
        amp_Ay = float(2*np.abs(C)/(P["B0"]*P["de"]))

        # Independent Bz m=1 amplitude.
        cb = np.fft.rfft(F["Bz"][:,j0])/nx
        amp_Bz = float(2*np.abs(cb[1])/P["B0"])

        raw_phase.append(phase)
        rows.append([
            step_from_path(fn), tci, phase, amp_Ay, amp_Bz,
            xmin/P["de"], Lx/P["de"],
        ])

    if not rows:
        raise RuntimeError("No frames satisfy requested time interval")

    a = np.asarray(rows,float)
    step,t,phase,ampAy,ampBz,xmin_de,Lx_de = a.T

    ph_un = np.unwrap(np.asarray(raw_phase))
    k1de = 2*np.pi/np.mean(Lx_de)
    # xO/de = xmin/de - phase/(k1*de).
    xO = xmin_de - ph_un/k1de
    xX = xO + 0.5*np.mean(Lx_de)

    xOd,cO = detrend_linear(t,xO)
    xXd,cX = detrend_linear(t,xX)
    rmsO = float(np.sqrt(np.mean(xOd*xOd)))
    rmsX = float(np.sqrt(np.mean(xXd*xXd)))

    predO,coefO,ampO,phiO,r2O = harmonic_fit(
        t,xO,args.reference_period
    )
    predX,coefX,ampX,phiX,r2X = harmonic_fit(
        t,xX,args.reference_period
    )

    out=np.column_stack([
        step,t,phase,ph_un,xO,xX,xOd,xXd,ampAy,ampBz
    ])
    np.savetxt(
        "ishizawa_m1_motion_history.txt",out,
        header=(
            "step omega_ci_t raw_phase_m1 unwrapped_phase_m1 "
            "xO_m1_de xX_m1_de detrended_xO_de detrended_xX_de "
            "Ay_m1_B0de Bz_m1_B0"
        )
    )

    fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True)
    axs[0].plot(t,xO,"o-",ms=3,label=r"$x_O$ from $m=1$ phase")
    axs[0].plot(t,np.polyval(cO,t),"--",label="linear drift")
    axs[0].set_ylabel(r"$x_O/d_e$")
    axs[0].legend(fontsize=8)
    axs[1].plot(t,xOd,"o-",ms=3,label="detrended O motion")
    axs[1].plot(t,predO-np.polyval(cO,t),"--",
                label=fr"$T\omega_{{ci}}={args.reference_period:g}$ fit")
    axs[1].set_ylabel(r"$\delta x_O/d_e$")
    axs[1].legend(fontsize=8)
    axs[2].plot(t,ampAy,label=r"$2|\hat A_{y,1}|/(B_0d_e)$")
    axs[2].plot(t,ampBz,label=r"$2|\hat B_{z,1}|/B_0$")
    axs[2].set_ylabel("m=1 amplitude")
    axs[2].set_xlabel(r"$\omega_{ci}t$")
    axs[2].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=.25)
    fig.suptitle("Continuous m=1 phase tracking of the system-scale island")
    fig.tight_layout()
    fig.savefig("ishizawa_m1_motion.png",dpi=210)
    plt.close(fig)

    with open("ishizawa_m1_motion_summary.txt","w") as f:
        f.write("Ishizawa-scale m=1 phase-based island motion\n")
        f.write("===========================================\n\n")
        f.write(f"interval omega_ci*t = {t[0]:.8f} .. {t[-1]:.8f}\n")
        f.write(f"samples = {len(t)}\n")
        f.write(f"mean Lx/de = {np.mean(Lx_de):.8f}\n")
        f.write(f"k1*de = {k1de:.8f}\n\n")
        f.write("Continuous O/X motion from m=1 Fourier phase\n")
        f.write("--------------------------------------------\n")
        f.write(f"O detrended rms displacement/de = {rmsO:.8e}\n")
        f.write(f"X detrended rms displacement/de = {rmsX:.8e}\n")
        f.write(f"O linear drift dx/d(omega_ci*t) [de] = {cO[0]:.8e}\n")
        f.write(f"X linear drift dx/d(omega_ci*t) [de] = {cX[0]:.8e}\n\n")
        f.write(f"Reference breathing period = {args.reference_period:.8f}\n")
        f.write(f"O harmonic displacement amplitude/de = {ampO:.8e}\n")
        f.write(f"O harmonic fit R2 = {r2O:.6f}\n")
        f.write(f"O harmonic phase [rad] = {phiO:.8f}\n")
        f.write(f"X harmonic displacement amplitude/de = {ampX:.8e}\n")
        f.write(f"X harmonic fit R2 = {r2X:.6f}\n\n")
        f.write("Interpretation\n")
        f.write("--------------\n")
        f.write("This phase-based trajectory is continuous by construction and avoids\n")
        f.write("frame-to-frame switching between equivalent/local extrema. Compare the\n")
        f.write("detrended rms and harmonic displacement amplitude with the mean island\n")
        f.write("width (~28 de). If both are small, the late mode is predominantly\n")
        f.write("breathing rather than translational sloshing.\n")

    print("Saved ishizawa_m1_motion_history.txt")
    print("Saved ishizawa_m1_motion.png")
    print("Saved ishizawa_m1_motion_summary.txt")


if __name__ == "__main__":
    main()
