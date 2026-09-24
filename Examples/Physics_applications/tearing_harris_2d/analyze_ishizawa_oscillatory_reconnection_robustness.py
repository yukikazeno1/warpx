#!/usr/bin/env python3
"""Robustness scan for late-time oscillatory reconnection diagnostics.

Scans:
  * x low-pass cutoff m_max
  * local z-averaging half-width for E_y at the O/X points

For every combination, this script recomputes the *matching* low-pass flux
Psi_m(t) from A_y and compares
  dPsi_m/d(omega_ci t)
against
  s [E_y(X)-E_y(O)] / (B0 de omega_ci).

It reports both raw pointwise agreement and the coherent breathing-frequency
agreement:
  - raw correlation / regression / RMS mismatch
  - fitted breathing period amplitude ratio
      A_E / [(2 pi/T) A_Psi]
  - phase(E-rate)-phase(Psi), expected +pi/2
  - phase(E-rate)-phase(dPsi), expected 0

The goal is to establish whether the oscillatory-reconnection conclusion is
robust to reasonable spatial filtering and local E_y sampling choices.

Outputs
-------
ishizawa_oscillatory_reconnection_robustness_table.txt
ishizawa_oscillatory_reconnection_robustness_summary.txt
ishizawa_oscillatory_reconnection_robustness_heatmaps.png
ishizawa_oscillatory_reconnection_robustness_selected.png
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import params, reconstruct_Ay
from analyze_ishizawa_breathing_phases import load_full_fields
from analyze_ishizawa_saturation import lowpass_x, choose_system_scale_OX


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs_ishizawa_scale/ppc49")
    p.add_argument("--tmin", type=float, default=2.8)
    p.add_argument("--tmax", type=float, default=4.0)
    p.add_argument("--period", type=float, default=0.63)
    p.add_argument("--max-modes", default="1,2,3,4,5",
                   help="comma-separated low-pass cutoffs")
    p.add_argument("--z-averages-de", default="0,0.25,0.5,1.0",
                   help="comma-separated E_y z half-widths in de")
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


def harmonic_fit(t, y, T):
    u = t-t[0]
    om = 2*np.pi/T
    M = np.column_stack([np.ones_like(u), u, np.sin(om*u), np.cos(om*u)])
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


def linreg(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    A = np.column_stack([x, np.ones_like(x)])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A@c
    ssr = np.sum((y-pred)**2)
    sst = np.sum((y-np.mean(y))**2)
    r2 = 1-ssr/sst if sst > 0 else np.nan
    return float(c[0]), float(c[1]), float(r2)


def parse_csv_numbers(s, cast=float):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params()

    modes = parse_csv_numbers(args.max_modes, int)
    zavgs = parse_csv_numbers(args.z_averages_de, float)

    files = sorted(
        glob.glob(str(Path(args.run_dir)/"diags"/"diag1*")),
        key=numeric_key
    )
    files = [f for f in files if step_from_plotfile(f) >= 0]
    if not files:
        raise FileNotFoundError("No diag1* plotfiles found")

    # data[(mmax,zavg)] -> rows of [t, psi, E-rate, xO, xX]
    data = {(m,z): [] for m in modes for z in zavgs}

    for fn in files:
        F = load_full_fields(fn)
        t = P["wci"] * F["ds"].current_time.to_value("s")
        if t < args.tmin or t > args.tmax:
            continue

        Ay = reconstruct_Ay(F["Bx"], F["Bz"], F["x"], F["z"])

        for mmax in modes:
            Ay_lp = lowpass_x(Ay, mmax)
            Ey_lp = lowpass_x(F["Ey"], mmax)
            j0, io, ix, orient, AO, AX, psi_si = choose_system_scale_OX(
                Ay_lp, F["z"]
            )
            psi = psi_si/(P["B0"]*P["de"])

            for zavg in zavgs:
                if zavg > 0:
                    zm = np.abs(F["z"]) <= zavg*P["de"]
                    EyO = float(np.mean(Ey_lp[io,zm]))
                    EyX = float(np.mean(Ey_lp[ix,zm]))
                else:
                    EyO = float(Ey_lp[io,j0])
                    EyX = float(Ey_lp[ix,j0])

                erate = orient*(EyX-EyO)/(P["B0"]*P["de"]*P["wci"])
                data[(mmax,zavg)].append([
                    t, psi, erate, F["x"][io]/P["de"],
                    F["x"][ix]/P["de"], orient
                ])

    rows = []
    per_case = {}
    for mmax in modes:
        for zavg in zavgs:
            a = np.asarray(data[(mmax,zavg)], float)
            if len(a) < 8:
                continue
            t,psi,erate,xO,xX,orient = a.T
            dpsi = np.gradient(psi,t,edge_order=2)

            corr = float(np.corrcoef(dpsi,erate)[0,1])
            slope,intercept,r2raw = linreg(erate,dpsi)
            rms_d = float(np.sqrt(np.mean(dpsi*dpsi)))
            rel_rms = float(np.sqrt(np.mean((dpsi-erate)**2))/rms_d)

            _,Apsi,phipsi,Rpsi,cpsi = harmonic_fit(t,psi,args.period)
            _,Ad,phid,Rd,cd = harmonic_fit(t,dpsi,args.period)
            _,Ae,phie,Re,ce = harmonic_fit(t,erate,args.period)

            expected = (2*np.pi/args.period)*Apsi
            amp_ratio = Ae/expected if expected > 0 else np.nan
            phase_E_Psi = wrap_phase(phie-phipsi)
            phase_err_quadrature = wrap_phase(phase_E_Psi-np.pi/2)
            phase_E_dpsi = wrap_phase(phie-phid)

            orient_unique = len(set(orient.astype(int)))
            xO_jump = float(np.max(np.abs(np.diff(xO)))) if len(xO)>1 else 0.0
            xX_jump = float(np.max(np.abs(np.diff(xX)))) if len(xX)>1 else 0.0

            # Composite low-frequency quality score; lower is better.
            score = (
                abs(amp_ratio-1.0)
                + abs(phase_err_quadrature)/np.pi
                + abs(phase_E_dpsi)/np.pi
                + max(0.0, 0.7-Re)
            )

            row = [
                mmax,zavg,len(t),corr,slope,intercept,r2raw,rel_rms,
                Apsi,Rpsi,Ad,Rd,Ae,Re,amp_ratio,
                phase_E_Psi,phase_err_quadrature,phase_E_dpsi,
                orient_unique,xO_jump,xX_jump,score
            ]
            rows.append(row)
            per_case[(mmax,zavg)] = dict(
                t=t,psi=psi,dpsi=dpsi,erate=erate,
                amp_ratio=amp_ratio,phase_err=phase_err_quadrature,
                phase_dpsi=phase_E_dpsi,Re=Re,corr=corr,score=score
            )

    arr = np.asarray(rows,float)
    header = (
        "mmax zavg_de nsamples raw_corr raw_slope raw_intercept raw_R2 "
        "raw_rel_rms Apsi R2psi Adpsi R2dpsi Aerate R2erate amp_ratio "
        "phase_E_minus_Psi phase_error_from_pi_over_2 "
        "phase_E_minus_dPsi orientation_unique_count "
        "max_xO_jump_de max_xX_jump_de quality_score"
    )
    np.savetxt(
        "ishizawa_oscillatory_reconnection_robustness_table.txt",
        arr, header=header
    )

    # Heatmaps.
    mi = {m:i for i,m in enumerate(modes)}
    zi = {z:i for i,z in enumerate(zavgs)}
    shape=(len(modes),len(zavgs))
    Hratio=np.full(shape,np.nan)
    Hphase=np.full(shape,np.nan)
    HR2=np.full(shape,np.nan)
    Hcorr=np.full(shape,np.nan)
    Hscore=np.full(shape,np.nan)
    for r in rows:
        m,z=int(r[0]),float(r[1])
        i,j=mi[m],zi[z]
        Hcorr[i,j]=r[3]
        HR2[i,j]=r[13]
        Hratio[i,j]=r[14]
        Hphase[i,j]=r[16]
        Hscore[i,j]=r[21]

    fig,axs=plt.subplots(2,2,figsize=(11,8.5))
    specs=[
        (Hratio,"$A_E/[(2\pi/T)A_\Psi]$",0.7,1.3),
        (Hphase,r"$\Delta\phi_E-\pi/2$ [rad]",-0.5,0.5),
        (HR2,r"$R^2$ of $E$-rate harmonic",0,1),
        (Hcorr,"raw corr$(d\Psi,E)$",0,1),
    ]
    for ax,(H,title,vmin,vmax) in zip(axs.ravel(),specs):
        im=ax.imshow(H,origin="lower",aspect="auto",vmin=vmin,vmax=vmax)
        ax.set_xticks(range(len(zavgs)),[f"{z:g}" for z in zavgs])
        ax.set_yticks(range(len(modes)),[str(m) for m in modes])
        ax.set_xlabel(r"$E_y$ z-average half-width / $d_e$")
        ax.set_ylabel(r"low-pass $m_{max}$")
        ax.set_title(title)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if np.isfinite(H[i,j]):
                    ax.text(j,i,f"{H[i,j]:.2f}",ha="center",va="center",
                            fontsize=8)
        fig.colorbar(im,ax=ax,shrink=.85)
    fig.suptitle("Robustness of the breathing-frequency oscillatory reconnection test")
    fig.tight_layout()
    fig.savefig("ishizawa_oscillatory_reconnection_robustness_heatmaps.png",dpi=210)
    plt.close(fig)

    # Pick best score and plot against baseline m=3,z=0 if available.
    best_row=min(rows,key=lambda r:r[21])
    best=(int(best_row[0]),float(best_row[1]))
    base=(3,0.0) if (3,0.0) in per_case else best

    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for key,label,ls in [(base,f"baseline m={base[0]}, zavg={base[1]:g}","-"),
                         (best,f"best m={best[0]}, zavg={best[1]:g}","--")]:
        D=per_case[key]
        axs[0].plot(D["t"],D["dpsi"],ls,label=label+" numeric dPsi")
        axs[0].plot(D["t"],D["erate"],ls,label=label+" Ey rate",alpha=.8)
        axs[1].plot(D["t"],D["psi"],ls,label=label+" Psi")
    axs[0].axhline(0,lw=.8)
    axs[0].set_ylabel("reconnection rate")
    axs[0].legend(fontsize=7,ncol=2)
    axs[1].set_ylabel(r"$\Psi/(B_0d_e)$")
    axs[1].set_xlabel(r"$\omega_{ci}t$")
    axs[1].legend(fontsize=8)
    for ax in axs: ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig("ishizawa_oscillatory_reconnection_robustness_selected.png",dpi=210)
    plt.close(fig)

    # Summary.
    ratios=np.array([r[14] for r in rows])
    perrs=np.array([r[16] for r in rows])
    dperrs=np.array([r[17] for r in rows])
    re=np.array([r[13] for r in rows])
    corr=np.array([r[3] for r in rows])

    # "Reasonable robust" subset: harmonic E fit R2 >= 0.5 and no orientation switch.
    robust=[r for r in rows if r[13]>=0.5 and int(r[18])==1]
    with open("ishizawa_oscillatory_reconnection_robustness_summary.txt","w") as f:
        f.write("Ishizawa oscillatory-reconnection robustness scan\n")
        f.write("================================================\n\n")
        f.write(f"interval omega_ci*t = {args.tmin:.8f} .. {args.tmax:.8f}\n")
        f.write(f"reference period = {args.period:.8f}\n")
        f.write(f"mmax values = {modes}\n")
        f.write(f"z-average half-widths/de = {zavgs}\n")
        f.write(f"number of scan cases = {len(rows)}\n\n")
        f.write("All cases\n")
        f.write("---------\n")
        f.write(f"amp ratio min/median/max = {np.nanmin(ratios):.6f} "
                f"{np.nanmedian(ratios):.6f} {np.nanmax(ratios):.6f}\n")
        f.write(f"|phase error from pi/2| median/max [rad] = "
                f"{np.nanmedian(np.abs(perrs)):.6f} "
                f"{np.nanmax(np.abs(perrs)):.6f}\n")
        f.write(f"|phase E-dPsi| median/max [rad] = "
                f"{np.nanmedian(np.abs(dperrs)):.6f} "
                f"{np.nanmax(np.abs(dperrs)):.6f}\n")
        f.write(f"E-rate harmonic R2 median = {np.nanmedian(re):.6f}\n")
        f.write(f"raw corr median = {np.nanmedian(corr):.6f}\n\n")
        f.write("Best composite case\n")
        f.write("-------------------\n")
        f.write(f"mmax = {best[0]}\n")
        f.write(f"zavg/de = {best[1]:.6f}\n")
        f.write(f"quality score = {best_row[21]:.8f}\n")
        f.write(f"raw corr = {best_row[3]:.8f}\n")
        f.write(f"raw slope = {best_row[4]:.8f}\n")
        f.write(f"E harmonic R2 = {best_row[13]:.8f}\n")
        f.write(f"amp ratio = {best_row[14]:.8f}\n")
        f.write(f"phase error from pi/2 = {best_row[16]:.8f} rad\n")
        f.write(f"phase E-dPsi = {best_row[17]:.8f} rad\n\n")
        if robust:
            rr=np.array(robust,float)
            f.write("Robust subset: E harmonic R2>=0.5 and no orientation switches\n")
            f.write("-----------------------------------------------------------\n")
            f.write(f"cases = {len(robust)}\n")
            f.write(f"amp ratio median = {np.median(rr[:,14]):.8f}\n")
            f.write(f"amp ratio 16-84% = "
                    f"{np.percentile(rr[:,14],16):.8f} .. "
                    f"{np.percentile(rr[:,14],84):.8f}\n")
            f.write(f"|phase error pi/2| median = "
                    f"{np.median(np.abs(rr[:,16])):.8f} rad\n")
            f.write(f"|phase E-dPsi| median = "
                    f"{np.median(np.abs(rr[:,17])):.8f} rad\n")
            f.write(f"raw corr median = {np.median(rr[:,3]):.8f}\n")
        f.write("\nInterpretation\n")
        f.write("--------------\n")
        f.write("The oscillatory-reconnection interpretation is numerically robust if the\n")
        f.write("reasonable scan cases keep amp_ratio near unity, phase_error_from_pi/2\n")
        f.write("near zero, and phase_E_minus_dPsi near zero.  Raw pointwise correlation\n")
        f.write("is expected to be more sensitive to PIC noise than these coherent-mode\n")
        f.write("metrics.\n")

    print("Saved ishizawa_oscillatory_reconnection_robustness_table.txt")
    print("Saved ishizawa_oscillatory_reconnection_robustness_summary.txt")
    print("Saved ishizawa_oscillatory_reconnection_robustness_heatmaps.png")
    print("Saved ishizawa_oscillatory_reconnection_robustness_selected.png")


if __name__=="__main__":
    main()
