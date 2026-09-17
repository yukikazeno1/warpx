#!/usr/bin/env python3
"""Animate quantitative X/O-point tracking during magnetic-island coalescence.

Compared with animate_ishizawa_tearing.py, this movie overlays the robust
midplane O/X detections and adds time histories of island count, dominant mode,
spectral centroid, island flux and X-point E_y.  It is intended for the
nonlinear 40k run where the early m~5-7 island chain merges into fewer,
larger islands and lower Fourier modes.
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import (
    params,
    load_fields,
    reconstruct_Ay,
    corrected_jy,
)
from analyze_ishizawa_island_coalescence import (
    detect_islands,
    spectrum_core,
    load_ey,
)

MU0 = 1.25663706212e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs_ishizawa_scale/ppc49")
    p.add_argument("--output", default="ishizawa_island_coalescence.mp4")
    p.add_argument("--max-mode", type=int, default=20)
    p.add_argument("--topology-max-mode", type=int, default=20)
    p.add_argument("--core-width-de", type=float, default=2.0)
    p.add_argument("--zoom-de", type=float, default=4.0)
    p.add_argument("--tmin", type=float, default=0.05)
    p.add_argument("--tmax", type=float, default=None)
    p.add_argument("--relative-flux-threshold", type=float, default=0.12)
    p.add_argument("--absolute-flux-threshold", type=float, default=1e-3)
    p.add_argument("--min-separation-de", type=float, default=2.0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--contours", type=int, default=35)
    return p.parse_args()


def numeric_key(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def step_from_path(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    if not m:
        return -1
    s = m.group(1)
    return int(s[1:]) if len(s) > 1 and s.startswith("1") else int(s)


def load_frame(path, P, args):
    F = load_fields(path)
    Ey = load_ey(F)
    Ay = reconstruct_Ay(F["Bx"], F["Bz"], F["x"], F["z"])
    dAy = Ay - np.mean(Ay, axis=0, keepdims=True)
    jy = corrected_jy(F["Bx"], F["Bz"], F["dx"], F["dz"])
    det = detect_islands(
        Ay, F["x"], F["z"], P["B0"], P["de"],
        topology_max_mode=args.topology_max_mode,
        relative_threshold=args.relative_flux_threshold,
        absolute_threshold=args.absolute_flux_threshold,
        min_separation_de=args.min_separation_de,
    )
    return F, Ey, Ay, dAy, jy, det


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params()
    vA = P["B0"] / np.sqrt(MU0 * P["n0"] * P["mi"])

    files_all = sorted(glob.glob(str(Path(args.run_dir)/"diags"/"diag1*")), key=numeric_key)
    if not files_all:
        raise FileNotFoundError(f"No diag1 plotfiles under {args.run_dir}/diags")

    files, times, steps = [], [], []
    diag = []
    print("Pre-scanning frames for O/X and spectral histories...")
    for fn in files_all:
        F, Ey, Ay, dAy, jy, det = load_frame(fn, P, args)
        tci = P["wci"] * F["ds"].current_time.to_value("s")
        if tci < args.tmin:
            continue
        if args.tmax is not None and tci > args.tmax:
            continue

        amps = spectrum_core(F["Bz"], F["z"], args.max_mode,
                             args.core_width_de*P["de"], P["B0"])
        modes = np.arange(1, args.max_mode+1, dtype=float)
        good = np.isfinite(amps) & (amps > 0)
        if np.any(good):
            pwr = np.where(good, amps**2, 0.0)
            mdom = int(np.nanargmax(amps)+1)
            mbar = float(np.sum(modes*pwr)/np.sum(pwr))
        else:
            mdom, mbar = -1, np.nan

        j0 = det["j0"]
        xids = det["x_indices"]
        if len(xids):
            eyx = np.abs(Ey[xids,j0])/(vA*P["B0"])
            max_eyx = float(np.nanmax(eyx))
        else:
            max_eyx = np.nan
        psis = np.asarray([q["psi"] for q in det["islands"]], dtype=float)
        psi_max = float(np.max(psis)) if len(psis) else 0.0

        files.append(fn)
        times.append(tci)
        steps.append(step_from_path(fn))
        diag.append((len(det["islands"]), len(xids), mdom, mbar, psi_max, max_eyx))
        if len(files)%10 == 0:
            print(f"  retained {len(files)} frames; latest omega_ci*t={tci:.4f}")

    if not files:
        raise RuntimeError("No frames after tmin/tmax filtering")

    stride = max(1,args.stride)
    files = files[::stride]
    times = np.asarray(times)[::stride]
    steps = np.asarray(steps)[::stride]
    diag = np.asarray(diag,dtype=float)[::stride]

    # Fixed color scales from final frame.
    Ff, Eyf, Ayf, dAyf, jyf, detf = load_frame(files[-1], P, args)
    zdef = Ff["z"]/P["de"]
    zoomf = np.abs(zdef) <= args.zoom_de
    jyfn = jyf[:,zoomf]/P["J0"]
    jy_hi = max(1.0,float(np.nanpercentile(jyfn,99.8)))
    jy_lo = min(-0.05*jy_hi,float(np.nanpercentile(jyfn,0.2)))
    jy_norm = Normalize(vmin=jy_lo,vmax=jy_hi)

    dan = dAyf[:,zoomf]/(P["B0"]*P["de"])
    da_abs = max(1e-6,float(np.nanmax(np.abs(dan))))
    day_norm = SymLogNorm(linthresh=max(1e-5,da_abs*1e-3),
                          linscale=1.0,vmin=-da_abs,vmax=da_abs,base=10)

    fig = plt.figure(figsize=(15,9.5))
    gs = fig.add_gridspec(2,2,height_ratios=[1.0,0.72],hspace=0.30,wspace=0.27)
    ax_top = fig.add_subplot(gs[0,0])
    ax_day = fig.add_subplot(gs[0,1])
    ax_count = fig.add_subplot(gs[1,0])
    ax_flux = fig.add_subplot(gs[1,1])

    smj = ScalarMappable(norm=jy_norm,cmap="viridis"); smj.set_array([])
    c1 = fig.colorbar(smj,ax=ax_top,fraction=0.046,pad=0.04); c1.set_label(r"$J_y/J_0$")
    sma = ScalarMappable(norm=day_norm,cmap="coolwarm"); sma.set_array([])
    c2 = fig.colorbar(sma,ax=ax_day,fraction=0.046,pad=0.04); c2.set_label(r"$\delta A_y/(B_0d_e)$")

    def draw(i):
        F, Ey, Ay, dAy, jy, det = load_frame(files[i], P, args)
        xde = F["x"]/P["de"]
        zde = F["z"]/P["de"]
        zoom = np.abs(zde)<=args.zoom_de
        X,Z = np.meshgrid(xde,zde,indexing="ij")
        jyn = jy/P["J0"]
        day = dAy/(P["B0"]*P["de"])

        ax_top.clear(); ax_day.clear(); ax_count.clear(); ax_flux.clear()

        ax_top.pcolormesh(X[:,zoom],Z[:,zoom],jyn[:,zoom],shading="auto",
                          cmap="viridis",norm=jy_norm)
        alo,ahi = float(np.nanmin(Ay[:,zoom])),float(np.nanmax(Ay[:,zoom]))
        if ahi>alo:
            ax_top.contour(X[:,zoom],Z[:,zoom],Ay[:,zoom],
                           levels=np.linspace(alo,ahi,args.contours),
                           colors="k",linewidths=0.42,alpha=0.75)
        j0 = det["j0"]
        if det["islands"]:
            io=[q["io"] for q in det["islands"]]
            ax_top.scatter(xde[io],np.full(len(io),zde[j0]),s=55,facecolors="none",
                           edgecolors="tab:orange",linewidths=1.6,label="O")
        if len(det["x_indices"]):
            ax_top.scatter(xde[det["x_indices"]],np.full(len(det["x_indices"]),zde[j0]),
                           s=55,marker="x",color="tab:cyan",linewidths=1.8,label="X")
        ax_top.set_xlabel(r"$x/d_e$"); ax_top.set_ylabel(r"$z/d_e$")
        ax_top.set_title("full topology + tracked O/X points")
        ax_top.legend(loc="upper right",fontsize=8)

        ax_day.pcolormesh(X[:,zoom],Z[:,zoom],day[:,zoom],shading="auto",
                          cmap="coolwarm",norm=day_norm)
        vf=float(np.nanmax(np.abs(day[:,zoom])))
        if vf>0:
            ax_day.contour(X[:,zoom],Z[:,zoom],day[:,zoom],
                           levels=np.linspace(-vf,vf,args.contours),
                           colors="k",linewidths=0.38,alpha=0.55)
        ax_day.set_xlabel(r"$x/d_e$"); ax_day.set_ylabel(r"$z/d_e$")
        ax_day.set_title(r"perturbation topology $\delta A_y$")

        # Count/mode history.
        ax_count.step(times[:i+1],diag[:i+1,0],where="mid",marker="o",label=r"$N_O$")
        ax_count.plot(times[:i+1],diag[:i+1,2],"o-",label=r"$m_{dom}$")
        ax_count.plot(times[:i+1],diag[:i+1,3],"o-",label=r"$\bar m$")
        ax_count.axvline(times[i],color="k",ls="--",lw=0.9,alpha=0.6)
        ax_count.set_xlim(times[0],times[-1]); ax_count.set_ylim(bottom=0)
        ax_count.set_xlabel(r"$\omega_{ci}t$"); ax_count.set_ylabel("island count / mode")
        ax_count.grid(alpha=0.25); ax_count.legend(fontsize=8,ncol=3)

        # Flux and X-point field.
        ax_flux.semilogy(times[:i+1],np.maximum(diag[:i+1,4],1e-12),"o-",
                         label=r"max $\Psi/(B_0d_e)$")
        axf2=ax_flux.twinx()
        axf2.plot(times[:i+1],diag[:i+1,5],"^-",label=r"max $|E_y(X)|/(v_AB_0)$")
        ax_flux.axvline(times[i],color="k",ls="--",lw=0.9,alpha=0.6)
        ax_flux.set_xlim(times[0],times[-1])
        ax_flux.set_xlabel(r"$\omega_{ci}t$"); ax_flux.set_ylabel(r"max island flux / $B_0d_e$")
        axf2.set_ylabel(r"max $|E_y(X)|/(v_AB_0)$")
        ax_flux.grid(alpha=0.25)
        h1,l1=ax_flux.get_legend_handles_labels(); h2,l2=axf2.get_legend_handles_labels()
        ax_flux.legend(h1+h2,l1+l2,fontsize=8,loc="upper left")

        fig.suptitle(fr"island coalescence: $\omega_{{ci}}t={times[i]:.3f}$; "
                     fr"$N_O={int(diag[i,0])}$, $m_{{dom}}={int(diag[i,2])}$, "
                     fr"$\bar m={diag[i,3]:.2f}$",fontsize=15)
        print(f"frame {i+1:3d}/{len(files)} step={steps[i]:6d} wci*t={times[i]:.4f} "
              f"NO={int(diag[i,0])} mdom={int(diag[i,2])}")
        return []

    ani=animation.FuncAnimation(fig,draw,frames=len(files),interval=1000.0/args.fps,
                                blit=False,repeat=True)
    out=Path(args.output)
    if out.suffix.lower()==".mp4":
        try:
            ani.save(out,writer=animation.FFMpegWriter(fps=args.fps,bitrate=2600),dpi=args.dpi)
        except Exception as exc:
            raise RuntimeError("MP4 writing failed; install ffmpeg or use a .gif output") from exc
    else:
        ani.save(out,writer=animation.PillowWriter(fps=args.fps),dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
