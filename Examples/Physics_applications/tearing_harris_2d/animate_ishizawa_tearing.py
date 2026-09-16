#!/usr/bin/env python3
"""Animate tearing-mode evolution in the Ishizawa-scale Harris-sheet pilot.

The animation combines three complementary diagnostics for every WarpX
plotfile:

1. full magnetic topology near the current sheet: corrected J_y/J0 background
   with A_y contours;
2. perturbation flux delta A_y = A_y - <A_y>_x, shown with a fixed symmetric
   logarithmic normalization so both the early linear stage and the nonlinear
   islands remain visible;
3. the time history of selected x-Fourier B_z modes, with the current frame
   highlighted.

The default output is a GIF (Pillow writer).  If ffmpeg is installed, an MP4
can be requested simply by using an output name ending in .mp4.
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

# Reuse the tested field/topology routines.  This module already contains the
# corrected x-z curl sign: (curl B)_y = dBx/dz - dBz/dx.
from analyze_ishizawa_scale_40000 import (
    params,
    load_fields,
    reconstruct_Ay,
    corrected_jy,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs_ishizawa_scale/ppc49")
    p.add_argument("--output", default="ishizawa_tearing_evolution.gif")
    p.add_argument("--modes", default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--core-width-de", type=float, default=2.0)
    p.add_argument("--zoom-de", type=float, default=4.0)
    p.add_argument("--tmin", type=float, default=0.0,
                   help="minimum omega_ci*t included in the movie")
    p.add_argument("--tmax", type=float, default=None,
                   help="maximum omega_ci*t included in the movie")
    p.add_argument("--stride", type=int, default=1,
                   help="use every Nth available plotfile")
    p.add_argument("--fps", type=int, default=7)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--contours", type=int, default=31)
    p.add_argument("--day-linthresh", type=float, default=None,
                   help="SymLog linear threshold for deltaAy/(B0 de); default is 1e-3 of final max")
    return p.parse_args()


def numeric_key(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def step_from_path(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    if not m:
        return -1
    n = int(m.group(1))
    # diag names are normally diag1XXXXXX, so strip the leading diagnostic id.
    s = m.group(1)
    return int(s[1:]) if len(s) > 1 and s.startswith("1") else n


def spectrum_core(bz, z, modes, core_half_width, B0):
    core = np.abs(z) <= core_half_width
    ft = np.fft.rfft(bz, axis=0) / bz.shape[0]
    out = np.full(len(modes), np.nan)
    for i, m in enumerate(modes):
        if m >= ft.shape[0]:
            continue
        local = 2.0 * np.abs(ft[m, :]) / B0
        out[i] = np.sqrt(np.mean(local[core] ** 2))
    return out


def load_derived(path, P, zoom_de):
    F = load_fields(path)
    Ay = reconstruct_Ay(F["Bx"], F["Bz"], F["x"], F["z"])
    dAy = Ay - np.mean(Ay, axis=0, keepdims=True)
    jy = corrected_jy(F["Bx"], F["Bz"], F["dx"], F["dz"])
    xde = F["x"] / P["de"]
    zde = F["z"] / P["de"]
    zoom = np.abs(zde) <= zoom_de
    return F, Ay, dAy, jy, xde, zde, zoom


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params()
    modes = [int(x) for x in args.modes.split(",") if x.strip()]

    files_all = sorted(
        glob.glob(str(Path(args.run_dir) / "diags" / "diag1*")),
        key=numeric_key,
    )
    if not files_all:
        raise FileNotFoundError(f"No diag1* plotfiles under {args.run_dir}/diags")

    # First pass: time coordinate and mode history.  Keeping this independent
    # from the animation update makes the bottom panel physically meaningful
    # and fixes its axes throughout the movie.
    times = []
    steps = []
    mode_hist = []
    kept_files = []
    print("Pre-scanning plotfiles for mode history...")
    for k, fn in enumerate(files_all, 1):
        F = load_fields(fn)
        tci = P["wci"] * F["ds"].current_time.to_value("s")
        if tci < args.tmin:
            continue
        if args.tmax is not None and tci > args.tmax:
            continue
        kept_files.append(fn)
        times.append(tci)
        steps.append(step_from_path(fn))
        mode_hist.append(
            spectrum_core(
                F["Bz"], F["z"], modes,
                args.core_width_de * P["de"], P["B0"],
            )
        )
        if len(kept_files) % 10 == 0 or k == len(files_all):
            print(f"  retained {len(kept_files)} frames; latest omega_ci*t={tci:.4f}")

    if not kept_files:
        raise RuntimeError("No frames remain after tmin/tmax filtering")

    kept_files = kept_files[::max(1, args.stride)]
    times = np.asarray(times)[::max(1, args.stride)]
    steps = np.asarray(steps)[::max(1, args.stride)]
    mode_hist = np.asarray(mode_hist)[::max(1, args.stride)]

    # Use the final state to define fixed movie color scales.  A SymLog norm
    # prevents the early linear stage from disappearing once islands become
    # nonlinear by omega_ci*t ~ 1.
    Ff, Ayf, dAyf, jyf, xdef, zdef, zoomf = load_derived(
        kept_files[-1], P, args.zoom_de
    )
    jy_final = jyf[:, zoomf] / P["J0"]
    jy_hi = max(1.0, float(np.nanpercentile(jy_final, 99.8)))
    jy_lo = min(-0.05 * jy_hi, float(np.nanpercentile(jy_final, 0.2)))
    jy_norm = Normalize(vmin=jy_lo, vmax=jy_hi)

    d_final = dAyf[:, zoomf] / (P["B0"] * P["de"])
    day_abs = max(1e-6, float(np.nanmax(np.abs(d_final))))
    linthresh = args.day_linthresh
    if linthresh is None:
        linthresh = max(1e-5, day_abs * 1e-3)
    day_norm = SymLogNorm(
        linthresh=linthresh, linscale=1.0,
        vmin=-day_abs, vmax=day_abs, base=10,
    )

    # Figure: two topology views above, modal time history below.
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.72], hspace=0.30, wspace=0.25)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_day = fig.add_subplot(gs[0, 1])
    ax_mode = fig.add_subplot(gs[1, :])

    sm_j = ScalarMappable(norm=jy_norm, cmap="viridis")
    sm_j.set_array([])
    cb_j = fig.colorbar(sm_j, ax=ax_top, fraction=0.046, pad=0.04)
    cb_j.set_label(r"$J_y/J_0$")

    sm_a = ScalarMappable(norm=day_norm, cmap="coolwarm")
    sm_a.set_array([])
    cb_a = fig.colorbar(sm_a, ax=ax_day, fraction=0.046, pad=0.04)
    cb_a.set_label(r"$\delta A_y/(B_0d_e)$")

    amp_floor = max(1e-7, float(np.nanmin(mode_hist[mode_hist > 0])) * 0.5)
    amp_ceil = max(1.0, float(np.nanmax(mode_hist)) * 1.5)

    def draw_frame(i):
        F, Ay, dAy, jy, xde, zde, zoom = load_derived(
            kept_files[i], P, args.zoom_de
        )
        X, Z = np.meshgrid(xde, zde, indexing="ij")
        jyn = jy / P["J0"]
        dan = dAy / (P["B0"] * P["de"])

        ax_top.clear()
        ax_day.clear()
        ax_mode.clear()

        # Full topology: current background plus A_y magnetic-flux contours.
        ax_top.pcolormesh(
            X[:, zoom], Z[:, zoom], jyn[:, zoom],
            shading="auto", cmap="viridis", norm=jy_norm,
        )
        alo = float(np.nanmin(Ay[:, zoom]))
        ahi = float(np.nanmax(Ay[:, zoom]))
        if ahi > alo:
            ax_top.contour(
                X[:, zoom], Z[:, zoom], Ay[:, zoom],
                levels=np.linspace(alo, ahi, args.contours),
                colors="k", linewidths=0.45, alpha=0.85,
            )
        ax_top.set_xlabel(r"$x/d_e$")
        ax_top.set_ylabel(r"$z/d_e$")
        ax_top.set_title("full topology: $J_y/J_0$ + $A_y$ contours")

        # Perturbation topology with a fixed symmetric-log normalization.
        ax_day.pcolormesh(
            X[:, zoom], Z[:, zoom], dan[:, zoom],
            shading="auto", cmap="coolwarm", norm=day_norm,
        )
        vmax_frame = float(np.nanmax(np.abs(dan[:, zoom])))
        if vmax_frame > 0:
            # Contours follow the instantaneous amplitude so the linear stage
            # remains geometrically visible while the colors retain a global
            # fixed physical normalization.
            lev = np.linspace(-vmax_frame, vmax_frame, args.contours)
            ax_day.contour(
                X[:, zoom], Z[:, zoom], dan[:, zoom],
                levels=lev, colors="k", linewidths=0.42, alpha=0.65,
            )
        ax_day.set_xlabel(r"$x/d_e$")
        ax_day.set_ylabel(r"$z/d_e$")
        ax_day.set_title(r"perturbation topology: $\delta A_y$")

        # Modal evolution up to the current frame.
        for j, m in enumerate(modes):
            ax_mode.semilogy(
                times[:i+1], mode_hist[:i+1, j], "o-", ms=3, lw=1.5,
                label=f"m={m}",
            )
        ax_mode.axvline(times[i], color="k", ls="--", lw=1.0, alpha=0.7)
        ax_mode.set_xlim(times[0], times[-1] if times[-1] > times[0] else times[0] + 1)
        ax_mode.set_ylim(amp_floor, amp_ceil)
        ax_mode.set_xlabel(r"$\omega_{ci}t$")
        ax_mode.set_ylabel(r"core $B_z$ Fourier amplitude / $B_0$")
        ax_mode.grid(alpha=0.25)
        ax_mode.legend(ncol=5, fontsize=8, loc="upper left")

        row = mode_hist[i]
        finite = np.isfinite(row)
        if np.any(finite):
            jf = np.flatnonzero(finite)[np.argmax(row[finite])]
            dom = modes[jf]
            dom_amp = row[jf]
            extra = fr"; dominant $m={dom}$, $A_m/B_0={dom_amp:.3g}$"
        else:
            extra = ""

        fig.suptitle(
            fr"Ishizawa-scale tearing evolution: $\omega_{{ci}}t={times[i]:.3f}$"
            + extra,
            fontsize=15,
        )
        print(
            f"frame {i+1:3d}/{len(kept_files)}  step={steps[i]:6d}  "
            f"omega_ci*t={times[i]:.4f}"
        )
        return []

    ani = animation.FuncAnimation(
        fig, draw_frame, frames=len(kept_files), interval=1000.0/args.fps,
        blit=False, repeat=True,
    )

    out = Path(args.output)
    if out.suffix.lower() == ".mp4":
        try:
            writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400)
            ani.save(out, writer=writer, dpi=args.dpi)
        except Exception as exc:
            raise RuntimeError(
                "MP4 writing failed. Install ffmpeg or use --output something.gif"
            ) from exc
    else:
        writer = animation.PillowWriter(fps=args.fps)
        ani.save(out, writer=writer, dpi=args.dpi)

    plt.close(fig)
    print(f"Saved animation: {out}")
    print(f"frames={len(kept_files)}, omega_ci*t=[{times[0]:.4f}, {times[-1]:.4f}]")
    print(f"fixed Jy scale=[{jy_lo:.3g}, {jy_hi:.3g}] J0")
    print(f"fixed deltaAy scale=+/-{day_abs:.3g} B0 de, linthresh={linthresh:.3g}")


if __name__ == "__main__":
    main()
