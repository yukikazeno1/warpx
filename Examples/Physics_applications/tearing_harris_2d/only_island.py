#!/usr/bin/env python3
"""Animate magnetic topology and O/X-point tracking during island coalescence.

This version retains only the primary topology view:
- Corrected J_y/J0 background
- Reconstructed A_y magnetic flux contours (magnetic field lines)
- Tracked midplane O-points (island centers) and X-points (reconnection sites)
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs_ishizawa_scale/ppc49")
    p.add_argument("--output", default="ishizawa_island_topology.mp4")
    p.add_argument("--topology-max-mode", type=int, default=20)
    p.add_argument("--zoom-de", type=float, default=32.0)
    p.add_argument("--tmin", type=float, default=0.05)
    p.add_argument("--tmax", type=float, default=None)
    p.add_argument("--relative-flux-threshold", type=float, default=0.12)
    p.add_argument("--absolute-flux-threshold", type=float, default=1e-3)
    p.add_argument("--min-separation-de", type=float, default=2.0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dpi", type=int, default=120)
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
    """Load fields, reconstruct vector potential, and detect O/X points."""
    F = load_fields(path)
    Ay = reconstruct_Ay(F["Bx"], F["Bz"], F["x"], F["z"])
    jy = corrected_jy(F["Bx"], F["Bz"], F["dx"], F["dz"])
    det = detect_islands(
        Ay, F["x"], F["z"], P["B0"], P["de"],
        topology_max_mode=args.topology_max_mode,
        relative_threshold=args.relative_flux_threshold,
        absolute_threshold=args.absolute_flux_threshold,
        min_separation_de=args.min_separation_de,
    )
    return F, Ay, jy, det


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params()

    files_all = sorted(glob.glob(str(Path(args.run_dir) / "diags" / "diag1*")), key=numeric_key)
    if not files_all:
        raise FileNotFoundError(f"No diag1 plotfiles under {args.run_dir}/diags")

    # 轻量化预扫描：仅获取时间信息并根据 tmin/tmax 进行帧过滤
    files, times, steps = [], [], []
    print("Pre-scanning plotfiles for time filtering...")
    for fn in files_all:
        F = load_fields(fn)
        tci = P["wci"] * F["ds"].current_time.to_value("s")
        if tci < args.tmin:
            continue
        if args.tmax is not None and tci > args.tmax:
            continue
        files.append(fn)
        times.append(tci)
        steps.append(step_from_path(fn))
        if len(files) % 10 == 0 or fn == files_all[-1]:
            print(f"  retained {len(files)} frames; latest omega_ci*t={tci:.4f}")

    if not files:
        raise RuntimeError("No frames after tmin/tmax filtering")

    stride = max(1, args.stride)
    files = files[::stride]
    times = np.asarray(times)[::stride]
    steps = np.asarray(steps)[::stride]

    # 根据最后一帧确定全局固定的 Jy 颜色映射标尺
    Ff, Ayf, jyf, _ = load_frame(files[-1], P, args)
    zdef = Ff["z"] / P["de"]
    zoomf = np.abs(zdef) <= args.zoom_de
    jyfn = jyf[:, zoomf] / P["J0"]
    jy_hi = max(1.0, float(np.nanpercentile(jyfn, 99.8)))
    jy_lo = min(-0.05 * jy_hi, float(np.nanpercentile(jyfn, 0.2)))
    jy_norm = Normalize(vmin=jy_lo, vmax=jy_hi)

    # 建立单图画布（适合纵横比较大的磁流片几何）
    fig, ax = plt.subplots(figsize=(10, 5))
    smj = ScalarMappable(norm=jy_norm, cmap="viridis")
    smj.set_array([])
    cbar = fig.colorbar(smj, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"$J_y/J_0$", fontsize=11)

    def draw(i):
        F, Ay, jy, det = load_frame(files[i], P, args)
        xde = F["x"] / P["de"]
        zde = F["z"] / P["de"]
        zoom = np.abs(zde) <= args.zoom_de
        X, Z = np.meshgrid(xde, zde, indexing="ij")
        jyn = jy / P["J0"]

        ax.clear()

        # 1. 面外电流密度背景
        ax.pcolormesh(
            X[:, zoom], Z[:, zoom], jyn[:, zoom],
            shading="auto", cmap="viridis", norm=jy_norm,
        )

        # 2. 磁矢势 Ay 等值线（即磁力线）
        alo, ahi = float(np.nanmin(Ay[:, zoom])), float(np.nanmax(Ay[:, zoom]))
        if ahi > alo:
            ax.contour(
                X[:, zoom], Z[:, zoom], Ay[:, zoom],
                levels=np.linspace(alo, ahi, args.contours),
                colors="k", linewidths=0.45, alpha=0.75,
            )

        # 3. 标记 O 点（磁岛中心）与 X 点（重联点）
        j0 = det["j0"]
        if det["islands"]:
            io = [q["io"] for q in det["islands"]]
            ax.scatter(
                xde[io], np.full(len(io), zde[j0]),
                s=55, facecolors="none", edgecolors="tab:orange",
                linewidths=1.6, label="O-point",
            )
        if len(det["x_indices"]):
            ax.scatter(
                xde[det["x_indices"]], np.full(len(det["x_indices"]), zde[j0]),
                s=55, marker="x", color="tab:cyan",
                linewidths=1.8, label="X-point",
            )

        ax.set_xlabel(r"$x/d_e$", fontsize=11)
        ax.set_ylabel(r"$z/d_e$", fontsize=11)
        n_o = len(det["islands"])
        n_x = len(det["x_indices"])
        ax.set_title(
            fr"Full topology + tracked O/X points: $\omega_{{ci}}t={times[i]:.3f}$ "
            fr"($N_O={n_o}$, $N_X={n_x}$)",
            fontsize=12,
        )

        if det["islands"] or len(det["x_indices"]):
            ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

        print(
            f"frame {i+1:3d}/{len(files)} step={steps[i]:6d} "
            f"wci*t={times[i]:.4f} NO={n_o} NX={n_x}"
        )
        return []

    fig.tight_layout()
    ani = animation.FuncAnimation(
        fig, draw, frames=len(files), interval=1000.0 / args.fps,
        blit=False, repeat=True,
    )

    out = Path(args.output)
    if out.suffix.lower() == ".mp4":
        try:
            writer = animation.FFMpegWriter(fps=args.fps, bitrate=2600)
            ani.save(out, writer=writer, dpi=args.dpi)
        except Exception as exc:
            raise RuntimeError("MP4 writing failed; install ffmpeg or use a .gif output") from exc
    else:
        writer = animation.PillowWriter(fps=args.fps)
        ani.save(out, writer=writer, dpi=args.dpi)

    plt.close(fig)
    print(f"Saved animation to {out}")


if __name__ == "__main__":
    main()