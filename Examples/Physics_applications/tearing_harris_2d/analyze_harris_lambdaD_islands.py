#!/usr/bin/env python3
"""Automatic O/X-point and reconnection-flux diagnostics for the
500 lambda_D box / 40 lambda_D Harris-sheet run.

The detector uses the midplane x-dependent flux delta A_y.  Because the
Harris equilibrium A_y is x-independent at z=0, its x extrema are the same as
those of the full A_y on the midplane.  A Fourier low-pass is applied only for
topology detection so PIC cell-scale noise is not counted as islands.
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_harris_lambdaD_early import params, load_fields, reconstruct_Ay


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--box-lambdaD", type=float, default=500.0)
    p.add_argument("--sheet-lambdaD", type=float, default=40.0)
    p.add_argument("--tmin", type=float, default=1.0)
    p.add_argument("--topology-max-mode", type=int, default=8)
    p.add_argument("--relative-flux-threshold", type=float, default=0.15)
    p.add_argument("--absolute-flux-threshold", type=float, default=2e-3,
                   help="in units B0*L0")
    p.add_argument("--min-separation-lambdaD", type=float, default=20.0)
    p.add_argument("--zoom-sheet", type=float, default=2.0)
    return p.parse_args()


def numeric_key(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def lowpass_periodic(y, max_mode):
    ft = np.fft.rfft(np.asarray(y, dtype=float))
    if max_mode + 1 < len(ft):
        ft[max_mode + 1:] = 0.0
    return np.fft.irfft(ft, n=len(y))


def periodic_extrema(y):
    ym = np.roll(y, 1)
    yp = np.roll(y, -1)
    maxima = np.flatnonzero((y >= ym) & (y >= yp) & ((y > ym) | (y > yp)))
    minima = np.flatnonzero((y <= ym) & (y <= yp) & ((y < ym) | (y < yp)))
    return maxima, minima


def prev_next(indices, i, n):
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return None, None
    dl = (i - indices) % n
    dr = (indices - i) % n
    dl[dl == 0] = n
    dr[dr == 0] = n
    return int(indices[np.argmin(dl)]), int(indices[np.argmin(dr)])


def pdist(xa, xb, Lx):
    d = abs(float(xa) - float(xb))
    return min(d, Lx - d)


def detect(Ay, x, z, B0, L0, max_mode, rel_thr, abs_thr, min_sep):
    j0 = int(np.argmin(np.abs(z)))
    amid = Ay[:, j0] - np.mean(Ay[:, j0])
    af = lowpass_periodic(amid, max_mode)
    maxima, minima = periodic_extrema(af)
    nx = len(x)
    dx = x[1] - x[0]
    Lx = (x[-1] - x[0]) + dx

    raw = []
    for io in maxima:
        il, ir = prev_next(minima, int(io), nx)
        if il is None:
            continue
        left = (af[io] - af[il]) / (B0 * L0)
        right = (af[io] - af[ir]) / (B0 * L0)
        psi = min(left, right)
        if psi > 0:
            raw.append(dict(io=int(io), il=il, ir=ir, psi=float(psi)))

    strongest = max((q["psi"] for q in raw), default=0.0)
    threshold = max(abs_thr, rel_thr * strongest)
    cand = sorted([q for q in raw if q["psi"] >= threshold],
                  key=lambda q: q["psi"], reverse=True)

    accepted = []
    for q in cand:
        if all(pdist(x[q["io"]], x[r["io"]], Lx) >= min_sep for r in accepted):
            accepted.append(q)
    accepted.sort(key=lambda q: x[q["io"]])
    xidx = sorted({q["il"] for q in accepted} | {q["ir"] for q in accepted})
    return j0, af, accepted, np.asarray(xidx, dtype=int), threshold


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params(args.sheet_lambdaD)
    files = sorted(glob.glob(str(Path(args.run_dir)/"diags"/"diag1*")),
                   key=numeric_key)
    if not files:
        raise FileNotFoundError("No diag1* plotfiles found")

    rows = []
    final = None
    min_sep = args.min_separation_lambdaD * P["lambdaD"]

    for fn in files:
        F = load_fields(fn)
        t = P["wci"] * F["ds"].current_time.to_value("s")
        if t < args.tmin:
            continue
        Ay = reconstruct_Ay(F["Bx"], F["Bz"], F["x"], F["z"])
        j0, af, islands, xidx, thr = detect(
            Ay, F["x"], F["z"], P["B0"], P["L"],
            args.topology_max_mode, args.relative_flux_threshold,
            args.absolute_flux_threshold, min_sep
        )
        psis = np.array([q["psi"] for q in islands], dtype=float)
        psi_max = float(psis.max()) if len(psis) else 0.0
        psi_mean = float(psis.mean()) if len(psis) else 0.0

        ft = np.fft.rfft(F["Bz"], axis=0) / F["Bz"].shape[0]
        core = np.abs(F["z"]) <= P["L"]
        amps = []
        for m in range(1, min(20, ft.shape[0]-1)+1):
            local = 2*np.abs(ft[m, :])/P["B0"]
            amps.append(np.sqrt(np.mean(local[core]**2)))
        amps = np.asarray(amps)
        mdom = int(np.argmax(amps)+1) if len(amps) else -1
        pwr = amps**2
        modes = np.arange(1, len(amps)+1)
        mbar = float(np.sum(modes*pwr)/np.sum(pwr)) if np.sum(pwr)>0 else np.nan

        rows.append((t, len(islands), len(xidx), mdom, mbar, psi_max, psi_mean, thr))
        final = (F, Ay, j0, islands, xidx)
        print(f"wci*t={t:.4f}  N_O={len(islands):2d}  N_X={len(xidx):2d}  "
              f"m_dom={mdom:2d}  <m>={mbar:.3f}  psi_max={psi_max:.4e}")

    if not rows:
        raise RuntimeError("No frames satisfy --tmin")

    arr = np.asarray(rows, dtype=float)
    np.savetxt(
        "harris_500ld_40ld_island_history.txt", arr,
        header="omega_ci_t N_O N_X m_dom m_bar psi_max_B0L psi_mean_B0L threshold_B0L"
    )

    t, NO, NX, md, mb, pmax, pmean, thr = arr.T
    fig, ax = plt.subplots(figsize=(9,5.7))
    ax.step(t, NO, where="mid", marker="o", label=r"$N_O$")
    ax.step(t, NX, where="mid", marker="s", label=r"$N_X$")
    ax.plot(t, md, "o-", label=r"$m_{dom}$")
    ax.plot(t, mb, "o-", label=r"$\bar m$")
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel("count / mode")
    ax.grid(alpha=.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig("harris_500ld_40ld_island_count_vs_modes.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5,5.5))
    ax.semilogy(t, np.maximum(pmax,1e-12), "o-", label=r"$\Psi_{max}/(B_0L_0)$")
    ax.semilogy(t, np.maximum(pmean,1e-12), "s-", label=r"$\langle\Psi\rangle/(B_0L_0)$")
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel("reconnected/island flux")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("harris_500ld_40ld_reconnected_flux.png", dpi=200)
    plt.close(fig)

    F, Ay, j0, islands, xidx = final
    xld = F["x"]/P["lambdaD"]
    zld = F["z"]/P["lambdaD"]
    zoom = np.abs(zld) <= args.zoom_sheet*args.sheet_lambdaD
    X,Z = np.meshgrid(xld,zld,indexing="ij")
    dAy = Ay - np.mean(Ay,axis=0,keepdims=True)
    dN = dAy/(P["B0"]*P["L"])
    vmax = np.nanpercentile(np.abs(dN[:,zoom]),99.5)

    fig,ax = plt.subplots(figsize=(11,5.7))
    pcm=ax.pcolormesh(X[:,zoom],Z[:,zoom],dN[:,zoom],shading="auto",
                      cmap="RdBu_r",vmin=-vmax,vmax=vmax)
    arrA=Ay[:,zoom]/(P["B0"]*P["L"])
    lo,hi=np.nanpercentile(arrA,[1,99])
    if hi>lo:
        ax.contour(X[:,zoom],Z[:,zoom],arrA,levels=np.linspace(lo,hi,45),
                   colors="k",linewidths=.5,alpha=.7)
    if islands:
        io=[q["io"] for q in islands]
        ax.scatter(xld[io],np.full(len(io),zld[j0]),s=80,facecolors="none",
                   edgecolors="orange",linewidths=2,label="O")
    if len(xidx):
        ax.scatter(xld[xidx],np.full(len(xidx),zld[j0]),s=70,marker="x",
                   color="cyan",linewidths=2,label="X")
    fig.colorbar(pcm,ax=ax,label=r"$\delta A_y/(B_0L_0)$")
    ax.set_xlabel(r"$x/\lambda_D$")
    ax.set_ylabel(r"$z/\lambda_D$")
    ax.set_title(fr"annotated O/X topology, $\omega_{{ci}}t={t[-1]:.3f}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig("harris_500ld_40ld_final_OX_topology.png",dpi=220)
    plt.close(fig)

    with open("harris_500ld_40ld_island_summary.txt","w") as f:
        f.write("Harris 500 lambda_D / 40 lambda_D island diagnostics\n")
        f.write("===================================================\n\n")
        f.write(f"frames = {len(rows)}\n")
        f.write(f"omega_ci*t range = {t[0]:.8f} .. {t[-1]:.8f}\n")
        f.write(f"topology max mode = {args.topology_max_mode}\n")
        f.write(f"final N_O = {int(NO[-1])}\n")
        f.write(f"final N_X = {int(NX[-1])}\n")
        f.write(f"final m_dom = {int(md[-1])}\n")
        f.write(f"final m_bar = {mb[-1]:.8f}\n")
        f.write(f"final psi_max/(B0 L0) = {pmax[-1]:.8e}\n")
        f.write(f"final psi_mean/(B0 L0) = {pmean[-1]:.8e}\n")
        f.write(f"final threshold/(B0 L0) = {thr[-1]:.8e}\n")

    print("Saved island history/summary and annotated topology.")


if __name__ == "__main__":
    main()
