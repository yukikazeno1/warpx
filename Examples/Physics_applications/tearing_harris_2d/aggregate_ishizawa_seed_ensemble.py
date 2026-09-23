#!/usr/bin/env python3
"""
Aggregate paired random-seed collisionless/collisional comparison summaries.

Expected layout:
  runs_ishizawa_scale_ensemble/
    seed000101/comparison/collision_compare_summary.txt
    seed000202/comparison/collision_compare_summary.txt
    ...

Produces:
  ensemble_growth_statistics.txt
  ensemble_scalar_statistics.txt
  ensemble_delta_gamma.png
  ensemble_flux_ratio.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs_ishizawa_scale_ensemble")
    p.add_argument("--seeds", default="101 202 303 404 505 606")
    p.add_argument("--max-mode", type=int, default=12)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def parse_summary(path, max_mode):
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    modes = {}
    scalars = {}

    in_modes = False
    in_scalars = False

    for line in lines:
        s = line.strip()
        if s.startswith("mode gamma_no"):
            in_modes = True
            in_scalars = False
            continue
        if s.startswith("Final scalar comparison"):
            in_modes = False
            in_scalars = True
            continue

        if in_modes and s:
            parts = s.split()
            if len(parts) >= 13 and parts[0].isdigit():
                m = int(parts[0])
                if m <= max_mode:
                    vals = [float(x) for x in parts[1:]]
                    modes[m] = {
                        "gamma_no": vals[0],
                        "gamma_coll": vals[1],
                        "delta_gamma": vals[2],
                        "R2_no": vals[3],
                        "R2_coll": vals[4],
                        "final_A_ratio": vals[5],
                        "dphi_raw": vals[6],
                        "dphi_aligned": vals[7],
                        "C_no": vals[8],
                        "C_coll": vals[9],
                        "Peven_no": vals[10],
                        "Peven_coll": vals[11],
                    }

        if in_scalars and s:
            # Example:
            # Psi        no=... coll=... delta=...
            m = re.match(
                r"(\S+)\s+no=([+\-0-9.eE]+)\s+coll=([+\-0-9.eE]+)\s+delta=([+\-0-9.eE]+)",
                s,
            )
            if m:
                key = m.group(1)
                scalars[key] = {
                    "no": float(m.group(2)),
                    "coll": float(m.group(3)),
                    "delta": float(m.group(4)),
                }

    return modes, scalars


def mean_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    mean = np.mean(x)
    sd = np.std(x, ddof=1) if len(x) > 1 else 0.0
    sem = sd / np.sqrt(len(x)) if len(x) > 0 else np.nan
    return mean, sd, sem, len(x)


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    out = Path(args.output_dir).resolve() if args.output_dir else root/"ensemble_statistics"
    out.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split()]
    loaded = []

    for seed in seeds:
        tag = f"{seed:06d}"
        summary = root/f"seed{tag}"/"comparison"/"collision_compare_summary.txt"
        if not summary.exists():
            print(f"WARNING: missing {summary}")
            continue
        modes, scalars = parse_summary(summary, args.max_mode)
        loaded.append((seed, modes, scalars))
        print(f"Loaded seed {seed}: {summary}")

    if not loaded:
        raise RuntimeError("No seed comparison summaries found")

    # Growth statistics
    with open(out/"ensemble_growth_statistics.txt", "w", encoding="utf-8") as f:
        f.write("Paired random-seed tearing growth statistics\n")
        f.write("===========================================\n")
        f.write(f"Nseed_loaded = {len(loaded)}\n\n")
        f.write(
            "mode N "
            "gamma_no_mean gamma_no_sd "
            "gamma_coll_mean gamma_coll_sd "
            "delta_gamma_mean delta_gamma_sd delta_gamma_sem "
            "R2_no_mean R2_coll_mean\n"
        )

        growth_rows = []
        for m in range(1, args.max_mode+1):
            g0=[]; g1=[]; dg=[]; r0=[]; r1=[]
            for _, modes, _ in loaded:
                if m not in modes:
                    continue
                q=modes[m]
                g0.append(q["gamma_no"])
                g1.append(q["gamma_coll"])
                dg.append(q["delta_gamma"])
                r0.append(q["R2_no"])
                r1.append(q["R2_coll"])

            a0,s0,_,n0=mean_sem(g0)
            a1,s1,_,_=mean_sem(g1)
            ad,sd,semd,_=mean_sem(dg)
            ar0,_,_,_=mean_sem(r0)
            ar1,_,_,_=mean_sem(r1)

            f.write(
                f"{m:3d} {n0:3d} "
                f"{a0:.8e} {s0:.8e} "
                f"{a1:.8e} {s1:.8e} "
                f"{ad:.8e} {sd:.8e} {semd:.8e} "
                f"{ar0:.6f} {ar1:.6f}\n"
            )
            growth_rows.append((m,ad,sd,semd,n0))

    # Scalar statistics
    keys = sorted(set().union(*(scalars.keys() for _,_,scalars in loaded)))
    with open(out/"ensemble_scalar_statistics.txt", "w", encoding="utf-8") as f:
        f.write("Paired random-seed final scalar statistics\n")
        f.write("==========================================\n")
        f.write(f"Nseed_loaded = {len(loaded)}\n\n")
        f.write("quantity N no_mean no_sd coll_mean coll_sd delta_mean delta_sd delta_sem ratio_mean ratio_sd\n")

        scalar_stats = {}
        for key in keys:
            a=[]; b=[]; d=[]; r=[]
            for _,_,scalars in loaded:
                if key not in scalars:
                    continue
                q=scalars[key]
                a.append(q["no"]); b.append(q["coll"]); d.append(q["delta"])
                if q["no"] != 0:
                    r.append(q["coll"]/q["no"])
            am,asd,_,n=mean_sem(a)
            bm,bsd,_,_=mean_sem(b)
            dm,dsd,dsem,_=mean_sem(d)
            rm,rsd,_,_=mean_sem(r)
            scalar_stats[key]=(rm,rsd)
            f.write(
                f"{key:12s} {n:3d} "
                f"{am:.8e} {asd:.8e} "
                f"{bm:.8e} {bsd:.8e} "
                f"{dm:.8e} {dsd:.8e} {dsem:.8e} "
                f"{rm:.8e} {rsd:.8e}\n"
            )

    # Plot delta gamma seed scatter + ensemble mean/SD
    fig, ax = plt.subplots(figsize=(10,6))
    modes = np.arange(1,args.max_mode+1)

    for seed, mdict, _ in loaded:
        y = np.array([mdict[m]["delta_gamma"] if m in mdict else np.nan for m in modes])
        ax.plot(modes, y, "o-", alpha=.35, label=f"seed {seed}")

    means=[]; sds=[]
    for m in modes:
        vals=[
            mdict[m]["delta_gamma"]
            for _,mdict,_ in loaded
            if m in mdict and np.isfinite(mdict[m]["delta_gamma"])
        ]
        mean,sd,_,_=mean_sem(vals)
        means.append(mean); sds.append(sd)

    means=np.asarray(means); sds=np.asarray(sds)
    ax.errorbar(modes,means,yerr=sds,fmt="ko-",lw=2,capsize=4,label="ensemble mean ± SD")
    ax.axhline(0,color="k",lw=.8)
    ax.set_xlabel("mode m")
    ax.set_ylabel(r"$\Delta\gamma/\omega_{ci}$")
    ax.set_xticks(modes)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8,ncol=2)
    fig.tight_layout()
    fig.savefig(out/"ensemble_delta_gamma.png",dpi=200)
    plt.close(fig)

    # Plot final flux ratio per seed if available.
    if all("Psi" in scalars for _,_,scalars in loaded):
        seed_x=[]
        ratio=[]
        for seed,_,scalars in loaded:
            no=scalars["Psi"]["no"]
            seed_x.append(seed)
            ratio.append(scalars["Psi"]["coll"]/no if no != 0 else np.nan)

        fig,ax=plt.subplots(figsize=(9,5.5))
        ax.plot(seed_x,ratio,"o")
        rm,rsd,_,_=mean_sem(ratio)
        ax.axhline(rm,ls="--",label=f"mean={rm:.3f}")
        ax.axhline(1.0,color="k",lw=.8)
        ax.set_xlabel("random seed")
        ax.set_ylabel(r"$\Psi_{coll}/\Psi_{nocoll}$")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out/"ensemble_flux_ratio.png",dpi=200)
        plt.close(fig)

    print()
    print(f"Loaded {len(loaded)} seed pairs.")
    print(f"Saved ensemble statistics to: {out}")


if __name__ == "__main__":
    main()
