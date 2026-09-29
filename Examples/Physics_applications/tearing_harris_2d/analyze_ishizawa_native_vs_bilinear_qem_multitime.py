#!/usr/bin/env python3
"""Multi-time WarpX-native vs cell-centered bilinear Q_EM comparison.

Imports the validated single-snapshot helpers and reconstructs the fixed mode
once.  It then compares native and bilinear gathers on every probe plotfile
matching --probe-glob, avoiding repeated reads of the two full dense crossing
snapshots.

Outputs
-------
<out_prefix>_history.txt
<out_prefix>_summary.txt
<out_prefix>.png
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import QE, ME, params
from analyze_ishizawa_nonaffine_breathing_mode import step_from_parent
from analyze_ishizawa_native_vs_bilinear_qem import (
    build_mode, load_field_mesh_only, species_compare
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--probe-glob",required=True)
    p.add_argument("--phase-root",required=True)
    p.add_argument("--em-history",required=True)
    p.add_argument("--period",type=float,required=True)
    p.add_argument("--tmin",type=float,default=4.05,
                   help="time-window lower bound used to normalize saturation width into q")
    p.add_argument("--tmax",type=float,default=5.95,
                   help="time-window upper bound used to normalize saturation width into q")
    p.add_argument("--mode-coarsen",type=int,default=4)
    p.add_argument("--core-z-de",type=float,default=12.0)
    p.add_argument("--mode-density-cut",type=float,default=0.02)
    p.add_argument("--smooth-passes",type=int,default=2)
    p.add_argument("--x-edge-taper-de",type=float,default=0.0)
    p.add_argument("--sample-fraction",type=float,default=0.02)
    p.add_argument("--out-prefix",default="ishizawa_native_vs_bilinear_qem_multitime")
    return p.parse_args()


def main():
    args=parse_args()
    yt.funcs.mylog.setLevel(40)
    P=params()

    probes=sorted(
        [p for p in glob.glob(args.probe_glob) if os.path.isdir(p)],
        key=step_from_parent
    )
    if not probes:
        raise RuntimeError(f"no probe plotfiles matched {args.probe_glob!r}")

    Mmode,xix,xiz,crossings=build_mode(args,P)

    rows=[]
    field_names=["Ex","Ey","Ez","Bx","By","Bz"]
    field_diff_ion=[]
    field_diff_ele=[]

    for ip,probe in enumerate(probes,1):
        Mfield=load_field_mesh_only(probe)
        ds=yt.load(probe)
        tau=float(P["wci"]*ds.current_time.to_value("s"))
        step=step_from_parent(probe)

        ion=species_compare(probe,"ions",P["mi"],+QE,Mfield,Mmode,xix,xiz)
        ele=species_compare(probe,"electrons",ME,-QE,Mfield,Mmode,xix,xiz)

        qn_i=ion["Qn"]/P["wci"]; qc_i=ion["Qc"]/P["wci"]
        qn_e=ele["Qn"]/P["wci"]; qc_e=ele["Qc"]/P["wci"]
        qn=qn_i+qn_e; qc=qc_i+qc_e

        qne=(ion["Qn_E"]+ele["Qn_E"])/P["wci"]
        qce=(ion["Qc_E"]+ele["Qc_E"])/P["wci"]
        qnb=(ion["Qn_B"]+ele["Qn_B"])/P["wci"]
        qcb=(ion["Qc_B"]+ele["Qc_B"])/P["wci"]

        def rr(a,b):
            return a/b if abs(b)>np.finfo(float).tiny else np.nan

        rows.append([
            step,tau,
            qn_i,qc_i,rr(qc_i,qn_i),
            qn_e,qc_e,rr(qc_e,qn_e),
            qn,qc,rr(qc,qn),
            qne,qce,qnb,qcb,
            ion["N"],ele["N"]
        ])
        field_diff_ion.append([ion["fields"][k]["rel_wrms"] for k in field_names])
        field_diff_ele.append([ele["fields"][k]["rel_wrms"] for k in field_names])

        print(
            f"[{ip:02d}/{len(probes):02d}] step={step} tau={tau:.6f} "
            f"ratio total={rr(qc,qn):+.8f} ion={rr(qc_i,qn_i):+.8f} "
            f"electron={rr(qc_e,qn_e):+.8f}"
        )

    A=np.asarray(rows,float)
    Fi=np.asarray(field_diff_ion,float)
    Fe=np.asarray(field_diff_ele,float)

    np.savetxt(
        args.out_prefix+"_history.txt",A,
        header=(
            "step omega_ci_t "
            "Qnative_i Qbilinear_i ratio_i "
            "Qnative_e Qbilinear_e ratio_e "
            "Qnative_total Qbilinear_total ratio_total "
            "QE_native_total QE_bilinear_total "
            "QvXB_native_total QvXB_bilinear_total "
            "Nion Nelectron"
        )
    )

    good=np.isfinite(A[:,10])
    diff=A[good,10]-1.0
    with open(args.out_prefix+"_summary.txt","w") as f:
        f.write("Multi-time native vs bilinear Q_EM gather comparison\n")
        f.write("===================================================\n\n")
        f.write(f"probe count = {len(A)}\n")
        f.write(f"tau range = {A[:,1].min():.8f} .. {A[:,1].max():.8f}\n")
        f.write(f"mode crossings = {crossings[0]['step']} {crossings[1]['step']}\n")
        f.write(f"sample fraction = {args.sample_fraction:.8f}\n\n")
        for label,col in [("ion",4),("electron",7),("total",10)]:
            x=A[:,col]
            g=np.isfinite(x)
            d=x[g]-1
            f.write(
                f"{label} bilinear/native min/median/max = "
                f"{x[g].min():.8f} {np.median(x[g]):.8f} {x[g].max():.8f}\n"
            )
            f.write(
                f"{label} |bilinear/native-1| median/max = "
                f"{np.median(np.abs(d)):.8e} {np.max(np.abs(d)):.8e}\n"
            )
        f.write("\nfield relWRMSdiff median/max across time\n")
        for j,k in enumerate(field_names):
            f.write(
                f"{k}: ion {np.median(Fi[:,j]):.8e} {np.max(Fi[:,j]):.8e}; "
                f"electron {np.median(Fe[:,j]):.8e} {np.max(Fe[:,j]):.8e}\n"
            )

    t=A[:,1]
    fig,axs=plt.subplots(2,1,figsize=(9,8),sharex=True)
    axs[0].plot(t,A[:,4]-1,"o-",label="ion")
    axs[0].plot(t,A[:,7]-1,"o-",label="electron")
    axs[0].plot(t,A[:,10]-1,"o-",label="total")
    axs[0].axhline(0,lw=.8)
    axs[0].set_ylabel("bilinear/native - 1")
    axs[0].set_title("Native-vs-bilinear electromagnetic projection")
    axs[0].grid(alpha=.25); axs[0].legend()

    for j,k in enumerate(field_names):
        axs[1].plot(t,Fi[:,j],"o-",label=f"ion {k}")
    axs[1].set_xlabel(r"$\omega_{ci}t$")
    axs[1].set_ylabel("field relWRMS difference")
    axs[1].grid(alpha=.25)
    axs[1].legend(ncol=3,fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_prefix+".png",dpi=210)
    plt.close(fig)

    print("Saved",args.out_prefix+"_summary.txt")
    print("Saved",args.out_prefix+"_history.txt")
    print("Saved",args.out_prefix+".png")


if __name__=="__main__":
    main()
