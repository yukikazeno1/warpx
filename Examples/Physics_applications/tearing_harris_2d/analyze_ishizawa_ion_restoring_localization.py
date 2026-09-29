#!/usr/bin/env python3
"""Localize the q-matched ion pressure-like restoring stress relative to island topology.

This is a lightweight follow-up to analyze_ishizawa_ion_kinetic_decomposition.py.
It consumes the saved spatial-phasor NPZ and loads only the two q~0 crossing
plotfiles needed for magnetic topology.  No full particle arrays are reread.

It overlays O/X points and the large-scale separatrix from both crossings on
the restoring/damping maps and reports objective localization metrics:
- signed/absolute restoring budget in |z| bands;
- concentration of positive restoring contribution in the strongest cells;
- nearest-O versus nearest-X partition using periodic x distance;
- location of the strongest restoring/damping cells and their distance to O/X.

The topology definition matches analyze_ishizawa_saturation.py:
large-scale Ay with x Fourier modes 0..M retained.
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_ishizawa_scale_40000 import params, reconstruct_Ay
from analyze_ishizawa_particle_moments import load_mesh
from analyze_ishizawa_nonaffine_breathing_mode import step_from_parent
from analyze_ishizawa_saturation import lowpass_x, choose_system_scale_OX


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--npz",required=True)
    p.add_argument("--summary",required=True,
                   help="q-matched ion decomposition summary containing mode crossings")
    p.add_argument("--phase-root",required=True)
    p.add_argument("--mode-coarsen",type=int,default=4)
    p.add_argument("--large-scale-max-mode",type=int,default=3)
    p.add_argument("--out-prefix",default="ishizawa_ion_restoring_localization")
    return p.parse_args()


def parse_crossings(path):
    pat=re.compile(r"mode crossings\s*=\s*(\d+)\s+(\d+)")
    with open(path,"r") as f:
        for line in f:
            m=pat.search(line)
            if m:
                return int(m.group(1)),int(m.group(2))
    raise RuntimeError(f"could not parse mode crossings from {path}")


def find_plotfile(root,step):
    files=set(
        glob.glob(str(Path(root)/"step*"/"diags"/"diag1*"))
        + glob.glob(str(Path(root)/"diags"/"diag1*"))
    )
    matches=[p for p in files if step_from_parent(p)==step]
    if len(matches)!=1:
        raise RuntimeError(
            f"expected exactly one plotfile for step {step}; found {len(matches)}: {matches}"
        )
    return matches[0]


def periodic_dx(x,x0,L):
    return (x-x0+0.5*L)%L-0.5*L


def circular_mean_x(vals,xlo,xhi):
    vals=np.asarray(vals,float)
    L=xhi-xlo
    th=2*np.pi*(vals-xlo)/L
    z=np.mean(np.exp(1j*th))
    thm=np.angle(z)%(2*np.pi)
    return xlo+L*thm/(2*np.pi)


def topology_state(path,coarsen,max_mode,P):
    M=load_mesh(path,coarsen)
    Ay=reconstruct_Ay(M["Bx"],M["Bz"],M["x"],M["z"])
    Ay_lp=lowpass_x(Ay,max_mode)
    j0,io,ix,orient,AO,AX,psi=choose_system_scale_OX(Ay_lp,M["z"])
    return dict(
        M=M,Ay=Ay_lp,j0=j0,io=io,ix=ix,orient=orient,
        AO=AO,AX=AX,psi=psi,
        xO=M["x"][io]/P["de"],xX=M["x"][ix]/P["de"],
        sep=AX,
    )


def strongest_cell(a,x,z,kind="max"):
    ij=np.unravel_index(
        np.nanargmax(a) if kind=="max" else np.nanargmin(a),a.shape
    )
    return float(a[ij]),float(x[ij[0]]),float(z[ij[1]])


def main():
    args=parse_args()
    P=params()
    D=np.load(args.npz)
    x=np.asarray(D["x_de"],float)
    z=np.asarray(D["z_de"],float)
    rxx=np.asarray(D["rand_xx_restoring"],float)
    dxx=np.asarray(D["rand_xx_damping"],float)
    rtot=np.asarray(D["rand_total_restoring"],float)
    btot=np.asarray(D["bulk_total_restoring"],float)

    s1,s2=parse_crossings(args.summary)
    f1=find_plotfile(args.phase_root,s1)
    f2=find_plotfile(args.phase_root,s2)
    T1=topology_state(f1,args.mode_coarsen,args.large_scale_max_mode,P)
    T2=topology_state(f2,args.mode_coarsen,args.large_scale_max_mode,P)

    xlo=float(x.min()-0.5*np.median(np.diff(x)))
    xhi=float(x.max()+0.5*np.median(np.diff(x)))
    L=xhi-xlo
    xO=circular_mean_x([T1["xO"],T2["xO"]],xlo,xhi)
    xX=circular_mean_x([T1["xX"],T2["xX"]],xlo,xhi)

    X,Z=np.meshgrid(x,z,indexing="ij")
    dO=np.abs(periodic_dx(X,xO,L))
    dX=np.abs(periodic_dx(X,xX,L))
    Oside=dO<=dX
    Xside=~Oside

    # Basic localization metrics.
    signed=float(np.sum(rxx))
    pos=float(np.sum(np.clip(rxx,0,None)))
    neg=float(np.sum(np.clip(rxx,None,0)))
    ab=float(np.sum(np.abs(rxx)))

    zbands={}
    for zz in [1,2,3,4,5,8,12]:
        m=np.abs(z)<=zz
        zbands[zz]=(
            float(np.sum(rxx[:,m])),
            float(np.sum(np.abs(rxx[:,m]))/max(ab,1e-300))
        )

    positive=np.sort(np.clip(rxx,0,None).ravel())[::-1]
    csum=np.cumsum(positive)
    ptot=max(float(csum[-1]),1e-300)
    concentration={}
    for frac in [0.5,0.8,0.9,0.95]:
        n=int(np.searchsorted(csum,frac*ptot)+1)
        concentration[frac]=(n,n/rxx.size)

    O_signed=float(np.sum(rxx[Oside]))
    X_signed=float(np.sum(rxx[Xside]))
    O_abs=float(np.sum(np.abs(rxx[Oside])))
    X_abs=float(np.sum(np.abs(rxx[Xside])))

    rmax=strongest_cell(rxx,x,z,"max")
    rmin=strongest_cell(rxx,x,z,"min")
    dmax=strongest_cell(dxx,x,z,"max")
    dmin=strongest_cell(dxx,x,z,"min")

    def distances(xp,zp):
        return (
            float(np.hypot(periodic_dx(xp,xO,L),zp)),
            float(np.hypot(periodic_dx(xp,xX,L),zp)),
        )

    with open(args.out_prefix+"_summary.txt","w") as f:
        f.write("Ion restoring-stress spatial localization\n")
        f.write("========================================\n\n")
        f.write(f"crossings = {s1} {s2}\n")
        f.write(f"topology low-pass max mode = {args.large_scale_max_mode}\n")
        f.write(f"crossing1 O_x/de X_x/de = {T1['xO']:.8f} {T1['xX']:.8f}\n")
        f.write(f"crossing2 O_x/de X_x/de = {T2['xO']:.8f} {T2['xX']:.8f}\n")
        f.write(f"circular-mean O_x/de X_x/de = {xO:.8f} {xX:.8f}\n\n")

        f.write("random xx restoring map\n")
        f.write("-----------------------\n")
        f.write(f"signed sum / |Zkin| = {signed:.10e}\n")
        f.write(f"positive sum / |Zkin| = {pos:.10e}\n")
        f.write(f"negative sum / |Zkin| = {neg:.10e}\n")
        f.write(f"absolute sum / |Zkin| = {ab:.10e}\n")
        f.write(f"nearest-O signed / |Zkin| = {O_signed:.10e}\n")
        f.write(f"nearest-X signed / |Zkin| = {X_signed:.10e}\n")
        f.write(f"nearest-O absolute fraction = {O_abs/max(ab,1e-300):.8f}\n")
        f.write(f"nearest-X absolute fraction = {X_abs/max(ab,1e-300):.8f}\n")
        for zz,(ss,af) in zbands.items():
            f.write(f"|z|<= {zz:2d} de: signed={ss:.10e} abs_fraction={af:.8f}\n")
        for frac,(n,nfrac) in concentration.items():
            f.write(
                f"strongest positive cells for {100*frac:4.0f}% positive budget: "
                f"{n} cells = {100*nfrac:.4f}% of grid\n"
            )

        for label,val in [
            ("restoring maximum",rmax),("restoring minimum",rmin),
            ("damping maximum",dmax),("damping minimum",dmin)
        ]:
            dvO,dvX=distances(val[1],val[2])
            f.write(
                f"{label}: value={val[0]:+.10e} at "
                f"(x/de,z/de)=({val[1]:+.5f},{val[2]:+.5f}); "
                f"distance_to_O/de={dvO:.5f} distance_to_X/de={dvX:.5f}\n"
            )

        f.write("\nGlobal spatial sums\n")
        f.write(f"rand total restoring sum / |Zkin| = {np.sum(rtot):.10e}\n")
        f.write(f"bulk total restoring sum / |Zkin| = {np.sum(btot):.10e}\n")
        f.write(f"rand xx damping sum / |Zkin| = {np.sum(dxx):.10e}\n")

    # Overlay both q~0 separatrices and O/X points.
    def plot_overlay(arr,name,title):
        fig,ax=plt.subplots(figsize=(9.2,6.0))
        vmax=np.nanpercentile(np.abs(arr),99.5)
        im=ax.imshow(
            arr.T,origin="lower",
            extent=[xlo,xhi,z.min()-0.25,z.max()+0.25],
            aspect="auto",vmin=-vmax,vmax=vmax,cmap="RdBu_r"
        )
        for j,(Ts,ls) in enumerate([(T1,"-"),(T2,"--")],1):
            ay=Ts["Ay"]
            xx=Ts["M"]["x"]/P["de"]
            zz=Ts["M"]["z"]/P["de"]
            XX,ZZ=np.meshgrid(xx,zz,indexing="ij")
            ax.contour(
                XX,ZZ,ay,levels=[Ts["sep"]],
                colors="k",linewidths=1.1,linestyles=ls
            )
            ax.plot(Ts["xO"],0,marker="o",ms=6,mfc="none",mec="k")
            ax.plot(Ts["xX"],0,marker="x",ms=6,mec="k")
        ax.set_xlim(xlo,xhi)
        ax.set_ylim(-8,8)
        ax.set_xlabel(r"$x/d_e$")
        ax.set_ylabel(r"$z/d_e$")
        ax.set_title(title+"; contours: two q~0 separatrices")
        cb=fig.colorbar(im,ax=ax)
        cb.set_label(r"cell contribution / $|Z_{kin,i}|$")
        fig.tight_layout()
        fig.savefig(name,dpi=210)
        plt.close(fig)

    plot_overlay(
        rxx,args.out_prefix+"_rand_xx_restoring_topology.png",
        "Ion random/pressure-like xx restoring projection"
    )
    plot_overlay(
        rtot,args.out_prefix+"_rand_total_restoring_topology.png",
        "Ion random/pressure-like total restoring projection"
    )
    plot_overlay(
        dxx,args.out_prefix+"_rand_xx_damping_topology.png",
        "Ion random/pressure-like xx damping projection"
    )
    plot_overlay(
        btot,args.out_prefix+"_bulk_total_restoring_topology.png",
        "Ion bulk/advective total restoring projection"
    )

    print("Saved",args.out_prefix+"_summary.txt")
    print("Saved topology-overlay maps.")


if __name__=="__main__":
    main()
