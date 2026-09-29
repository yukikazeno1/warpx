#!/usr/bin/env python3
"""Flux-shell localization of the q-matched ion restoring stress.

Consumes the saved spatial phasors from the ion kinetic decomposition and
uses the two q~0 crossing magnetic topologies to answer a more precise
question than nearest-O/nearest-X distance:

    Is the dominant ion random/pressure-like xx restoring contribution
    located in the island core, mid-island flux shells, or close to the
    separatrix?

For each crossing we reconstruct the large-scale A_y, identify the connected
island region containing the O point (with periodic x connectivity), and define

    chi = (A_y - A_O)/(A_X - A_O),

so chi=0 at the O point and chi=1 at the separatrix.  Budgets are reported in
chi shells.  The script also recenters x periodically on the O point so an
island centered at the periodic seam is shown continuously rather than split
between the left and right edges.

No particle arrays are read.
"""

import argparse
import glob
import re
from collections import deque
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
    p.add_argument("--summary",required=True)
    p.add_argument("--phase-root",required=True)
    p.add_argument("--mode-coarsen",type=int,default=4)
    p.add_argument("--large-scale-max-mode",type=int,default=3)
    p.add_argument("--out-prefix",default="ishizawa_ion_restoring_flux_shells")
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
            f"expected one plotfile for step {step}; found {len(matches)}"
        )
    return matches[0]


def periodic_dx(x,x0,L):
    return (x-x0+0.5*L)%L-0.5*L


def topology(path,coarsen,max_mode,P):
    M=load_mesh(path,coarsen)
    Ay=lowpass_x(
        reconstruct_Ay(M["Bx"],M["Bz"],M["x"],M["z"]),
        max_mode
    )
    j0,io,ix,orient,AO,AX,psi=choose_system_scale_OX(Ay,M["z"])
    return dict(
        M=M,Ay=Ay,j0=j0,io=io,ix=ix,orient=orient,
        AO=AO,AX=AX,psi=psi,
        xO=M["x"][io]/P["de"],
        xX=M["x"][ix]/P["de"]
    )


def periodic_component(mask,seed):
    """4-neighbor connected component with periodic x and open z."""
    nx,nz=mask.shape
    out=np.zeros_like(mask,dtype=bool)
    if not mask[seed]:
        raise RuntimeError("O-point seed is not in candidate island mask")
    q=deque([seed])
    out[seed]=True
    while q:
        i,j=q.popleft()
        for ii,jj in (((i-1)%nx,j),((i+1)%nx,j),(i,j-1),(i,j+1)):
            if jj<0 or jj>=nz:
                continue
            if mask[ii,jj] and not out[ii,jj]:
                out[ii,jj]=True
                q.append((ii,jj))
    return out


def island_flux_coordinate(T):
    Ay=T["Ay"]; AO=T["AO"]; AX=T["AX"]
    den=AX-AO
    if abs(den)<=np.finfo(float).tiny:
        raise RuntimeError("A_X-A_O is too small")

    chi=(Ay-AO)/den

    # Candidate region on the O side of the separatrix.  A small tolerance
    # keeps the discrete contour connected without swallowing the X point.
    tol=0.02
    if den>0:
        cand=Ay <= AX + tol*abs(den)
    else:
        cand=Ay >= AX - tol*abs(den)

    island=periodic_component(cand,(T["io"],T["j0"]))
    return chi,island


def shell_budget(arr,chi,island,lo,hi):
    m=island & np.isfinite(chi) & (chi>=lo) & (chi<hi)
    signed=float(np.sum(arr[m]))
    absolute=float(np.sum(np.abs(arr[m])))
    positive=float(np.sum(np.clip(arr[m],0,None)))
    negative=float(np.sum(np.clip(arr[m],None,0)))
    return signed,absolute,positive,negative,int(np.count_nonzero(m))


def shift_on_O(x,arr,xO,L):
    xs=periodic_dx(x,xO,L)
    order=np.argsort(xs)
    return xs[order],arr[order],order


def main():
    args=parse_args()
    P=params()
    D=np.load(args.npz)
    x=np.asarray(D["x_de"],float)
    z=np.asarray(D["z_de"],float)
    rxx=np.asarray(D["rand_xx_restoring"],float)
    rtot=np.asarray(D["rand_total_restoring"],float)
    dxx=np.asarray(D["rand_xx_damping"],float)

    s1,s2=parse_crossings(args.summary)
    T=[
        topology(find_plotfile(args.phase_root,s1),args.mode_coarsen,
                 args.large_scale_max_mode,P),
        topology(find_plotfile(args.phase_root,s2),args.mode_coarsen,
                 args.large_scale_max_mode,P)
    ]
    chis=[]; islands=[]
    for Ts in T:
        ch,isl=island_flux_coordinate(Ts)
        chis.append(ch); islands.append(isl)

    dx=float(np.median(np.diff(x)))
    L=float((x.max()-x.min())+dx)
    total_signed=float(np.sum(rxx))
    total_abs=float(np.sum(np.abs(rxx)))
    pos_total=float(np.sum(np.clip(rxx,0,None)))

    # Strongest restoring cell.
    imax=np.unravel_index(np.nanargmax(rxx),rxx.shape)
    xmax=float(x[imax[0]]); zmax=float(z[imax[1]])

    shells=[(0,.25),(.25,.5),(.5,.75),(.75,1.02)]

    with open(args.out_prefix+"_summary.txt","w") as f:
        f.write("Ion restoring stress in magnetic-island flux shells\n")
        f.write("==================================================\n\n")
        f.write(f"crossings = {s1} {s2}\n")
        f.write(f"large-scale max x mode = {args.large_scale_max_mode}\n")
        f.write(f"global rand_xx restoring signed / |Zkin| = {total_signed:.10e}\n")
        f.write(f"global rand_xx restoring absolute / |Zkin| = {total_abs:.10e}\n")
        f.write(f"global rand_xx restoring positive / |Zkin| = {pos_total:.10e}\n")
        f.write(
            f"strongest restoring cell = ({xmax:+.5f},{zmax:+.5f}) de, "
            f"value={rxx[imax]:+.10e}\n\n"
        )

        for n,(step,Ts,chi,isl) in enumerate(
            zip((s1,s2),T,chis,islands),1
        ):
            xO=Ts["xO"]; xX=Ts["xX"]
            half=abs(periodic_dx(xX,xO,L))
            dspot=abs(periodic_dx(xmax,xO,L))
            f.write(f"[crossing {n}: step {step}]\n")
            f.write(f"O_x/de = {xO:+.8f}; X_x/de = {xX:+.8f}\n")
            f.write(f"periodic O-X distance/de = {half:.8f}\n")
            f.write(
                f"hotspot periodic |x-xO|/de = {dspot:.8f}; "
                f"fraction of O-X distance = {dspot/max(half,1e-300):.8f}\n"
            )
            f.write(
                f"hotspot chi = {chi[imax]:.8f}; "
                f"inside connected island = {int(isl[imax])}\n"
            )

            ins_signed=float(np.sum(rxx[isl]))
            ins_abs=float(np.sum(np.abs(rxx[isl])))
            ins_pos=float(np.sum(np.clip(rxx[isl],0,None)))
            f.write(
                f"inside island signed fraction of global signed = "
                f"{ins_signed/max(total_signed,1e-300):.8f}\n"
            )
            f.write(
                f"inside island absolute fraction of global absolute = "
                f"{ins_abs/max(total_abs,1e-300):.8f}\n"
            )
            f.write(
                f"inside island positive fraction of global positive = "
                f"{ins_pos/max(pos_total,1e-300):.8f}\n"
            )

            f.write("flux shells (chi=0 O-point, chi=1 separatrix):\n")
            for lo,hi in shells:
                ss,aa,pp,nn,nc=shell_budget(rxx,chi,isl,lo,hi)
                f.write(
                    f"  {lo:.2f} <= chi < {hi:.2f}: cells={nc:5d} "
                    f"signed={ss:+.10e} "
                    f"global_signed_frac={ss/max(total_signed,1e-300):+.8f} "
                    f"global_abs_frac={aa/max(total_abs,1e-300):.8f} "
                    f"global_positive_frac={pp/max(pos_total,1e-300):.8f}\n"
                )
            f.write("\n")

    # O-centered plots: use each crossing topology separately.
    def plot_shell_overlay(arr,name,title):
        fig,axs=plt.subplots(2,1,figsize=(10,9),sharex=True,sharey=True)
        vmax=np.nanpercentile(np.abs(arr),99.5)
        for ax,Ts,chi,isl,step in zip(axs,T,chis,islands,(s1,s2)):
            xs,ash,order=shift_on_O(x,arr,Ts["xO"],L)
            chs=chi[order]
            isl_s=isl[order]
            im=ax.imshow(
                ash.T,origin="lower",
                extent=[xs.min()-0.5*dx,xs.max()+0.5*dx,
                        z.min()-0.25,z.max()+0.25],
                aspect="auto",vmin=-vmax,vmax=vmax,cmap="RdBu_r"
            )

            XX,ZZ=np.meshgrid(xs,z,indexing="ij")
            masked=np.where(isl_s,chs,np.nan)
            for lev,ls in zip([.25,.5,.75,1.0],[":","-.","--","-"]):
                try:
                    ax.contour(
                        XX,ZZ,masked,levels=[lev],colors="k",
                        linewidths=1.0,linestyles=ls
                    )
                except Exception:
                    pass

            xXc=float(periodic_dx(Ts["xX"],Ts["xO"],L))
            ax.plot(0,0,"ko",mfc="none",ms=7,label="O")
            ax.plot(xXc,0,"kx",ms=7,label="X")
            ax.set_xlim(-0.5*L,0.5*L)
            ax.set_ylim(-8,8)
            ax.set_ylabel(r"$z/d_e$")
            ax.set_title(f"step {step}: O-centered; chi contours 0.25, 0.5, 0.75, 1")
            ax.grid(alpha=.12)
        axs[-1].set_xlabel(r"periodic $(x-x_O)/d_e$")
        fig.suptitle(title)
        fig.colorbar(im,ax=axs,label=r"cell contribution / $|Z_{kin,i}|$")
        fig.tight_layout()
        fig.savefig(name,dpi=210)
        plt.close(fig)

    plot_shell_overlay(
        rxx,args.out_prefix+"_rand_xx_restoring_Ocentered.png",
        "Ion random/pressure-like xx restoring projection"
    )
    plot_shell_overlay(
        rtot,args.out_prefix+"_rand_total_restoring_Ocentered.png",
        "Ion random/pressure-like total restoring projection"
    )
    plot_shell_overlay(
        dxx,args.out_prefix+"_rand_xx_damping_Ocentered.png",
        "Ion random/pressure-like xx damping projection"
    )

    print("Saved",args.out_prefix+"_summary.txt")
    print("Saved O-centered flux-shell maps.")


if __name__=="__main__":
    main()
