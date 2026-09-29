#!/usr/bin/env python3
"""Generate presentation-ready schematic figures for the Ishizawa-scale
late-time island-breathing story.

These are conceptual diagrams, not simulation data.  They are intended for
slides that explain the logic of the analysis:
  1. research_story_flow
  2. breathing_cycle
  3. weak_form_budget
  4. restoring_mechanism_chain

Each figure is written as both PNG (300 dpi) and SVG.
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",default="presentation_schematics")
    p.add_argument("--prefix",default="ishizawa_ppt")
    return p.parse_args()


def save(fig,out_dir,name):
    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(out_dir/f"{name}.png",dpi=300,bbox_inches="tight")
    fig.savefig(out_dir/f"{name}.svg",bbox_inches="tight")
    plt.close(fig)


def box(ax,xy,w,h,text,fontsize=15,rounding=.08,lw=1.8):
    x,y=xy
    p=FancyBboxPatch(
        (x,y),w,h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        facecolor="white",edgecolor="black",linewidth=lw
    )
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fontsize)
    return p


def arrow(ax,a,b,text=None,fontsize=12,rad=0.0,lw=2.0):
    p=FancyArrowPatch(
        a,b,arrowstyle="-|>",mutation_scale=16,
        linewidth=lw,color="black",
        connectionstyle=f"arc3,rad={rad}"
    )
    ax.add_patch(p)
    if text:
        xm=(a[0]+b[0])/2; ym=(a[1]+b[1])/2
        ax.text(xm,ym+0.035,text,ha="center",va="bottom",fontsize=fontsize)
    return p


def story_flow(out_dir,prefix):
    fig,ax=plt.subplots(figsize=(13.5,4.2))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

    labels=[
        "Extend late-time\nsimulation",
        "Persistent island\nbreathing appears",
        "Verify it is a\ncoherent mode",
        "Close the particle\nweak-form budget",
        "Identify dominant\nrestoring channel",
        "Localize the\nkinetic mechanism",
    ]
    xs=np.linspace(.02,.82,len(labels))
    w=.15; h=.30; y=.37
    for i,(x,lbl) in enumerate(zip(xs,labels)):
        box(ax,(x,y),w,h,lbl,fontsize=14)
        if i<len(labels)-1:
            arrow(ax,(x+w,y+h/2),(xs[i+1],y+h/2))

    ax.text(
        .5,.88,
        "From 'is it saturated?' to a kinetic restoring mechanism",
        ha="center",va="center",fontsize=20,weight="bold"
    )
    ax.text(
        .5,.12,
        "steady-state search  →  oscillation  →  validation  →  mechanism",
        ha="center",va="center",fontsize=14
    )
    save(fig,out_dir,f"{prefix}_research_story_flow")


def draw_island(ax,cx,cy,a,b,label=None):
    # Simple closed magnetic-flux contours.
    for s,lw in [(1.0,2.0),(.75,1.5),(.5,1.2)]:
        e=Ellipse((cx,cy),2*a*s,2*b*s,fill=False,edgecolor="black",linewidth=lw)
        ax.add_patch(e)
    if label:
        ax.text(cx,cy-b-0.12,label,ha="center",va="top",fontsize=13)


def breathing_cycle(out_dir,prefix):
    fig,ax=plt.subplots(figsize=(12,6.5))
    ax.set_xlim(0,12); ax.set_ylim(0,7); ax.axis("off")

    states=[
        (2.0,5.0,1.7,1.25,"Expanded\n$q>0$"),
        (5.0,5.0,1.35,1.0,"Equilibrium\n$q\approx0$"),
        (8.0,5.0,1.05,.78,"Contracted\n$q<0$"),
        (5.0,1.8,1.35,1.0,"Equilibrium\n$q\approx0$"),
    ]
    for cx,cy,a,b,label in states:
        draw_island(ax,cx,cy,a,b,label)

    arrow(ax,(3.5,5.0),(3.95,5.0),"restoring")
    arrow(ax,(6.4,5.0),(6.85,5.0),"inertia")
    arrow(ax,(7.2,4.0),(5.9,2.7),"restoring",rad=.12)
    arrow(ax,(4.1,2.7),(2.6,4.0),"inertia",rad=.12)

    ax.text(
        6,6.5,
        "Breathing-cycle schematic",
        ha="center",fontsize=21,weight="bold"
    )
    ax.text(
        6,.55,
        r"dominant restoring phase:  $Q^{\\rm rand}_{i,xx}\propto -q$",
        ha="center",fontsize=17
    )
    save(fig,out_dir,f"{prefix}_breathing_cycle")


def weak_form_budget(out_dir,prefix):
    fig,ax=plt.subplots(figsize=(12,6.4))
    ax.set_xlim(0,12); ax.set_ylim(0,7); ax.axis("off")

    ax.text(
        6,6.5,
        "Particle weak-form momentum budget",
        ha="center",fontsize=21,weight="bold"
    )

    box(ax,(.7,2.75),2.4,1.0,r"$dP_{\xi}/dt$",fontsize=24)
    ax.text(3.45,3.25,"=",fontsize=28,va="center")

    box(ax,(4.0,4.65),2.2,.9,r"$Q_{\rm EM}$",fontsize=21)
    box(ax,(4.0,3.05),2.2,.9,r"$Q_{{\rm kin},i}$",fontsize=21,lw=2.7)
    box(ax,(4.0,1.45),2.2,.9,r"$Q_{{\rm kin},e}$",fontsize=21)

    arrow(ax,(3.65,3.25),(4.0,5.1),lw=1.6)
    arrow(ax,(3.65,3.25),(4.0,3.5),lw=2.6)
    arrow(ax,(3.65,3.25),(4.0,1.9),lw=1.6)

    box(ax,(7.1,3.05),1.7,.9,r"$Q_{xx}$",fontsize=20,lw=2.7)
    box(ax,(9.3,4.55),1.7,.8,r"$Q_{xz}$",fontsize=18)
    box(ax,(9.3,3.10),1.7,.8,r"$Q_{zx}$",fontsize=18)
    box(ax,(9.3,1.65),1.7,.8,r"$Q_{zz}$",fontsize=18)
    arrow(ax,(6.2,3.5),(7.1,3.5),text="tensor split",fontsize=11,lw=2.4)
    arrow(ax,(8.8,3.5),(9.3,4.95),lw=1.4)
    arrow(ax,(8.8,3.5),(9.3,3.50),lw=1.4)
    arrow(ax,(8.8,3.5),(9.3,2.05),lw=1.4)

    ax.text(7.95,2.25,"dominant",ha="center",fontsize=13,weight="bold")
    ax.text(
        6,.45,
        r"validated fundamental closure:  $|Z_{RHS}-Z_{LHS}|/|Z_{LHS}|\simeq 3.3\times10^{-3}$",
        ha="center",fontsize=15
    )
    save(fig,out_dir,f"{prefix}_weak_form_budget")


def mechanism_chain(out_dir,prefix):
    fig,ax=plt.subplots(figsize=(13.5,6.2))
    ax.set_xlim(0,13.5); ax.set_ylim(0,7); ax.axis("off")

    ax.text(
        6.75,6.55,
        "Current working picture of the restoring mechanism",
        ha="center",fontsize=21,weight="bold"
    )

    box(ax,(.5,3.9),2.2,1.0,"Island breathing\n$q(t)$",fontsize=17)
    box(ax,(3.25,3.9),2.2,1.0,"Deep closed-island\ncore  $\chi<0.25$",fontsize=16)
    box(ax,(6.0,3.9),2.2,1.0,r"Ion random $xx$\nstress",fontsize=17,lw=2.5)
    box(ax,(8.75,3.9),2.2,1.0,"Generalized\nrestoring force",fontsize=16)
    box(ax,(11.5,3.9),1.5,1.0,r"$-q$",fontsize=22)

    for x1,x2 in [(2.7,3.25),(5.45,6.0),(8.2,8.75),(10.95,11.5)]:
        arrow(ax,(x1,4.4),(x2,4.4))

    box(ax,(4.9,1.35),2.7,1.0,"Density / occupancy\nmodulation  ~51%",fontsize=15)
    box(ax,(8.0,1.35),2.7,1.0,"Specific random stress\nmodulation  ~45%",fontsize=15)
    box(ax,(11.1,1.35),1.8,1.0,"Nonlinear\ncross  ~5%",fontsize=14)

    arrow(ax,(6.25,2.35),(6.85,3.9),rad=-.12)
    arrow(ax,(9.35,2.35),(7.45,3.9),rad=.12)
    arrow(ax,(12.0,2.35),(7.8,3.9),rad=.18)

    ax.text(
        6.75,.55,
        "Interpretation: compression/occupancy changes and velocity-space deformation contribute comparably",
        ha="center",fontsize=14
    )
    save(fig,out_dir,f"{prefix}_restoring_mechanism_chain")


def main():
    args=parse_args()
    story_flow(args.out_dir,args.prefix)
    breathing_cycle(args.out_dir,args.prefix)
    weak_form_budget(args.out_dir,args.prefix)
    mechanism_chain(args.out_dir,args.prefix)
    print("Saved schematic PNG/SVG files under",args.out_dir)


if __name__=="__main__":
    main()
