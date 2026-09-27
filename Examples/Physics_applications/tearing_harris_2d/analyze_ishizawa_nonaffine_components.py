#!/usr/bin/env python3
"""Decompose the non-affine breathing restoring force by physical channel.

This is the follow-up to analyze_ishizawa_nonaffine_breathing_mode.py.
Using the same four particle-rich breathing phases, it reconstructs the
velocity-derived non-affine displacement xi and decomposes

  - ion/electron pressure projections into Pxx, Pxz and Pzz pieces;
  - electromagnetic projection into magnetic-pressure, magnetic-tension
    and charge-electric pieces.

For a tapered periodic mode, integration by parts gives

  Q_P = int [ Pxx d_x xi_x
            + Pxz (d_z xi_x + d_x xi_z)
            + Pzz d_z xi_z ] dA.

This form is useful because it identifies whether the restoring pressure
response is normal stress or shear/off-diagonal stress.

Outputs
-------
ishizawa_nonaffine_components.txt
ishizawa_nonaffine_components_summary.txt
ishizawa_nonaffine_pressure_components.png
ishizawa_nonaffine_em_components.png
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import QE, ME, C, MU0, params
from analyze_ishizawa_particle_moments import load_mesh, deposit_species
from analyze_ishizawa_nonaffine_breathing_mode import (
    step_from_parent, harmonic_fit, smooth2, weighted_mean, linfit
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--phase-root",default="runs_ishizawa_particle_phases/ppc49")
    p.add_argument("--em-history",required=True)
    p.add_argument("--coarsen",type=int,default=4)
    p.add_argument("--period",type=float,default=0.63)
    p.add_argument("--core-z-de",type=float,default=12.0)
    p.add_argument("--mode-density-cut",type=float,default=0.02)
    p.add_argument("--smooth-passes",type=int,default=2)
    return p.parse_args()


def project(xix,xiz,fx,fz,dx,dz):
    return float(np.sum(xix*fx+xiz*fz)*dx*dz)


def slope_stats(q,y):
    s,b,r2=linfit(q,y)
    corr=float(np.corrcoef(q,y)[0,1]) if len(q)>2 else np.nan
    return s,r2,corr


def main():
    args=parse_args()
    yt.funcs.mylog.setLevel(40)
    P=params()
    p=dict(qe=QE,me=ME,mi=P["mi"],c=C,n0=P["n0"],
           de=P["de"],wci=P["wci"],B0=P["B0"])

    files=sorted(
        glob.glob(str(Path(args.phase_root)/"step*"/"diags"/"diag1*")),
        key=step_from_parent
    )
    if len(files)<4:
        raise RuntimeError(f"need four particle-rich phase plotfiles; found {len(files)}")

    eh=np.loadtxt(args.em_history)
    if eh.ndim==1: eh=eh[None,:]
    ht=eh[:,1]; hq=eh[:,4]
    qfit,hqdot,hqddot,Rq,cq=harmonic_fit(ht,hq,args.period)

    states=[]
    for fn in files:
        M=load_mesh(fn,args.coarsen)
        I=deposit_species(fn,"ions",P["mi"],M,p)
        E=deposit_species(fn,"electrons",ME,M,p)
        tci=P["wci"]*M["t"]
        ni=np.abs(M["rho_i"])/QE
        ne=np.abs(M["rho_e"])/QE
        states.append(dict(
            step=step_from_parent(fn),M=M,I=I,E=E,ni=ni,ne=ne,t=tci,
            q=float(np.interp(tci,ht,hq)),
            qdot=float(np.interp(tci,ht,hqdot)),
            qdd=float(np.interp(tci,ht,hqddot))
        ))

    # Reconstruct the same non-affine displacement from opposite q-crossings.
    pos=[s for s in states if s["qdot"]>0]
    neg=[s for s in states if s["qdot"]<0]
    if not pos or not neg:
        raise RuntimeError("need opposite-sign qdot crossings")
    crossings=[min(neg,key=lambda s:abs(s["q"])),
               min(pos,key=lambda s:abs(s["q"]))]

    xis=[]
    for s in crossings:
        z=s["M"]["z"]
        core=(np.abs(z)[None,:] <= args.core_z_de*P["de"])
        w=s["ni"]*core
        ux0=weighted_mean(s["I"]["ux"],w)
        uz0=weighted_mean(s["I"]["uz"],w)
        qdot_phys=P["wci"]*s["qdot"]
        xis.append(((s["I"]["ux"]-ux0)/qdot_phys,
                    (s["I"]["uz"]-uz0)/qdot_phys))

    xix=smooth2(0.5*(xis[0][0]+xis[1][0]),args.smooth_passes)
    xiz=smooth2(0.5*(xis[0][1]+xis[1][1]),args.smooth_passes)

    nmean=0.5*(crossings[0]["ni"]+crossings[1]["ni"])
    dens_env=nmean/(nmean+args.mode_density_cut*P["n0"])
    z=crossings[0]["M"]["z"]
    zmax=args.core_z_de*P["de"]
    zenv=np.zeros_like(z)
    inside=np.abs(z)<zmax
    zenv[inside]=0.5*(1+np.cos(np.pi*z[inside]/zmax))
    env=dens_env*zenv[None,:]
    xix*=env; xiz*=env

    M0=crossings[0]["M"]
    dx,dz=M0["dx"],M0["dz"]
    dxx=np.gradient(xix,dx,axis=0,edge_order=2)
    dxz=np.gradient(xix,dz,axis=1,edge_order=2)
    dzx=np.gradient(xiz,dx,axis=0,edge_order=2)
    dzz=np.gradient(xiz,dz,axis=1,edge_order=2)

    rows=[]
    for s in states:
        M,I,E,ni,ne=s["M"],s["I"],s["E"],s["ni"],s["ne"]
        rhoi=P["mi"]*ni
        rhoe=ME*ne
        rhot=rhoi+rhoe

        modal_mass=float(np.sum(rhot*(xix*xix+xiz*xiz))*dx*dz)
        fac=1.0/(modal_mass*P["wci"]**2)

        # Pressure-tensor pieces, integration-by-parts form.
        Pxx_i=P["mi"]*ni*I["cxx"]; Pxz_i=P["mi"]*ni*I["cxz"]; Pzz_i=P["mi"]*ni*I["czz"]
        Pxx_e=ME*ne*E["cxx"]; Pxz_e=ME*ne*E["cxz"]; Pzz_e=ME*ne*E["czz"]

        Qi_xx=float(np.sum(Pxx_i*dxx)*dx*dz)
        Qi_xz=float(np.sum(Pxz_i*(dxz+dzx))*dx*dz)
        Qi_zz=float(np.sum(Pzz_i*dzz)*dx*dz)
        Qe_xx=float(np.sum(Pxx_e*dxx)*dx*dz)
        Qe_xz=float(np.sum(Pxz_e*(dxz+dzx))*dx*dz)
        Qe_zz=float(np.sum(Pzz_e*dzz)*dx*dz)

        # Field-force decomposition.
        Bx,By,Bz=M["Bx"],M["By"],M["Bz"]
        B2=Bx*Bx+By*By+Bz*Bz
        pm=B2/(2*MU0)
        fpm_x=-np.gradient(pm,dx,axis=0,edge_order=2)
        fpm_z=-np.gradient(pm,dz,axis=1,edge_order=2)

        dBx_dx=np.gradient(Bx,dx,axis=0,edge_order=2)
        dBx_dz=np.gradient(Bx,dz,axis=1,edge_order=2)
        dBz_dx=np.gradient(Bz,dx,axis=0,edge_order=2)
        dBz_dz=np.gradient(Bz,dz,axis=1,edge_order=2)
        ften_x=(Bx*dBx_dx+Bz*dBx_dz)/MU0
        ften_z=(Bx*dBz_dx+Bz*dBz_dz)/MU0

        rhoc=QE*(ni-ne)
        frhoe_x=rhoc*M["Ex"]
        frhoe_z=rhoc*M["Ez"]

        # Particle-current EM projection as an independent check.
        Jx=QE*(ni*I["ux"]-ne*E["ux"])
        Jy=QE*(ni*I["uy"]-ne*E["uy"])
        Jz=QE*(ni*I["uz"]-ne*E["uz"])
        fEM_x=frhoe_x + Jy*Bz - Jz*By
        fEM_z=frhoe_z + Jx*By - Jy*Bx

        a_i_xx=Qi_xx*fac; a_i_xz=Qi_xz*fac; a_i_zz=Qi_zz*fac
        a_e_xx=Qe_xx*fac; a_e_xz=Qe_xz*fac; a_e_zz=Qe_zz*fac
        a_pmag=project(xix,xiz,fpm_x,fpm_z,dx,dz)*fac
        a_ten=project(xix,xiz,ften_x,ften_z,dx,dz)*fac
        a_rhoe=project(xix,xiz,frhoe_x,frhoe_z,dx,dz)*fac
        a_em=project(xix,xiz,fEM_x,fEM_z,dx,dz)*fac

        rows.append([
            s["step"],s["t"],s["q"],s["qdd"],modal_mass,
            a_i_xx,a_i_xz,a_i_zz,a_e_xx,a_e_xz,a_e_zz,
            a_i_xx+a_i_xz+a_i_zz,
            a_e_xx+a_e_xz+a_e_zz,
            a_pmag,a_ten,a_rhoe,a_pmag+a_ten+a_rhoe,a_em
        ])

    a=np.asarray(rows,float)
    a=a[np.argsort(a[:,1])]
    np.savetxt(
        "ishizawa_nonaffine_components.txt",a,
        header=(
            "step omega_ci_t q measured_ddq modal_mass "
            "a_i_Pxx a_i_Pxz a_i_Pzz a_e_Pxx a_e_Pxz a_e_Pzz "
            "a_i_pressure a_e_pressure "
            "a_mag_pressure a_mag_tension a_rhoE a_EM_identity a_EM_particle"
        )
    )

    t=a[:,1]; q=a[:,2]
    names=[
        ("ion Pxx",5),("ion Pxz shear",6),("ion Pzz",7),
        ("electron Pxx",8),("electron Pxz shear",9),("electron Pzz",10),
        ("ion pressure total",11),("electron pressure total",12),
        ("magnetic pressure",13),("magnetic tension",14),("rhoE",15),
        ("EM identity sum",16),("EM particle-current",17)
    ]

    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for name,col in names[:6]:
        axs[0 if name.startswith("ion") else 1].plot(t,a[:,col],"o-",label=name)
    for ax in axs:
        ax.axhline(0,lw=.7); ax.grid(alpha=.25); ax.legend(fontsize=8)
        ax.set_ylabel("modal acceleration")
    axs[1].set_xlabel(r"$\omega_{ci}t$")
    fig.suptitle("Non-affine pressure-tensor channel decomposition")
    fig.tight_layout()
    fig.savefig("ishizawa_nonaffine_pressure_components.png",dpi=210)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5.5))
    ax.plot(t,a[:,13],"o-",label="magnetic pressure")
    ax.plot(t,a[:,14],"o-",label="magnetic tension")
    ax.plot(t,a[:,15],"o-",label=r"$\rho_c E$")
    ax.plot(t,a[:,16],"o-",lw=2,label="field identity sum")
    ax.plot(t,a[:,17],"s--",label="particle-current EM")
    ax.axhline(0,lw=.7); ax.grid(alpha=.25)
    ax.set_xlabel(r"$\omega_{ci}t$"); ax.set_ylabel("modal acceleration")
    ax.legend(fontsize=8)
    ax.set_title("Non-affine electromagnetic-force decomposition")
    fig.tight_layout()
    fig.savefig("ishizawa_nonaffine_em_components.png",dpi=210)
    plt.close(fig)

    with open("ishizawa_nonaffine_components_summary.txt","w") as f:
        f.write("Non-affine breathing-force component decomposition\n")
        f.write("=================================================\n\n")
        f.write(f"period omega_ci*T = {args.period:.8f}\n")
        f.write(f"q harmonic R2 = {Rq:.8f}\n")
        f.write(f"coarsen = {args.coarsen}\n")
        f.write(f"core |z|/de <= {args.core_z_de:.6f}\n\n")

        f.write("force-displacement slopes\n")
        f.write("-------------------------\n")
        f.write("channel                    slope_vs_q       R2       corr(q,channel)\n")
        for name,col in names:
            s,r2,corr=slope_stats(q,a[:,col])
            f.write(f"{name:26s} {s:+14.6e} {r2:9.5f} {corr:+12.6f}\n")

        f.write("\npressure fractions of restoring slope\n")
        si=sum(linfit(q,a[:,c])[0] for c in (5,6,7))
        se=sum(linfit(q,a[:,c])[0] for c in (8,9,10))
        f.write(f"ion pressure slope = {si:+.8e}\n")
        f.write(f"electron pressure slope = {se:+.8e}\n")
        f.write(f"total pressure slope = {si+se:+.8e}\n")
        if abs(si+se)>0:
            f.write(f"ion fraction of total pressure slope = {si/(si+se):.8f}\n")
            f.write(f"electron fraction of total pressure slope = {se/(si+se):.8f}\n")

        emdiff=np.sqrt(np.mean((a[:,16]-a[:,17])**2))/max(
            np.sqrt(np.mean(a[:,17]**2)),1e-300
        )
        f.write("\nEM identity consistency\n")
        f.write("-----------------------\n")
        f.write(f"relative RMS [mag pressure+tension+rhoE vs particle-current EM] = {emdiff:.8e}\n")

        f.write("\nInterpretation\n")
        f.write("--------------\n")
        f.write(
            "Negative slope is restoring for the chosen width coordinate. "
            "The Pxx/Pxz/Pzz split identifies whether the non-affine pressure "
            "restoring response comes mainly from normal stress or the "
            "off-diagonal shear stress.  The magnetic-pressure/tension split "
            "is the non-affine counterpart of the earlier affine field-force "
            "diagnostic.  With only four phases these slopes are a mechanism "
            "screen; use a denser particle-rich phase series before quoting "
            "precision stiffnesses.\n"
        )

    print("Saved ishizawa_nonaffine_components.txt")
    print("Saved ishizawa_nonaffine_components_summary.txt")
    print("Saved ishizawa_nonaffine_pressure_components.png")
    print("Saved ishizawa_nonaffine_em_components.png")


if __name__=="__main__":
    main()
