#!/usr/bin/env python3
"""Exact first-moment closure for the saturated non-affine breathing mode.

This diagnostic is deliberately different from the reduced q'' = force/M tests.
It projects the *particle momentum conservation law* itself,

    d/dt ∫ xi·g dA
      = ∫ xi·(rho_c E + JxB) dA
        - ∫ xi·div(T_i + T_e) dA,

where
    g_s = n_s <p>_s
and the relativistic kinetic momentum-flux tensor is
    (T_s)_{ij} = n_s <p_i v_j>_s.

The fixed non-affine test displacement xi(x,z) is reconstructed from the two
opposite near-zero-q ion-flow crossings, exactly as in the existing non-affine
breathing diagnostic.  Because xi is fixed in time, no assumed modal mass or
q'' normalization is needed.

Using tau = omega_ci t, the tested equation is

    dP_xi/dtau = (Q_EM + Q_kin,i + Q_kin,e)/omega_ci.

The momentum-flux projection is evaluated by integration by parts,
which is much less noisy than differentiating PIC moments:

    Q_kin = ∫ [Txx d_x xi_x + Txz d_z xi_x
               + Tzx d_x xi_z + Tzz d_z xi_z] dA.

The direct breathing-frequency complex closure is the primary result.

Outputs
-------
ishizawa_modal_momentum_closure_summary.txt
ishizawa_modal_momentum_closure_history.txt
ishizawa_modal_momentum_closure_time.png
ishizawa_modal_momentum_closure_complex.png
ishizawa_modal_momentum_closure_mode.png
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import QE, ME, C, params
from analyze_ishizawa_particle_moments import load_mesh, particle_array
from analyze_ishizawa_nonaffine_breathing_mode import (
    step_from_parent, harmonic_fit, smooth2, weighted_mean
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--phase-root",default="runs_ishizawa_particle_phases/ppc49")
    p.add_argument("--em-history",required=True,
                   help="history containing omega_ci*t in col 2 and q in col 5")
    p.add_argument("--coarsen",type=int,default=4)
    p.add_argument("--period",type=float,default=0.63)
    p.add_argument("--tmin",type=float,default=3.20)
    p.add_argument("--tmax",type=float,default=3.80)
    p.add_argument("--core-z-de",type=float,default=12.0)
    p.add_argument("--mode-density-cut",type=float,default=0.02)
    p.add_argument("--smooth-passes",type=int,default=2)
    return p.parse_args()


def bincell_species(path,species,mass,mesh):
    """Cell-bin relativistic first and momentum-flux moments.

    Absolute density is taken from the native rho_s mesh.  Particle weights are
    only used for within-cell normalized averages, so the unknown 2-D macro
    volume normalization cancels.
    """
    ds=yt.load(path)
    ad=ds.all_data()
    xp=particle_array(ad,species,"particle_position_x","m")
    zp=particle_array(ad,species,"particle_position_y","m")
    px=particle_array(ad,species,"particle_momentum_x","kg*m/s")
    py=particle_array(ad,species,"particle_momentum_y","kg*m/s")
    pz=particle_array(ad,species,"particle_momentum_z","kg*m/s")
    w=particle_array(ad,species,"particle_weight")

    gamma=np.sqrt(1.0+(px*px+py*py+pz*pz)/(mass*C)**2)
    vx=px/(gamma*mass); vy=py/(gamma*mass); vz=pz/(gamma*mass)

    nx,nz=mesh["nx"],mesh["nz"]
    ix=np.floor((xp-mesh["xlo"])/mesh["dx"]).astype(np.int64)
    iz=np.floor((zp-mesh["zlo"])/mesh["dz"]).astype(np.int64)
    ix=np.clip(ix,0,nx-1); iz=np.clip(iz,0,nz-1)
    cell=ix*nz+iz
    ncell=nx*nz

    def bc(a):
        return np.bincount(cell,weights=a,minlength=ncell).reshape(nx,nz)

    sw=bc(w)
    inv=1.0/np.maximum(sw,np.finfo(float).tiny)

    def av(a):
        return bc(w*a)*inv

    out=dict(
        ux=av(vx), uy=av(vy), uz=av(vz),
        px=av(px), py=av(py), pz=av(pz),
        Txx=av(px*vx), Txz=av(px*vz),
        Tzx=av(pz*vx), Tzz=av(pz*vz),
        Nmacro=len(w), sw=sw
    )
    return out


def fit_phasor(t,y,T):
    """Fit trend + fundamental and return Z with y_h=Re[Z exp(i omega u)]."""
    t=np.asarray(t,float); y=np.asarray(y,float)
    u=t-t[0]
    om=2*np.pi/T
    M=np.column_stack([np.ones_like(u),u,np.cos(om*u),np.sin(om*u)])
    c,*_=np.linalg.lstsq(M,y,rcond=None)
    pred=M@c
    ssr=np.sum((y-pred)**2)
    sst=np.sum((y-y.mean())**2)
    r2=1-ssr/sst if sst>0 else np.nan
    Z=complex(c[2],-c[3])
    return Z,float(r2),pred,c


def hseries(Z,t,T,t0):
    om=2*np.pi/T
    return np.real(Z*np.exp(1j*om*(np.asarray(t)-t0)))


def project_em(xix,xiz,M,ni,ne,I,E,dx,dz):
    rhoc=QE*(ni-ne)
    Jx=QE*(ni*I["ux"]-ne*E["ux"])
    Jy=QE*(ni*I["uy"]-ne*E["uy"])
    Jz=QE*(ni*I["uz"]-ne*E["uz"])
    fx=rhoc*M["Ex"] + Jy*M["Bz"] - Jz*M["By"]
    fz=rhoc*M["Ez"] + Jx*M["By"] - Jy*M["Bx"]
    return float(np.sum(xix*fx+xiz*fz)*dx*dz)


def project_kinetic_ibp(xix,xiz,n,S,dx,dz):
    """Project -div[n<p_i v_j>] by parts onto fixed xi."""
    dxx=np.gradient(xix,dx,axis=0,edge_order=2)
    dxz=np.gradient(xix,dz,axis=1,edge_order=2)
    dzx=np.gradient(xiz,dx,axis=0,edge_order=2)
    dzz=np.gradient(xiz,dz,axis=1,edge_order=2)
    Txx=n*S["Txx"]; Txz=n*S["Txz"]
    Tzx=n*S["Tzx"]; Tzz=n*S["Tzz"]
    return float(np.sum(Txx*dxx + Txz*dxz + Tzx*dzx + Tzz*dzz)*dx*dz)


def modal_momentum(xix,xiz,ni,ne,I,E,dx,dz):
    gx=ni*I["px"] + ne*E["px"]
    gz=ni*I["pz"] + ne*E["pz"]
    return float(np.sum(xix*gx+xiz*gz)*dx*dz)


def main():
    args=parse_args()
    yt.funcs.mylog.setLevel(40)
    P=params()

    files=sorted(
        glob.glob(str(Path(args.phase_root)/"step*"/"diags"/"diag1*")),
        key=step_from_parent
    )
    if len(files)<6:
        raise RuntimeError(
            f"need >=6 particle-rich phase plotfiles for a meaningful fundamental closure; found {len(files)}"
        )

    eh=np.loadtxt(args.em_history)
    if eh.ndim==1: eh=eh[None,:]
    ht=eh[:,1]; hq=eh[:,4]
    qfit,hqdot,hqddot,Rq,cq=harmonic_fit(ht,hq,args.period)

    states=[]
    for i,fn in enumerate(files,1):
        M=load_mesh(fn,args.coarsen)
        tci=P["wci"]*M["t"]
        if tci<args.tmin-1e-9 or tci>args.tmax+1e-9:
            continue
        I=bincell_species(fn,"ions",P["mi"],M)
        E=bincell_species(fn,"electrons",ME,M)
        ni=np.abs(M["rho_i"])/QE
        ne=np.abs(M["rho_e"])/QE
        states.append(dict(
            fn=fn,step=step_from_parent(fn),M=M,I=I,E=E,ni=ni,ne=ne,t=tci,
            q=float(np.interp(tci,ht,hq)),
            qdot=float(np.interp(tci,ht,hqdot))
        ))
        print(f"[{i:02d}/{len(files):02d}] step={step_from_parent(fn)} omega_ci*t={tci:.6f}")

    if len(states)<6:
        raise RuntimeError(f"only {len(states)} particle-rich states remain inside requested time window")

    states=sorted(states,key=lambda s:s["t"])

    # Reconstruct a fixed non-affine test displacement xi from the two
    # opposite near-zero-q ion-flow crossings.
    pos=[s for s in states if s["qdot"]>0]
    neg=[s for s in states if s["qdot"]<0]
    if not pos or not neg:
        raise RuntimeError("states do not contain opposite-sign qdot crossings")
    s_pos=min(pos,key=lambda s:abs(s["q"]))
    s_neg=min(neg,key=lambda s:abs(s["q"]))
    crossings=[s_neg,s_pos]

    xis=[]
    for s in crossings:
        core=(np.abs(s["M"]["z"])[None,:] <= args.core_z_de*P["de"])
        w=s["ni"]*core
        ux=s["I"]["ux"]; uz=s["I"]["uz"]
        ux0=weighted_mean(ux,w); uz0=weighted_mean(uz,w)
        qdot_phys=P["wci"]*s["qdot"]
        xis.append(((ux-ux0)/qdot_phys,(uz-uz0)/qdot_phys))

    xix=0.5*(xis[0][0]+xis[1][0])
    xiz=0.5*(xis[0][1]+xis[1][1])
    xix=smooth2(xix,args.smooth_passes)
    xiz=smooth2(xiz,args.smooth_passes)

    nmean=0.5*(crossings[0]["ni"]+crossings[1]["ni"])
    dens_env=nmean/(nmean+args.mode_density_cut*P["n0"])
    z=crossings[0]["M"]["z"]
    zmax=args.core_z_de*P["de"]
    zenv=np.zeros_like(z)
    inside=np.abs(z)<zmax
    zenv[inside]=0.5*(1+np.cos(np.pi*z[inside]/zmax))
    env=dens_env*zenv[None,:]
    xix*=env; xiz*=env

    rows=[]
    for s in states:
        M=s["M"]; I=s["I"]; E=s["E"]
        dx,dz=M["dx"],M["dz"]
        Pxi=modal_momentum(xix,xiz,s["ni"],s["ne"],I,E,dx,dz)
        Qem=project_em(xix,xiz,M,s["ni"],s["ne"],I,E,dx,dz)
        Qki=project_kinetic_ibp(xix,xiz,s["ni"],I,dx,dz)
        Qke=project_kinetic_ibp(xix,xiz,s["ne"],E,dx,dz)
        # tau = omega_ci t -> dP/dtau = Q/omega_ci.
        Rem=Qem/P["wci"]
        Rki=Qki/P["wci"]
        Rke=Qke/P["wci"]
        rows.append([s["step"],s["t"],s["q"],Pxi,Rem,Rki,Rke,Rem+Rki+Rke])

    a=np.asarray(rows,float)
    np.savetxt(
        "ishizawa_modal_momentum_closure_history.txt",a,
        header="step omega_ci_t q P_xi QEM_over_wci Qkin_i_over_wci Qkin_e_over_wci Qrhs_over_wci"
    )

    t=a[:,1]; Pxi=a[:,3]
    Zp,Rp,_,_=fit_phasor(t,Pxi,args.period)
    omega=2*np.pi/args.period
    Zlhs=1j*omega*Zp

    Zem,RemR2,_,_=fit_phasor(t,a[:,4],args.period)
    Zki,RkiR2,_,_=fit_phasor(t,a[:,5],args.period)
    Zke,RkeR2,_,_=fit_phasor(t,a[:,6],args.period)
    Zrhs=Zem+Zki+Zke
    Zres=Zlhs-Zrhs
    rel=abs(Zres)/max(abs(Zlhs),1e-300)

    # Raw finite-difference derivative is shown only as a sampling/noise check.
    dPraw=np.gradient(Pxi,t,edge_order=2)
    raw_rel=np.sqrt(np.mean((dPraw-a[:,7])**2))/max(np.sqrt(np.mean(dPraw*dPraw)),1e-300)

    with open("ishizawa_modal_momentum_closure_summary.txt","w") as f:
        f.write("Non-affine projected particle-momentum closure\n")
        f.write("==============================================\n\n")
        f.write("Equation tested in tau=omega_ci*t:\n")
        f.write("  d/dtau int xi.g dA = (Q_EM + Q_kin,i + Q_kin,e)/omega_ci\n")
        f.write("with exact relativistic moments g=n<p>, T_ij=n<p_i v_j>.\n\n")
        f.write(f"points = {len(a)}\n")
        f.write(f"window omega_ci*t = {t.min():.8f} .. {t.max():.8f}\n")
        f.write(f"period omega_ci*T = {args.period:.8f}\n")
        f.write(f"omega_b/omega_ci = {omega:.8f}\n")
        f.write(f"q harmonic R2 = {Rq:.8f}\n")
        f.write(f"P_xi harmonic R2 = {Rp:.8f}\n")
        f.write(f"EM harmonic R2 = {RemR2:.8f}\n")
        f.write(f"ion kinetic-flux harmonic R2 = {RkiR2:.8f}\n")
        f.write(f"electron kinetic-flux harmonic R2 = {RkeR2:.8f}\n")
        f.write(f"crossing steps = {s_neg['step']} {s_pos['step']}\n\n")

        f.write("Breathing-frequency complex amplitudes\n")
        f.write("--------------------------------------\n")
        f.write("channel                 Re(Z)              Im(Z)             |Z|\n")
        for name,Z in [
            ("dPxi/dtau",Zlhs),("EM",Zem),("ion kinetic",Zki),
            ("electron kinetic",Zke),("RHS sum",Zrhs),("residual",Zres)
        ]:
            f.write(f"{name:18s} {Z.real:+17.8e} {Z.imag:+17.8e} {abs(Z):17.8e}\n")

        f.write("\n")
        f.write(f"fundamental complex closure relative error = {rel:.8e}\n")
        f.write(f"phase(rhs/lhs) [rad] = {np.angle(Zrhs/Zlhs):+.8f}\n")
        f.write(f"|rhs|/|lhs| = {abs(Zrhs)/max(abs(Zlhs),1e-300):.8f}\n")
        f.write(f"raw finite-difference closure relRMS (diagnostic only) = {raw_rel:.8e}\n\n")
        f.write("Interpretation\n")
        f.write("--------------\n")
        f.write(
            "This is a first-moment conservation test, not a reduced oscillator fit. "
            "If the complex closure error is small, the previous large reduced-model "
            "residual came mainly from mapping the full kinetic momentum equation onto "
            "q'' with an assumed modal inertia. If this first-moment closure is still "
            "poor, inspect particle binning/deposition consistency, omitted boundary "
            "flux, time-dependent xi, and particle sampling density before assigning "
            "the residual to new physics.\n"
        )

    # Time-domain harmonic closure.
    u0=t[0]
    lhs_h=hseries(Zlhs,t,args.period,u0)
    em_h=hseries(Zem,t,args.period,u0)
    ki_h=hseries(Zki,t,args.period,u0)
    ke_h=hseries(Zke,t,args.period,u0)
    rhs_h=em_h+ki_h+ke_h
    scale=max(np.max(np.abs(lhs_h)),1e-300)

    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
    axs[0].plot(t,Pxi,"o-",label=r"$P_\xi$")
    axs[0].plot(t,hseries(Zp,t,args.period,u0)+np.mean(Pxi),"--",label="fundamental shape (offset)")
    axs[0].set_ylabel("modal particle momentum")
    axs[0].grid(alpha=.25); axs[0].legend()

    axs[1].plot(t,lhs_h/scale,"o-",lw=2,label=r"$dP_\xi/d\tau$")
    axs[1].plot(t,em_h/scale,"o-",label="EM")
    axs[1].plot(t,ki_h/scale,"o-",label="ion kinetic flux")
    axs[1].plot(t,ke_h/scale,"o-",label="electron kinetic flux")
    axs[1].plot(t,rhs_h/scale,"k--",lw=2,label="RHS sum")
    axs[1].axhline(0,lw=.7)
    axs[1].set_xlabel(r"$\omega_{ci}t$")
    axs[1].set_ylabel("fundamental / |LHS|")
    axs[1].grid(alpha=.25); axs[1].legend(fontsize=8,ncol=2)
    fig.suptitle("Non-affine first-moment closure at the breathing frequency")
    fig.tight_layout()
    fig.savefig("ishizawa_modal_momentum_closure_time.png",dpi=210)
    plt.close(fig)

    # Complex-plane closure.
    fig,ax=plt.subplots(figsize=(7,6.5))
    scale=max(abs(Zlhs),1e-300)
    for name,Z in [
        ("LHS dPxi/dtau",Zlhs),("EM",Zem),("ion kinetic",Zki),
        ("electron kinetic",Zke),("RHS sum",Zrhs),("residual",Zres)
    ]:
        zz=Z/scale
        ax.arrow(0,0,zz.real,zz.imag,length_includes_head=True,
                 head_width=.035,head_length=.05,alpha=.85)
        ax.text(zz.real*1.04,zz.imag*1.04,name,fontsize=9)
    ax.axhline(0,lw=.7); ax.axvline(0,lw=.7)
    ax.set_xlabel(r"Re$(Z)/|Z_{LHS}|$")
    ax.set_ylabel(r"Im$(Z)/|Z_{LHS}|$")
    ax.set_title(f"Projected momentum closure; complex error={rel:.3f}")
    ax.grid(alpha=.25); ax.set_aspect("equal",adjustable="datalim")
    fig.tight_layout()
    fig.savefig("ishizawa_modal_momentum_closure_complex.png",dpi=210)
    plt.close(fig)

    # Test-function geometry.
    M0=crossings[0]["M"]
    extent=[M0["x"][0]/P["de"],M0["x"][-1]/P["de"],
            M0["z"][0]/P["de"],M0["z"][-1]/P["de"]]
    fig,axs=plt.subplots(1,2,figsize=(10,4.4))
    im=axs[0].imshow((xix/P["de"]).T,origin="lower",aspect="auto",extent=extent)
    axs[0].set_title(r"$\xi_x/d_e$"); fig.colorbar(im,ax=axs[0],shrink=.8)
    im=axs[1].imshow((xiz/P["de"]).T,origin="lower",aspect="auto",extent=extent)
    axs[1].set_title(r"$\xi_z/d_e$"); fig.colorbar(im,ax=axs[1],shrink=.8)
    for ax in axs:
        ax.set_xlabel(r"$x/d_e$"); ax.set_ylabel(r"$z/d_e$")
    fig.suptitle("Fixed non-affine test displacement used in momentum closure")
    fig.tight_layout()
    fig.savefig("ishizawa_modal_momentum_closure_mode.png",dpi=210)
    plt.close(fig)

    print("="*78)
    print("Non-affine projected particle-momentum closure")
    print("="*78)
    print(f"points                         : {len(a)}")
    print(f"P_xi harmonic R2               : {Rp:.6f}")
    print(f"fundamental complex error      : {rel:.6f}")
    print(f"|RHS|/|LHS|                    : {abs(Zrhs)/max(abs(Zlhs),1e-300):.6f}")
    print(f"phase(RHS/LHS) [rad]           : {np.angle(Zrhs/Zlhs):+.6f}")
    print(f"raw finite-difference relRMS   : {raw_rel:.6f}")
    print("Saved ishizawa_modal_momentum_closure_summary.txt")
    print("Saved ishizawa_modal_momentum_closure_history.txt")
    print("Saved ishizawa_modal_momentum_closure_time.png")
    print("Saved ishizawa_modal_momentum_closure_complex.png")
    print("Saved ishizawa_modal_momentum_closure_mode.png")


if __name__=="__main__":
    main()
