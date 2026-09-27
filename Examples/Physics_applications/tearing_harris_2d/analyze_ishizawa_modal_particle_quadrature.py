#!/usr/bin/env python3
"""Direct particle-quadrature weak-form momentum closure for the breathing mode.

This is a stricter follow-up to analyze_ishizawa_modal_momentum_closure.py.
Instead of mixing native mesh density with NGP-binned normalized particle
moments, every term in the projected particle momentum equation is evaluated
from the *same macroparticles*:

  P_xi = sum_p W_p xi(x_p) . p_p

  Q_EM = sum_p W_p q_s xi(x_p) . [E(x_p) + v_p x B(x_p)]

  Q_kin = sum_p W_p [
      p_x (v_x d_x xi_x + v_z d_z xi_x)
    + p_z (v_x d_x xi_z + v_z d_z xi_z)
  ]

For a fixed test displacement xi on a finite x-z domain the exact weak-form
identity contains the kinetic-momentum surface flux

  dP_xi/dt = Q_EM + Q_kin - Q_surf,

  Q_surf = integral_boundary xi_i Pi_ij n_j dS.

With tau=omega_ci*t,

  dP_xi/dtau = (Q_EM + Q_kin - Q_surf)/omega_ci.

Q_surf is estimated directly from the same macroparticles in thin boundary
strips.  On periodic x, the right-minus-left contribution is a periodic-seam
diagnostic: it should cancel for a truly periodic test function and particle
stress.  The z contribution is the physical top-minus-bottom surface term.

All particle-weight normalization cancels from the relative closure so long as
the same WarpX particle weights are used consistently.  Grid E and B are
bilinearly interpolated to particle positions.  xi and grad(xi) are likewise
interpolated from the reconstructed non-affine breathing test function.

The purpose is to determine whether the ~25% residual seen in the mixed
mesh/particle first-moment diagnostic is mainly a deposition-consistency error
rather than missing physics.

Outputs
-------
ishizawa_particle_quadrature_closure_summary.txt
ishizawa_particle_quadrature_closure_history.txt
ishizawa_particle_quadrature_closure_time.png
ishizawa_particle_quadrature_closure_complex.png
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import QE, ME, C, params
from analyze_ishizawa_particle_moments import (
    load_mesh, deposit_species, particle_array
)
from analyze_ishizawa_nonaffine_breathing_mode import (
    step_from_parent, harmonic_fit, smooth2, weighted_mean
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--phase-root",default="runs_ishizawa_particle_phases/ppc49")
    p.add_argument("--em-history",required=True)
    p.add_argument("--period",type=float,default=0.63)
    p.add_argument("--tmin",type=float,default=3.20)
    p.add_argument("--tmax",type=float,default=3.80)
    p.add_argument("--mode-coarsen",type=int,default=4)
    p.add_argument("--field-coarsen",type=int,default=1,
                   help="coarsening used for E/B interpolation; 1 is recommended")
    p.add_argument("--core-z-de",type=float,default=12.0)
    p.add_argument("--mode-density-cut",type=float,default=0.02)
    p.add_argument("--smooth-passes",type=int,default=2)
    p.add_argument("--x-edge-taper-de",type=float,default=0.0,
                   help="cosine taper width from each periodic-x boundary, in de; "
                        "0 keeps the original untapered test function")
    p.add_argument("--periodic-x-gradient",action="store_true",
                   help="use a centered periodic derivative in x for grad(xi); "
                        "recommended for the periodic-x domain")
    p.add_argument("--boundary-strip-de",type=float,default=2.0,
                   help="width of the particle strip used to estimate each "
                        "weak-form surface flux, in de; default 2 de")
    return p.parse_args()


def fit_phasor(t,y,T):
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


def harmonic_series(Z,t,T,t0):
    om=2*np.pi/T
    return np.real(Z*np.exp(1j*om*(np.asarray(t)-t0)))


def interp_cc(a,xp,zp,x,z,periodic_x=True):
    """Bilinear interpolation from regular cell centers."""
    nx,nz=a.shape
    dx=x[1]-x[0]; dz=z[1]-z[0]

    fx=(xp-x[0])/dx
    if periodic_x:
        i0=np.floor(fx).astype(np.int64)
        tx=fx-i0
        i0=np.mod(i0,nx)
        i1=np.mod(i0+1,nx)
    else:
        fx=np.clip(fx,0,nx-1)
        i0=np.floor(fx).astype(np.int64)
        tx=fx-i0
        i1=np.minimum(i0+1,nx-1)

    fz=np.clip((zp-z[0])/dz,0,nz-1)
    j0=np.floor(fz).astype(np.int64)
    tz=fz-j0
    j1=np.minimum(j0+1,nz-1)

    return (
        (1-tx)*(1-tz)*a[i0,j0]
        + tx*(1-tz)*a[i1,j0]
        + (1-tx)*tz*a[i0,j1]
        + tx*tz*a[i1,j1]
    )


def particle_weak_terms(path,species,mass,charge,Mfield,Mmode,xix,xiz,grads,
                        boundary_strip_de,de):
    ds=yt.load(path)
    ad=ds.all_data()

    xp=particle_array(ad,species,"particle_position_x","m")
    zp=particle_array(ad,species,"particle_position_y","m")
    px=particle_array(ad,species,"particle_momentum_x","kg*m/s")
    py=particle_array(ad,species,"particle_momentum_y","kg*m/s")
    pz=particle_array(ad,species,"particle_momentum_z","kg*m/s")
    w =particle_array(ad,species,"particle_weight")

    gamma=np.sqrt(1.0+(px*px+py*py+pz*pz)/(mass*C)**2)
    vx=px/(gamma*mass); vy=py/(gamma*mass); vz=pz/(gamma*mass)

    # Fixed test function and gradients at particle positions.
    xi_x=interp_cc(xix,xp,zp,Mmode["x"],Mmode["z"])
    xi_z=interp_cc(xiz,xp,zp,Mmode["x"],Mmode["z"])
    dxx =interp_cc(grads[0],xp,zp,Mmode["x"],Mmode["z"])
    dxz =interp_cc(grads[1],xp,zp,Mmode["x"],Mmode["z"])
    dzx =interp_cc(grads[2],xp,zp,Mmode["x"],Mmode["z"])
    dzz =interp_cc(grads[3],xp,zp,Mmode["x"],Mmode["z"])

    # Fields at particles.  Plotfile E/B are treated as cell-centered samples.
    Ex=interp_cc(Mfield["Ex"],xp,zp,Mfield["x"],Mfield["z"])
    Ey=interp_cc(Mfield["Ey"],xp,zp,Mfield["x"],Mfield["z"])
    Ez=interp_cc(Mfield["Ez"],xp,zp,Mfield["x"],Mfield["z"])
    Bx=interp_cc(Mfield["Bx"],xp,zp,Mfield["x"],Mfield["z"])
    By=interp_cc(Mfield["By"],xp,zp,Mfield["x"],Mfield["z"])
    Bz=interp_cc(Mfield["Bz"],xp,zp,Mfield["x"],Mfield["z"])

    fx=charge*(Ex + vy*Bz - vz*By)
    fz=charge*(Ez + vx*By - vy*Bx)

    Pxi=np.sum(w*(xi_x*px + xi_z*pz))
    Qem=np.sum(w*(xi_x*fx + xi_z*fz))

    # Weak form of -div Pi after integration by parts.
    Qkin=np.sum(w*(
        px*(vx*dxx + vz*dxz)
        + pz*(vx*dzx + vz*dzz)
    ))

    # Particle-strip estimator of the weak-form surface term
    #
    #   Q_surf = int_boundary xi_i Pi_ij n_j dS.
    #
    # A boundary strip of width h converts the particle volume sum to a
    # surface-flux estimate by division by h.  Because exactly the same WarpX
    # particle weights are used as in Pxi/Qkin, the otherwise unknown 2-D
    # macro-particle normalization cancels in the relative closure.
    h=float(boundary_strip_de)*float(de)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("boundary_strip_de must be positive")
    hx=min(h,0.25*(Mfield["xhi"]-Mfield["xlo"]))
    hz=min(h,0.25*(Mfield["zhi"]-Mfield["zlo"]))

    pxi=xi_x*px + xi_z*pz
    fxsurf=w*vx*pxi
    fzsurf=w*vz*pxi

    left = xp <  Mfield["xlo"] + hx
    right= xp >= Mfield["xhi"] - hx
    bot  = zp <  Mfield["zlo"] + hz
    top  = zp >= Mfield["zhi"] - hz

    Bxl=np.sum(fxsurf[left])/hx
    Bxr=np.sum(fxsurf[right])/hx
    Bzb=np.sum(fzsurf[bot])/hz
    Bzt=np.sum(fzsurf[top])/hz

    # Outward-normal surface integral:
    # x: + at right, - at left; z: + at top, - at bottom.
    Bx=Bxr-Bxl
    Bz=Bzt-Bzb
    Bsurf=Bx+Bz
    surf=dict(total=float(Bsurf),x=float(Bx),z=float(Bz),
              left=float(Bxl),right=float(Bxr),
              bottom=float(Bzb),top=float(Bzt))

    return float(Pxi),float(Qem),float(Qkin),surf,int(len(w))


def main():
    args=parse_args()
    yt.funcs.mylog.setLevel(40)
    P=params()
    if args.boundary_strip_de <= 0.0:
        raise ValueError("--boundary-strip-de must be > 0")

    files=sorted(
        glob.glob(str(Path(args.phase_root)/"step*"/"diags"/"diag1*")),
        key=step_from_parent
    )
    if len(files)<6:
        raise RuntimeError(f"need >=6 particle-rich plotfiles; found {len(files)}")

    eh=np.loadtxt(args.em_history)
    if eh.ndim==1: eh=eh[None,:]
    ht=eh[:,1]; hq=eh[:,4]
    qfit,hqdot,hqddot,Rq,cq=harmonic_fit(ht,hq,args.period)

    # First pass: load only mode-resolution moments and identify states/crossings.
    states=[]
    for fn in files:
        M=load_mesh(fn,args.mode_coarsen)
        tci=P["wci"]*M["t"]
        if tci<args.tmin-1e-9 or tci>args.tmax+1e-9:
            continue
        I=deposit_species(fn,"ions",P["mi"],M,
                          dict(qe=QE,me=ME,mi=P["mi"],c=C,n0=P["n0"],
                               de=P["de"],wci=P["wci"],B0=P["B0"]))
        ni=np.abs(M["rho_i"])/QE
        states.append(dict(
            fn=fn,step=step_from_parent(fn),M=M,I=I,ni=ni,t=tci,
            q=float(np.interp(tci,ht,hq)),
            qdot=float(np.interp(tci,ht,hqdot))
        ))

    if len(states)<6:
        raise RuntimeError(f"only {len(states)} states remain in requested time window")
    states=sorted(states,key=lambda s:s["t"])

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
        ww=s["ni"]*core
        ux=s["I"]["ux"]; uz=s["I"]["uz"]
        ux0=weighted_mean(ux,ww); uz0=weighted_mean(uz,ww)
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

    if args.x_edge_taper_de > 0.0:
        x=crossings[0]["M"]["x"]
        xlo=crossings[0]["M"]["xlo"]
        xhi=crossings[0]["M"]["xhi"]
        w=args.x_edge_taper_de*P["de"]
        dedge=np.minimum(x-xlo,xhi-x)
        s=np.clip(dedge/max(w,np.finfo(float).tiny),0.0,1.0)
        xenv=0.5*(1.0-np.cos(np.pi*s))
        env*=xenv[:,None]

    xix*=env; xiz*=env

    dx=crossings[0]["M"]["dx"]; dz=crossings[0]["M"]["dz"]

    def ddx(a):
        if args.periodic_x_gradient:
            return (np.roll(a,-1,axis=0)-np.roll(a,1,axis=0))/(2.0*dx)
        return np.gradient(a,dx,axis=0,edge_order=2)

    grads=(
        ddx(xix),
        np.gradient(xix,dz,axis=1,edge_order=2),
        ddx(xiz),
        np.gradient(xiz,dz,axis=1,edge_order=2),
    )

    # Release heavy binned ion moment objects before direct particle pass.
    for s in states:
        s.pop("I",None); s.pop("ni",None)

    rows=[]
    for k,s in enumerate(states,1):
        Mfield=load_mesh(s["fn"],args.field_coarsen)
        Pi,QEi,QKi,Si,Ni=particle_weak_terms(
            s["fn"],"ions",P["mi"],+QE,Mfield,s["M"],xix,xiz,grads,
            args.boundary_strip_de,P["de"]
        )
        Pe,QEe,QKe,Se,Ne=particle_weak_terms(
            s["fn"],"electrons",ME,-QE,Mfield,s["M"],xix,xiz,grads,
            args.boundary_strip_de,P["de"]
        )
        Ptot=Pi+Pe
        Qem=QEi+QEe
        Qki=QKi/P["wci"]; Qke=QKe/P["wci"]; Rem=Qem/P["wci"]
        rhs_vol=Rem+Qki+Qke
        Bsi=Si["total"]/P["wci"]; Bse=Se["total"]/P["wci"]
        Bxi=Si["x"]/P["wci"]; Bxe=Se["x"]/P["wci"]
        Bzi=Si["z"]/P["wci"]; Bze=Se["z"]/P["wci"]
        rhs_corr=rhs_vol-(Bsi+Bse)
        rows.append([
            s["step"],s["t"],s["q"],Ptot,
            Rem,Qki,Qke,rhs_vol,
            Pi,Pe,QEi/P["wci"],QEe/P["wci"],Ni,Ne,
            Bsi,Bse,Bxi,Bxe,Bzi,Bze,rhs_corr
        ])
        print(
            f"[{k:02d}/{len(states):02d}] step={s['step']} tau={s['t']:.6f} "
            f"Ni={Ni} Ne={Ne}"
        )

    a=np.asarray(rows,float)
    np.savetxt(
        "ishizawa_particle_quadrature_closure_history.txt",a,
        header=(
            "step omega_ci_t q Pxi_total QEM_over_wci Qkin_i_over_wci "
            "Qkin_e_over_wci Qrhs_volume_over_wci Pxi_i Pxi_e "
            "QEM_i_over_wci QEM_e_over_wci Ni_macro Ne_macro "
            "Qsurf_i_over_wci Qsurf_e_over_wci "
            "Qsurf_x_i_over_wci Qsurf_x_e_over_wci "
            "Qsurf_z_i_over_wci Qsurf_z_e_over_wci "
            "Qrhs_corrected_over_wci"
        )
    )

    t=a[:,1]
    Zp,Rp,_,_=fit_phasor(t,a[:,3],args.period)
    omega=2*np.pi/args.period
    Zlhs=1j*omega*Zp

    Zem,RemR2,_,_=fit_phasor(t,a[:,4],args.period)
    Zki,RkiR2,_,_=fit_phasor(t,a[:,5],args.period)
    Zke,RkeR2,_,_=fit_phasor(t,a[:,6],args.period)
    Zrhs_vol=Zem+Zki+Zke

    Zsi,RsiR2,_,_=fit_phasor(t,a[:,14],args.period)
    Zse,RseR2,_,_=fit_phasor(t,a[:,15],args.period)
    Zsx_i,_,_,_=fit_phasor(t,a[:,16],args.period)
    Zsx_e,_,_,_=fit_phasor(t,a[:,17],args.period)
    Zsz_i,_,_,_=fit_phasor(t,a[:,18],args.period)
    Zsz_e,_,_,_=fit_phasor(t,a[:,19],args.period)
    Zsurf=Zsi+Zse
    Zsx=Zsx_i+Zsx_e
    Zsz=Zsz_i+Zsz_e

    Zrhs_corr=Zrhs_vol-Zsurf
    Zres_vol=Zlhs-Zrhs_vol
    Zres_corr=Zlhs-Zrhs_corr
    rel_vol=abs(Zres_vol)/max(abs(Zlhs),1e-300)
    rel_corr=abs(Zres_corr)/max(abs(Zlhs),1e-300)
    amp_vol=abs(Zrhs_vol)/max(abs(Zlhs),1e-300)
    amp_corr=abs(Zrhs_corr)/max(abs(Zlhs),1e-300)
    phase_vol=np.angle(Zrhs_vol/Zlhs)
    phase_corr=np.angle(Zrhs_corr/Zlhs)

    dPraw=np.gradient(a[:,3],t,edge_order=2)
    raw_rel_vol=np.sqrt(np.mean((dPraw-a[:,7])**2))/max(
        np.sqrt(np.mean(dPraw*dPraw)),1e-300
    )
    raw_rel_corr=np.sqrt(np.mean((dPraw-a[:,20])**2))/max(
        np.sqrt(np.mean(dPraw*dPraw)),1e-300
    )

    with open("ishizawa_particle_quadrature_closure_summary.txt","w") as f:
        f.write("Direct particle-quadrature weak-form momentum closure\n")
        f.write("=====================================================\n\n")
        f.write("Equation: dP_xi/dtau = (Q_EM + Q_kin,i + Q_kin,e - Q_surf)/omega_ci\n")
        f.write("All P, force, and kinetic-flux moments use the same WarpX particle weights.\n\n")
        f.write(f"points = {len(a)}\n")
        f.write(f"window omega_ci*t = {t.min():.8f} .. {t.max():.8f}\n")
        f.write(f"period omega_ci*T = {args.period:.8f}\n")
        f.write(f"omega_b/omega_ci = {omega:.8f}\n")
        f.write(f"q harmonic R2 = {Rq:.8f}\n")
        f.write(f"P_xi harmonic R2 = {Rp:.8f}\n")
        f.write(f"EM harmonic R2 = {RemR2:.8f}\n")
        f.write(f"ion kinetic harmonic R2 = {RkiR2:.8f}\n")
        f.write(f"electron kinetic harmonic R2 = {RkeR2:.8f}\n")
        f.write(f"mode crossings = {s_neg['step']} {s_pos['step']}\n")
        f.write(f"mode coarsen = {args.mode_coarsen}\n")
        f.write(f"field coarsen = {args.field_coarsen}\n")
        f.write(f"x edge taper / de = {args.x_edge_taper_de:.8f}\n")
        f.write(f"periodic x gradient = {int(args.periodic_x_gradient)}\n")
        f.write(f"boundary strip / de = {args.boundary_strip_de:.8f}\n\n")

        f.write("Breathing-frequency complex amplitudes\n")
        f.write("--------------------------------------\n")
        f.write("channel                 Re(Z)              Im(Z)             |Z|\n")
        for name,Z in [
            ("dPxi/dtau",Zlhs),("EM",Zem),("ion kinetic",Zki),
            ("electron kinetic",Zke),("surface ion",Zsi),
            ("surface electron",Zse),("surface x total",Zsx),
            ("surface z total",Zsz),("surface total",Zsurf),
            ("volume RHS",Zrhs_vol),("corrected RHS",Zrhs_corr),
            ("volume residual",Zres_vol),("corrected residual",Zres_corr)
        ]:
            f.write(f"{name:18s} {Z.real:+17.8e} {Z.imag:+17.8e} {abs(Z):17.8e}\n")
        f.write("\n")
        f.write(f"volume-only fundamental complex closure relative error = {rel_vol:.8e}\n")
        f.write(f"surface-corrected fundamental complex closure relative error = {rel_corr:.8e}\n")
        f.write(f"|volume rhs|/|lhs| = {amp_vol:.8f}\n")
        f.write(f"|corrected rhs|/|lhs| = {amp_corr:.8f}\n")
        f.write(f"phase(volume rhs/lhs) [rad] = {phase_vol:+.8f}\n")
        f.write(f"phase(corrected rhs/lhs) [rad] = {phase_corr:+.8f}\n")
        f.write(f"surface ion harmonic R2 = {RsiR2:.8f}\n")
        f.write(f"surface electron harmonic R2 = {RseR2:.8f}\n")
        f.write(f"raw FD volume-only relRMS (diagnostic only) = {raw_rel_vol:.8e}\n")
        f.write(f"raw FD surface-corrected relRMS (diagnostic only) = {raw_rel_corr:.8e}\n\n")
        f.write("Interpretation guide\n")
        f.write("--------------------\n")
        f.write(
            "If this particle-quadrature closure is substantially better than the "
            "mixed mesh/particle closure, the old residual was mainly deposition/"
            "normalization inconsistency. If a comparable residual remains, the "
            "next checks are field interpolation/staggering and finite time sampling. "
            "The explicit surface term reported here is a particle-strip estimate; "
            "on periodic x its right-minus-left piece is a seam-consistency diagnostic, "
            "whereas the top-minus-bottom z piece is the physical surface flux. "
            "Because xi is fixed in this equation, "
            "a physical time-dependent mode shape is not itself a missing term in "
            "this exact fixed-test-function conservation identity.\n"
        )

    t0=t[0]
    lhs_h=harmonic_series(Zlhs,t,args.period,t0)
    em_h=harmonic_series(Zem,t,args.period,t0)
    ki_h=harmonic_series(Zki,t,args.period,t0)
    ke_h=harmonic_series(Zke,t,args.period,t0)
    rhs_vol_h=em_h+ki_h+ke_h
    surf_h=harmonic_series(Zsurf,t,args.period,t0)
    rhs_corr_h=rhs_vol_h-surf_h
    scale=max(np.max(np.abs(lhs_h)),1e-300)

    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
    axs[0].plot(t,a[:,3],"o-",label=r"$P_\xi$ direct particles")
    axs[0].grid(alpha=.25); axs[0].legend()
    axs[0].set_ylabel("modal particle momentum")

    axs[1].plot(t,lhs_h/scale,"o-",lw=2,label=r"$dP_\xi/d\tau$")
    axs[1].plot(t,em_h/scale,"o-",label="EM")
    axs[1].plot(t,ki_h/scale,"o-",label="ion kinetic flux")
    axs[1].plot(t,ke_h/scale,"o-",label="electron kinetic flux")
    axs[1].plot(t,-surf_h/scale,"o-",label="- surface flux")
    axs[1].plot(t,rhs_vol_h/scale,"--",lw=1.5,label="volume RHS")
    axs[1].plot(t,rhs_corr_h/scale,"k--",lw=2,label="surface-corrected RHS")
    axs[1].axhline(0,lw=.7); axs[1].grid(alpha=.25)
    axs[1].legend(fontsize=8,ncol=2)
    axs[1].set_xlabel(r"$\omega_{ci}t$")
    axs[1].set_ylabel("fundamental / |LHS|")
    fig.suptitle("Direct particle-quadrature momentum closure")
    fig.tight_layout()
    fig.savefig("ishizawa_particle_quadrature_closure_time.png",dpi=210)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,6.5))
    s=max(abs(Zlhs),1e-300)
    for name,Z in [
        ("LHS",Zlhs),("EM",Zem),("ion kinetic",Zki),
        ("electron kinetic",Zke),("-surface",-Zsurf),
        ("volume RHS",Zrhs_vol),("corrected RHS",Zrhs_corr),
        ("corrected residual",Zres_corr)
    ]:
        zz=Z/s
        ax.arrow(0,0,zz.real,zz.imag,length_includes_head=True,
                 head_width=.035,head_length=.05,alpha=.85)
        ax.text(1.04*zz.real,1.04*zz.imag,name,fontsize=9)
    ax.axhline(0,lw=.7); ax.axvline(0,lw=.7)
    ax.set_xlabel(r"Re$(Z)/|Z_{LHS}|$")
    ax.set_ylabel(r"Im$(Z)/|Z_{LHS}|$")
    ax.set_title(
        f"Particle-quadrature closure; volume={rel_vol:.3f}, "
        f"surface-corrected={rel_corr:.3f}"
    )
    ax.grid(alpha=.25); ax.set_aspect("equal",adjustable="datalim")
    fig.tight_layout()
    fig.savefig("ishizawa_particle_quadrature_closure_complex.png",dpi=210)
    plt.close(fig)

    print("="*78)
    print("Direct particle-quadrature weak-form momentum closure")
    print("="*78)
    print(f"points                    : {len(a)}")
    print(f"P_xi harmonic R2          : {Rp:.6f}")
    print(f"volume-only complex error : {rel_vol:.6f}")
    print(f"surface-corrected error   : {rel_corr:.6f}")
    print(f"|volume RHS|/|LHS|        : {amp_vol:.6f}")
    print(f"|corrected RHS|/|LHS|     : {amp_corr:.6f}")
    print(f"phase corrected [rad]     : {phase_corr:+.6f}")
    print(f"raw FD volume relRMS      : {raw_rel_vol:.6f}")
    print(f"raw FD corrected relRMS   : {raw_rel_corr:.6f}")
    print(f"x edge taper / de         : {args.x_edge_taper_de:.3f}")
    print(f"boundary strip / de       : {args.boundary_strip_de:.3f}")
    print(f"periodic x gradient       : {args.periodic_x_gradient}")
    print("Saved ishizawa_particle_quadrature_closure_summary.txt")
    print("Saved ishizawa_particle_quadrature_closure_history.txt")
    print("Saved ishizawa_particle_quadrature_closure_time.png")
    print("Saved ishizawa_particle_quadrature_closure_complex.png")


if __name__=="__main__":
    main()
