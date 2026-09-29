#!/usr/bin/env python3
"""Ion kinetic/stress decomposition for the q-fixed Ishizawa breathing mode.

Starting from the exact particle weak-form kinetic term

    Q_kin,i = sum_p W_p [
        p_x (v_x d_x xi_x + v_z d_z xi_x)
      + p_z (v_x d_x xi_z + v_z d_z xi_z)
    ],

this diagnostic performs two complementary decompositions.

1) Exact particle tensor split (machine-precision identity)

       Q_kin,i = Q_xx + Q_xz + Q_zx + Q_zz

   with
       Q_xx = sum W p_x v_x d_x xi_x
       Q_xz = sum W p_x v_z d_z xi_x
       Q_zx = sum W p_z v_x d_x xi_z
       Q_zz = sum W p_z v_z d_z xi_z.

2) Coarse-cell bulk/random stress split on the same grid used to construct xi.

   In each coarse cell c,
       Pi_ij(c) = sum_{p in c} W p_i v_j,
       Pi_ij^bulk(c) = (sum W p_i)(sum W v_j)/(sum W),
       Pi_ij^rand(c) = Pi_ij(c) - Pi_ij^bulk(c).

   Contracting these cell moments with grad(xi) gives a controlled
   coarse-grained split into bulk/advective and random/pressure-like stress.
   The script reports the binned reconstruction error relative to the exact
   particle quadrature, so this physical split is never silently treated as
   exact if the coarse grid is insufficient.

The breathing coordinate is reconstructed with the same q-fixed logic as the
validated closure diagnostic.

Outputs
-------
ishizawa_ion_kinetic_decomposition_history.txt
ishizawa_ion_kinetic_decomposition_summary.txt
ishizawa_ion_kinetic_tensor_time.png
ishizawa_ion_kinetic_tensor_complex.png
ishizawa_ion_bulk_random_time.png
ishizawa_ion_bulk_random_complex.png
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
from analyze_ishizawa_modal_particle_quadrature import (
    fit_phasor, interp_cc, load_breathing_coordinate_history
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--phase-root",required=True)
    p.add_argument("--em-history",required=True)
    p.add_argument("--period",type=float,required=True)
    p.add_argument("--tmin",type=float,default=4.05)
    p.add_argument("--tmax",type=float,default=5.90)
    p.add_argument("--q-tmin",type=float,default=None,
                   help="lower bound used only to normalize saturation width into q; default: --tmin")
    p.add_argument("--q-tmax",type=float,default=None,
                   help="upper bound used only to normalize saturation width into q; default: --tmax")
    p.add_argument("--mode-coarsen",type=int,default=4)
    p.add_argument("--core-z-de",type=float,default=12.0)
    p.add_argument("--mode-density-cut",type=float,default=0.02)
    p.add_argument("--smooth-passes",type=int,default=2)
    p.add_argument("--x-edge-taper-de",type=float,default=0.0)
    p.add_argument("--periodic-x-gradient",action="store_true")
    p.add_argument("--out-prefix",default="ishizawa_ion_kinetic_decomposition")
    return p.parse_args()


def build_fixed_mode(args,P):
    files=sorted(
        set(
            glob.glob(str(Path(args.phase_root)/"step*"/"diags"/"diag1*"))
            + glob.glob(str(Path(args.phase_root)/"diags"/"diag1*"))
        ),
        key=step_from_parent
    )
    if len(files)<6:
        raise RuntimeError(f"need >=6 particle-rich plotfiles; found {len(files)}")

    q_tmin=args.tmin if args.q_tmin is None else args.q_tmin
    q_tmax=args.tmax if args.q_tmax is None else args.q_tmax
    ht,hq,q_source_kind=load_breathing_coordinate_history(
        args.em_history,q_tmin,q_tmax
    )
    _,hqdot,_,Rq,_=harmonic_fit(ht,hq,args.period)

    states=[]
    for fn in files:
        ds=yt.load(fn)
        tau=float(P["wci"]*ds.current_time.to_value("s"))
        if tau<args.tmin-1e-9 or tau>args.tmax+1e-9:
            continue
        states.append(dict(
            fn=fn,step=step_from_parent(fn),t=tau,
            q=float(np.interp(tau,ht,hq)),
            qdot=float(np.interp(tau,ht,hqdot))
        ))
    if len(states)<6:
        raise RuntimeError(f"only {len(states)} states in requested window")
    states=sorted(states,key=lambda s:s["t"])

    pos=[s for s in states if s["qdot"]>0]
    neg=[s for s in states if s["qdot"]<0]
    if not pos or not neg:
        raise RuntimeError("states do not contain opposite-sign qdot crossings")
    s_pos=min(pos,key=lambda s:abs(s["q"]))
    s_neg=min(neg,key=lambda s:abs(s["q"]))
    crossing_meta=[s_neg,s_pos]

    xis=[]
    nis=[]
    meshes=[]
    for s in crossing_meta:
        M=load_mesh(s["fn"],args.mode_coarsen)
        I=deposit_species(
            s["fn"],"ions",P["mi"],M,
            dict(qe=QE,me=ME,mi=P["mi"],c=C,n0=P["n0"],
                 de=P["de"],wci=P["wci"],B0=P["B0"])
        )
        ni=np.abs(M["rho_i"])/QE
        core=(np.abs(M["z"])[None,:] <= args.core_z_de*P["de"])
        ww=ni*core
        ux0=weighted_mean(I["ux"],ww)
        uz0=weighted_mean(I["uz"],ww)
        qdot_phys=P["wci"]*s["qdot"]
        xis.append(((I["ux"]-ux0)/qdot_phys,(I["uz"]-uz0)/qdot_phys))
        nis.append(ni)
        meshes.append(M)

    xix=0.5*(xis[0][0]+xis[1][0])
    xiz=0.5*(xis[0][1]+xis[1][1])
    xix=smooth2(xix,args.smooth_passes)
    xiz=smooth2(xiz,args.smooth_passes)

    nmean=0.5*(nis[0]+nis[1])
    dens_env=nmean/(nmean+args.mode_density_cut*P["n0"])

    M=meshes[0]
    z=M["z"]
    zmax=args.core_z_de*P["de"]
    zenv=np.zeros_like(z)
    inside=np.abs(z)<zmax
    zenv[inside]=0.5*(1.0+np.cos(np.pi*z[inside]/zmax))
    env=dens_env*zenv[None,:]

    if args.x_edge_taper_de>0:
        x=M["x"]
        w=args.x_edge_taper_de*P["de"]
        dedge=np.minimum(x-M["xlo"],M["xhi"]-x)
        ss=np.clip(dedge/max(w,np.finfo(float).tiny),0.0,1.0)
        env*= (0.5*(1.0-np.cos(np.pi*ss)))[:,None]

    xix*=env
    xiz*=env

    dx=M["dx"]; dz=M["dz"]
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
    return states,M,xix,xiz,grads,crossing_meta,q_source_kind,Rq,q_tmin,q_tmax


def bincount2(cell,val,ncell,nx,nz):
    return np.bincount(
        cell,weights=val,minlength=ncell
    ).reshape(nx,nz)


def ion_terms(path,P,Mmode,grads):
    ds=yt.load(path)
    ad=ds.all_data()

    xp=particle_array(ad,"ions","particle_position_x","m")
    zp=particle_array(ad,"ions","particle_position_y","m")
    px=particle_array(ad,"ions","particle_momentum_x","kg*m/s")
    py=particle_array(ad,"ions","particle_momentum_y","kg*m/s")
    pz=particle_array(ad,"ions","particle_momentum_z","kg*m/s")
    w =particle_array(ad,"ions","particle_weight")

    gamma=np.sqrt(1.0+(px*px+py*py+pz*pz)/(P["mi"]*C)**2)
    vx=px/(gamma*P["mi"])
    vz=pz/(gamma*P["mi"])

    gxx=interp_cc(grads[0],xp,zp,Mmode["x"],Mmode["z"])
    gxz=interp_cc(grads[1],xp,zp,Mmode["x"],Mmode["z"])
    gzx=interp_cc(grads[2],xp,zp,Mmode["x"],Mmode["z"])
    gzz=interp_cc(grads[3],xp,zp,Mmode["x"],Mmode["z"])

    qxx=float(np.sum(w*px*vx*gxx))
    qxz=float(np.sum(w*px*vz*gxz))
    qzx=float(np.sum(w*pz*vx*gzx))
    qzz=float(np.sum(w*pz*vz*gzz))
    qdirect=qxx+qxz+qzx+qzz

    # Coarse-cell relativistic momentum-flux moments.
    nx,nz=Mmode["nx"],Mmode["nz"]
    ix=np.floor((xp-Mmode["xlo"])/Mmode["dx"]).astype(np.int64)
    iz=np.floor((zp-Mmode["zlo"])/Mmode["dz"]).astype(np.int64)
    ix=np.mod(ix,nx)
    iz=np.clip(iz,0,nz-1)
    cell=ix*nz+iz
    ncell=nx*nz

    W=bincount2(cell,w,ncell,nx,nz)
    Spx=bincount2(cell,w*px,ncell,nx,nz)
    Spz=bincount2(cell,w*pz,ncell,nx,nz)
    Svx=bincount2(cell,w*vx,ncell,nx,nz)
    Svz=bincount2(cell,w*vz,ncell,nx,nz)

    Sxx=bincount2(cell,w*px*vx,ncell,nx,nz)
    Sxz=bincount2(cell,w*px*vz,ncell,nx,nz)
    Szx=bincount2(cell,w*pz*vx,ncell,nx,nz)
    Szz=bincount2(cell,w*pz*vz,ncell,nx,nz)

    def bulk(Sp,Sv):
        out=np.zeros_like(W)
        m=W>0
        out[m]=Sp[m]*Sv[m]/W[m]
        return out

    Bxx=bulk(Spx,Svx); Bxz=bulk(Spx,Svz)
    Bzx=bulk(Spz,Svx); Bzz=bulk(Spz,Svz)
    Rxx=Sxx-Bxx; Rxz=Sxz-Bxz
    Rzx=Szx-Bzx; Rzz=Szz-Bzz

    Gxx,Gxz,Gzx,Gzz=grads
    qbin_xx=float(np.sum(Sxx*Gxx))
    qbin_xz=float(np.sum(Sxz*Gxz))
    qbin_zx=float(np.sum(Szx*Gzx))
    qbin_zz=float(np.sum(Szz*Gzz))

    qbulk_xx=float(np.sum(Bxx*Gxx))
    qbulk_xz=float(np.sum(Bxz*Gxz))
    qbulk_zx=float(np.sum(Bzx*Gzx))
    qbulk_zz=float(np.sum(Bzz*Gzz))

    qrand_xx=float(np.sum(Rxx*Gxx))
    qrand_xz=float(np.sum(Rxz*Gxz))
    qrand_zx=float(np.sum(Rzx*Gzx))
    qrand_zz=float(np.sum(Rzz*Gzz))

    return dict(
        N=len(w),
        exact=np.array([qxx,qxz,qzx,qzz],float),
        direct=qdirect,
        binned=np.array([qbin_xx,qbin_xz,qbin_zx,qbin_zz],float),
        bulk=np.array([qbulk_xx,qbulk_xz,qbulk_zx,qbulk_zz],float),
        rand=np.array([qrand_xx,qrand_xz,qrand_zx,qrand_zz],float),
    )


def phasor_stats(t,q,series,T):
    Zq,_,_,_=fit_phasor(t,q,T)
    out={}
    for name,y in series.items():
        Z,R2,_,_=fit_phasor(t,y,T)
        out[name]=dict(
            Z=Z,R2=R2,
            amp_over_q=abs(Z)/max(abs(Zq),1e-300),
            phase_q=float(np.angle(Z/Zq)) if abs(Zq)>0 else np.nan,
            restoring_alignment=float(np.cos(np.angle(Z/(-Zq)))) if abs(Zq)>0 else np.nan,
            damping_alignment=float(
                np.cos(np.angle(Z/(-1j*Zq)))
            ) if abs(Zq)>0 else np.nan,
        )
    return Zq,out


def main():
    args=parse_args()
    yt.funcs.mylog.setLevel(40)
    P=params()

    states,Mmode,xix,xiz,grads,crossings,qkind,Rq,q_tmin,q_tmax=build_fixed_mode(args,P)

    rows=[]
    for k,s in enumerate(states,1):
        R=ion_terms(s["fn"],P,Mmode,grads)
        ex=R["exact"]/P["wci"]
        bn=R["binned"]/P["wci"]
        bu=R["bulk"]/P["wci"]
        rr=R["rand"]/P["wci"]
        direct=R["direct"]/P["wci"]
        rows.append([
            s["step"],s["t"],s["q"],direct,
            *ex,
            np.sum(bn),*bn,
            np.sum(bu),*bu,
            np.sum(rr),*rr,
            R["N"]
        ])
        print(
            f"[{k:02d}/{len(states):02d}] step={s['step']} tau={s['t']:.6f} "
            f"Nion={R['N']} bin/direct={np.sum(bn)/direct:+.6f}"
        )

    A=np.asarray(rows,float)
    header=(
        "step omega_ci_t q Qkin_i_direct "
        "Qxx_exact Qxz_exact Qzx_exact Qzz_exact "
        "Qbin_total Qxx_bin Qxz_bin Qzx_bin Qzz_bin "
        "Qbulk_total Qxx_bulk Qxz_bulk Qzx_bulk Qzz_bulk "
        "Qrand_total Qxx_rand Qxz_rand Qzx_rand Qzz_rand Nion_macro"
    )
    np.savetxt(args.out_prefix+"_history.txt",A,header=header)

    t=A[:,1]; q=A[:,2]
    names_tensor=["Qxx","Qxz","Qzx","Qzz"]
    tensor={n:A[:,4+i] for i,n in enumerate(names_tensor)}
    exact_total=A[:,3]

    bulk={
        "bulk_total":A[:,13],
        "bulk_xx":A[:,14],"bulk_xz":A[:,15],
        "bulk_zx":A[:,16],"bulk_zz":A[:,17],
    }
    rand={
        "rand_total":A[:,18],
        "rand_xx":A[:,19],"rand_xz":A[:,20],
        "rand_zx":A[:,21],"rand_zz":A[:,22],
    }
    binned_total=A[:,8]

    Zq,st_tensor=phasor_stats(t,q,tensor,args.period)
    _,st_br=phasor_stats(
        t,q,{"direct":exact_total,"binned":binned_total,**bulk,**rand},args.period
    )
    Zkin=st_br["direct"]["Z"]

    # Machine identity of exact tensor split and coarse-bin reconstruction.
    exact_sum=np.sum(A[:,4:8],axis=1)
    ident_rel=np.sqrt(np.mean((exact_sum-exact_total)**2))/max(
        np.sqrt(np.mean(exact_total**2)),1e-300
    )
    bin_rel=np.sqrt(np.mean((binned_total-exact_total)**2))/max(
        np.sqrt(np.mean(exact_total**2)),1e-300
    )
    bulk_rand_rel=np.sqrt(np.mean((A[:,13]+A[:,18]-binned_total)**2))/max(
        np.sqrt(np.mean(binned_total**2)),1e-300
    )

    with open(args.out_prefix+"_summary.txt","w") as f:
        f.write("Ion kinetic/stress decomposition for q-fixed breathing mode\n")
        f.write("==========================================================\n\n")
        f.write(f"points = {len(A)}\n")
        f.write(f"window omega_ci*t = {t.min():.8f} .. {t.max():.8f}\n")
        f.write(f"period omega_ci*T = {args.period:.8f}\n")
        f.write(f"q harmonic R2 = {Rq:.8f}\n")
        f.write(f"mode crossings = {crossings[0]['step']} {crossings[1]['step']}\n")
        f.write(f"breathing coordinate source = {qkind}\n")
        f.write(f"q normalization window = {q_tmin:.8f} .. {q_tmax:.8f}\n")
        f.write(f"mode coarsen = {args.mode_coarsen}\n")
        f.write(f"periodic x gradient = {int(args.periodic_x_gradient)}\n\n")

        f.write("Consistency checks\n")
        f.write("------------------\n")
        f.write(f"exact tensor-sum time-domain relRMS = {ident_rel:.8e}\n")
        f.write(f"binned-vs-direct time-domain relRMS = {bin_rel:.8e}\n")
        f.write(f"bulk+random vs binned relRMS = {bulk_rand_rel:.8e}\n\n")

        f.write("Exact ion tensor phasors at breathing frequency\n")
        f.write("----------------------------------------------\n")
        f.write("channel      Re(Z)            Im(Z)            |Z|/|Zkin|   R2       phase(Q/q)  align(-q) align(-qdot)\n")
        for n in names_tensor:
            st=st_tensor[n]; Z=st["Z"]
            f.write(
                f"{n:8s} {Z.real:+14.6e} {Z.imag:+14.6e} "
                f"{abs(Z)/max(abs(Zkin),1e-300):12.6f} {st['R2']:8.5f} "
                f"{st['phase_q']:+11.6f} {st['restoring_alignment']:+9.5f} "
                f"{st['damping_alignment']:+11.5f}\n"
            )

        f.write("\nBulk/random coarse-grained phasors\n")
        f.write("-----------------------------------\n")
        for n in ["direct","binned","bulk_total","rand_total"]:
            st=st_br[n]; Z=st["Z"]
            f.write(
                f"{n:12s} Re={Z.real:+.8e} Im={Z.imag:+.8e} "
                f"|Z|/|Zdirect|={abs(Z)/max(abs(Zkin),1e-300):.8f} "
                f"R2={st['R2']:.8f} phase(Q/q)={st['phase_q']:+.8f} "
                f"align(-q)={st['restoring_alignment']:+.8f} "
                f"align(-qdot)={st['damping_alignment']:+.8f}\n"
            )

        f.write("\nBulk tensor components\n")
        for n in ["bulk_xx","bulk_xz","bulk_zx","bulk_zz"]:
            st=st_br[n]; Z=st["Z"]
            f.write(
                f"{n:12s} |Z|/|Zdirect|={abs(Z)/max(abs(Zkin),1e-300):.8f} "
                f"phase(Q/q)={st['phase_q']:+.8f} R2={st['R2']:.8f}\n"
            )

        f.write("\nRandom/pressure-like tensor components\n")
        for n in ["rand_xx","rand_xz","rand_zx","rand_zz"]:
            st=st_br[n]; Z=st["Z"]
            f.write(
                f"{n:12s} |Z|/|Zdirect|={abs(Z)/max(abs(Zkin),1e-300):.8f} "
                f"phase(Q/q)={st['phase_q']:+.8f} R2={st['R2']:.8f}\n"
            )

        f.write("\nInterpretation\n")
        f.write("--------------\n")
        f.write(
            "The Qxx/Qxz/Qzx/Qzz split is an exact particle identity. "
            "The bulk/random split is a coarse-cell relativistic momentum-flux "
            "decomposition on the mode grid; use the reported binned-vs-direct "
            "error to judge its quantitative accuracy. align(-q)=+1 denotes a "
            "pure restoring-phase contribution at the breathing frequency; "
            "align(-qdot)=+1 denotes a pure damping-phase contribution.\n"
        )

    # Exact tensor harmonic time histories.
    Zs=[st_tensor[n]["Z"] for n in names_tensor]
    scale=max(abs(Zkin),1e-300)
    t0=t[0]; om=2*np.pi/args.period
    def hs(Z):
        return np.real(Z*np.exp(1j*om*(t-t0)))/scale

    fig,ax=plt.subplots(figsize=(9,5.7))
    for n,Z in zip(names_tensor,Zs):
        ax.plot(t,hs(Z),"o-",label=n)
    ax.plot(t,hs(Zkin),"k--",lw=2,label="sum / ion kinetic")
    ax.axhline(0,lw=.7)
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel(r"fundamental / $|Z_{kin,i}|$")
    ax.set_title("Ion kinetic tensor decomposition")
    ax.grid(alpha=.25); ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(args.out_prefix+"_tensor_time.png",dpi=210)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,6.5))
    for n,Z in zip(names_tensor,Zs):
        z=Z/scale
        ax.arrow(0,0,z.real,z.imag,length_includes_head=True,
                 head_width=.025,head_length=.04)
        ax.text(z.real,z.imag,n)
    z=Zkin/scale
    ax.arrow(0,0,z.real,z.imag,length_includes_head=True,
             head_width=.025,head_length=.04)
    ax.text(z.real,z.imag,"ion kinetic")
    ax.axhline(0,lw=.7); ax.axvline(0,lw=.7)
    ax.set_xlabel(r"Re$(Z)/|Z_{kin,i}|$")
    ax.set_ylabel(r"Im$(Z)/|Z_{kin,i}|$")
    ax.set_title("Ion kinetic tensor phasors")
    ax.grid(alpha=.25); ax.set_aspect("equal",adjustable="datalim")
    fig.tight_layout()
    fig.savefig(args.out_prefix+"_tensor_complex.png",dpi=210)
    plt.close(fig)

    # Bulk/random harmonic comparison.
    fig,ax=plt.subplots(figsize=(9,5.7))
    for n in ["direct","binned","bulk_total","rand_total"]:
        ax.plot(t,hs(st_br[n]["Z"]),"o-",label=n)
    ax.axhline(0,lw=.7)
    ax.set_xlabel(r"$\omega_{ci}t$")
    ax.set_ylabel(r"fundamental / $|Z_{kin,i}|$")
    ax.set_title("Ion bulk vs random/pressure-like momentum flux")
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_prefix+"_bulk_random_time.png",dpi=210)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,6.5))
    for n in ["bulk_total","rand_total","binned","direct"]:
        Z=st_br[n]["Z"]/scale
        ax.arrow(0,0,Z.real,Z.imag,length_includes_head=True,
                 head_width=.025,head_length=.04)
        ax.text(Z.real,Z.imag,n)
    ax.axhline(0,lw=.7); ax.axvline(0,lw=.7)
    ax.set_xlabel(r"Re$(Z)/|Z_{kin,i}|$")
    ax.set_ylabel(r"Im$(Z)/|Z_{kin,i}|$")
    ax.set_title("Ion bulk/random stress phasors")
    ax.grid(alpha=.25); ax.set_aspect("equal",adjustable="datalim")
    fig.tight_layout()
    fig.savefig(args.out_prefix+"_bulk_random_complex.png",dpi=210)
    plt.close(fig)

    print("Saved",args.out_prefix+"_summary.txt")
    print("Saved",args.out_prefix+"_history.txt")
    print("Saved",args.out_prefix+"_tensor_time.png")
    print("Saved",args.out_prefix+"_tensor_complex.png")
    print("Saved",args.out_prefix+"_bulk_random_time.png")
    print("Saved",args.out_prefix+"_bulk_random_complex.png")


if __name__=="__main__":
    main()
