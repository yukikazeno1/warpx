#!/usr/bin/env python3
"""Non-affine breathing-mode projection for the saturated Ishizawa-scale island.

This diagnostic addresses a key limitation of the affine test xi_z = z q.
The breathing eigen-displacement is reconstructed directly from the ion bulk
velocity at the two near-zero-q crossings:

    xi(x,z) ~= u_i(x,z) / qdot_phys,

after removing rigid x-z translation.  The two opposite crossings are averaged
so incoherent PIC noise is reduced.

The full 2-D mode xi=(xi_x,xi_z) is then used to project the total momentum
balance,

    rho du/dt = rho_c E + JxB - div(P_i+P_e)
                - div(rho_i u_i u_i + rho_e u_e u_e),

onto the measured breathing coordinate q.  This restores the spatial
weighting that is lost when one assumes xi_z proportional to z.

The script uses the existing four particle-rich phase snapshots.  It is a
mechanism screen; if the non-affine projection closes substantially better,
a denser particle-rich phase scan should follow before quoting a final
kinetic stiffness or eigenfrequency.

Outputs
-------
ishizawa_nonaffine_mode_summary.txt
ishizawa_nonaffine_mode_history.txt
ishizawa_nonaffine_mode_shape.png
ishizawa_nonaffine_force_balance.png
ishizawa_nonaffine_force_vs_q.png
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import QE, ME, C, params
from analyze_ishizawa_particle_moments import load_mesh, deposit_species


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--phase-root",default="runs_ishizawa_particle_phases/ppc49")
    p.add_argument("--em-history",required=True)
    p.add_argument("--coarsen",type=int,default=4)
    p.add_argument("--period",type=float,default=0.63)
    p.add_argument("--core-z-de",type=float,default=12.0)
    p.add_argument("--mode-density-cut",type=float,default=0.02,
                   help="smooth density cutoff n_i/n0 used for mode envelope")
    p.add_argument("--smooth-passes",type=int,default=2)
    return p.parse_args()


def step_from_parent(path):
    for part in Path(path).parts[::-1]:
        m=re.fullmatch(r"step(\d+)",part)
        if m:
            return int(m.group(1))
    base=os.path.basename(path.rstrip("/"))
    # WarpX Full diagnostics use names such as diag1162000, i.e. the
    # diagnostic name "diag1" followed by step 162000.  Strip that prefix
    # before falling back to a generic trailing-integer parser.
    m=re.fullmatch(r"diag1(\d+)",base)
    if m:
        return int(m.group(1))
    m=re.search(r"(\d+)$",base)
    return int(m.group(1)) if m else -1


def harmonic_fit(t,y,T):
    t=np.asarray(t,float); y=np.asarray(y,float)
    u=t-t[0]
    om=2*np.pi/T
    M=np.column_stack([np.ones_like(u),u,np.sin(om*u),np.cos(om*u)])
    c,*_=np.linalg.lstsq(M,y,rcond=None)
    fit=M@c
    qdot=om*(c[2]*np.cos(om*u)-c[3]*np.sin(om*u))
    qddot=-(om**2)*(c[2]*np.sin(om*u)+c[3]*np.cos(om*u))
    ssr=np.sum((y-fit)**2)
    sst=np.sum((y-np.mean(y))**2)
    r2=1-ssr/sst if sst>0 else np.nan
    return fit,qdot,qddot,float(r2),c


def smooth2(a,n):
    a=np.asarray(a,float).copy()
    for _ in range(max(0,n)):
        # x is periodic; z uses edge replication.
        xp=np.roll(a,1,axis=0); xm=np.roll(a,-1,axis=0)
        zp=np.empty_like(a); zm=np.empty_like(a)
        zp[:,:-1]=a[:,1:]; zp[:,-1]=a[:,-1]
        zm[:,1:]=a[:,:-1]; zm[:,0]=a[:,0]
        a=(4*a+xp+xm+zp+zm)/8.0
    return a


def weighted_mean(a,w):
    den=np.sum(w)
    return float(np.sum(w*a)/den) if den>0 else 0.0


def weighted_corr(a,b,w):
    w=np.asarray(w,float)
    sw=np.sum(w)
    if sw<=0: return np.nan
    am=np.sum(w*a)/sw; bm=np.sum(w*b)/sw
    da=a-am; db=b-bm
    va=np.sum(w*da*da); vb=np.sum(w*db*db)
    if va<=0 or vb<=0: return np.nan
    return float(np.sum(w*da*db)/np.sqrt(va*vb))


def weighted_rrms(obs,pred,w):
    num=np.sum(w*(obs-pred)**2)
    den=np.sum(w*obs**2)
    return float(np.sqrt(num/max(den,1e-300)))


def linfit(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    A=np.column_stack([x,np.ones_like(x)])
    c,*_=np.linalg.lstsq(A,y,rcond=None)
    pred=A@c
    ssr=np.sum((y-pred)**2)
    sst=np.sum((y-np.mean(y))**2)
    r2=1-ssr/sst if sst>0 else np.nan
    return float(c[0]),float(c[1]),float(r2)


def divergence_stress(Txx,Txz,Tzz,dx,dz):
    fx=-(np.gradient(Txx,dx,axis=0,edge_order=2)
         +np.gradient(Txz,dz,axis=1,edge_order=2))
    fz=-(np.gradient(Txz,dx,axis=0,edge_order=2)
         +np.gradient(Tzz,dz,axis=1,edge_order=2))
    return fx,fz


def project_force(xix,xiz,fx,fz,dx,dz):
    return float(np.sum(xix*fx+xiz*fz)*dx*dz)


def ibp_project(xix,xiz,Txx,Txz,Tzz,dx,dz):
    dxx=np.gradient(xix,dx,axis=0,edge_order=2)
    dxz=np.gradient(xix,dz,axis=1,edge_order=2)
    dzx=np.gradient(xiz,dx,axis=0,edge_order=2)
    dzz=np.gradient(xiz,dz,axis=1,edge_order=2)
    return float(np.sum(Txx*dxx+Txz*dxz+Txz*dzx+Tzz*dzz)*dx*dz)


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
        raise RuntimeError(f"need four particle-rich phases; found {len(files)}")

    eh=np.loadtxt(args.em_history)
    if eh.ndim==1: eh=eh[None,:]
    ht=eh[:,1]; hq=eh[:,4]
    qfit,hqdot,hqddot,Rq,cq=harmonic_fit(ht,hq,args.period)

    states=[]
    for fn in files:
        M=load_mesh(fn,args.coarsen)
        ION=deposit_species(fn,"ions",P["mi"],M,p)
        ELE=deposit_species(fn,"electrons",ME,M,p)
        tci=P["wci"]*M["t"]
        q=float(np.interp(tci,ht,hq))
        qdot=float(np.interp(tci,ht,hqdot))
        qdd=float(np.interp(tci,ht,hqddot))
        ni=np.abs(M["rho_i"])/QE
        ne=np.abs(M["rho_e"])/QE
        states.append(dict(fn=fn,step=step_from_parent(fn),M=M,
                           I=ION,E=ELE,t=tci,q=q,qdot=qdot,qdd=qdd,
                           ni=ni,ne=ne))

    # Select the best opposite-sign near-zero-q crossings.
    pos=[s for s in states if s["qdot"]>0]
    neg=[s for s in states if s["qdot"]<0]
    if not pos or not neg:
        raise RuntimeError("phase snapshots do not bracket opposite qdot crossings")
    s_pos=min(pos,key=lambda s:abs(s["q"]))
    s_neg=min(neg,key=lambda s:abs(s["q"]))
    crossings=[s_neg,s_pos]

    # Build displacement eigenfunction from ion bulk flow / qdot_phys.
    xis=[]
    for s in crossings:
        ni=s["ni"]
        core=(np.abs(s["M"]["z"])[None,:] <= args.core_z_de*P["de"])
        w=ni*core
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

    # Mode-quality metrics at the two crossings.
    quality=[]
    for s in crossings:
        core=(np.abs(s["M"]["z"])[None,:] <= zmax)
        w=s["ni"]*core
        predx=P["wci"]*s["qdot"]*xix
        predz=P["wci"]*s["qdot"]*xiz
        ux=s["I"]["ux"]; uz=s["I"]["uz"]
        ux0=weighted_mean(ux,w); uz0=weighted_mean(uz,w)
        ox=ux-ux0; oz=uz-uz0
        quality.append(dict(
            step=s["step"],t=s["t"],
            corr_x=weighted_corr(ox,predx,w),
            corr_z=weighted_corr(oz,predz,w),
            rrms_x=weighted_rrms(ox,predx,w),
            rrms_z=weighted_rrms(oz,predz,w),
        ))

    # How affine is the measured vertical mode?
    Z=np.broadcast_to(z[None,:],xiz.shape)
    wmode=nmean*env
    denom=np.sum(wmode*Z*Z)
    affine_a=np.sum(wmode*Z*xiz)/max(denom,1e-300)
    xiz_aff=affine_a*Z
    affine_corr=weighted_corr(xiz,xiz_aff,wmode)
    affine_rrms=weighted_rrms(xiz,xiz_aff,wmode)

    # Horizontal fraction of modal inertia using mean ion density.
    mx=float(np.sum(P["mi"]*nmean*xix*xix)*crossings[0]["M"]["dx"]*crossings[0]["M"]["dz"])
    mz=float(np.sum(P["mi"]*nmean*xiz*xiz)*crossings[0]["M"]["dx"]*crossings[0]["M"]["dz"])
    horiz_frac=mx/max(mx+mz,1e-300)

    rows=[]
    ibp_checks=[]
    for s in states:
        M=s["M"]; I=s["I"]; E=s["E"]; ni=s["ni"]; ne=s["ne"]
        dx,dz=M["dx"],M["dz"]

        rhoi=P["mi"]*ni
        rhoe=ME*ne
        rhot=rhoi+rhoe
        modal_mass=float(np.sum(rhot*(xix*xix+xiz*xiz))*dx*dz)
        fac=1.0/(modal_mass*P["wci"]**2)

        # Pressure tensors in the simulation x-z plane.
        Pxx_i=P["mi"]*ni*I["cxx"]; Pxz_i=P["mi"]*ni*I["cxz"]; Pzz_i=P["mi"]*ni*I["czz"]
        Pxx_e=ME*ne*E["cxx"]; Pxz_e=ME*ne*E["cxz"]; Pzz_e=ME*ne*E["czz"]

        fPix,fPiz=divergence_stress(Pxx_i,Pxz_i,Pzz_i,dx,dz)
        fPex,fPez=divergence_stress(Pxx_e,Pxz_e,Pzz_e,dx,dz)
        QPi=project_force(xix,xiz,fPix,fPiz,dx,dz)
        QPe=project_force(xix,xiz,fPex,fPez,dx,dz)

        QPi_ibp=ibp_project(xix,xiz,Pxx_i,Pxz_i,Pzz_i,dx,dz)
        QPe_ibp=ibp_project(xix,xiz,Pxx_e,Pxz_e,Pzz_e,dx,dz)

        # Bulk momentum-flux tensors.
        Bxx_i=rhoi*I["ux"]**2; Bxz_i=rhoi*I["ux"]*I["uz"]; Bzz_i=rhoi*I["uz"]**2
        Bxx_e=rhoe*E["ux"]**2; Bxz_e=rhoe*E["ux"]*E["uz"]; Bzz_e=rhoe*E["uz"]**2
        fBix,fBiz=divergence_stress(Bxx_i,Bxz_i,Bzz_i,dx,dz)
        fBex,fBez=divergence_stress(Bxx_e,Bxz_e,Bzz_e,dx,dz)
        QBi=project_force(xix,xiz,fBix,fBiz,dx,dz)
        QBe=project_force(xix,xiz,fBex,fBez,dx,dz)

        # Particle-current electromagnetic force.
        Jx=QE*(ni*I["ux"]-ne*E["ux"])
        Jy=QE*(ni*I["uy"]-ne*E["uy"])
        Jz=QE*(ni*I["uz"]-ne*E["uz"])
        rhoc=QE*(ni-ne)
        fEMx=rhoc*M["Ex"] + Jy*M["Bz"] - Jz*M["By"]
        fEMz=rhoc*M["Ez"] + Jx*M["By"] - Jy*M["Bx"]
        QEM=project_force(xix,xiz,fEMx,fEMz,dx,dz)

        aPi=QPi*fac; aPe=QPe*fac
        aB=(QBi+QBe)*fac; aEM=QEM*fac
        atot=aEM+aPi+aPe+aB

        rows.append([
            s["step"],s["t"],s["q"],s["qdot"],s["qdd"],modal_mass,
            aEM,aPi,aPe,aPi+aPe,aB,atot,
            QPi_ibp*fac,QPe_ibp*fac
        ])
        ibp_checks.append((aPi,QPi_ibp*fac,aPe,QPe_ibp*fac))

    a=np.asarray(rows,float)
    a=a[np.argsort(a[:,1])]

    np.savetxt(
        "ishizawa_nonaffine_mode_history.txt",a,
        header=(
            "step omega_ci_t q dq_dtau measured_ddq modal_mass "
            "a_EM a_Pi a_Pe a_Ptotal a_bulk a_total "
            "a_Pi_ibp a_Pe_ibp"
        )
    )

    # Plots.
    fig,axs=plt.subplots(1,3,figsize=(14,4.5))
    extent=[crossings[0]["M"]["x"][0]/P["de"],crossings[0]["M"]["x"][-1]/P["de"],
            z[0]/P["de"],z[-1]/P["de"]]
    im=axs[0].imshow((xix/P["de"]).T,origin="lower",aspect="auto",extent=extent)
    axs[0].set_title(r"$\xi_x/d_e$"); fig.colorbar(im,ax=axs[0],shrink=.8)
    im=axs[1].imshow((xiz/P["de"]).T,origin="lower",aspect="auto",extent=extent)
    axs[1].set_title(r"$\xi_z/d_e$"); fig.colorbar(im,ax=axs[1],shrink=.8)
    im=axs[2].imshow((nmean/P["n0"]).T,origin="lower",aspect="auto",extent=extent)
    axs[2].set_title(r"crossing mean $n_i/n_0$"); fig.colorbar(im,ax=axs[2],shrink=.8)
    for ax in axs:
        ax.set_xlabel(r"$x/d_e$"); ax.set_ylabel(r"$z/d_e$")
    fig.suptitle("Velocity-reconstructed non-affine breathing displacement")
    fig.tight_layout()
    fig.savefig("ishizawa_nonaffine_mode_shape.png",dpi=210)
    plt.close(fig)

    t=a[:,1]; q=a[:,2]; ddq=a[:,4]
    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
    axs[0].plot(t,ddq,"o-",lw=2,label="measured q harmonic acceleration")
    axs[0].plot(t,a[:,6],"o-",label="EM")
    axs[0].plot(t,a[:,9],"o-",label="pressure")
    axs[0].plot(t,a[:,10],"o-",label="bulk stress")
    axs[0].plot(t,a[:,11],"o-",lw=2,label="sum projected forces")
    axs[0].axhline(0,lw=.7); axs[0].grid(alpha=.25); axs[0].legend(fontsize=8)
    axs[0].set_ylabel("generalized q acceleration")

    axs[1].plot(t,a[:,7],"o-",label="ion pressure direct")
    axs[1].plot(t,a[:,12],"o--",label="ion pressure IBP")
    axs[1].plot(t,a[:,8],"s-",label="electron pressure direct")
    axs[1].plot(t,a[:,13],"s--",label="electron pressure IBP")
    axs[1].axhline(0,lw=.7); axs[1].grid(alpha=.25); axs[1].legend(fontsize=8)
    axs[1].set_ylabel("pressure projection check")
    axs[1].set_xlabel(r"$\omega_{ci}t$")
    fig.suptitle("Non-affine modal projection of the momentum balance")
    fig.tight_layout()
    fig.savefig("ishizawa_nonaffine_force_balance.png",dpi=210)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,5.5))
    ax.scatter(q,ddq,s=65,label="measured acceleration")
    for col,name in [(6,"EM"),(9,"pressure"),(10,"bulk"),(11,"total")]:
        ax.plot(q,a[:,col],"o-",label=name)
    ax.axhline(0,lw=.7); ax.axvline(0,lw=.7)
    ax.set_xlabel("q"); ax.set_ylabel("modal acceleration")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    ax.set_title("Non-affine force-displacement screen")
    fig.tight_layout()
    fig.savefig("ishizawa_nonaffine_force_vs_q.png",dpi=210)
    plt.close(fig)

    # Four-point diagnostics.
    stats={}
    for col,name in [(6,"EM"),(9,"pressure"),(10,"bulk"),(11,"total")]:
        slope,intercept,r2=linfit(q,a[:,col])
        corr=float(np.corrcoef(ddq,a[:,col])[0,1])
        stats[name]=(slope,r2,corr)

    rel_i=np.sqrt(np.mean((a[:,7]-a[:,12])**2))/max(np.sqrt(np.mean(a[:,7]**2)),1e-300)
    rel_e=np.sqrt(np.mean((a[:,8]-a[:,13])**2))/max(np.sqrt(np.mean(a[:,8]**2)),1e-300)
    rel_total=np.sqrt(np.mean((ddq-a[:,11])**2))/max(np.sqrt(np.mean(ddq**2)),1e-300)
    corr_total=float(np.corrcoef(ddq,a[:,11])[0,1])

    with open("ishizawa_nonaffine_mode_summary.txt","w") as f:
        f.write("Ishizawa-scale non-affine breathing-mode projection\n")
        f.write("===================================================\n\n")
        f.write(f"period omega_ci*T = {args.period:.8f}\n")
        f.write(f"q harmonic R2 = {Rq:.8f}\n")
        f.write(f"coarsen = {args.coarsen}\n")
        f.write(f"core |z|/de <= {args.core_z_de:.6f}\n")
        f.write(f"mode density cutoff n_i/n0 = {args.mode_density_cut:.6f}\n")
        f.write(f"smoothing passes = {args.smooth_passes}\n\n")
        f.write("Crossings used to reconstruct xi=u_i/qdot_phys\n")
        for ql in quality:
            f.write(
                f"step={ql['step']} tci={ql['t']:.6f} "
                f"corr_x={ql['corr_x']:.6f} corr_z={ql['corr_z']:.6f} "
                f"rrms_x={ql['rrms_x']:.6f} rrms_z={ql['rrms_z']:.6f}\n"
            )
        f.write("\n")
        f.write(f"vertical-mode affine correlation = {affine_corr:.8f}\n")
        f.write(f"vertical-mode affine relative RMS error = {affine_rrms:.8f}\n")
        f.write(f"horizontal modal-inertia fraction = {horiz_frac:.8f}\n\n")
        f.write("phase state force projections\n")
        f.write("step tci q measured_ddq a_EM a_Ptot a_bulk a_total\n")
        for r in a:
            f.write(
                f"{int(r[0]):6d} {r[1]:8.4f} {r[2]:+10.5f} {r[4]:+12.5e} "
                f"{r[6]:+12.5e} {r[9]:+12.5e} {r[10]:+12.5e} {r[11]:+12.5e}\n"
            )
        f.write("\nFour-point slope/correlation screen\n")
        f.write("channel      slope_vs_q       R2       corr_with_measured_ddq\n")
        for name,(slope,r2,corr) in stats.items():
            f.write(f"{name:10s} {slope:+14.6e} {r2:9.5f} {corr:+14.6f}\n")
        f.write("\n")
        f.write(f"ion pressure direct-vs-IBP relRMS = {rel_i:.8e}\n")
        f.write(f"electron pressure direct-vs-IBP relRMS = {rel_e:.8e}\n")
        f.write(f"total projected-force relRMS mismatch = {rel_total:.8e}\n")
        f.write(f"corr(total projected force, measured ddq) = {corr_total:.8f}\n\n")
        f.write("Interpretation guide\n")
        f.write("--------------------\n")
        f.write(
            "If the reconstructed mode has much better closure than the affine "
            "xi_z=z test, the previous residual was primarily a coordinate/mode-"
            "shape problem.  If the non-affine total still fails while the "
            "crossing velocity reconstruction is coherent, then a denser time "
            "series is needed to test temporal inertia and trapped-particle/"
            "orbit physics directly.\n"
        )

    print("Saved ishizawa_nonaffine_mode_summary.txt")
    print("Saved ishizawa_nonaffine_mode_history.txt")
    print("Saved ishizawa_nonaffine_mode_shape.png")
    print("Saved ishizawa_nonaffine_force_balance.png")
    print("Saved ishizawa_nonaffine_force_vs_q.png")


if __name__=="__main__":
    main()
