#!/usr/bin/env python3
"""Compare WarpX-native particle field gather with cell-centered bilinear gather.

The native-gather probe plotfile contains Ex/Ey/Ez/Bx/By/Bz already gathered
onto the sampled particles by WarpX::storeFieldOnParticles().  For the *same*
particles, this script also bilinearly interpolates the ordinary cell-centered
plotfile fields and compares the resulting electromagnetic weak-form term

    Q_EM = sum_p W_p q_s xi(x_p) . [ E_p + v_p x B_p ]

using the same fixed non-affine test displacement xi used by
analyze_ishizawa_modal_particle_quadrature.py.

Only two full dense plotfiles are read to reconstruct xi (the opposite-qdot
crossings); all field-gather comparisons are then performed on the inexpensive
random particle sample in the probe plotfile.

Outputs
-------
ishizawa_native_vs_bilinear_qem_summary.txt
ishizawa_native_vs_bilinear_qem_species.txt
ishizawa_native_vs_bilinear_qem.png
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

from analyze_ishizawa_scale_40000 import QE, ME, C, params
from analyze_ishizawa_particle_moments import load_mesh, deposit_species, particle_array
from analyze_ishizawa_nonaffine_breathing_mode import (
    step_from_parent, harmonic_fit, smooth2, weighted_mean
)
from analyze_ishizawa_modal_particle_quadrature import interp_cc


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--probe", required=True,
                   help="native-gather probe plotfile, e.g. diags/native202000")
    p.add_argument("--phase-root", required=True,
                   help="full dense particle run containing diags/diag1*")
    p.add_argument("--em-history", required=True,
                   help="ishizawa_saturation_history.txt used to define q(t)")
    p.add_argument("--period", type=float, required=True)
    p.add_argument("--mode-coarsen", type=int, default=4)
    p.add_argument("--core-z-de", type=float, default=12.0)
    p.add_argument("--mode-density-cut", type=float, default=0.02)
    p.add_argument("--smooth-passes", type=int, default=2)
    p.add_argument("--x-edge-taper-de", type=float, default=0.0)
    p.add_argument("--sample-fraction", type=float, default=0.02)
    p.add_argument("--full-history", default=None,
                   help="optional full-particle closure history for an absolute QEM check")
    p.add_argument("--out-prefix", default="ishizawa_native_vs_bilinear_qem")
    return p.parse_args()


def build_mode(args, P):
    files=sorted(
        set(
            glob.glob(str(Path(args.phase_root)/"step*"/"diags"/"diag1*"))
            + glob.glob(str(Path(args.phase_root)/"diags"/"diag1*"))
        ),
        key=step_from_parent
    )
    if len(files)<2:
        raise RuntimeError(f"need dense plotfiles under {args.phase_root}")

    eh=np.loadtxt(args.em_history)
    if eh.ndim==1:
        eh=eh[None,:]
    ht=eh[:,1]
    hq=eh[:,4]
    _,hqdot,_,_,_=harmonic_fit(ht,hq,args.period)

    # Metadata-only first pass to locate opposite-sign qdot crossings.
    meta=[]
    for fn in files:
        ds=yt.load(fn)
        tau=float(P["wci"]*ds.current_time.to_value("s"))
        if tau < ht.min()-1e-9 or tau > ht.max()+1e-9:
            continue
        meta.append(dict(
            fn=fn,
            step=step_from_parent(fn),
            tau=tau,
            q=float(np.interp(tau,ht,hq)),
            qdot=float(np.interp(tau,ht,hqdot))
        ))
    pos=[s for s in meta if s["qdot"]>0]
    neg=[s for s in meta if s["qdot"]<0]
    if not pos or not neg:
        raise RuntimeError("could not find opposite-sign qdot states")
    spos=min(pos,key=lambda s:abs(s["q"]))
    sneg=min(neg,key=lambda s:abs(s["q"]))
    crossings=[sneg,spos]

    xis=[]
    nis=[]
    meshes=[]
    for s in crossings:
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
    return M,xix,xiz,crossings


def wrms(w,a):
    den=np.sum(w)
    return np.sqrt(np.sum(w*a*a)/max(den,np.finfo(float).tiny))


def wcorr(w,a,b):
    sw=np.sum(w)
    if sw<=0:
        return np.nan
    ma=np.sum(w*a)/sw
    mb=np.sum(w*b)/sw
    da=a-ma; db=b-mb
    den=np.sqrt(np.sum(w*da*da)*np.sum(w*db*db))
    return np.sum(w*da*db)/den if den>0 else np.nan


def species_compare(probe,species,mass,charge,Mfield,Mmode,xix,xiz):
    ds=yt.load(probe)
    ad=ds.all_data()

    xp=particle_array(ad,species,"particle_position_x","m")
    zp=particle_array(ad,species,"particle_position_y","m")
    px=particle_array(ad,species,"particle_momentum_x","kg*m/s")
    py=particle_array(ad,species,"particle_momentum_y","kg*m/s")
    pz=particle_array(ad,species,"particle_momentum_z","kg*m/s")
    w =particle_array(ad,species,"particle_weight")

    gamma=np.sqrt(1.0+(px*px+py*py+pz*pz)/(mass*C)**2)
    vx=px/(gamma*mass)
    vy=py/(gamma*mass)
    vz=pz/(gamma*mass)

    xi_x=interp_cc(xix,xp,zp,Mmode["x"],Mmode["z"])
    xi_z=interp_cc(xiz,xp,zp,Mmode["x"],Mmode["z"])

    native={}
    units={"Ex":"V/m","Ey":"V/m","Ez":"V/m",
           "Bx":"T","By":"T","Bz":"T"}
    for name,unit in units.items():
        native[name]=particle_array(ad,species,"particle_"+name,unit)

    cc={}
    for name in units:
        cc[name]=interp_cc(Mfield[name],xp,zp,Mfield["x"],Mfield["z"])

    def qparts(F):
        qE=np.sum(w*charge*(xi_x*F["Ex"] + xi_z*F["Ez"]))
        qB=np.sum(w*charge*(
            xi_x*(vy*F["Bz"]-vz*F["By"])
            + xi_z*(vx*F["By"]-vy*F["Bx"])
        ))
        return qE,qB,qE+qB

    nE,nB,nT=qparts(native)
    cE,cB,cT=qparts(cc)

    field_stats={}
    for name in units:
        scale=max(wrms(w,native[name]),np.finfo(float).tiny)
        field_stats[name]=dict(
            rel_wrms=wrms(w,cc[name]-native[name])/scale,
            corr=wcorr(w,native[name],cc[name]),
            native_wrms=scale,
            cc_wrms=wrms(w,cc[name])
        )

    return dict(
        N=len(w),sumw=float(np.sum(w)),
        Qn_E=float(nE),Qn_B=float(nB),Qn=float(nT),
        Qc_E=float(cE),Qc_B=float(cB),Qc=float(cT),
        fields=field_stats
    )


def main():
    args=parse_args()
    yt.funcs.mylog.setLevel(40)
    if not (0<args.sample_fraction<=1):
        raise ValueError("--sample-fraction must be in (0,1]")

    P=params()
    Mmode,xix,xiz,crossings=build_mode(args,P)
    Mfield=load_mesh(args.probe,1)

    ds=yt.load(args.probe)
    tau=float(P["wci"]*ds.current_time.to_value("s"))
    step=step_from_parent(args.probe)

    ion=species_compare(args.probe,"ions",P["mi"],+QE,Mfield,Mmode,xix,xiz)
    ele=species_compare(args.probe,"electrons",ME,-QE,Mfield,Mmode,xix,xiz)

    Qn=ion["Qn"]+ele["Qn"]
    Qc=ion["Qc"]+ele["Qc"]
    QnE=ion["Qn_E"]+ele["Qn_E"]
    QcE=ion["Qc_E"]+ele["Qc_E"]
    QnB=ion["Qn_B"]+ele["Qn_B"]
    QcB=ion["Qc_B"]+ele["Qc_B"]

    eps=np.finfo(float).tiny
    scale=1.0/args.sample_fraction

    full_qem=None
    if args.full_history:
        h=np.loadtxt(args.full_history)
        if h.ndim==1: h=h[None,:]
        # Current closure history: col 1=tau, col 4=QEM/omega_ci.
        full_qem=float(np.interp(tau,h[:,1],h[:,4]))

    def ratio(a,b):
        return a/b if abs(b)>eps else np.nan

    with open(args.out_prefix+"_summary.txt","w") as f:
        f.write("WarpX-native vs cell-centered bilinear Q_EM comparison\n")
        f.write("=====================================================\n\n")
        f.write(f"probe = {args.probe}\n")
        f.write(f"step = {step}\n")
        f.write(f"omega_ci*t = {tau:.10f}\n")
        f.write(f"sample fraction = {args.sample_fraction:.8f}\n")
        f.write(f"mode crossings = {crossings[0]['step']} {crossings[1]['step']}\n")
        f.write(f"x edge taper / de = {args.x_edge_taper_de:.8f}\n\n")

        for name,R in [("ions",ion),("electrons",ele)]:
            f.write(f"[{name}]\n")
            f.write(f"Nsample = {R['N']}\n")
            f.write(f"QEM native / omega_ci = {R['Qn']/P['wci']:+.10e}\n")
            f.write(f"QEM bilinear / omega_ci = {R['Qc']/P['wci']:+.10e}\n")
            f.write(f"bilinear/native = {ratio(R['Qc'],R['Qn']):+.10e}\n")
            f.write(f"(bilinear-native)/native = {ratio(R['Qc']-R['Qn'],R['Qn']):+.10e}\n")
            f.write(f"QE native/bilinear = {R['Qn_E']/P['wci']:+.10e} {R['Qc_E']/P['wci']:+.10e}\n")
            f.write(f"QvXB native/bilinear = {R['Qn_B']/P['wci']:+.10e} {R['Qc_B']/P['wci']:+.10e}\n")
            for fld,S in R["fields"].items():
                f.write(
                    f"{fld} relWRMSdiff={S['rel_wrms']:.8e} "
                    f"corr={S['corr']:.8f} "
                    f"nativeWRMS={S['native_wrms']:.8e} "
                    f"ccWRMS={S['cc_wrms']:.8e}\n"
                )
            f.write("\n")

        f.write("[total same sampled particles]\n")
        f.write(f"QEM native / omega_ci = {Qn/P['wci']:+.10e}\n")
        f.write(f"QEM bilinear / omega_ci = {Qc/P['wci']:+.10e}\n")
        f.write(f"bilinear/native = {ratio(Qc,Qn):+.10e}\n")
        f.write(f"(bilinear-native)/native = {ratio(Qc-Qn,Qn):+.10e}\n")
        f.write(f"QE native/bilinear / omega_ci = {QnE/P['wci']:+.10e} {QcE/P['wci']:+.10e}\n")
        f.write(f"QvXB native/bilinear / omega_ci = {QnB/P['wci']:+.10e} {QcB/P['wci']:+.10e}\n")
        f.write("\n[random-sample scaled estimate]\n")
        f.write(f"scaled native QEM / omega_ci = {scale*Qn/P['wci']:+.10e}\n")
        f.write(f"scaled bilinear QEM / omega_ci = {scale*Qc/P['wci']:+.10e}\n")
        if full_qem is not None:
            f.write(f"full-particle existing bilinear QEM / omega_ci = {full_qem:+.10e}\n")
            f.write(f"scaled-sample bilinear / full existing = {ratio(scale*Qc/P['wci'],full_qem):+.10e}\n")
            f.write(f"scaled native / full existing = {ratio(scale*Qn/P['wci'],full_qem):+.10e}\n")

    rows=[]
    for ispec,(name,R) in enumerate([("ions",ion),("electrons",ele)]):
        rows.append([
            ispec,R["N"],R["Qn"]/P["wci"],R["Qc"]/P["wci"],
            ratio(R["Qc"],R["Qn"]),
            ratio(R["Qc"]-R["Qn"],R["Qn"]),
            R["Qn_E"]/P["wci"],R["Qc_E"]/P["wci"],
            R["Qn_B"]/P["wci"],R["Qc_B"]/P["wci"]
        ])
    np.savetxt(
        args.out_prefix+"_species.txt",np.asarray(rows,float),
        header=(
            "species_index(0=ion,1=electron) Nsample "
            "Qnative_over_wci Qbilinear_over_wci bilinear_over_native "
            "relative_difference QE_native QE_bilinear QvXB_native QvXB_bilinear"
        )
    )

    labels=["ion","electron","total"]
    qn=np.array([ion["Qn"],ele["Qn"],Qn])/P["wci"]
    qc=np.array([ion["Qc"],ele["Qc"],Qc])/P["wci"]
    x=np.arange(3)
    fig,axs=plt.subplots(2,1,figsize=(8.5,8))
    bw=.36
    axs[0].bar(x-bw/2,qn,bw,label="WarpX native gather")
    axs[0].bar(x+bw/2,qc,bw,label="cell-centered bilinear")
    axs[0].set_xticks(x,labels)
    axs[0].set_ylabel(r"$Q_{EM}/\\omega_{ci}$ (sample)")
    axs[0].set_title("Same-particle electromagnetic weak-form projection")
    axs[0].grid(axis="y",alpha=.25)
    axs[0].legend()

    flds=["Ex","Ey","Ez","Bx","By","Bz"]
    xi=np.arange(len(flds))
    axs[1].plot(xi,[ion["fields"][k]["rel_wrms"] for k in flds],"o-",label="ions")
    axs[1].plot(xi,[ele["fields"][k]["rel_wrms"] for k in flds],"o-",label="electrons")
    axs[1].set_xticks(xi,flds)
    axs[1].set_ylabel("weighted RMS(native-bilinear) / RMS(native)")
    axs[1].set_title("Particle-field gather discrepancy")
    axs[1].grid(alpha=.25)
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(args.out_prefix+".png",dpi=210)
    plt.close(fig)

    print("Saved",args.out_prefix+"_summary.txt")
    print("Saved",args.out_prefix+"_species.txt")
    print("Saved",args.out_prefix+".png")


if __name__=="__main__":
    main()
