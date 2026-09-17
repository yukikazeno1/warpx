#!/usr/bin/env python3
"""Early diagnostics for a Debye-length-normalized Harris-sheet run.

Designed for cases such as:
  box = 500 lambda_D x 500 lambda_D
  Harris tanh scale L = 40 lambda_D
  mi/me = 800, Ti=Te, omega_pe/omega_ce=3.5

The script does NOT assume L=d_e.  It reports spectra in k*lambda_D and k*L,
checks the Harris Bx and Jy equilibrium, fits early exponential growth of Bz
Fourier modes, and produces perturbation/full magnetic topology plots.
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt

QE = 1.602176634e-19
ME = 9.1093837139e-31
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
C = 299792458.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--box-lambdaD", type=float, default=500.0)
    p.add_argument("--sheet-lambdaD", type=float, default=40.0)
    p.add_argument("--max-mode", type=int, default=20)
    p.add_argument("--core-sheet-halfwidth", type=float, default=1.0,
                   help="Fourier/RMS core half-width in units of Harris L")
    p.add_argument("--fit-t0", type=float, default=0.05)
    p.add_argument("--fit-t1", type=float, default=0.25)
    p.add_argument("--min-r2", type=float, default=0.90)
    p.add_argument("--min-growth", type=float, default=1.5)
    p.add_argument("--zoom-sheet", type=float, default=2.0,
                   help="topology half-height in units of Harris L")
    return p.parse_args()


def numeric_key(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def as_xz(a, nx, nz):
    a = np.asarray(a).squeeze()
    if a.shape == (nx, nz):
        return a
    if a.shape == (nz, nx):
        return a.T
    raise RuntimeError(f"unexpected field shape {a.shape}")


def params(sheet_lambdaD):
    n0 = 1.0e19
    mi = 800.0 * ME
    wpe = np.sqrt(n0 * QE**2 / (EPS0 * ME))
    de = C / wpe
    di = np.sqrt(mi / ME) * de
    wce = wpe / 3.5
    wci = wce / 800.0
    B0 = ME * wce / QE
    Te = B0**2 / (4.0 * MU0 * n0)
    lambdaD = np.sqrt(EPS0 * Te / (n0 * QE**2))
    L = sheet_lambdaD * lambdaD
    J0 = B0 / (MU0 * L)
    return dict(n0=n0, mi=mi, wpe=wpe, de=de, di=di, wce=wce, wci=wci,
                B0=B0, Te=Te, lambdaD=lambdaD, L=L, J0=J0)


def load_fields(path):
    ds = yt.load(path)
    nx = int(ds.domain_dimensions[0])
    nz = int(ds.domain_dimensions[1])
    xlo = ds.domain_left_edge[0].to_value("m")
    xhi = ds.domain_right_edge[0].to_value("m")
    zlo = ds.domain_left_edge[1].to_value("m")
    zhi = ds.domain_right_edge[1].to_value("m")
    dx = (xhi-xlo)/nx
    dz = (zhi-zlo)/nz
    x = xlo + (np.arange(nx)+0.5)*dx
    z = zlo + (np.arange(nz)+0.5)*dz
    g = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
                         dims=ds.domain_dimensions)

    def fld(name, unit=None):
        q = g["boxlib", name]
        try:
            a = q.to_value(unit) if unit is not None else q.to_ndarray()
        except Exception:
            a = q.to_ndarray()
        return as_xz(a, nx, nz)

    return dict(ds=ds, x=x, z=z, dx=dx, dz=dz,
                Bx=fld("Bx", "T"), By=fld("By", "T"), Bz=fld("Bz", "T"),
                Ey=fld("Ey", "V/m"), jy=fld("jy"))


def corrected_jy(Bx, Bz, dx, dz):
    dBx_dz = np.gradient(Bx, dz, axis=1, edge_order=2)
    dBz_dx = np.gradient(Bz, dx, axis=0, edge_order=2)
    return (dBx_dz - dBz_dx) / MU0


def reconstruct_Ay(Bx, Bz, x, z):
    nx, nz = Bx.shape
    dx = x[1]-x[0]
    dz = z[1]-z[0]
    j0 = int(np.argmin(np.abs(z)))
    ay_mid = np.zeros(nx)
    for i in range(1, nx):
        ay_mid[i] = ay_mid[i-1] + 0.5*(Bz[i-1,j0]+Bz[i,j0])*dx
    ay_mid -= np.linspace(0.0, ay_mid[-1]-ay_mid[0], nx)
    ay_mid -= np.mean(ay_mid)
    Ay = np.zeros_like(Bx)
    Ay[:,j0] = ay_mid
    for j in range(j0+1, nz):
        Ay[:,j] = Ay[:,j-1] - 0.5*(Bx[:,j-1]+Bx[:,j])*dz
    for j in range(j0-1, -1, -1):
        Ay[:,j] = Ay[:,j+1] + 0.5*(Bx[:,j+1]+Bx[:,j])*dz
    return Ay


def fwhm(z, y):
    y = np.asarray(y)
    ymax = np.nanmax(y)
    if not np.isfinite(ymax) or ymax <= 0:
        return np.nan
    ids = np.flatnonzero(y >= 0.5*ymax)
    return np.nan if ids.size < 2 else z[ids[-1]]-z[ids[0]]


def fit_growth(t, a, t0, t1):
    mask = np.isfinite(t) & np.isfinite(a) & (a > 0) & (t >= t0) & (t <= t1)
    if np.count_nonzero(mask) < 4:
        return np.nan, np.nan, np.nan
    x = t[mask]
    y = np.log(a[mask])
    s, b = np.polyfit(x, y, 1)
    yp = s*x+b
    ssr = np.sum((y-yp)**2)
    sst = np.sum((y-np.mean(y))**2)
    r2 = 1.0-ssr/sst if sst > 0 else np.nan
    growth = float(np.exp(s*(x[-1]-x[0])))
    return float(s), float(r2), growth


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params(args.sheet_lambdaD)
    files = sorted(glob.glob(str(Path(args.run_dir)/"diags"/"diag1*")), key=numeric_key)
    if not files:
        raise FileNotFoundError(f"No diag1* under {args.run_dir}/diags")

    tci=[]; spectra=[]; bz_rms=[]; by_rms=[]; ey_rms=[]; bx_err=[]; jy_err=[]; widths=[]; flux_span=[]
    final=None
    core_half = args.core_sheet_halfwidth * P["L"]

    for k, fn in enumerate(files, 1):
        F = load_fields(fn)
        x,z=F["x"],F["z"]
        tci.append(P["wci"]*F["ds"].current_time.to_value("s"))
        core = np.abs(z) <= core_half

        ft = np.fft.rfft(F["Bz"], axis=0)/F["Bz"].shape[0]
        aa=np.full(args.max_mode, np.nan)
        for m in range(1,args.max_mode+1):
            if m < ft.shape[0]:
                local=2.0*np.abs(ft[m,:])/P["B0"]
                aa[m-1]=np.sqrt(np.mean(local[core]**2))
        spectra.append(aa)
        bz_rms.append(np.sqrt(np.mean(F["Bz"][:,core]**2))/P["B0"])
        by_rms.append(np.sqrt(np.mean(F["By"][:,core]**2))/P["B0"])
        ey_rms.append(np.sqrt(np.mean(F["Ey"][:,core]**2))/(C*P["B0"]))

        bx_mean=np.mean(F["Bx"],axis=0)
        bx_ref=P["B0"]*np.tanh(z/P["L"])
        bx_err.append(np.sqrt(np.mean((bx_mean-bx_ref)**2))/P["B0"])

        jyc=corrected_jy(F["Bx"],F["Bz"],F["dx"],F["dz"])
        jy_mean=np.mean(jyc,axis=0)
        jy_ref=P["J0"]/np.cosh(z/P["L"])**2
        sheet=np.abs(z)<=4.0*P["L"]
        jy_err.append(np.sqrt(np.mean((jy_mean[sheet]-jy_ref[sheet])**2))/P["J0"])
        widths.append(fwhm(z,jy_mean)/P["lambdaD"])

        Ay=reconstruct_Ay(F["Bx"],F["Bz"],x,z)
        dAy=Ay-np.mean(Ay,axis=0,keepdims=True)
        j0=int(np.argmin(np.abs(z)))
        flux_span.append((np.max(dAy[:,j0])-np.min(dAy[:,j0]))/(P["B0"]*P["L"]))
        final=(F,jyc,Ay,dAy)
        print(f"{k:3d}/{len(files)} omega_ci*t={tci[-1]:.5f} Bz_rms/B0={bz_rms[-1]:.3e}")

    tci=np.asarray(tci); A=np.asarray(spectra)
    bz_rms=np.asarray(bz_rms); by_rms=np.asarray(by_rms); ey_rms=np.asarray(ey_rms)
    bx_err=np.asarray(bx_err); jy_err=np.asarray(jy_err); widths=np.asarray(widths); flux_span=np.asarray(flux_span)

    modes=np.arange(1,args.max_mode+1)
    kld = 2*np.pi*modes/args.box_lambdaD
    kL = kld*args.sheet_lambdaD
    dprimeL = 2*(1.0/kL-kL)
    fits=[]
    for i,m in enumerate(modes):
        s,r2,g=fit_growth(tci,A[:,i],args.fit_t0,args.fit_t1)
        fits.append((s,r2,g))

    table=np.column_stack([tci,bz_rms,by_rms,ey_rms,bx_err,jy_err,widths,flux_span,A])
    hdr=("omega_ci_t Bz_rms_B0 By_rms_B0 Ey_rms_cB0 Bx_err_B0 Jy_err_J0 Jy_FWHM_lambdaD "
         "deltaAy_span_B0L "+" ".join(f"A_m{m}" for m in modes))
    np.savetxt("harris_500ld_40ld_mode_history.txt",table,header=hdr)

    with open("harris_500ld_40ld_early_summary.txt","w") as f:
        f.write("Harris 500 lambda_D box / 40 lambda_D sheet early diagnostics\n")
        f.write("===========================================================\n\n")
        f.write(f"plotfiles = {len(files)}\n")
        f.write(f"final omega_ci*t = {tci[-1]:.8f}\n")
        f.write(f"lambda_D/de = {P['lambdaD']/P['de']:.8f}\n")
        f.write(f"box/de = {args.box_lambdaD*P['lambdaD']/P['de']:.8f}\n")
        f.write(f"sheet_L/de = {args.sheet_lambdaD*P['lambdaD']/P['de']:.8f}\n")
        f.write(f"sheet_L/di = {P['L']/P['di']:.8f}\n")
        f.write(f"final Bz core rms/B0 = {bz_rms[-1]:.8e}\n")
        f.write(f"final By core rms/B0 = {by_rms[-1]:.8e}\n")
        f.write(f"final Ey core rms/cB0 = {ey_rms[-1]:.8e}\n")
        f.write(f"final Bx profile error = {bx_err[-1]:.8e}\n")
        f.write(f"final Jy profile error = {jy_err[-1]:.8e}\n")
        f.write(f"final Jy FWHM/lambda_D = {widths[-1]:.8f}\n")
        f.write(f"analytic Jy FWHM/lambda_D = {1.762747174*args.sheet_lambdaD:.8f}\n")
        f.write(f"final deltaAy span/(B0 L) = {flux_span[-1]:.8e}\n\n")
        f.write("mode k*lambdaD k*L DeltaPrime*L final_A/B0 gamma/wci R2 growth accepted\n")
        for i,m in enumerate(modes):
            s,r2,g=fits[i]
            acc=np.isfinite(s) and s>0 and r2>=args.min_r2 and g>=args.min_growth
            f.write(f"{m:3d} {kld[i]:.8f} {kL[i]:.8f} {dprimeL[i]: .8f} {A[-1,i]:.8e} "
                    f"{s: .8e} {r2:.5f} {g:.4f} {'YES' if acc else 'no'}\n")

    fig,ax=plt.subplots(figsize=(8,5.5))
    for m in range(1,min(10,args.max_mode)+1):
        ax.semilogy(tci,A[:,m-1],"o-",ms=4,label=f"m={m}")
    ax.set_xlabel(r"$\omega_{ci}t$"); ax.set_ylabel(r"core $B_z$ Fourier amplitude / $B_0$")
    ax.grid(alpha=.25); ax.legend(ncol=2,fontsize=8); fig.tight_layout()
    fig.savefig("harris_500ld_40ld_low_modes.png",dpi=200); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5.5))
    gam=np.array([x[0] for x in fits]); good=np.array([np.isfinite(x[0]) and x[0]>0 and x[1]>=args.min_r2 and x[2]>=args.min_growth for x in fits])
    ax.plot(kL,gam,"o-",label="fixed-window slope")
    if np.any(good): ax.plot(kL[good],gam[good],"s",ms=7,label="accepted")
    ax.axhline(0,lw=.8); ax.axvline(1,ls="--",lw=.8,label=r"$kL=1$")
    ax.set_xlabel(r"$kL$"); ax.set_ylabel(r"$\gamma/\omega_{ci}$")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig("harris_500ld_40ld_gamma_vs_kL.png",dpi=200); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5.5))
    ax.semilogy(tci,bz_rms,"o-",label=r"$B_{z,rms}/B_0$")
    ax.semilogy(tci,by_rms,"o-",label=r"$B_{y,rms}/B_0$")
    ax.semilogy(tci,ey_rms,"o-",label=r"$E_{y,rms}/cB_0$")
    ax.semilogy(tci,bx_err,"o-",label=r"$B_x$ profile error")
    ax.semilogy(tci,jy_err,"o-",label=r"$J_y$ profile error")
    ax.set_xlabel(r"$\omega_{ci}t$"); ax.set_ylabel("normalized amplitude/error")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig("harris_500ld_40ld_field_history.png",dpi=200); plt.close(fig)

    F,jyc,Ay,dAy=final; zld=F["z"]/P["lambdaD"]
    bxm=np.mean(F["Bx"],axis=0); jym=np.mean(jyc,axis=0)
    fig,ax=plt.subplots(figsize=(7.5,5.5))
    ax.plot(zld,bxm/P["B0"],label="final x-avg")
    ax.plot(zld,np.tanh(F["z"]/P["L"]),"--",label="analytic tanh")
    ax.set_xlim(-3*args.sheet_lambdaD,3*args.sheet_lambdaD); ax.set_xlabel(r"$z/\lambda_D$"); ax.set_ylabel(r"$\langle B_x\rangle/B_0$")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig("harris_500ld_40ld_final_Bx.png",dpi=200); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,5.5))
    ax.plot(zld,jym/P["J0"],label="final curl-B Jy")
    ax.plot(zld,1/np.cosh(F["z"]/P["L"])**2,"--",label=r"analytic $sech^2$")
    ax.set_xlim(-3*args.sheet_lambdaD,3*args.sheet_lambdaD); ax.set_xlabel(r"$z/\lambda_D$"); ax.set_ylabel(r"$\langle J_y\rangle/J_0$")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig("harris_500ld_40ld_final_Jy.png",dpi=200); plt.close(fig)

    xld=F["x"]/P["lambdaD"]; zoom=np.abs(zld)<=args.zoom_sheet*args.sheet_lambdaD
    X,Z=np.meshgrid(xld,zld,indexing="ij")
    dn=dAy/(P["B0"]*P["L"]); vmax=np.nanmax(np.abs(dn[:,zoom]))
    fig,ax=plt.subplots(figsize=(11,5.5)); pcm=ax.pcolormesh(X[:,zoom],Z[:,zoom],dn[:,zoom],shading="auto")
    if vmax>0: ax.contour(X[:,zoom],Z[:,zoom],dn[:,zoom],levels=np.linspace(-vmax,vmax,31),colors="k",linewidths=.45,alpha=.6)
    fig.colorbar(pcm,ax=ax,label=r"$\delta A_y/(B_0L)$"); ax.set_xlabel(r"$x/\lambda_D$"); ax.set_ylabel(r"$z/\lambda_D$")
    ax.set_title(fr"perturbation topology, $\omega_{{ci}}t={tci[-1]:.3f}$"); fig.tight_layout(); fig.savefig("harris_500ld_40ld_deltaAy_topology.png",dpi=200); plt.close(fig)

    print("Saved harris_500ld_40ld_early_summary.txt and diagnostic figures")


if __name__ == "__main__":
    main()
