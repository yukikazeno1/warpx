#!/usr/bin/env python3
"""Corrected current-sheet diagnostic for the Ishizawa-scale pilot.

Uses (curl B)_y = dBx/dz - dBz/dx.  This corrects the sign in the first
version of analyze_ishizawa_scale_early.py.  It reports the final current
profile error and FWHM and overlays deposited Jy when present.
"""
import argparse, glob, os, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yt

QE=1.602176634e-19
ME=9.1093837139e-31
EPS0=8.8541878128e-12
MU0=1.25663706212e-6
C=299792458.0

def key(p):
    m=re.search(r"(\d+)$", os.path.basename(p.rstrip('/')))
    return int(m.group(1)) if m else -1

def as_xz(a,nx,nz):
    a=np.asarray(a).squeeze()
    if a.shape==(nx,nz): return a
    if a.shape==(nz,nx): return a.T
    raise RuntimeError(a.shape)

def fwhm(z,y):
    ymax=np.nanmax(y)
    ids=np.flatnonzero(y>=0.5*ymax)
    return np.nan if ids.size<2 else z[ids[-1]]-z[ids[0]]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',default='runs_ishizawa_scale/ppc49')
    args=ap.parse_args()
    files=sorted(glob.glob(str(Path(args.run_dir)/'diags'/'diag1*')),key=key)
    if not files: raise FileNotFoundError('no diag1 plotfiles')
    ds=yt.load(files[-1]); yt.funcs.mylog.setLevel(40)
    nx,nz=map(int,ds.domain_dimensions[:2])
    xlo=ds.domain_left_edge[0].to_value('m'); xhi=ds.domain_right_edge[0].to_value('m')
    zlo=ds.domain_left_edge[1].to_value('m'); zhi=ds.domain_right_edge[1].to_value('m')
    dx=(xhi-xlo)/nx; dz=(zhi-zlo)/nz
    z=zlo+(np.arange(nz)+0.5)*dz
    g=ds.covering_grid(level=0,left_edge=ds.domain_left_edge,dims=ds.domain_dimensions)
    Bx=as_xz(g['boxlib','Bx'].to_value('T'),nx,nz)
    Bz=as_xz(g['boxlib','Bz'].to_value('T'),nx,nz)
    try: jy_dep=as_xz(g['boxlib','jy'].to_value('A/m**2'),nx,nz)
    except Exception: jy_dep=as_xz(g['boxlib','jy'].to_ndarray(),nx,nz)

    n0=1e19; wpe=np.sqrt(n0*QE**2/(EPS0*ME)); de=C/wpe
    wce=wpe/3.5; B0=ME*wce/QE; J0=B0/(MU0*de)
    dBx_dz=np.gradient(Bx,dz,axis=1,edge_order=2)
    dBz_dx=np.gradient(Bz,dx,axis=0,edge_order=2)
    jy=(dBx_dz-dBz_dx)/MU0
    jm=np.mean(jy,axis=0); jd=np.mean(jy_dep,axis=0)
    ref=J0/np.cosh(z/de)**2
    sheet=np.abs(z)<=4*de
    err=np.sqrt(np.mean((jm[sheet]-ref[sheet])**2))/J0
    width=fwhm(z,jm)/de
    dep_rel=np.sqrt(np.mean((jd[sheet]-jm[sheet])**2))/np.sqrt(np.mean(jm[sheet]**2))
    print(f'final plotfile              = {files[-1]}')
    print(f'corrected Jy RMS error/J0   = {err:.8e}')
    print(f'corrected Jy FWHM/de        = {width:.8f}')
    print(f'deposited-vs-curl relRMS    = {dep_rel:.8e}')
    print('analytic Harris FWHM/de     = 1.76274717')

    fig,ax=plt.subplots(figsize=(8,5.5))
    ax.plot(z/de,jm/J0,label=r'corrected $(\nabla\times B)_y/(\mu_0J_0)$')
    ax.plot(z/de,jd/J0,label=r'deposited $J_y/J_0$',alpha=0.8)
    ax.plot(z/de,ref/J0,'--',label=r'initial $\mathrm{sech}^2$')
    ax.set_xlim(-8,8); ax.set_xlabel(r'$z/d_e$'); ax.set_ylabel(r'$J_y/J_0$')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig('ishizawa_scale_final_current_corrected.png',dpi=200)

if __name__=='__main__': main()
