#!/usr/bin/env python3
"""Validate Stage-A open-boundary Harris equilibrium for the Ishizawa work.

The diagnostic is intentionally focused on boundary contamination rather than
tearing growth.  It measures whether switching from periodic/PEC/reflecting to
absorbing field/particle boundaries immediately destroys the Harris sheet.

Outputs
-------
ishizawa_open_equilibrium_summary.txt
ishizawa_open_equilibrium_history.txt
ishizawa_open_equilibrium_history.png
ishizawa_open_equilibrium_profiles.png

divB is normalized by B0/L.  It is monitored because the initial MLMG
projection cleaner is intentionally disabled for Silver-Mueller boundaries.
"""

from __future__ import annotations

import argparse
import glob
import math
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
    p.add_argument("--interior-fraction", type=float, default=0.80,
                   help="fraction of each half-domain treated as the interior")
    return p.parse_args()


def numeric_key(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def parse_numeric(text, key, default):
    pat = re.compile(
        rf"^\s*{re.escape(key)}\s*=\s*([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)",
        re.M,
    )
    m = pat.search(text)
    return float(m.group(1).replace("D", "E").replace("d", "e")) if m else default


def params(input_path):
    text = Path(input_path).read_text(encoding="utf-8", errors="replace")
    n0 = parse_numeric(text, "my_constants.n0", 1.0e19)
    mi_me = 800.0
    wpe_wce = parse_numeric(text, "my_constants.wpe_wce", 3.5)
    wpe = math.sqrt(n0*QE**2/(EPS0*ME))
    wce = wpe/wpe_wce
    wci = wce/mi_me
    de = C/wpe
    di = math.sqrt(mi_me)*de
    B0 = ME*wce/QE
    Te = B0**2/(4.0*MU0*n0)
    lamD = math.sqrt(EPS0*Te/(n0*QE**2))
    L = de
    J0 = B0/(MU0*L)
    return dict(
        n0=n0, mi_me=mi_me, wpe=wpe, wce=wce, wci=wci,
        de=de, di=di, B0=B0, Te=Te, lambdaD=lamD, L=L, J0=J0
    )


def as_xz(a, nx, nz):
    a = np.asarray(a).squeeze()
    if a.shape == (nx, nz):
        return a
    if a.shape == (nz, nx):
        return a.T
    raise RuntimeError(f"unexpected field shape {a.shape}")


def load(path):
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

    g = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )

    def fld(name, unit=None, required=True):
        try:
            q = g[("boxlib", name)]
        except Exception:
            if required:
                raise
            return None
        try:
            a = q.to_value(unit) if unit else q.to_ndarray()
        except Exception:
            a = q.to_ndarray()
        return as_xz(a, nx, nz)

    return dict(
        ds=ds, x=x, z=z, dx=dx, dz=dz,
        Bx=fld("Bx","T"), By=fld("By","T"), Bz=fld("Bz","T"),
        Ex=fld("Ex","V/m"), Ey=fld("Ey","V/m"), Ez=fld("Ez","V/m"),
        jy=fld("jy","A/m**2"), rho=fld("rho","C/m**3"),
        rho_e=fld("rho_electrons","C/m**3"),
        rho_i=fld("rho_ions","C/m**3"),
        divB=fld("divB","T/m",required=False),
    )


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, float)**2)))


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)

    run = Path(args.run_dir).resolve()
    inp = run/"inputs_open_equilibrium"
    if not inp.exists():
        found = sorted(run.glob("inputs*"))
        if not found:
            raise FileNotFoundError(f"no runtime input under {run}")
        inp = found[0]

    P = params(inp)
    files = sorted(glob.glob(str(run/"diags"/"diag1*")), key=numeric_key)
    if not files:
        raise FileNotFoundError(f"no diag1* plotfiles under {run/'diags'}")

    rows = []
    snapshots = []

    for fn in files:
        F = load(fn)
        x, z = F["x"], F["z"]
        xmax = max(abs(x[0]),abs(x[-1]))
        zmax = max(abs(z[0]),abs(z[-1]))

        interior_x = np.abs(x) <= args.interior_fraction*xmax
        interior_z = np.abs(z) <= args.interior_fraction*zmax
        interior2d = interior_x[:,None] & interior_z[None,:]
        edge2d = ~interior2d

        Bxref = P["B0"]*np.tanh(z/P["L"])
        Jyref = P["J0"]/np.cosh(z/P["L"])**2

        # Do not let downstream x-boundary layers contaminate the nominal
        # "interior" 1-D Harris profiles: average only over interior x.
        bxm = np.mean(F["Bx"][interior_x,:],axis=0)

        dBx_dz = np.gradient(F["Bx"],F["dz"],axis=1,edge_order=2)
        dBz_dx = np.gradient(F["Bz"],F["dx"],axis=0,edge_order=2)
        jycurl = (dBx_dz-dBz_dx)/MU0
        jym = np.mean(jycurl[interior_x,:],axis=0)

        ne = -F["rho_e"]/QE
        ni = F["rho_i"]/QE
        ntot0 = P["n0"]/np.cosh(z/P["L"])**2
        ne_m = np.mean(ne[interior_x,:],axis=0)
        ni_m = np.mean(ni[interior_x,:],axis=0)

        E2 = F["Ex"]**2+F["Ey"]**2+F["Ez"]**2
        Bpert2 = F["By"]**2+F["Bz"]**2
        tci = P["wci"]*F["ds"].current_time.to_value("s")
        divB_norm = (
            rms(F["divB"][interior2d])/(P["B0"]/P["L"])
            if F["divB"] is not None else np.nan
        )

        row = dict(
            omega_ci_t=tci,
            Bx_interior_rms_B0=rms((bxm-Bxref)[interior_z])/P["B0"],
            Jy_interior_rms_J0=rms((jym-Jyref)[interior_z])/P["J0"],
            density_e_rms_n0=rms((ne_m-ntot0)[interior_z])/P["n0"],
            density_i_rms_n0=rms((ni_m-ntot0)[interior_z])/P["n0"],
            neutrality_rms_n0=rms(ni-ne)/P["n0"],
            E_interior_rms_cB0=rms(np.sqrt(E2[interior2d]))/(C*P["B0"]),
            Bpert_interior_rms_B0=rms(np.sqrt(Bpert2[interior2d]))/P["B0"],
            E_edge_rms_cB0=rms(np.sqrt(E2[edge2d]))/(C*P["B0"]),
            Bpert_edge_rms_B0=rms(np.sqrt(Bpert2[edge2d]))/P["B0"],
            divB_interior_rms_norm=divB_norm,
            electron_inventory=np.sum(ne),
            ion_inventory=np.sum(ni),
        )
        rows.append(row)
        snapshots.append((fn,F,bxm,jym,ne_m,ni_m,Bxref,Jyref,ntot0))

        print(
            f"{Path(fn).name}: wci*t={tci:.6f}  "
            f"Bx={row['Bx_interior_rms_B0']:.3e}  "
            f"Jy={row['Jy_interior_rms_J0']:.3e}  "
            f"Eint={row['E_interior_rms_cB0']:.3e}  "
            f"Eedge={row['E_edge_rms_cB0']:.3e}  "
            f"divB={row['divB_interior_rms_norm']:.3e}"
        )

    keys = list(rows[0])
    arr = np.array([[r[k] for k in keys] for r in rows],float)
    arr[:,keys.index("electron_inventory")] /= arr[0,keys.index("electron_inventory")]
    arr[:,keys.index("ion_inventory")] /= arr[0,keys.index("ion_inventory")]

    np.savetxt(
        "ishizawa_open_equilibrium_history.txt",
        arr,
        header=" ".join(keys),
    )

    final = rows[-1]
    with open("ishizawa_open_equilibrium_summary.txt","w",encoding="utf-8") as f:
        f.write("Ishizawa 2005 Stage-A open-boundary equilibrium summary\n")
        f.write("=======================================================\n\n")
        f.write(f"run_dir = {run}\n")
        f.write(f"plotfiles = {len(files)}\n")
        f.write(f"final omega_ci*t = {final['omega_ci_t']:.8e}\n")
        f.write(f"L/de = {P['L']/P['de']:.8f}\n")
        f.write(
            "half-height/di = "
            f"{max(abs(snapshots[-1][1]['z'][0]),abs(snapshots[-1][1]['z'][-1]))/P['di']:.8f}\n\n"
        )
        for k in keys[1:-2]:
            f.write(f"final {k} = {final[k]:.8e}\n")
        f.write(
            "electron inventory fraction = "
            f"{arr[-1,keys.index('electron_inventory')]:.8e}\n"
        )
        f.write(
            "ion inventory fraction = "
            f"{arr[-1,keys.index('ion_inventory')]:.8e}\n"
        )
        f.write(
            "\nInterpretation: Stage A passes only if interior Harris structure remains stable\n"
        )
        f.write(
            "while losses/distortion remain confined to the boundary region.  Do not use\n"
        )
        f.write(
            "this run as the final driven Ishizawa reproduction.\n"
        )

    t = arr[:,0]
    fig,axs = plt.subplots(2,1,figsize=(9,8),sharex=True)

    for key,label in [
        ("Bx_interior_rms_B0","Bx profile"),
        ("Jy_interior_rms_J0","Jy profile"),
        ("neutrality_rms_n0","neutrality"),
        ("E_interior_rms_cB0","E interior"),
        ("Bpert_interior_rms_B0","Bpert interior"),
        ("divB_interior_rms_norm","divB interior"),
    ]:
        axs[0].semilogy(
            t,np.maximum(arr[:,keys.index(key)],1e-30),
            "o-",label=label
        )

    axs[0].set_ylabel("normalized RMS")
    axs[0].grid(alpha=.25)
    axs[0].legend(fontsize=8)

    axs[1].plot(
        t,arr[:,keys.index("electron_inventory")],
        "o-",label="electron inventory"
    )
    axs[1].plot(
        t,arr[:,keys.index("ion_inventory")],
        "o-",label="ion inventory"
    )
    axs[1].set_xlabel(r"$\omega_{ci}t$")
    axs[1].set_ylabel("fraction of initial")
    axs[1].grid(alpha=.25)
    axs[1].legend()

    fig.tight_layout()
    fig.savefig("ishizawa_open_equilibrium_history.png",dpi=200)
    plt.close(fig)

    chosen = [snapshots[0],snapshots[-1]] if len(snapshots)>1 else [snapshots[0]]
    fig,axs = plt.subplots(3,1,figsize=(9,11),sharex=True)

    for fn,F,bxm,jym,ne_m,ni_m,Bxref,Jyref,ntot0 in chosen:
        tag = Path(fn).name
        z = F["z"]/P["de"]
        axs[0].plot(z,bxm/P["B0"],label=f"Bx {tag}")
        axs[1].plot(z,jym/P["J0"],label=f"curl-B Jy {tag}")
        axs[2].plot(z,ne_m/P["n0"],label=f"ne {tag}")
        axs[2].plot(z,ni_m/P["n0"],"--",label=f"ni {tag}")

    z = chosen[-1][1]["z"]/P["de"]
    axs[0].plot(z,chosen[-1][6]/P["B0"],"k--",label="Harris tanh")
    axs[1].plot(z,chosen[-1][7]/P["J0"],"k--",label="Harris sech2")
    axs[2].plot(z,chosen[-1][8]/P["n0"],"k--",label="Harris density")

    axs[0].set_ylabel("Bx/B0")
    axs[1].set_ylabel("Jy/J0")
    axs[2].set_ylabel("n/n0")
    axs[2].set_xlabel("z/de")

    for ax in axs:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8,ncol=2)

    fig.tight_layout()
    fig.savefig("ishizawa_open_equilibrium_profiles.png",dpi=200)
    plt.close(fig)

    print("Saved Ishizawa Stage-A open-boundary equilibrium diagnostics.")


if __name__ == "__main__":
    main()
