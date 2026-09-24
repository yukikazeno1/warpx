#!/usr/bin/env python3
"""Field-line Alfvén-transit diagnostic for the saturated island breathing mode.

This script tests whether the measured breathing period can be associated with
an Alfvénic restoring time inside the nonlinear system-scale magnetic island.

For representative breathing phases, it:
  1. reconstructs A_y from B_x and B_z;
  2. low-pass filters A_y in x to isolate the system-scale island;
  3. rolls x so the O point is centered;
  4. extracts closed flux surfaces
         A_y = A_X + f (A_O - A_X)
     for several flux fractions f;
  5. integrates the local Alfvén travel time around each closed surface.

For a 2-D field with a possible Hall B_y, the physical travel time along the
3-D magnetic field can be written using the projected x-z arc length ds_p as

    dt = ds_p * sqrt(mu0 rho_i) / B_p,

where B_p = sqrt(B_x^2+B_z^2).  The explicit B_y cancels between the longer
3-D field-line length and the total-field Alfvén speed.  Therefore the closed
loop travel time is

    tau_A = integral ds_p / v_Ap
          = integral ds_p * sqrt(mu0 m_i n_i) / B_p.

The script reports omega_ci*tau_A as well as tau_A/2 and tau_A/4, because a
standing/restoring oscillation can couple to different harmonics of a closed
field-line transit time.

Outputs
-------
ishizawa_alfven_transit_table.txt
ishizawa_alfven_transit_summary.txt
ishizawa_alfven_transit_vs_flux.png
ishizawa_alfven_transit_contours.png
ishizawa_alfven_transit_ratio.png
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import yt

from analyze_ishizawa_scale_40000 import QE, MU0, params, reconstruct_Ay
from analyze_ishizawa_breathing_phases import (
    load_history, select_cycle, load_full_fields
)
from analyze_ishizawa_saturation import lowpass_x, choose_system_scale_OX


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs_ishizawa_scale/ppc49")
    p.add_argument("--history", required=True)
    p.add_argument("--cycle-tmin", type=float, default=2.8)
    p.add_argument("--cycle-tmax", type=float, default=4.0)
    p.add_argument("--period-min", type=float, default=0.45)
    p.add_argument("--period-max", type=float, default=0.85)
    p.add_argument("--prominence-frac", type=float, default=0.18)
    p.add_argument("--period", type=float, default=0.63,
                   help="measured breathing period in omega_ci^-1")
    p.add_argument("--max-mode", type=int, default=2,
                   help="x low-pass cutoff used for the system-scale island")
    p.add_argument("--flux-fractions", default="0.2,0.4,0.6,0.8",
                   help="comma-separated f in Ay=AX+f(AO-AX)")
    p.add_argument("--density-floor-frac", type=float, default=1.0e-5,
                   help="numerical floor n_i/n0 used only if interpolation "
                        "returns extremely small positive density")
    return p.parse_args()


def numeric_key(path):
    m = re.search(r"(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def step_from_plotfile(path):
    name = os.path.basename(path.rstrip("/"))
    if not name.startswith("diag1"):
        return -1
    s = name[len("diag1"):]
    return int(s) if s.isdigit() else -1


def parse_csv_float(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def roll_to_center(a, iO):
    nx = a.shape[0]
    shift = nx//2 - int(iO)
    return np.roll(a, shift, axis=0), shift


def polygon_area(seg):
    x = seg[:,0]
    y = seg[:,1]
    if len(seg) < 3:
        return 0.0
    return 0.5*abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))


def extract_closed_contour(x, z, A, level, o_point, dx, dz):
    """Return the closed contour segment containing the O point."""
    fig, ax = plt.subplots(figsize=(2,2))
    try:
        cs = ax.contour(x, z, A.T, levels=[level])
        segs = cs.allsegs[0] if cs.allsegs else []
    finally:
        plt.close(fig)

    candidates = []
    tol = 3.0*max(abs(dx),abs(dz))
    for seg in segs:
        seg = np.asarray(seg,float)
        if len(seg) < 12:
            continue
        # Close the segment for geometric tests/integration.
        if np.linalg.norm(seg[-1]-seg[0]) > tol:
            # Open contours should not be used as island flux surfaces.
            continue
        poly = MplPath(seg, closed=True)
        if poly.contains_point(o_point):
            candidates.append(seg)

    if not candidates:
        return None

    # If numerical contouring returns nested/duplicate candidates, choose the
    # largest closed polygon that contains O.
    return max(candidates, key=polygon_area)


def close_segment(seg):
    if np.linalg.norm(seg[-1]-seg[0]) == 0:
        return seg
    return np.vstack([seg,seg[0]])


def integrate_alfven_time(seg, x, z, Bp, ni, P, density_floor_frac):
    seg = close_segment(np.asarray(seg,float))
    p0 = seg[:-1]
    p1 = seg[1:]
    mid = 0.5*(p0+p1)
    dl = np.sqrt(np.sum((p1-p0)**2,axis=1))

    interp_B = RegularGridInterpolator(
        (x,z), Bp, bounds_error=False, fill_value=np.nan
    )
    interp_n = RegularGridInterpolator(
        (x,z), ni, bounds_error=False, fill_value=np.nan
    )
    b = interp_B(mid)
    n = interp_n(mid)

    good = np.isfinite(b) & np.isfinite(n) & np.isfinite(dl)
    if np.count_nonzero(good) < 0.95*len(dl):
        return None
    b=b[good]; n=n[good]; dl=dl[good]

    n_floor = density_floor_frac*P["n0"]
    n_used = np.maximum(n,n_floor)
    b_floor = 1.0e-10*P["B0"]
    b_used = np.maximum(np.abs(b),b_floor)

    rho_i = P["mi"]*n_used
    dt = dl*np.sqrt(MU0*rho_i)/b_used
    tau = float(np.sum(dt))
    length = float(np.sum(dl))

    v_eff = length/tau if tau>0 else np.nan
    vA0 = P["B0"]/np.sqrt(MU0*P["mi"]*P["n0"])

    return dict(
        tau_ci=tau*P["wci"],
        tau_s=tau,
        length_de=length/P["de"],
        v_eff_over_vA0=v_eff/vA0,
        mean_Bp_B0=float(np.average(b/P["B0"],weights=dl)),
        min_Bp_B0=float(np.min(b/P["B0"])),
        mean_ni_n0=float(np.average(n/P["n0"],weights=dl)),
        min_ni_n0=float(np.min(n/P["n0"])),
        floor_fraction=float(np.mean(n<n_floor)),
    )


def main():
    args = parse_args()
    yt.funcs.mylog.setLevel(40)
    P = params()
    flux_fractions = parse_csv_float(args.flux_fractions)

    H = load_history(args.history)
    labels, inds, next_peak = select_cycle(
        H, args.cycle_tmin, args.cycle_tmax,
        args.period_min, args.period_max, args.prominence_frac
    )

    plotfiles = sorted(
        glob.glob(str(Path(args.run_dir)/"diags"/"diag1*")),
        key=numeric_key
    )
    bystep = {step_from_plotfile(p):p for p in plotfiles}

    rows = []
    contour_store = {}

    for iph,(label,ih) in enumerate(zip(labels,inds)):
        step = int(H["step"][ih])
        if step not in bystep:
            avail=np.asarray(sorted(bystep))
            step=int(avail[np.argmin(np.abs(avail-step))])

        F = load_full_fields(bystep[step])
        tci = P["wci"]*F["ds"].current_time.to_value("s")

        Ay = reconstruct_Ay(F["Bx"],F["Bz"],F["x"],F["z"])
        Ay_lp = lowpass_x(Ay,args.max_mode)
        Bx_lp = lowpass_x(F["Bx"],args.max_mode)
        Bz_lp = lowpass_x(F["Bz"],args.max_mode)
        ni = np.maximum(F["rho_i"]/QE,0.0)

        j0,io,ix,orient,AO,AX,psi_si = choose_system_scale_OX(
            Ay_lp,F["z"]
        )

        AyC,shift = roll_to_center(Ay_lp,io)
        BxC,_ = roll_to_center(Bx_lp,io)
        BzC,_ = roll_to_center(Bz_lp,io)
        niC,_ = roll_to_center(ni,io)

        nx=len(F["x"])
        xrel=(np.arange(nx)-nx//2)*F["dx"]
        z=F["z"]
        o_point=(0.0,float(z[j0]))
        Bp=np.sqrt(BxC*BxC+BzC*BzC)

        contour_store[label] = dict(
            x=xrel,z=z,Ay=AyC,AO=AO,AX=AX,t=tci,step=step,
            contours=[]
        )

        for f in flux_fractions:
            level = AX + f*(AO-AX)
            seg = extract_closed_contour(
                xrel,z,AyC,level,o_point,F["dx"],F["dz"]
            )
            if seg is None:
                print(
                    f"WARNING: no closed O-containing contour: "
                    f"{label}, f={f:.3f}"
                )
                continue

            M = integrate_alfven_time(
                seg,xrel,z,Bp,niC,P,args.density_floor_frac
            )
            if M is None:
                print(
                    f"WARNING: failed Alfvén integral: {label}, f={f:.3f}"
                )
                continue

            contour_store[label]["contours"].append((f,seg))

            rows.append([
                iph,step,tci,H["psi"][ih],H["width"][ih],f,
                M["tau_ci"],0.5*M["tau_ci"],0.25*M["tau_ci"],
                M["length_de"],M["v_eff_over_vA0"],
                M["mean_Bp_B0"],M["min_Bp_B0"],
                M["mean_ni_n0"],M["min_ni_n0"],
                M["floor_fraction"],
                args.period/M["tau_ci"],
                args.period/(0.5*M["tau_ci"]),
                args.period/(0.25*M["tau_ci"]),
            ])

            print(
                f"{label:11s} tci={tci:.3f} f={f:.2f} "
                f"loop={M['tau_ci']:.4f} half={0.5*M['tau_ci']:.4f} "
                f"quarter={0.25*M['tau_ci']:.4f} "
                f"L={M['length_de']:.2f} de"
            )

    arr=np.asarray(rows,float)
    if len(arr)==0:
        raise RuntimeError("No closed flux-surface Alfvén integrals succeeded")

    np.savetxt(
        "ishizawa_alfven_transit_table.txt",arr,
        header=(
            "phase_index step omega_ci_t Psi width_de flux_fraction "
            "tau_loop_omegaci tau_half_omegaci tau_quarter_omegaci "
            "contour_length_de v_eff_over_vA0 "
            "mean_Bp_B0 min_Bp_B0 mean_ni_n0 min_ni_n0 density_floor_fraction "
            "Tbreath_over_tau_loop Tbreath_over_tau_half "
            "Tbreath_over_tau_quarter"
        )
    )

    # Transit-time curves versus flux fraction.
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for iph,label in enumerate(labels):
        m=arr[:,0].astype(int)==iph
        if not np.any(m):
            continue
        aa=arr[m]
        order=np.argsort(aa[:,5]); aa=aa[order]
        axs[0].plot(aa[:,5],aa[:,6],"o-",label=label)
        axs[1].plot(aa[:,5],aa[:,7],"o-",label=label+" half")
        axs[1].plot(aa[:,5],aa[:,8],"s--",label=label+" quarter")
    axs[0].axhline(args.period,ls="--",label="breathing T")
    axs[1].axhline(args.period,ls="--",label="breathing T")
    axs[0].set_xlabel("flux fraction f")
    axs[0].set_ylabel(r"$\omega_{ci}\tau_{A,loop}$")
    axs[1].set_xlabel("flux fraction f")
    axs[1].set_ylabel(r"candidate standing/transit time $\omega_{ci}\tau$")
    axs[0].legend(fontsize=8)
    axs[1].legend(fontsize=7,ncol=2)
    for ax in axs: ax.grid(alpha=.25)
    fig.suptitle("Alfvén travel times on nonlinear island flux surfaces")
    fig.tight_layout()
    fig.savefig("ishizawa_alfven_transit_vs_flux.png",dpi=210)
    plt.close(fig)

    # Contour overview for all four phases.
    fig,axs=plt.subplots(1,4,figsize=(16,4.1),sharex=True,sharey=True)
    for ax,label in zip(axs,labels):
        D=contour_store[label]
        X,Z=np.meshgrid(D["x"]/P["de"],D["z"]/P["de"],indexing="ij")
        ax.contour(X,Z,D["Ay"],levels=24,linewidths=.45,alpha=.35)
        for f,seg in D["contours"]:
            ax.plot(seg[:,0]/P["de"],seg[:,1]/P["de"],lw=1.6,label=f"f={f:g}")
        ax.plot(0,D["z"][np.argmin(np.abs(D["z"]))]/P["de"],"ko",ms=3)
        ax.set_title(label+"\n"+fr"$\omega_{{ci}}t={D['t']:.3f}$")
        ax.set_xlabel(r"$x/d_e$")
        ax.set_xlim(-32,32)
        ax.set_ylim(-12,12)
        ax.grid(alpha=.15)
    axs[0].set_ylabel(r"$z/d_e$")
    axs[-1].legend(fontsize=7)
    fig.suptitle("Closed flux surfaces used for Alfvén-transit integration")
    fig.tight_layout()
    fig.savefig("ishizawa_alfven_transit_contours.png",dpi=210)
    plt.close(fig)

    # Direct ratio to the measured breathing period.
    fig,ax=plt.subplots(figsize=(8,5.2))
    for iph,label in enumerate(labels):
        m=arr[:,0].astype(int)==iph
        if not np.any(m):
            continue
        aa=arr[m]; aa=aa[np.argsort(aa[:,5])]
        ax.plot(aa[:,5],aa[:,16],"o-",label=r"$T/\tau_{loop}$ "+label)
        ax.plot(aa[:,5],aa[:,17],"s--",label=r"$T/(\tau_{loop}/2)$ "+label)
    ax.axhline(1,ls="--")
    ax.set_xlabel("flux fraction f")
    ax.set_ylabel("measured breathing period / candidate Alfvén time")
    ax.grid(alpha=.25)
    ax.legend(fontsize=7,ncol=2)
    fig.tight_layout()
    fig.savefig("ishizawa_alfven_transit_ratio.png",dpi=210)
    plt.close(fig)

    # Summary: find closest candidate among loop, half-loop and quarter-loop.
    candidates=[]
    for r in rows:
        for name,col in [("loop",6),("half",7),("quarter",8)]:
            candidates.append((
                abs(r[col]-args.period)/args.period,
                int(r[0]),r[5],name,r[col],r[2]
            ))
    candidates.sort(key=lambda x:x[0])

    with open("ishizawa_alfven_transit_summary.txt","w") as f:
        f.write("Ishizawa-scale nonlinear-island Alfvén transit diagnostic\n")
        f.write("=======================================================\n\n")
        f.write(f"measured breathing period T*omega_ci = {args.period:.8f}\n")
        f.write(f"x low-pass max mode = {args.max_mode}\n")
        f.write(f"flux fractions = {flux_fractions}\n")
        f.write(
            f"selected cycle peak1/trough/nextpeak = "
            f"{H['t'][inds[0]]:.8f} / {H['t'][inds[2]]:.8f} / "
            f"{H['t'][next_peak]:.8f}\n\n"
        )

        f.write("Best candidate matches to the measured period\n")
        f.write("---------------------------------------------\n")
        for rel,iph,ff,name,tau,tci in candidates[:12]:
            f.write(
                f"phase={labels[iph]:11s} t={tci:.5f} f={ff:.3f} "
                f"candidate={name:7s} tau*omega_ci={tau:.6f} "
                f"relative_error={rel:.5f}\n"
            )

        f.write("\nPer-phase/per-surface values\n")
        f.write("----------------------------\n")
        for r in rows:
            f.write(
                f"{labels[int(r[0])]:11s} t={r[2]:.5f} f={r[5]:.3f} "
                f"loop={r[6]:.6f} half={r[7]:.6f} quarter={r[8]:.6f} "
                f"L/de={r[9]:.4f} veff/vA0={r[10]:.4f} "
                f"<Bp>/B0={r[11]:.4f} minBp/B0={r[12]:.4e} "
                f"<ni>/n0={r[13]:.4f} min_ni/n0={r[14]:.4e} "
                f"floorfrac={r[15]:.4e}\n"
            )

        f.write("\nInterpretation guide\n")
        f.write("--------------------\n")
        f.write(
            "A physically meaningful Alfvénic identification should not rely on "
            "one accidental contour.  Look for a broad band of intermediate "
            "flux surfaces and multiple breathing phases for which the same "
            "candidate (loop, half-loop, or quarter-loop) stays near the "
            "measured T.  Strong divergence only near f->0 or f->1 is expected "
            "because B_p becomes small near separatrix/X or O critical points.\n"
        )

    print("Saved ishizawa_alfven_transit_table.txt")
    print("Saved ishizawa_alfven_transit_summary.txt")
    print("Saved ishizawa_alfven_transit_vs_flux.png")
    print("Saved ishizawa_alfven_transit_contours.png")
    print("Saved ishizawa_alfven_transit_ratio.png")


if __name__=="__main__":
    main()
