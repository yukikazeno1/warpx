#!/usr/bin/env python3
"""Inspect an Ishizawa native-gather OpenPMD diagnostic.

Prints iteration numbers, species, particle record/component names, unit_SI
metadata, and particle counts.  This is intentionally schema-only: it lets us
verify the exact names that WarpX wrote for gathered Ex/Ey/Ez/Bx/By/Bz before
the quantitative native-vs-cell-centered Q_EM comparison.
"""

import argparse
from pathlib import Path

import numpy as np
import openpmd_api as io


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("series",
                   help="OpenPMD series pattern, e.g. diags/native_%T.h5")
    return p.parse_args()


def component_shape(comp):
    try:
        return tuple(comp.shape)
    except Exception:
        try:
            return tuple(comp.get_extent())
        except Exception:
            return None


def main():
    args=parse_args()
    series=io.Series(args.series, io.Access.read_only)
    its=sorted(series.iterations)
    print("series:",args.series)
    print("iterations:",its)

    for itn in its:
        it=series.iterations[itn]
        print(f"\n=== iteration {itn} ===")
        try:
            print("time =",it.time,"time_unit_SI =",it.time_unit_SI)
        except Exception:
            pass
        print("particle species:",list(it.particles))

        for sname in it.particles:
            sp=it.particles[sname]
            print(f"\n  [{sname}] records:")
            for rname in sp:
                rec=sp[rname]
                comps=list(rec)
                print(f"    {rname}: components={comps}")
                for cname in comps:
                    comp=rec[cname]
                    unit=getattr(comp,"unit_SI",None)
                    print(
                        f"      {cname}: shape={component_shape(comp)} "
                        f"unit_SI={unit}"
                    )

    series.flush()


if __name__=="__main__":
    main()
