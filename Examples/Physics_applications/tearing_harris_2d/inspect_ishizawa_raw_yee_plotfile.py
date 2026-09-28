#!/usr/bin/env python3
"""Inspect a WarpX plotfile containing raw staggered Yee fields.

This is a schema/metadata probe used before implementing the exact
shape-2 energy-conserving gather in Python.  It prints all native yt fields,
highlights electromagnetic/raw fields, and reports domain geometry.
"""

import argparse
import re
import numpy as np
import yt


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("plotfile", help="WarpX plotfile, e.g. diags/native202000")
    return p.parse_args()


def main():
    args=parse_args()
    ds=yt.load(args.plotfile)
    print("plotfile:",args.plotfile)
    print("current_time:",ds.current_time)
    print("domain_left_edge:",ds.domain_left_edge)
    print("domain_right_edge:",ds.domain_right_edge)
    print("domain_dimensions:",ds.domain_dimensions)
    print("dimensionality:",ds.dimensionality)

    fields=list(ds.field_list)
    print("\n=== ds.field_list ===")
    for f in fields:
        print(f)

    print("\n=== likely EM/raw fields ===")
    patt=re.compile(r"(raw|(^|[_])(Ex|Ey|Ez|Bx|By|Bz)($|[_]))",re.I)
    selected=[]
    for f in fields:
        name="/".join(map(str,f))
        if patt.search(name):
            selected.append(f)
            print(f)

    print("\n=== field access probe ===")
    ad=ds.all_data()
    for f in selected:
        try:
            arr=ad[f]
            a=np.asarray(arr)
            print(
                f"{f}: shape={a.shape} size={a.size} "
                f"min={np.nanmin(a):.8e} max={np.nanmax(a):.8e}"
            )
        except Exception as exc:
            print(f"{f}: ACCESS FAILED: {type(exc).__name__}: {exc}")


if __name__=="__main__":
    main()
