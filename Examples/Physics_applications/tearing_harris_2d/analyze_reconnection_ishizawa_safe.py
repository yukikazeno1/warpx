#!/usr/bin/env python3
"""Compatibility wrapper for analyze_reconnection_ishizawa.py.

Fixes NumPy's read-only broadcast view in infer_current_scale without changing
the main analyzer.  This wrapper can be removed once the main file is updated
in-place.
"""

import numpy as np
import analyze_reconnection_ishizawa as base


def infer_current_scale_safe(jraw, jcurl, z, L):
    """Infer one native-J scale using a writable/functional boolean mask."""
    core = np.abs(z) <= 2.0 * L
    # np.broadcast_to returns a read-only view.  Avoid in-place '&=' here.
    mask = (
        np.broadcast_to(core[None, :], jraw.shape)
        & np.isfinite(jraw)
        & np.isfinite(jcurl)
    )
    den = np.sum(jraw[mask] ** 2)
    if den <= 0:
        return 1.0
    return float(np.sum(jraw[mask] * jcurl[mask]) / den)


base.infer_current_scale = infer_current_scale_safe

if __name__ == "__main__":
    base.main()
