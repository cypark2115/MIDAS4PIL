# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""
Numba-accelerated kernels for integration and caking.

Single-pass histogramming replaces the numpy chain of:
    mask → fancy-index → range-filter → searchsorted → bincount
with one loop over the 2D arrays — no temporary arrays allocated.

All functions are @njit compiled on first call (~2-5 s warm-up).
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _jit_histogram_1d(image, mask, tth_lut, tth_edges,
                      eta_lut, eta_min, eta_max):
    """Single-pass 1D histogram for integration.

    Iterates over every pixel once. For each unmasked pixel whose
    2θ falls within [tth_edges[0], tth_edges[-1]) and whose eta
    falls within [eta_min, eta_max], finds the bin via binary search
    and accumulates intensity + count.

    Parameters
    ----------
    image     : float64 2-D array (nrows, ncols)
    mask      : bool/int 2-D array (1=bad, 0=good)
    tth_lut   : float64 2-D array, 2θ in degrees
    tth_edges : float64 1-D array, bin edges (n_bins + 1)
    eta_lut   : float64 2-D array, eta in degrees
    eta_min   : float64, lower eta limit
    eta_max   : float64, upper eta limit

    Returns
    -------
    I_sum  : float64 1-D array (n_bins,)
    px_cnt : float64 1-D array (n_bins,)
    """
    nrows = image.shape[0]
    ncols = image.shape[1]
    n_bins = len(tth_edges) - 1
    I_sum  = np.zeros(n_bins, dtype=np.float64)
    px_cnt = np.zeros(n_bins, dtype=np.float64)

    tth_lo = tth_edges[0]
    tth_hi = tth_edges[n_bins]  # == tth_edges[-1]
    use_eta = (eta_min > -179.99) or (eta_max < 179.99)

    for r in range(nrows):
        for c in range(ncols):
            if mask[r, c] != 0:
                continue
            tth = tth_lut[r, c]
            if tth < tth_lo or tth >= tth_hi:
                continue
            if use_eta:
                eta = eta_lut[r, c]
                if eta < eta_min or eta > eta_max:
                    continue
            b = np.searchsorted(tth_edges, tth, side='right') - 1
            if b >= n_bins:
                b = n_bins - 1
            I_sum[b]  += image[r, c]
            px_cnt[b] += 1.0

    return I_sum, px_cnt


@njit(cache=True)
def _jit_histogram_2d(image, mask, tth_lut, eta_lut, tth_edges,
                      eta_min, eta_bin_size, n_eta):
    """Single-pass 2D histogram for caking.

    Parameters
    ----------
    image        : float64 2-D array (nrows, ncols)
    mask         : bool/int 2-D array
    tth_lut      : float64 2-D array, 2θ degrees
    eta_lut      : float64 2-D array, eta degrees
    tth_edges    : float64 1-D array, 2θ bin edges
    eta_min      : float64, lower eta edge
    eta_bin_size : float64, eta bin width (degrees)
    n_eta        : int, number of eta bins

    Returns
    -------
    I_sum  : float64 1-D array (n_tth * n_eta,)
    px_cnt : float64 1-D array (n_tth * n_eta,)
    """
    nrows = image.shape[0]
    ncols = image.shape[1]
    n_tth = len(tth_edges) - 1
    n_total = n_tth * n_eta
    I_sum  = np.zeros(n_total, dtype=np.float64)
    px_cnt = np.zeros(n_total, dtype=np.float64)

    tth_lo = tth_edges[0]
    tth_hi = tth_edges[n_tth]
    eta_hi = eta_min + n_eta * eta_bin_size

    for r in range(nrows):
        for c in range(ncols):
            if mask[r, c] != 0:
                continue
            tth = tth_lut[r, c]
            if tth < tth_lo or tth >= tth_hi:
                continue
            eta = eta_lut[r, c]
            if eta < eta_min or eta >= eta_hi:
                continue
            i_tth = np.searchsorted(tth_edges, tth, side='right') - 1
            if i_tth >= n_tth:
                i_tth = n_tth - 1
            i_eta = int((eta - eta_min) / eta_bin_size)
            if i_eta >= n_eta:
                i_eta = n_eta - 1
            flat_idx = i_tth * n_eta + i_eta
            I_sum[flat_idx]  += image[r, c]
            px_cnt[flat_idx] += 1.0

    return I_sum, px_cnt


@njit(cache=True)
def _jit_precompute_bins_1d(mask, tth_lut, tth_edges,
                            eta_lut, eta_min, eta_max):
    """Precompute 1D bin index map.  -1 = excluded pixel.

    Returns
    -------
    bin_map : int32 2-D array (nrows, ncols), bin index or -1
    """
    nrows = mask.shape[0]
    ncols = mask.shape[1]
    n_bins = len(tth_edges) - 1
    bin_map = np.full((nrows, ncols), -1, dtype=np.int32)

    tth_lo = tth_edges[0]
    tth_hi = tth_edges[n_bins]
    use_eta = (eta_min > -179.99) or (eta_max < 179.99)

    for r in range(nrows):
        for c in range(ncols):
            if mask[r, c] != 0:
                continue
            tth = tth_lut[r, c]
            if tth < tth_lo or tth >= tth_hi:
                continue
            if use_eta:
                eta = eta_lut[r, c]
                if eta < eta_min or eta > eta_max:
                    continue
            b = np.searchsorted(tth_edges, tth, side='right') - 1
            if b >= n_bins:
                b = n_bins - 1
            bin_map[r, c] = b

    return bin_map


@njit(cache=True)
def _jit_precompute_bins_2d(mask, tth_lut, eta_lut, tth_edges,
                            eta_min, eta_bin_size, n_eta):
    """Precompute 2D (cake) flat bin index map.  -1 = excluded pixel.

    Returns
    -------
    flat_map : int32 2-D array (nrows, ncols), flat index or -1
    """
    nrows = mask.shape[0]
    ncols = mask.shape[1]
    n_tth = len(tth_edges) - 1
    flat_map = np.full((nrows, ncols), -1, dtype=np.int32)

    tth_lo = tth_edges[0]
    tth_hi = tth_edges[n_tth]
    eta_hi = eta_min + n_eta * eta_bin_size

    for r in range(nrows):
        for c in range(ncols):
            if mask[r, c] != 0:
                continue
            tth = tth_lut[r, c]
            if tth < tth_lo or tth >= tth_hi:
                continue
            eta = eta_lut[r, c]
            if eta < eta_min or eta >= eta_hi:
                continue
            i_tth = np.searchsorted(tth_edges, tth, side='right') - 1
            if i_tth >= n_tth:
                i_tth = n_tth - 1
            i_eta = int((eta - eta_min) / eta_bin_size)
            if i_eta >= n_eta:
                i_eta = n_eta - 1
            flat_map[r, c] = i_tth * n_eta + i_eta

    return flat_map


@njit(cache=True)
def _jit_accumulate(image, bin_map, n_bins):
    """Accumulate pixel intensities into precomputed bins.

    Parameters
    ----------
    image   : float64 2-D array
    bin_map : int32 2-D array, bin index per pixel (-1 = skip)
    n_bins  : int, number of bins

    Returns
    -------
    I_sum  : float64 1-D array (n_bins,)
    px_cnt : float64 1-D array (n_bins,)
    """
    nrows = image.shape[0]
    ncols = image.shape[1]
    I_sum  = np.zeros(n_bins, dtype=np.float64)
    px_cnt = np.zeros(n_bins, dtype=np.float64)

    for r in range(nrows):
        for c in range(ncols):
            b = bin_map[r, c]
            if b < 0:
                continue
            v = image[r, c]
            if v < 0.0:
                continue
            I_sum[b]  += v
            px_cnt[b] += 1.0

    return I_sum, px_cnt


@njit(cache=True)
def _jit_snip(y_in, n_iter):
    """SNIP background (numba-compiled).

    Same algorithm as snip_background() but the iterative loop
    runs in compiled code instead of Python.

    Parameters
    ----------
    y_in   : float64 1-D array, raw intensities
    n_iter : int, clipping iterations

    Returns
    -------
    bg : float64 1-D array, background estimate
    """
    n = len(y_in)
    y = np.empty(n, dtype=np.float64)

    # LLS transform
    for i in range(n):
        v = y_in[i]
        if v < 0.0:
            v = 0.0
        y[i] = np.log(np.log(np.sqrt(v + 1.0) + 1.0) + 1.0)

    # Iterative clipping with decreasing window
    for m in range(n_iter, 0, -1):
        for i in range(m, n - m):
            avg = (y[i - m] + y[i + m]) * 0.5
            if avg < y[i]:
                y[i] = avg

    # Inverse LLS
    bg = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = (np.exp(np.exp(y[i]) - 1.0) - 1.0) ** 2 - 1.0
        if v < 0.0:
            v = 0.0
        bg[i] = v

    return bg
