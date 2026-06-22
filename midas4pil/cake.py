"""
Phase 3 — Caked image.

Maps detector pixels to a 2-D (2θ, eta) histogram using the
precomputed lookup tables.  Per-bin statistic is the mean; Poisson SEM
(σ = √ΣI / N) is only valid for the mean of counts, not the median.

Note: median-per-bin was removed because it cannot be combined with
correct Poisson error propagation.
"""

import numpy as np

try:
    from ._jit import _jit_histogram_2d
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def cake(image, mask, tth_lut, eta_lut,
         tth_min, tth_max, tth_bin_size,
         eta_min=-180.0, eta_max=180.0, eta_bin_size=1.0):
    """Build a caked (2θ–eta) image from a detector frame. [unibin]

    Parameters
    ----------
    image        : 2-D array (nrows, ncols), raw detector counts
    mask         : 2-D array, 1 = bad pixel, 0 = good
    tth_lut      : 2-D array (nrows, ncols), 2θ in degrees from build_lut()
    eta_lut      : 2-D array (nrows, ncols), eta in degrees from build_lut()
    tth_min      : lower 2θ edge (degrees)
    tth_max      : upper 2θ edge (degrees)
    tth_bin_size : 2θ bin width (degrees)
    eta_min      : lower eta edge (degrees, default −180)
    eta_max      : upper eta edge (degrees, default +180)
    eta_bin_size : eta bin width (degrees, default 1)

    Returns
    -------
    cake_img    : 2-D array (n_tth, n_eta), mean intensity; NaN where no data
    tth_centres : 1-D array (n_tth,), bin centres in degrees
    eta_centres : 1-D array (n_eta,), bin centres in degrees
    cnt_map     : 2-D int32 array (n_tth, n_eta), pixel count per cell
    """

    n_tth = int(np.ceil((tth_max - tth_min) / tth_bin_size))
    n_eta = int(np.ceil((eta_max - eta_min) / eta_bin_size))
    if n_tth < 1 or n_eta < 1:
        raise ValueError("Bin size too large for the requested range.")

    tth_edges = np.linspace(tth_min, tth_max, n_tth + 1)

    if _HAS_NUMBA:
        I_sum, px_cnt = _jit_histogram_2d(
            image.astype(np.float64), mask,
            tth_lut, eta_lut, tth_edges,
            eta_min, eta_bin_size, n_eta)
        n_total = n_tth * n_eta
        with np.errstate(invalid="ignore"):
            cake_flat = np.where(px_cnt > 0, I_sum / px_cnt, np.nan)
        px_cnt_flat = px_cnt
    else:
        good = (mask == 0)
        tth_flat = tth_lut[good]
        eta_flat = eta_lut[good]
        I_flat   = image[good].astype(np.float64)

        in_range = (
            (tth_flat >= tth_min) & (tth_flat < tth_max) &
            (eta_flat >= eta_min) & (eta_flat < eta_max)
        )
        tth_flat = tth_flat[in_range]
        eta_flat = eta_flat[in_range]
        I_flat   = I_flat[in_range]

        i_tth = np.floor((tth_flat - tth_min) / tth_bin_size).astype(int)
        i_eta = np.floor((eta_flat - eta_min) / eta_bin_size).astype(int)
        i_tth = np.clip(i_tth, 0, n_tth - 1)
        i_eta = np.clip(i_eta, 0, n_eta - 1)

        flat_idx    = i_tth * n_eta + i_eta
        n_total     = n_tth * n_eta
        I_sum       = np.bincount(flat_idx, weights=I_flat, minlength=n_total)
        px_cnt_flat = np.bincount(flat_idx,                 minlength=n_total).astype(float)
        with np.errstate(invalid="ignore"):
            cake_flat = np.where(px_cnt_flat > 0, I_sum / px_cnt_flat, np.nan)

    cake_img    = cake_flat.reshape(n_tth, n_eta)
    px_cnt_map  = px_cnt_flat.reshape(n_tth, n_eta).astype(np.int32)

    eta_edges   = np.linspace(eta_min, eta_max, n_eta + 1)
    tth_centres = 0.5 * (tth_edges[:-1] + tth_edges[1:])
    eta_centres = 0.5 * (eta_edges[:-1] + eta_edges[1:])

    return cake_img, tth_centres, eta_centres, px_cnt_map


# ── Caked image with variable bins ────────────────────────────────────────

def cake_varbin(image, mask, tth_lut, eta_lut,
                tth_min, tth_max, px, lsd, dR=1.0,
                tth_edges=None,
                eta_min=-180.0, eta_max=180.0,
                eta_bin_size=None):
    """Build a caked (2th-eta) image with pixel-matched variable bins. [varbin]

    2th bins are matched to pixel angular resolution via uniform R-space
    binning (see varbin_tth_edges).  Eta bin size is set by the pixel
    azimuthal footprint at tth_min:

        delta_eta [rad] = px / (Lsd * tan(tth_min))

    This is the coarsest footprint in the range, so no bin is ever finer
    than one pixel anywhere.  Supplying eta_bin_size overrides the
    auto-computed value; the caller is then responsible for ensuring it
    does not produce sub-pixel bins at tth_min.

    Parameters
    ----------
    image        : 2-D array (nrows, ncols), raw detector counts
    mask         : 2-D array, 1 = bad pixel, 0 = good
    tth_lut      : 2-D array (nrows, ncols), 2th in degrees from build_lut()
    eta_lut      : 2-D array (nrows, ncols), eta in degrees from build_lut()
    tth_min      : lower 2th edge (degrees)
    tth_max      : upper 2th edge (degrees)
    px           : pixel size (um)
    lsd          : sample-to-detector distance (um)
    dR           : R-space bin width in pixels (default 1.0 = pixel-matched)
    tth_edges    : 1-D array, pre-computed 2th bin edges (degrees); if provided,
                   px/lsd/dR are ignored for edge generation
    eta_min      : lower eta edge (degrees, default -180)
    eta_max      : upper eta edge (degrees, default +180)
    eta_bin_size : eta bin width (degrees).  If None (default), auto-computed
                   from tth_min as px/(Lsd*tan(tth_min)) in radians, converted
                   to degrees.  Typical default result: ~1 deg at Lsd ~350 mm
                   with 172 um pixels (Pilatus) when tth_min ~1.6 deg.

    Returns
    -------
    cake_img     : 2-D array (n_tth, n_eta), mean intensity; NaN where no data
    tth_centres  : 1-D array (n_tth,), bin centres in degrees (non-uniform)
    eta_centres  : 1-D array (n_eta,), bin centres in degrees (uniform)
    eta_bin_size : float, eta bin width used (degrees)
    px_cnt_map   : 2-D int32 array (n_tth, n_eta), pixel count per cell
    """

    # 2th bin edges (variable, pixel-matched)
    if tth_edges is None:
        from .geometry import varbin_tth_edges
        tth_edges = varbin_tth_edges(tth_min, tth_max, px, lsd, dR)

    n_tth = len(tth_edges) - 1

    # Eta bin size: pixel azimuthal footprint at tth_min (coarsest in range)
    if eta_bin_size is None:
        eta_bin_size = np.degrees(px / (lsd * np.tan(np.radians(tth_min))))
    n_eta = int(np.ceil((eta_max - eta_min) / eta_bin_size))

    if n_tth < 1 or n_eta < 1:
        raise ValueError("Range too small for the computed bin sizes.")

    if _HAS_NUMBA:
        I_sum, px_cnt = _jit_histogram_2d(
            image.astype(np.float64), mask,
            tth_lut, eta_lut, tth_edges,
            eta_min, eta_bin_size, n_eta)
        n_total = n_tth * n_eta
        with np.errstate(invalid="ignore"):
            cake_flat = np.where(px_cnt > 0, I_sum / px_cnt, np.nan)
        px_cnt_flat = px_cnt
    else:
        good = (mask == 0)
        tth_flat = tth_lut[good]
        eta_flat = eta_lut[good]
        I_flat   = image[good].astype(np.float64)

        in_range = (
            (tth_flat >= tth_edges[0]) & (tth_flat < tth_edges[-1]) &
            (eta_flat >= eta_min) & (eta_flat < eta_max)
        )
        tth_flat = tth_flat[in_range]
        eta_flat = eta_flat[in_range]
        I_flat   = I_flat[in_range]

        i_tth = np.searchsorted(tth_edges, tth_flat, side='right') - 1
        i_eta = np.floor((eta_flat - eta_min) / eta_bin_size).astype(int)
        i_tth = np.clip(i_tth, 0, n_tth - 1)
        i_eta = np.clip(i_eta, 0, n_eta - 1)

        flat_idx    = i_tth * n_eta + i_eta
        n_total     = n_tth * n_eta
        I_sum       = np.bincount(flat_idx, weights=I_flat, minlength=n_total)
        px_cnt_flat = np.bincount(flat_idx, minlength=n_total).astype(float)
        with np.errstate(invalid="ignore"):
            cake_flat = np.where(px_cnt_flat > 0, I_sum / px_cnt_flat, np.nan)

    cake_img   = cake_flat.reshape(n_tth, n_eta)
    px_cnt_map = px_cnt_flat.reshape(n_tth, n_eta).astype(np.int32)

    tth_centres = 0.5 * (tth_edges[:-1] + tth_edges[1:])
    eta_edges   = np.linspace(eta_min, eta_max, n_eta + 1)
    eta_centres = 0.5 * (eta_edges[:-1] + eta_edges[1:])

    return cake_img, tth_centres, eta_centres, eta_bin_size, px_cnt_map
