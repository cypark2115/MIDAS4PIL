"""
Phase 2 — Integration engine.

Histogram-based (Option A) 1D and azimuthal-wedge integration using
a precomputed (2θ, eta) lookup table.  Per-bin statistic is the mean;
Poisson standard error of the mean (σ = √ΣI / N) is returned alongside
each lineout.

Note: median-per-bin was removed because Poisson SEM (σ = √ΣI / N) is
only valid for the mean of counts, not the median.  Hot pixels that slip
through the mask should be handled by mask editing.

SNIP background algorithm ported from MIDAS utils/extract_lineouts.py.
Reference: Morháč et al., NIM A 401 (1997) 113–132.
"""

import numpy as np

try:
    from ._jit import (_jit_histogram_1d, _jit_snip,
                       _jit_precompute_bins_1d, _jit_precompute_bins_2d,
                       _jit_accumulate)
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ── SNIP background ────────────────────────────────────────────────────────

def snip_background(intensities, n_iter=50):
    """SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping).

    Parameters
    ----------
    intensities : 1-D array of raw intensities
    n_iter      : clipping iterations (40–50 typical)

    Returns
    -------
    background : 1-D array, same shape as intensities
    """
    if _HAS_NUMBA:
        return _jit_snip(np.asarray(intensities, dtype=np.float64), n_iter)

    y = np.asarray(intensities, dtype=np.float64).copy()
    n = len(y)

    # LLS transform: sqrt → log → log; clamp negatives first
    y = np.clip(y, 0, None)
    y = np.log(np.log(np.sqrt(y + 1) + 1) + 1)

    # Iterative clipping with decreasing window
    for m in range(n_iter, 0, -1):
        avg = (y[:-2*m] + y[2*m:]) / 2.0
        y[m:n-m] = np.minimum(y[m:n-m], avg)

    # Inverse LLS
    bg = (np.exp(np.exp(y) - 1) - 1) ** 2 - 1
    return np.maximum(bg, 0.0)


def _snip_safe(I, n_iter):
    """Run SNIP on *I*, interpolating NaN/empty bins first.

    When a mask removes entire detector panels, some 2θ bins may contain
    no pixels (NaN).  Replacing NaN with 0 (np.nan_to_num) creates
    spike-free dips that SNIP clips, producing an artifically low
    background.  Linear interpolation across the gap gives a physically
    reasonable baseline instead.  I_sub is still NaN at empty bins.
    """
    valid = np.isfinite(I)
    if not valid.any():
        return np.zeros_like(I)
    I_snip = I.copy()
    if not valid.all():
        xs = np.arange(len(I))
        I_snip[~valid] = np.interp(xs[~valid], xs[valid], I[valid])
    return snip_background(I_snip, n_iter=n_iter)


# ── Rebinning ──────────────────────────────────────────────────────────────

def rebin_lineout(tth, I, px_cnt, factor):
    """Rebin a lineout by grouping *factor* consecutive bins.

    Exact Poisson error propagation: groups adjacent I_sum and px_cnt values,
    then recomputes the mean and standard error of the mean.  No new
    physics is created — only existing pixels are re-grouped.

    Parameters
    ----------
    tth    : 1-D array, bin centres (degrees)
    I      : 1-D array, mean intensity per bin (NaN for empty bins)
    px_cnt : 1-D array, active pixel count per bin (float)
    factor : int >= 1, number of consecutive bins to merge

    Returns
    -------
    tth_r    : 1-D array, rebinned bin centres (px_cnt-weighted)
    I_r      : 1-D array, rebinned mean intensity
    px_cnt_r : 1-D array, rebinned active pixel count
    sigma_r  : 1-D array, Poisson s.e.m. = sqrt(I_sum) / px_cnt_r
    """
    factor = int(factor)
    with np.errstate(invalid='ignore'):
        sigma = np.where(px_cnt > 0,
                         np.sqrt(np.maximum(I * px_cnt, 0.0)) / px_cnt, np.nan)
    if factor <= 1:
        return tth.copy(), I.copy(), px_cnt.copy(), sigma

    n = len(tth)
    n_out = n // factor
    if n_out < 1:
        raise ValueError(
            f"Rebin factor {factor} exceeds number of bins {n}")

    sl = slice(None, n_out * factor)
    # Treat empty (NaN) bins as zero contribution
    good    = ~np.isnan(I[sl])
    px_cnt_ = np.where(good, px_cnt[sl], 0.0).reshape(n_out, factor)
    Isum_   = np.where(good, I[sl] * px_cnt[sl], 0.0).reshape(n_out, factor)
    tth_    = tth[sl].reshape(n_out, factor)

    px_cnt_r = px_cnt_.sum(axis=1)
    Isum_r   = Isum_.sum(axis=1)

    with np.errstate(invalid='ignore'):
        I_r     = np.where(px_cnt_r > 0, Isum_r / px_cnt_r, np.nan)
        sigma_r = np.where(px_cnt_r > 0,
                           np.sqrt(np.maximum(Isum_r, 0.0)) / px_cnt_r, np.nan)
        tth_r   = np.where(px_cnt_r > 0,
                           (tth_ * px_cnt_).sum(axis=1) / px_cnt_r,
                           tth_.mean(axis=1))

    return tth_r, I_r, px_cnt_r, sigma_r


# ── 1-D integration ────────────────────────────────────────────────────────

def integrate_1d(image, mask, tth_lut,
                 tth_min, tth_max, tth_bin_size,
                 eta_lut=None, eta_min=-180.0, eta_max=180.0,
                 snip_iter=50):
    """Histogram integration to a 1-D powder pattern. [unibin]

    Each unmasked pixel votes into the 2θ bin that contains its LUT value.
    An optional eta wedge [eta_min, eta_max] restricts which pixels
    contribute (useful for azimuthal sector selection).

    Parameters
    ----------
    image        : 2-D array (nrows, ncols), raw detector counts
    mask         : 2-D array, 1 = bad pixel, 0 = good
    tth_lut      : 2-D array (nrows, ncols), 2θ in degrees from build_lut()
    tth_min      : lower 2θ limit (degrees)
    tth_max      : upper 2θ limit (degrees)
    tth_bin_size : bin width (degrees)
    eta_lut      : 2-D array (nrows, ncols), eta in degrees (optional)
    eta_min      : lower eta limit for wedge selection (degrees, default −180)
    eta_max      : upper eta limit for wedge selection (degrees, default +180)
    snip_iter    : SNIP iterations (0 to skip background subtraction)

    Returns
    -------
    tth      : 1-D array, bin centres (degrees)
    I        : 1-D array, mean intensity per bin
    bg       : 1-D array, SNIP background
    I_sub    : 1-D array, background-subtracted intensity
    sigma    : 1-D array, 1-σ Poisson standard error of the mean per bin
    px_cnt   : 1-D array, active pixel count per bin (float; 0 for empty bins)
    """
    n_bins = int(np.ceil((tth_max - tth_min) / tth_bin_size))
    if n_bins < 1:
        raise ValueError(f"tth range [{tth_min}, {tth_max}] with bin size "
                         f"{tth_bin_size} gives no bins")

    tth_edges = np.linspace(tth_min, tth_max, n_bins + 1)

    if _HAS_NUMBA:
        _eta = eta_lut if eta_lut is not None else tth_lut
        _eta_lo = eta_min if eta_lut is not None else -180.0
        _eta_hi = eta_max if eta_lut is not None else 180.0
        I_sum, px_cnt = _jit_histogram_1d(
            image.astype(np.float64), mask,
            tth_lut, tth_edges, _eta, _eta_lo, _eta_hi)
        with np.errstate(invalid="ignore"):
            I_result = np.where(px_cnt > 0, I_sum / px_cnt, np.nan)
            sigma    = np.where(px_cnt > 0,
                                np.sqrt(np.maximum(I_sum, 0.0)) / px_cnt, np.nan)
    else:
        good = (mask == 0)
        if eta_lut is not None:
            good = good & (eta_lut >= eta_min) & (eta_lut <= eta_max)

        tth_flat = tth_lut[good]
        I_flat   = image[good].astype(np.float64)

        in_range = (tth_flat >= tth_min) & (tth_flat < tth_max)
        tth_flat = tth_flat[in_range]
        I_flat   = I_flat[in_range]

        bin_idx = np.floor((tth_flat - tth_min) / tth_bin_size).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        I_sum  = np.bincount(bin_idx, weights=I_flat, minlength=n_bins)
        px_cnt = np.bincount(bin_idx,                 minlength=n_bins).astype(float)
        with np.errstate(invalid="ignore"):
            I_result = np.where(px_cnt > 0, I_sum / px_cnt, np.nan)
            sigma    = np.where(px_cnt > 0,
                                np.sqrt(np.maximum(I_sum, 0.0)) / px_cnt, np.nan)

    tth_centres = 0.5 * (tth_edges[:-1] + tth_edges[1:])

    if snip_iter > 0:
        bg    = _snip_safe(I_result, snip_iter)
        I_sub = I_result - bg
    else:
        bg    = np.zeros(n_bins)
        I_sub = I_result.copy()

    return tth_centres, I_result, bg, I_sub, sigma, px_cnt


# ── 1-D integration with variable bins ────────────────────────────────────

def integrate_1d_varbin(image, mask, tth_lut,
                        tth_min, tth_max, px, lsd, dR=1.0,
                        tth_edges=None,
                        eta_lut=None, eta_min=-180.0, eta_max=180.0,
                        snip_iter=50):
    """Histogram integration with pixel-matched variable 2θ bins. [varbin]

    Each bin width matches the detector's angular resolution at that 2θ:
        δ(2θ) = dR · px · cos²(2θ) / Lsd

    With dR=1.0 (default), bins are pixel-matched: each bin spans the
    angular width of one pixel.  No artificial oversampling.

    Implemented via uniform R-space binning (equivalent to MIDAS RBinSize).

    Parameters
    ----------
    image     : 2-D array (nrows, ncols), raw detector counts
    mask      : 2-D array, 1 = bad pixel, 0 = good
    tth_lut   : 2-D array (nrows, ncols), 2θ in degrees from build_lut()
    tth_min   : lower 2θ limit (degrees)
    tth_max   : upper 2θ limit (degrees)
    px        : pixel size (µm)
    lsd       : sample-to-detector distance (µm)
    dR        : R-space bin width in pixels (default 1.0 = pixel-matched)
    tth_edges : 1-D array, pre-computed bin edges (degrees); if provided,
                px/lsd/dR are ignored for edge generation
    eta_lut   : 2-D array (nrows, ncols), eta in degrees (optional)
    eta_min   : lower eta limit for wedge selection (degrees, default −180)
    eta_max   : upper eta limit for wedge selection (degrees, default +180)
    snip_iter : SNIP iterations (0 to skip background subtraction)

    Returns
    -------
    tth      : 1-D array, bin centres (degrees) — non-uniformly spaced
    I        : 1-D array, mean intensity per bin
    bg       : 1-D array, SNIP background
    I_sub    : 1-D array, background-subtracted intensity
    sigma    : 1-D array, 1-σ Poisson standard error of the mean per bin
    px_cnt   : 1-D array, active pixel count per bin (float; 0 for empty bins)
    """

    # Build or accept bin edges
    if tth_edges is None:
        from .geometry import varbin_tth_edges
        tth_edges = varbin_tth_edges(tth_min, tth_max, px, lsd, dR)

    n_bins = len(tth_edges) - 1
    if n_bins < 1:
        raise ValueError(f"tth range [{tth_min}, {tth_max}] gives no bins")

    if _HAS_NUMBA:
        _eta = eta_lut if eta_lut is not None else tth_lut
        _eta_lo = eta_min if eta_lut is not None else -180.0
        _eta_hi = eta_max if eta_lut is not None else 180.0
        I_sum, px_cnt = _jit_histogram_1d(
            image.astype(np.float64), mask,
            tth_lut, tth_edges, _eta, _eta_lo, _eta_hi)
        with np.errstate(invalid="ignore"):
            I_result = np.where(px_cnt > 0, I_sum / px_cnt, np.nan)
            sigma    = np.where(px_cnt > 0,
                                np.sqrt(np.maximum(I_sum, 0.0)) / px_cnt, np.nan)
    else:
        good = (mask == 0)
        if eta_lut is not None:
            good = good & (eta_lut >= eta_min) & (eta_lut <= eta_max)

        tth_flat = tth_lut[good]
        I_flat   = image[good].astype(np.float64)

        in_range = (tth_flat >= tth_edges[0]) & (tth_flat < tth_edges[-1])
        tth_flat = tth_flat[in_range]
        I_flat   = I_flat[in_range]

        bin_idx = np.searchsorted(tth_edges, tth_flat, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        I_sum  = np.bincount(bin_idx, weights=I_flat, minlength=n_bins)
        px_cnt = np.bincount(bin_idx,                 minlength=n_bins).astype(float)
        with np.errstate(invalid="ignore"):
            I_result = np.where(px_cnt > 0, I_sum / px_cnt, np.nan)
            sigma    = np.where(px_cnt > 0,
                                np.sqrt(np.maximum(I_sum, 0.0)) / px_cnt, np.nan)

    tth_centres = 0.5 * (tth_edges[:-1] + tth_edges[1:])

    if snip_iter > 0:
        bg    = _snip_safe(I_result, snip_iter)
        I_sub = I_result - bg
    else:
        bg    = np.zeros(n_bins)
        I_sub = I_result.copy()

    return tth_centres, I_result, bg, I_sub, sigma, px_cnt


# ── Precomputed bin maps for fast per-frame reduction ──────────────────

def precompute_bin_maps(mask, tth_lut, eta_lut, geom,
                        lineout=True, cake_out=True):
    """Precompute bin index maps for fast per-frame reduction.

    Call once when geometry/mask change. The returned dict is passed to
    ``reduce_frame(..., bin_maps=maps)`` to skip per-frame bin assignment.

    Parameters
    ----------
    mask     : 2-D array, 1 = bad, 0 = good
    tth_lut  : 2-D array, 2θ lookup table (degrees)
    eta_lut  : 2-D array, eta lookup table (degrees)
    geom     : dict from make_geometry() / load_params()
    lineout  : bool, precompute 1-D lineout bins (default True)
    cake_out : bool, precompute 2-D cake bins (default True)

    Returns
    -------
    dict with precomputed arrays and metadata for reduce_frame()
    """
    if not _HAS_NUMBA:
        raise RuntimeError("precompute_bin_maps requires numba")

    mode = geom.get('mode', 'varbin')
    maps = {'mode': mode}

    # --- 1D lineout ---
    if lineout:
        if mode == 'varbin':
            from .geometry import varbin_tth_edges
            tth_edges = varbin_tth_edges(geom['tth_min'], geom['tth_max'],
                                         geom['px'], geom['lsd'])
        else:
            n_bins = int(np.ceil((geom['tth_max'] - geom['tth_min'])
                                 / geom['tth_bin_size']))
            tth_edges = np.linspace(geom['tth_min'], geom['tth_max'],
                                    n_bins + 1)

        bin_map_1d = _jit_precompute_bins_1d(
            mask, tth_lut, tth_edges, eta_lut, -180.0, 180.0)

        n_bins = len(tth_edges) - 1
        tth_centres = 0.5 * (tth_edges[:-1] + tth_edges[1:])
        maps['bin_map_1d'] = bin_map_1d
        maps['n_bins_1d'] = n_bins
        maps['tth_centres'] = tth_centres

    # --- 2D cake ---
    if cake_out:
        if mode == 'varbin':
            from .geometry import varbin_tth_edges
            tth_edges_2d = varbin_tth_edges(geom['tth_min'], geom['tth_max'],
                                            geom['px'], geom['lsd'])
            # Use explicit eta_bin_size from geom if set; otherwise auto from tth_min.
            # geom['eta_bin_size'] is written by the GUI when user adjusts the spinbox.
            eta_bin_size = geom.get('eta_bin_size') or np.degrees(
                geom['px'] / (geom['lsd']
                              * np.tan(np.radians(geom['tth_min']))))
        else:
            n_tth = int(np.ceil((geom['tth_max'] - geom['tth_min'])
                                / geom['tth_bin_size']))
            tth_edges_2d = np.linspace(geom['tth_min'], geom['tth_max'],
                                       n_tth + 1)
            eta_bin_size = geom.get('eta_bin_size', 1.0)

        eta_min = geom.get('eta_min', -180.0)
        eta_max = geom.get('eta_max', 180.0)
        n_eta = int(np.ceil((eta_max - eta_min) / eta_bin_size))
        n_tth_2d = len(tth_edges_2d) - 1

        flat_map = _jit_precompute_bins_2d(
            mask, tth_lut, eta_lut, tth_edges_2d,
            eta_min, eta_bin_size, n_eta)

        tth_centres_2d = 0.5 * (tth_edges_2d[:-1] + tth_edges_2d[1:])
        eta_edges = np.linspace(eta_min, eta_max, n_eta + 1)
        eta_centres = 0.5 * (eta_edges[:-1] + eta_edges[1:])

        maps['flat_map_2d'] = flat_map
        maps['n_tth_2d'] = n_tth_2d
        maps['n_eta'] = n_eta
        maps['n_total_2d'] = n_tth_2d * n_eta
        maps['tth_centres_2d'] = tth_centres_2d
        maps['eta_centres'] = eta_centres
        maps['eta_bin_size'] = eta_bin_size

    return maps


# ── One-call reduction facade ───────────────────────────────────────────

def reduce_frame(image, mask, tth_lut, eta_lut, geom,
                 lineout=True, cake_out=True, bin_maps=None,
                 eta_min=-180.0, eta_max=180.0):
    """One-call data reduction: lineout and/or cake from a precomputed LUT.

    Dispatches to varbin or unibin integration/caking based on
    ``geom['mode']`` (default ``'varbin'``).

    Parameters
    ----------
    image    : 2-D array (nrows, ncols), raw detector counts
    mask     : 2-D array, 1 = bad, 0 = good
    tth_lut  : 2-D array, 2θ lookup table (degrees)
    eta_lut  : 2-D array, eta lookup table (degrees)
    geom     : dict from make_geometry() / load_params().
               Optional key: 'eta_bin_size' (degrees) — sets the azimuthal bin
               width for varbin caking.  If absent, auto-computed from tth_min
               as px/(Lsd*tan(tth_min)) in radians.  Typical result: ~1 deg at
               Lsd ~350 mm with 172 um pixels (Pilatus) when tth_min ~1.6 deg.
    lineout  : bool, compute 1-D lineout (default True)
    cake_out : bool, compute caked image (default True)
    bin_maps : dict from precompute_bin_maps(), or None for original path
    eta_min  : lower eta limit for lineout azimuthal sector (degrees, default -180)
    eta_max  : upper eta limit for lineout azimuthal sector (degrees, default +180)

    Returns
    -------
    dict with keys (present only when requested):
        tth, I, bg, I_sub, sigma, px_cnt — lineout arrays
        cake_img, tth_cake, eta_cake     — caked image + axes
        px_cnt_cake                      — pixel count per cake cell (2-D, int32)
        eta_bin_size                     — (varbin only) eta bin width used (degrees)
    """
    # Always exclude below-threshold pixels (Pilatus/Eiger store them as -2)
    mask = np.asarray(mask, dtype=bool) | (image < 0)

    full_eta = (eta_min == -180.0 and eta_max == 180.0)

    # ── Fast path: precomputed bin maps (full eta only) ──
    if bin_maps is not None and _HAS_NUMBA and full_eta:
        img64 = image.astype(np.float64)
        result = {}

        if lineout and 'bin_map_1d' in bin_maps:
            I_sum, px_cnt = _jit_accumulate(img64, bin_maps['bin_map_1d'],
                                            bin_maps['n_bins_1d'])
            with np.errstate(invalid="ignore"):
                I_result = np.where(px_cnt > 0, I_sum / px_cnt, np.nan)
                sigma    = np.where(px_cnt > 0,
                                    np.sqrt(np.maximum(I_sum, 0.0)) / px_cnt, np.nan)
            tth_centres = bin_maps['tth_centres']
            bg = _snip_safe(I_result, n_iter=50)
            I_sub = I_result - bg
            result.update(tth=tth_centres, I=I_result, bg=bg, I_sub=I_sub,
                          sigma=sigma, px_cnt=px_cnt)

        if cake_out and 'flat_map_2d' in bin_maps:
            I_sum, px_cnt_flat = _jit_accumulate(img64, bin_maps['flat_map_2d'],
                                                 bin_maps['n_total_2d'])
            with np.errstate(invalid="ignore"):
                cake_flat = np.where(px_cnt_flat > 0, I_sum / px_cnt_flat, np.nan)
            cake_img    = cake_flat.reshape(bin_maps['n_tth_2d'], bin_maps['n_eta'])
            px_cnt_cake = px_cnt_flat.reshape(
                bin_maps['n_tth_2d'], bin_maps['n_eta']).astype(np.int32)
            result.update(cake_img=cake_img,
                          tth_cake=bin_maps['tth_centres_2d'],
                          eta_cake=bin_maps['eta_centres'],
                          px_cnt_cake=px_cnt_cake)
            if 'eta_bin_size' in bin_maps:
                result['eta_bin_size'] = bin_maps['eta_bin_size']

        return result

    # ── Original path (no precomputed maps, or eta sector) ──
    from .cake import cake as _cake, cake_varbin as _cake_varbin

    mode = geom.get('mode', 'varbin')
    result = {}

    if lineout:
        if mode == 'varbin':
            tth, I, bg, I_sub, sigma, px_cnt = integrate_1d_varbin(
                image, mask, tth_lut,
                geom['tth_min'], geom['tth_max'],
                geom['px'], geom['lsd'],
                eta_lut=eta_lut, eta_min=eta_min, eta_max=eta_max)
        else:
            tth, I, bg, I_sub, sigma, px_cnt = integrate_1d(
                image, mask, tth_lut,
                geom['tth_min'], geom['tth_max'],
                geom['tth_bin_size'],
                eta_lut=eta_lut, eta_min=eta_min, eta_max=eta_max)
        result.update(tth=tth, I=I, bg=bg, I_sub=I_sub, sigma=sigma, px_cnt=px_cnt)

    if cake_out:
        if mode == 'varbin':
            cake_img, tth_c, eta_c, eta_bs, px_cnt_cake = _cake_varbin(
                image, mask, tth_lut, eta_lut,
                geom['tth_min'], geom['tth_max'],
                geom['px'], geom['lsd'],
                eta_bin_size=geom.get('eta_bin_size'))
            result.update(cake_img=cake_img, tth_cake=tth_c,
                          eta_cake=eta_c, eta_bin_size=eta_bs,
                          px_cnt_cake=px_cnt_cake)
        else:
            cake_img, tth_c, eta_c, px_cnt_cake = _cake(
                image, mask, tth_lut, eta_lut,
                geom['tth_min'], geom['tth_max'],
                geom['tth_bin_size'])
            result.update(cake_img=cake_img, tth_cake=tth_c,
                          eta_cake=eta_c, px_cnt_cake=px_cnt_cake)

    return result
