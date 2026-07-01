# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""
Phase 1 — Geometry kernel.

Vectorized Python/numpy port of dg_pixel_to_REta() from
MIDAS FF_HEDM/src/DetectorGeometry.c (UChicago Argonne, LLC).
Credit: Hemant Sharma (MIDAS reference implementation).

Coordinate convention (pyFAI)
------------------------------
- Images are stored as standard numpy arrays: row 0 = top of detector,
  column 0 = left.  No top-bottom flip is applied.
- bc_y : beam-centre column, measured from the LEFT edge   (= Poni2 / px)
- bc_z : beam-centre row,    measured from the TOP  edge   (= Poni1 / px)
  (same as pyFAI/MIDAS after MIDAS's ImTransOpt=2 flip convention)
- Internally, Zc = (bc_z − row) * px so that +Zc points upward (toward
  smaller row indices, i.e. toward the top of the image array).

Azimuthal angle eta
-------------------
- eta = 0° at 3 o'clock (right side of detector)
- Positive direction: counter-clockwise (CCW) when viewed from the sample
  → 12 o'clock (top) = +90°, 6 o'clock (bottom) = −90°, 9 o'clock (left) = ±180°
- Matches Dioptas display convention (top of detector = positive eta)
- Implementation: eta = atan2(XYZ_z, −XYZ_y) after tilt rotation,
  range (−180°, +180°]
- Note: pyFAI calls this angle "chi", but the IUCr/MIDAS convention is "eta".

Tilt angles
-----------
tx, ty, tz are in DEGREES, matching MIDAS params-file convention
(dg_build_tilt_matrix multiplies by π/180 internally).
"""

import numpy as np

_DEG2RAD = np.pi / 180.0
_RAD2DEG = 180.0 / np.pi


# ── Tilt matrix ────────────────────────────────────────────────────────────

def build_tilt_matrix(tx_deg, ty_deg, tz_deg):
    """Return TRs = Rx(tx) @ Ry(ty) @ Rz(tz), angles in degrees.

    Matches dg_build_tilt_matrix() in DetectorGeometry.c.
    """
    tx = tx_deg * _DEG2RAD
    ty = ty_deg * _DEG2RAD
    tz = tz_deg * _DEG2RAD

    Rx = np.array([[1,          0,           0         ],
                   [0,  np.cos(tx), -np.sin(tx)],
                   [0,  np.sin(tx),  np.cos(tx)]])
    Ry = np.array([[ np.cos(ty), 0, np.sin(ty)],
                   [ 0,          1, 0          ],
                   [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0],
                   [np.sin(tz),  np.cos(tz), 0],
                   [0,           0,          1]])
    return Rx @ Ry @ Rz


# ── Core pixel → (R, eta) transform ───────────────────────────────────────

def pixel_to_r_eta(col, Z_phys, bc_y, bc_z, TRs, lsd, rho_d,
                   p0, p1, p2, p3, p4, px,
                   dlsd=0.0, dp2=0.0):
    """Vectorized port of dg_pixel_to_REta().

    Parameters
    ----------
    col    : column index (0 = left edge), scalar or broadcastable array
    Z_phys : array row index (0 = top edge); same as `row` in build_lut
    bc_y   : beam-centre column from left (pixels, = Poni2 / px)
    bc_z   : beam-centre row    from top  (pixels, = Poni1 / px)
    TRs    : 3×3 tilt matrix from build_tilt_matrix()
    lsd    : sample-to-detector distance (µm)
    rho_d  : distortion reference radius (µm)
    p0..p4 : distortion coefficients (p3 in degrees)
    px     : pixel size (µm)
    dlsd   : per-panel ΔLsd (µm), 0 for global geometry
    dp2    : per-panel Δp2,        0 for global geometry

    Returns
    -------
    R_px : distortion-corrected radial distance (pixels)
    eta  : azimuthal angle (degrees), 0° at 3 o'clock, CCW positive
    """
    panel_lsd = lsd + dlsd
    panel_p2  = p2  + dp2

    # Centre and convert to µm
    Yc = (-col   + bc_y) * px   # positive to the left of beam centre
    Zc = (bc_z - Z_phys) * px   # positive above beam centre (row 0 = top)

    # Tilt rotation: rotate the FULL position vector [lsd, Yc, Zc] — not just
    # the pixel offset.  This matches pyFAI's sample-centred geometry where
    # the lsd component mixes into the transverse directions under tilt:
    #   XYZ_y ← TRs[1,0]*lsd + …   carries the lsd·sin(tilt) correction term
    # that would be absent if only [0, Yc, Zc] were rotated.
    XYZ_x = TRs[0, 0] * panel_lsd + TRs[0, 1] * Yc + TRs[0, 2] * Zc
    XYZ_y = TRs[1, 0] * panel_lsd + TRs[1, 1] * Yc + TRs[1, 2] * Zc
    XYZ_z = TRs[2, 0] * panel_lsd + TRs[2, 1] * Yc + TRs[2, 2] * Zc

    Rad = (panel_lsd / XYZ_x) * np.sqrt(XYZ_y**2 + XYZ_z**2)

    # eta: CCW from 3 o'clock (+90° at 12 o'clock), range (−180, +180]
    eta = _RAD2DEG * np.arctan2(XYZ_z, -XYZ_y)

    # Distortion model — EtaT is MIDAS's internal azimuthal variable,
    # NOT the same as our eta.  It converts our convention to MIDAS's
    # distortion model so calibrated p0–p4 coefficients work as-is.
    # EtaT = 180 − eta;  see geometry docstring for derivation.
    RNorm = Rad / rho_d
    EtaT  = 180.0 - eta

    distort = (
        p0         * RNorm**2 * np.cos(2.0 * EtaT * _DEG2RAD)
        + p1       * RNorm**4 * np.cos((4.0 * EtaT + p3) * _DEG2RAD)
        + panel_p2 * RNorm**2
        + p4       * RNorm**6
        + 1.0
    )

    Rt = Rad * distort / px * (lsd / panel_lsd)
    return Rt, eta


# ── R → 2θ conversion ─────────────────────────────────────────────────────

def r_to_tth(R_px, px, lsd):
    """Convert radial distance R (pixels) to 2θ (degrees).

    Flat-detector geometry: tan(2θ) = R·px / Lsd
    → 2θ = atan(R·px / Lsd)
    """
    return _RAD2DEG * np.arctan(R_px * px / lsd)


# ── Pixel angular resolution ─────────────────────────────────────────────

def pixel_resolution(tth_deg, px, lsd):
    """Pixel angular resolution at given 2θ on a flat detector.

    Accounts for oblique incidence and increased sample-to-pixel distance
    at high scattering angles.

    Parameters
    ----------
    tth_deg : scalar or array, 2θ in degrees
    px      : pixel size (µm)
    lsd     : sample-to-detector distance (µm)

    Returns
    -------
    delta_tth : same shape, δ(2θ) in degrees per pixel (radial direction)
    delta_eta : same shape, δ(eta) in degrees per pixel (tangential direction)
    """
    tth_rad = np.radians(tth_deg)
    cos_tth = np.cos(tth_rad)
    tan_tth = np.tan(tth_rad)
    delta_tth = _RAD2DEG * px * cos_tth**2 / lsd
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_eta = np.where(tan_tth > 0,
                             _RAD2DEG * px / (lsd * tan_tth),
                             np.inf)
    return delta_tth, delta_eta


# ── LUT range helper ──────────────────────────────────────────────────────

def lut_tth_range(tth_lut, mask=None, percentile=0.02, margin=0.0,
                  px=None, lsd=None, max_eta_bin_deg=1.0):
    """Compute the 2θ range actually covered by the detector.

    Reads the min and max 2θ from a precomputed LUT so that integration
    limits are set by detector geometry rather than hardcoded defaults.

    In varbin mode the eta bin size is computed at tth_min (coarsest
    azimuthal resolution).  Providing *px* and *lsd* enforces a lower
    bound on tth_min so that the eta bin never exceeds *max_eta_bin_deg*:

        tth_min ≥ arctan(px / (lsd × max_eta_bin_rad))

    Parameters
    ----------
    tth_lut         : 2-D array, 2θ in degrees from build_lut()
    mask            : 2-D array, 1 = bad pixel, 0 = good (optional).
                      Masked pixels are excluded from the range computation.
    percentile      : fraction of pixels to clip at each tail (default 0.02 %).
                      Avoids extreme values from corner / beam-stop-adjacent pixels.
    margin          : additional margin to add at each end (degrees, default 0).
    px              : pixel size (µm); if provided together with *lsd*, the
                      eta-resolution constraint is applied to tth_min.
    lsd             : sample-to-detector distance (µm).
    max_eta_bin_deg : maximum allowed eta bin width (degrees, default 1.0).
                      Only used when *px* and *lsd* are supplied.

    Returns
    -------
    tth_min, tth_max : floats, rounded to 2 decimal places
    """
    tth = np.asarray(tth_lut, dtype=np.float64)
    if mask is not None:
        tth = tth[mask == 0]
    else:
        tth = tth.ravel()

    tth = tth[np.isfinite(tth)]
    if len(tth) == 0:
        return 0.1, 30.0  # fallback

    tth_min = float(np.percentile(tth, percentile))
    tth_max = float(np.percentile(tth, 100.0 - percentile))

    tth_min = max(0.1, tth_min - margin)

    # Enforce eta-resolution constraint for varbin: eta bin at tth_min must
    # not exceed max_eta_bin_deg.  This prevents the cake from becoming
    # coarser than physically meaningful when tth_min is near zero.
    if px is not None and lsd is not None:
        max_eta_rad = np.radians(max_eta_bin_deg)
        tth_min_eta = np.degrees(np.arctan(px / (lsd * max_eta_rad)))
        tth_min = max(tth_min, tth_min_eta)

    tth_min = round(tth_min, 2)
    tth_max = round(tth_max + margin, 2)
    return tth_min, tth_max


# ── Variable-bin 2θ edges ─────────────────────────────────────────────────

def varbin_tth_edges(tth_min, tth_max, px, lsd, dR=1.0):
    """Pixel-matched 2θ bin edges via uniform R-space binning. [varbin]

    Bins uniformly in R-space (detector radial distance in pixels), then
    converts to 2θ.  With dR=1.0, each bin spans exactly one pixel's
    angular width:  δ(2θ) = dR · px · cos²(2θ) / Lsd.

    This is equivalent to MIDAS's RBinSize approach.

    Parameters
    ----------
    tth_min : lower 2θ limit (degrees)
    tth_max : upper 2θ limit (degrees)
    px      : pixel size (µm)
    lsd     : sample-to-detector distance (µm)
    dR      : R-space bin width in pixels (default 1.0 = pixel-matched).
              Use dR < 1 for oversampling, dR > 1 for undersampling.

    Returns
    -------
    tth_edges : 1-D array, bin edges in 2θ (degrees), non-uniformly spaced.
                Length = n_bins + 1.
    """
    R_min = np.tan(np.radians(tth_min)) * lsd / px
    R_max = np.tan(np.radians(tth_max)) * lsd / px

    # Snap to R-space pixel grid: first edge at the next full pixel boundary
    # above R_min, last edge at the pixel boundary at or below R_max.
    R_start = np.ceil(R_min / dR) * dR
    R_stop  = np.floor(R_max / dR) * dR
    R_edges = np.arange(R_start, R_stop + dR * 0.5, dR)

    tth_edges = r_to_tth(R_edges, px, lsd)
    return tth_edges


# ── Full-detector LUT builder ──────────────────────────────────────────────

def build_lut(nrows, ncols, bc_y, bc_z, lsd, px,
              tx_deg, ty_deg, tz_deg,
              p0, p1, p2, p3, p4, rho_d):
    """Build (2θ, eta) lookup tables for the entire detector.

    Parameters
    ----------
    nrows, ncols           : detector size (NrPixelsZ, NrPixelsY in MIDAS)
    bc_y                   : beam-centre column from left (= Poni2 / px)
    bc_z                   : beam-centre row    from top  (= Poni1 / px)
    lsd                    : sample-to-detector distance (µm)
    px                     : pixel size (µm)
    tx_deg, ty_deg, tz_deg : tilt angles (degrees)
    p0..p4                 : distortion coefficients (p3 in degrees)
    rho_d                  : distortion reference radius (µm)

    Returns
    -------
    tth : shape (nrows, ncols), 2θ in degrees
    eta : shape (nrows, ncols), eta in degrees, 0° at 3 o'clock, CCW positive
    """
    col    = np.arange(ncols, dtype=np.float64)           # shape (ncols,)
    Z_phys = np.arange(nrows, dtype=np.float64)[:, None]  # row index from top; shape (nrows, 1)

    TRs = build_tilt_matrix(tx_deg, ty_deg, tz_deg)
    R, eta = pixel_to_r_eta(col, Z_phys, bc_y, bc_z, TRs, lsd, rho_d,
                             p0, p1, p2, p3, p4, px)
    tth = r_to_tth(R, px, lsd)
    return tth, eta


# ── Beam center finders ───────────────────────────────────────────────────

def find_beam_center(image, mask, rings, lsd, px,
                     bc_y_init=None, bc_z_init=None,
                     search_range=100, tth_window_deg=0.5):
    """Find beam center and Lsd by minimizing ring-position residuals.

    For each trial (bc_y, bc_z, Lsd), converts pixel distances to 2θ using
    the flat-detector approximation and measures how well intensity peaks
    align with the expected ring positions from the calibrant.  Works with
    partial rings and as few as one ring.

    Lsd is a free variable in the fine-refinement stage.  Fixing Lsd while
    searching only for (bc_y, bc_z) couples the two parameters: a wrong Lsd
    produces a systematically biased optimal center.  Fitting all three
    simultaneously constrains each ring to land at its absolute 2θ position,
    which is much more robust with weak or partial arcs.

    Parameters
    ----------
    image          : 2-D array, detector image
    mask           : 2-D bool array, True = bad pixel
    rings          : list of dicts with 'tth' key (degrees), from load_calibrant()
    lsd            : initial sample-to-detector distance (µm)
    px             : pixel size (µm)
    bc_y_init      : initial beam-center column (pixels); default ncols/2
    bc_z_init      : initial beam-center row (pixels); default nrows/2
    search_range   : coarse search half-width in pixels (default 100)
    tth_window_deg : half-width of 2θ window around each ring for peak
                     centroid search (default 0.5°); auto-reduced to half
                     the minimum ring spacing if rings are closely packed

    Returns
    -------
    bc_y, bc_z : refined beam center in pixels
    lsd_ref    : refined sample-to-detector distance in µm
    """
    from scipy.optimize import minimize as _minimize

    if not rings:
        raise ValueError("No rings provided — load a calibrant and click "
                         "'Show Rings' before using Find Center.")

    nrows, ncols = image.shape
    if bc_y_init is None:
        bc_y_init = ncols / 2.0
    if bc_z_init is None:
        bc_z_init = nrows / 2.0

    ring_tths = np.array(sorted(r['tth'] for r in rings))

    # Fine window: auto-reduced so adjacent rings never overlap in the centroid
    # measurement used by the Nelder-Mead refinement step.
    tth_window_fine = tth_window_deg
    if len(ring_tths) > 1:
        min_gap = np.min(np.diff(ring_tths))
        tth_window_fine = min(tth_window_fine, min_gap * 0.4)

    # Coarse window: never narrower than 0.3° regardless of ring spacing.
    # The coarse grid step is max(4, search_range // 12) pixels.  At
    # lsd=350 mm and px=172 µm a 0.3° window spans ~27 px, so a step of
    # ≤16 px guarantees at least one grid point overlaps every ring.
    # Without this floor a densely-packed CeO2 pattern can produce a window
    # of ~0.04° (~3.6 px) — narrower than the step — so the coarse search
    # finds no rings anywhere and leaves bc_y/bc_z unchanged.
    tth_window_coarse = max(tth_window_fine, 0.3)

    # Precompute valid pixel coordinates (downsampled for coarse search)
    valid = ~mask & np.isfinite(image) & (image > 0)
    rows_all, cols_all = np.where(valid)
    I_all = image[valid].astype(np.float64)

    ds = 4
    rows_c = rows_all[::ds].astype(np.float64)
    cols_c = cols_all[::ds].astype(np.float64)
    I_c    = I_all[::ds]

    rows_f = rows_all.astype(np.float64)
    cols_f = cols_all.astype(np.float64)

    def _score(params, rows_v, cols_v, I_v, window):
        bc_y, bc_z, lsd_v = params
        if lsd_v <= 0:
            return 1e9
        R   = np.sqrt((cols_v - bc_y)**2 + (rows_v - bc_z)**2)
        tth = np.degrees(np.arctan2(R * px, lsd_v))
        total = 0.0
        n_found = 0
        for expected in ring_tths:
            lo, hi = expected - window, expected + window
            sel = (tth >= lo) & (tth <= hi)
            if sel.sum() < 5:
                continue
            w = I_v[sel]
            w_sum = w.sum()
            if w_sum <= 0:
                continue
            observed = np.dot(tth[sel], w) / w_sum   # intensity-weighted centroid
            total += (observed - expected) ** 2
            n_found += 1
        return total / n_found if n_found > 0 else 1e9

    # Coarse grid search on downsampled pixels: (bc_y, bc_z) only, lsd fixed.
    # Lsd is freed in the subsequent Nelder-Mead refinement.
    step = max(4, search_range // 12)
    y_lo = max(0.0,       bc_y_init - search_range)
    y_hi = min(float(ncols), bc_y_init + search_range)
    z_lo = max(0.0,       bc_z_init - search_range)
    z_hi = min(float(nrows), bc_z_init + search_range)

    best_score = np.inf
    best = np.array([bc_y_init, bc_z_init])

    for by in np.arange(y_lo, y_hi + 1, step):
        for bz in np.arange(z_lo, z_hi + 1, step):
            s = _score([by, bz, lsd], rows_c, cols_c, I_c, tth_window_coarse)
            if s < best_score:
                best_score = s
                best = np.array([by, bz])

    # Fine refinement over (bc_y, bc_z, lsd_px) jointly on the same 4×
    # downsampled pixel set used for the coarse search.
    # Lsd is normalised to pixels (lsd_px = lsd / px) so all three parameters
    # share the same scale and xatol=0.5 makes sense for all of them.
    # Full-resolution pixels (~600 k for Pilatus 2M) with maxiter=1000 takes
    # ~50 s; the 4× downsampled set (~150 k) finishes in < 3 s with no loss
    # of meaningful accuracy (the coarse grid already pins the answer to
    # within a few pixels, and ring-based calibration later refines further).
    lsd_px0 = lsd / px

    def _score_px(params):
        bc_y_v, bc_z_v, lsd_px_v = params
        return _score([bc_y_v, bc_z_v, lsd_px_v * px], rows_c, cols_c, I_c, tth_window_fine)

    x0 = np.array([best[0], best[1], lsd_px0])
    result = _minimize(
        _score_px, x0, method='Nelder-Mead',
        options={'xatol': 0.3, 'fatol': 1e-10, 'maxiter': 2000,
                 'adaptive': True})

    return float(result.x[0]), float(result.x[1]), float(result.x[2]) * px


def find_beam_center_auto(image, mask, bc_y_init=None, bc_z_init=None,
                          search_range=100, downsample=8):
    """Estimate beam center from image sharpness (no calibrant required).

    Finds the point that maximizes the variance of the azimuthally-averaged
    radial intensity profile.  Works for any powder pattern without knowing
    the calibrant or wavelength, but is unreliable for partial rings or
    when fewer than ~3 rings are visible.

    Parameters
    ----------
    image        : 2-D array, detector image
    mask         : 2-D bool array, True = bad pixel
    bc_y_init    : initial guess for beam-center column (pixels from left).
                   Default: detector center (ncols / 2).
    bc_z_init    : initial guess for beam-center row (pixels from top).
                   Default: detector center (nrows / 2).
    search_range : half-width of search window in pixels (default 100).
    downsample   : downsampling factor for the coarse search (default 8).

    Returns
    -------
    bc_y, bc_z : refined beam center in pixels (full-resolution coords).
    """
    from scipy.optimize import minimize as _minimize

    nrows, ncols = image.shape
    if bc_y_init is None:
        bc_y_init = ncols / 2.0
    if bc_z_init is None:
        bc_z_init = nrows / 2.0

    # --- Step 1: coarse grid search on downsampled image ---
    ds = max(1, int(downsample))
    img_ds = image[::ds, ::ds].astype(np.float64)
    mask_ds = mask[::ds, ::ds]

    nr_ds, nc_ds = img_ds.shape
    valid_ds = ~mask_ds
    if valid_ds.sum() < 100:
        return bc_y_init, bc_z_init

    cols_ds = np.arange(nc_ds, dtype=np.float64)
    rows_ds = np.arange(nr_ds, dtype=np.float64)
    cc_ds, rr_ds = np.meshgrid(cols_ds, rows_ds)
    c_flat = cc_ds[valid_ds]
    r_flat = rr_ds[valid_ds]
    I_flat = img_ds[valid_ds]

    def _sharpness_ds(bc_y_ds, bc_z_ds):
        R = np.sqrt((c_flat - bc_y_ds)**2 + (r_flat - bc_z_ds)**2)
        R_max = R.max()
        if R_max < 10:
            return 0.0
        n_bins = max(int(R_max / 2), 10)
        bins = np.linspace(0, R_max, n_bins + 1)
        hist_I, _ = np.histogram(R, bins, weights=I_flat)
        hist_N, _ = np.histogram(R, bins)
        good = hist_N > 5
        if good.sum() < 10:
            return 0.0
        profile = hist_I[good] / hist_N[good]
        return np.var(profile)

    # Search grid in downsampled coordinates
    sr_ds = max(1, int(search_range / ds))
    bc_y_ds_init = bc_y_init / ds
    bc_z_ds_init = bc_z_init / ds

    y_lo = max(0, int(bc_y_ds_init - sr_ds))
    y_hi = min(nc_ds - 1, int(bc_y_ds_init + sr_ds))
    z_lo = max(0, int(bc_z_ds_init - sr_ds))
    z_hi = min(nr_ds - 1, int(bc_z_ds_init + sr_ds))

    best_score = -1.0
    best_y, best_z = bc_y_ds_init, bc_z_ds_init

    for by in range(y_lo, y_hi + 1):
        for bz in range(z_lo, z_hi + 1):
            s = _sharpness_ds(float(by), float(bz))
            if s > best_score:
                best_score = s
                best_y, best_z = float(by), float(bz)

    # Convert back to full-resolution coordinates
    bc_y_coarse = best_y * ds
    bc_z_coarse = best_z * ds

    # --- Step 2: refine on 4× downsampled image ---
    # Full-resolution optimization on a Pilatus 2M (~2.5 M pixels) with
    # Nelder-Mead would take ~50 s per run.  A 4× downsample keeps
    # ≈625 K pixels, runs in ≤5 s, and gives well under 2 px accuracy —
    # more than enough for the subsequent ring-position optimizer.
    ds2 = 4
    img2   = image[::ds2, ::ds2].astype(np.float64)
    mask2  = mask[::ds2, ::ds2]
    valid2 = ~mask2
    if valid2.sum() < 50:
        return bc_y_coarse, bc_z_coarse

    # Pixel coordinates in *full-resolution* units so the returned bc values
    # are directly in full-resolution pixels.
    r2_idx, c2_idx = np.where(valid2)
    c2_f = c2_idx.astype(np.float64) * ds2
    r2_f = r2_idx.astype(np.float64) * ds2
    I2   = img2[valid2]

    def _cost2(params):
        bc_y, bc_z = params
        R = np.sqrt((c2_f - bc_y)**2 + (r2_f - bc_z)**2)
        R_max = R.max()
        if R_max < 10:
            return 0.0
        n_bins = max(int(R_max / 4), 20)
        bins = np.linspace(0, R_max, n_bins + 1)
        hist_I, _ = np.histogram(R, bins, weights=I2)
        hist_N, _ = np.histogram(R, bins)
        good = hist_N > 5
        if good.sum() < 10:
            return 0.0
        profile = hist_I[good] / hist_N[good]
        return -np.var(profile)

    # fatol must be scale-compatible with –variance (O(counts²)).  1e-10 is
    # far too tight and would always exhaust maxiter with no real convergence.
    result = _minimize(_cost2, [bc_y_coarse, bc_z_coarse],
                       method='Nelder-Mead',
                       options={'xatol': 1.0, 'fatol': 1.0,
                                'maxiter': 200, 'adaptive': True})

    return float(result.x[0]), float(result.x[1])


def find_beam_center_sharpness(image, mask, px=172.0,
                               bc_y_init=None, bc_z_init=None,
                               search_range=200, coarse_step=20,
                               bin_width=2, downsample=4,
                               intensity_percentile=80):
    """Find beam center by maximizing azimuthal-mean radial profile sharpness.

    For the correct beam center, full diffraction rings collapse to sharp peaks
    in the radially-averaged I(R) profile.  The metric is the variance of I_bar(R):
    it is high when rings are tight (correct center) and low when they are smeared
    (wrong center).  No calibrant list or Lsd estimate is required.

    Algorithm
    ---------
    1. Threshold pixels below *intensity_percentile* to suppress background and
       focus the metric on diffraction signal.  This is critical for sparse
       (spotty) patterns from coarse-grained calibrants.
    2. Downsample the image for speed; precompute pixel coordinates once.
    3. Coarse grid search ±search_range around *bc_y_init* / *bc_z_init*
       (default: detector centre), step *coarse_step*.
    4. Nelder-Mead refinement starting from the best coarse point.

    Parameters
    ----------
    image                : 2-D float array
    mask                 : 2-D bool array (True = bad)
    px                   : pixel size in µm (not used internally, kept for API consistency)
    bc_y_init            : initial-guess beam-centre column (pixels from left).
                           Default: detector centre (ncols / 2).
    bc_z_init            : initial-guess beam-centre row (pixels from top).
                           Default: detector centre (nrows / 2).
    search_range         : half-side of the coarse search region in pixels
    coarse_step          : grid spacing for the coarse search in pixels
    bin_width            : radial bin width in pixels (default 2)
    downsample           : spatial downsampling factor applied before the search
    intensity_percentile : keep only pixels above this percentile of valid-pixel
                           intensities (default 80).  Lower values use more
                           background pixels; higher values focus on ring peaks.
                           Pass 0 to disable thresholding.

    Returns
    -------
    bc_y : float, beam-centre column (pixels from left)
    bc_z : float, beam-centre row    (pixels from top)
    """
    from scipy.optimize import minimize

    nrows, ncols = image.shape
    c0 = bc_y_init if bc_y_init is not None else ncols / 2.0
    r0 = bc_z_init if bc_z_init is not None else nrows / 2.0

    # Always exclude below-threshold pixels (Pilatus -2 sentinel)
    mask = np.asarray(mask, dtype=bool) | (image < 0)

    ds = max(1, int(downsample))
    img_ds = image[::ds, ::ds].astype(np.float64)
    msk_ds = mask[::ds, ::ds]

    valid = (img_ds > 0) & ~msk_ds & np.isfinite(img_ds)
    rows_v, cols_v = np.where(valid)
    if rows_v.size < 100:
        return c0, r0

    I_v = img_ds[valid]

    # Percentile threshold: suppress background, focus on diffraction peaks.
    # Critical for spotty (coarse-grained) calibrant patterns where background
    # pixels would dominate the azimuthal mean and blur the sharpness metric.
    if intensity_percentile > 0:
        thresh = np.percentile(I_v, intensity_percentile)
        bright = I_v >= thresh
        if bright.sum() >= 50:
            rows_v = rows_v[bright]
            cols_v = cols_v[bright]
            I_v    = I_v[bright]

    cols_f = cols_v.astype(np.float64) * ds   # back to full-res coordinates
    rows_f = rows_v.astype(np.float64) * ds

    # Exclude the innermost 10 px (beam stop / direct beam)
    _R_MIN_PX = 10.0

    def _sharpness(cx, cz):
        R_v   = np.hypot(cols_f - cx, rows_f - cz)
        outer = R_v > _R_MIN_PX
        if outer.sum() < 50:
            return 0.0
        R_use = R_v[outer]
        I_use = I_v[outer]
        idx   = (R_use / bin_width).astype(np.intp)
        n_bins = int(idx.max()) + 1
        I_sum  = np.zeros(n_bins)
        cnt    = np.zeros(n_bins)
        np.add.at(I_sum, idx, I_use)
        np.add.at(cnt,   idx, 1.0)
        with np.errstate(invalid='ignore'):
            I_bar = np.where(cnt > 1, I_sum / cnt, 0.0)
        active = I_bar > 0
        if active.sum() < 5:
            return 0.0
        return float(np.var(I_bar[active]))

    # ── Coarse grid ────────────────────────────────────────────────────────
    cx_grid = np.arange(c0 - search_range, c0 + search_range + 1, coarse_step)
    cz_grid = np.arange(r0 - search_range, r0 + search_range + 1, coarse_step)

    best_s, best_cx, best_cz = -1.0, c0, r0
    for cx in cx_grid:
        for cz in cz_grid:
            s = _sharpness(cx, cz)
            if s > best_s:
                best_s, best_cx, best_cz = s, cx, cz

    # ── Fine Nelder-Mead refinement ────────────────────────────────────────
    def _neg(p):
        return -_sharpness(p[0], p[1])

    res = minimize(_neg, [best_cx, best_cz],
                   method='Nelder-Mead',
                   options={'xatol': 0.3, 'fatol': 1.0,
                            'maxiter': 500, 'adaptive': True})

    bc_y = float(np.clip(res.x[0], 0.0, ncols - 1.0))
    bc_z = float(np.clip(res.x[1], 0.0, nrows - 1.0))
    return bc_y, bc_z


def find_beam_center_ellipse(image, mask, rings, px=172.0, rho_d=217578.0,
                              bc_y_init=None, bc_z_init=None, lsd_init=None,
                              ty_init=0.0, tz_init=0.0,
                              search_range=200, bc_step=20,
                              tilt_range_deg=8.0, tilt_step_deg=2.0,
                              intensity_percentile=99, tth_tol=0.3,
                              p0=0.0, p1=0.0, p2=0.0, p3=0.0, p4=0.0):
    """Find beam center + Lsd + tilts by maximising the ring-score metric.

    Maps each bright pixel to 2θ using the full tilt-aware pixel_to_r_eta()
    model and sums the intensity of pixels within *tth_tol* of any known ring.
    Unlike the flat-detector sharpness metric, this correctly handles rings
    that appear as ellipses on a tilted detector.

    Algorithm
    ---------
    1. Extract the top *intensity_percentile* % of valid pixels (ring spots).
    2. Coarse (ty, tz) tilt grid search ± *tilt_range_deg* around init values
       with BC fixed at *bc_y_init* / *bc_z_init*.
    3. Coarse (bc_y, bc_z) grid search around init BC at the best tilts.
    4. Joint Nelder-Mead refinement over all five parameters
       (bc_y, bc_z, lsd, ty, tz) starting from the best coarse point.

    Parameters
    ----------
    image               : 2-D array, detector image
    mask                : 2-D bool array, True = bad pixel
    rings               : list of dicts with key 'tth' (degrees) — calibrant rings
    px                  : pixel size (µm, default 172.0)
    rho_d               : distortion reference radius (µm, default 217578.0)
    bc_y_init           : initial beam-centre column (px); default ncols/2
    bc_z_init           : initial beam-centre row (px); default nrows/2
    lsd_init            : initial Lsd (µm); default 200000.0
    ty_init, tz_init    : initial tilt angles (degrees); default 0
    search_range        : half-width of (bc_y, bc_z) coarse search (px)
    bc_step             : coarse (bc_y, bc_z) grid step (px)
    tilt_range_deg      : half-width of tilt coarse search (degrees)
    tilt_step_deg       : coarse tilt grid step (degrees)
    intensity_percentile: keep pixels above this percentile (default 99)
    tth_tol             : 2θ tolerance for ring membership (degrees, default 0.3)
    p0..p4              : distortion coefficients (default 0)

    Returns
    -------
    bc_y, bc_z : refined beam centre (pixels)
    lsd        : refined Lsd (µm)
    ty_deg     : refined ty tilt (degrees)
    tz_deg     : refined tz tilt (degrees)
    """
    from scipy.optimize import minimize

    nrows, ncols = image.shape
    if bc_y_init is None:
        bc_y_init = ncols / 2.0
    if bc_z_init is None:
        bc_z_init = nrows / 2.0
    if lsd_init is None:
        lsd_init = 200000.0

    mask = np.asarray(mask, dtype=bool) | (image < 0)

    # Extract bright pixels at full resolution to preserve sparse ring spots.
    valid = (image > 0) & ~mask
    I_all = image[valid]
    if I_all.size < 50:
        return bc_y_init, bc_z_init, lsd_init, ty_init, tz_init

    thresh = np.percentile(I_all, intensity_percentile)
    bright = valid & (image >= thresh)
    rows_b, cols_b = np.where(bright)
    I_b = image[rows_b, cols_b].astype(np.float64)

    if len(I_b) < 20:
        return bc_y_init, bc_z_init, lsd_init, ty_init, tz_init

    rows_f = rows_b.astype(np.float64)
    cols_f = cols_b.astype(np.float64)
    ring_tths = np.array([r['tth'] for r in rings], dtype=np.float64)

    # Coarse tolerance: wide enough so that lsd_init wrong by ~30% still captures
    # ring spots (error in 2θ ≈ 30% × tth_first_ring ≈ 1.5° for 5° ring).
    # Narrow tolerance *tth_tol* is reserved for the lsd refinement scan and the
    # final Nelder-Mead, where BC and tilts have already been corrected.
    tth_tol_coarse = max(tth_tol, 1.5)

    def _ring_score(bc_y, bc_z, lsd, ty_deg, tz_deg, tol):
        if lsd <= 0:
            return 0.0
        TRs = build_tilt_matrix(0.0, ty_deg, tz_deg)
        R_px, _ = pixel_to_r_eta(cols_f, rows_f, bc_y, bc_z, TRs, lsd,
                                   rho_d, p0, p1, p2, p3, p4, px)
        tth = r_to_tth(R_px, px, lsd)
        on_ring = np.zeros(len(tth), dtype=bool)
        for tth_r in ring_tths:
            on_ring |= np.abs(tth - tth_r) < tol
        return float(I_b[on_ring].sum())

    # Evaluate the init geometry with narrow tol first so the coarse stages
    # can never produce a result worse than the user's current starting point.
    s_init_narrow = _ring_score(bc_y_init, bc_z_init, lsd_init,
                                 ty_init, tz_init, tth_tol)

    # ── Stage 1: coarse (ty, tz) tilt search — wide tol handles lsd error ────
    ty_grid = np.arange(ty_init - tilt_range_deg,
                        ty_init + tilt_range_deg + 1e-9, tilt_step_deg)
    tz_grid = np.arange(tz_init - tilt_range_deg,
                        tz_init + tilt_range_deg + 1e-9, tilt_step_deg)

    best_s = -1.0
    best_ty, best_tz = ty_init, tz_init
    for ty in ty_grid:
        for tz in tz_grid:
            s = _ring_score(bc_y_init, bc_z_init, lsd_init, ty, tz,
                            tth_tol_coarse)
            if s > best_s:
                best_s, best_ty, best_tz = s, ty, tz

    # ── Stage 2: coarse (bc_y, bc_z) search at best tilts ────────────────────
    cx_grid = np.arange(bc_y_init - search_range,
                        bc_y_init + search_range + 1, bc_step)
    cz_grid = np.arange(bc_z_init - search_range,
                        bc_z_init + search_range + 1, bc_step)

    best_bc_y, best_bc_z = bc_y_init, bc_z_init
    for cx in cx_grid:
        for cz in cz_grid:
            s = _ring_score(cx, cz, lsd_init, best_ty, best_tz,
                            tth_tol_coarse)
            if s > best_s:
                best_s, best_bc_y, best_bc_z = s, cx, cz

    # ── Stage 2.5: 1-D Lsd scan with narrow tol ──────────────────────────────
    # Always include lsd_init so the init geometry is a valid candidate.
    lsd_grid = np.unique(np.concatenate([
        np.linspace(lsd_init * 0.70, lsd_init * 1.30, 13),
        [lsd_init]]))
    best_lsd   = lsd_init
    best_s_lsd = s_init_narrow   # floor: never go below the init's narrow score
    for lsd_v in lsd_grid:
        s = _ring_score(best_bc_y, best_bc_z, lsd_v, best_ty, best_tz,
                        tth_tol)
        if s > best_s_lsd:
            best_s_lsd, best_lsd = s, lsd_v

    # If coarse+lsd scan found nothing better than init, fall back to init fully
    if best_s_lsd <= s_init_narrow:
        best_bc_y, best_bc_z = bc_y_init, bc_z_init
        best_ty, best_tz     = ty_init, tz_init
        best_lsd             = lsd_init

    # ── Stage 3: joint 5-D Nelder-Mead refinement ────────────────────────────
    # Start from the corrected (BC, tilts, Lsd) seed; use narrow tth_tol.
    lsd_px0 = best_lsd / px

    def _neg5(params):
        bc_y, bc_z, lsd_px, ty, tz = params
        return -_ring_score(bc_y, bc_z, lsd_px * px, ty, tz, tth_tol)

    x0 = [best_bc_y, best_bc_z, lsd_px0, best_ty, best_tz]
    res = minimize(_neg5, x0, method='Nelder-Mead',
                   options={'xatol': 0.3, 'fatol': 1.0,
                            'maxiter': 2000, 'adaptive': True})

    bc_y = float(np.clip(res.x[0], 0.0, ncols - 1.0))
    bc_z = float(np.clip(res.x[1], 0.0, nrows - 1.0))
    lsd  = float(max(res.x[2] * px, 1.0))
    ty   = float(res.x[3])
    tz   = float(res.x[4])
    return bc_y, bc_z, lsd, ty, tz



def lsd_from_first_ring_profile(image, mask, rings, bc_y, bc_z, px=172.0,
                                 lsd_hint=None, tth_max_deg=None):
    """Estimate sample-to-detector distance from the innermost Bragg ring profile.

    Builds the radially-averaged intensity profile I(R) around (bc_y, bc_z),
    finds the innermost peak using prominence-based detection, and returns Lsd
    from the known ring 2theta.  No fitting — the ring identification is the
    only input.

    Parameters
    ----------
    image       : 2-D float array
    mask        : 2-D bool array (True = bad pixel)
    rings       : list of dicts with key 'tth' (degrees), sorted ascending
    bc_y        : beam-centre column (pixels from left)
    bc_z        : beam-centre row    (pixels from top)
    px          : pixel size in µm (default 172.0)
    lsd_hint    : approximate Lsd in µm; used to set the search window
    tth_max_deg : upper search bound in degrees (bounds R_hi)

    Returns
    -------
    lsd : float (µm) if a peak is found, else None
    """
    from scipy.signal import find_peaks

    if not rings:
        return None

    tth_first = rings[0]['tth']
    tth_rad   = np.radians(tth_first)

    # Search window in radius (pixels)
    if lsd_hint is not None and lsd_hint > 0:
        R_expected = (lsd_hint / px) * np.tan(tth_rad)
        R_lo = max(10.0, R_expected * 0.4)
    else:
        R_lo = 10.0

    if tth_max_deg is not None and lsd_hint is not None and lsd_hint > 0:
        R_hi = (lsd_hint / px) * np.tan(np.radians(tth_max_deg)) * 1.3
    elif lsd_hint is not None and lsd_hint > 0:
        R_hi = (lsd_hint / px) * np.tan(tth_rad) * 2.5
    else:
        R_hi = min(image.shape) / 2.0

    # Build radial profile in 1-pixel bins
    nrows, ncols = image.shape
    cols, rows = np.meshgrid(np.arange(ncols, dtype=np.float64),
                             np.arange(nrows, dtype=np.float64))
    R_arr = np.hypot(cols - bc_y, rows - bc_z)

    valid = (~mask) & (image > 0) & np.isfinite(image)
    valid &= (R_arr >= R_lo) & (R_arr <= R_hi)

    if valid.sum() < 50:
        return None

    R_v = R_arr[valid]
    I_v = image[valid].astype(np.float64)

    n_bins = max(int(R_hi - R_lo) + 1, 10)
    counts, bin_edges = np.histogram(R_v, bins=n_bins, range=(R_lo, R_hi))
    I_sum,  _         = np.histogram(R_v, bins=n_bins, range=(R_lo, R_hi),
                                     weights=I_v)
    good = counts > 0
    profile = np.where(good, I_sum / np.where(good, counts, 1), 0.0)

    # Prominence-based peak detection in the valid window
    peaks, props = find_peaks(profile,
                              prominence=profile[good].max() * 0.05 if good.any() else 1.0,
                              width=1)

    if peaks.size == 0:
        return None

    # Use the innermost (first) peak
    pk = int(peaks[0])

    # Parabolic sub-pixel interpolation
    if 0 < pk < len(profile) - 1:
        y0, y1, y2 = profile[pk - 1], profile[pk], profile[pk + 1]
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        sub = (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        sub = float(np.clip(sub, -0.5, 0.5))
    else:
        sub = 0.0

    bin_width = (R_hi - R_lo) / n_bins
    R_peak = R_lo + (pk + 0.5 + sub) * bin_width

    if R_peak <= R_lo:
        return None

    return float(R_peak * px / np.tan(tth_rad))


# ── Circle fit for manual beam-center determination ───────────────────

def fit_circle(points):
    """Least-squares circle fit to N >= 3 points (Kasa method).

    Parameters
    ----------
    points : array-like, shape (N, 2)
        (y, z) coordinates of clicked points on a diffraction ring arc.

    Returns
    -------
    cy     : float, circle centre y (column)
    cz     : float, circle centre z (row)
    radius : float, fitted radius in pixels
    residual : float, mean absolute radial residual (pixels)

    Raises
    ------
    ValueError  if fewer than 3 points provided
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 points for circle fit")

    y = pts[:, 0]
    z = pts[:, 1]

    # Kasa algebraic circle fit: minimise sum (y^2 + z^2 - cy*y - cz*z - r^2)^2
    A = np.column_stack([y, z, np.ones(len(y))])
    b = y**2 + z**2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    cy = result[0] / 2.0
    cz = result[1] / 2.0
    radius = np.sqrt(result[2] + cy**2 + cz**2)

    residual = np.mean(np.abs(np.sqrt((y - cy)**2 + (z - cz)**2) - radius))

    return float(cy), float(cz), float(radius), float(residual)


def lsd_from_ring(radius_px, px, tth_deg):
    """Compute sample-to-detector distance from a ring radius and known 2theta.

    Parameters
    ----------
    radius_px : float, measured ring radius in pixels
    px        : float, pixel size in microns
    tth_deg   : float, known 2theta of the ring in degrees

    Returns
    -------
    lsd : float, sample-to-detector distance in microns
    """
    return radius_px * px / np.tan(np.radians(tth_deg))
