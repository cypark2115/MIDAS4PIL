"""
Panel calibration optimizer.

Fits detector geometry and per-panel misalignment corrections to a
calibrant diffraction image.  Solves the same calibration problem as
MIDAS CalibrantPanelShiftsOMP, but is an independent Python implementation.

Algorithm
---------
1. Compute (2theta, eta) for every unmasked pixel using current geometry.
2. Assign pixels near known calibrant rings to (ring, eta-bin) pairs.
3. Compute intensity-weighted mean pixel position per (ring, eta-bin).
4. Minimize strain = |1 - R_corrected / R_ideal| over all data points,
   varying global geometry + per-panel shifts.
5. Re-bin with updated geometry and repeat.
"""

import logging

logger = logging.getLogger(__name__)

import numpy as np
from scipy.optimize import minimize


class _CalibrationCancelled(BaseException):
    """Raised in the progress callback to signal cooperative cancellation.

    Inherits from BaseException (not Exception) so it passes through the
    broad ``except Exception: pass`` guards inside the optimizer loop.
    """

from .geometry import build_tilt_matrix, pixel_to_r_eta, r_to_tth


# ── Ring pixel assignment ─────────────────────────────────────────────────

def find_ring_pixels(tth_lut, eta_lut, mask, rings, tth_tol,
                     eta_bin_size=1.0):
    """Assign detector pixels to (ring, eta-bin) pairs.

    Parameters
    ----------
    tth_lut      : 2-D array (nrows, ncols), 2theta in degrees
    eta_lut      : 2-D array (nrows, ncols), eta in degrees
    mask         : 2-D bool array, True = bad pixel
    rings        : list of dicts from calibrant.ring_table(), each with 'tth'
    tth_tol      : half-width of 2theta acceptance window (degrees)
    eta_bin_size : eta bin width (degrees, default 1.0)

    Returns
    -------
    list of dicts, one per populated (ring, eta-bin), each with:
        'pixel_rows'   : 1-D int array, row indices of contributing pixels
        'pixel_cols'   : 1-D int array, col indices
        'ring_tth'     : float, ideal 2theta of this ring (degrees)
        'eta_centre'   : float, centre of this eta bin (degrees)
    """
    good = ~mask
    tth_flat = tth_lut[good]
    eta_flat = eta_lut[good]
    row_flat, col_flat = np.where(good)

    n_eta = int(np.ceil(360.0 / eta_bin_size))
    eta_edges = np.linspace(-180.0, 180.0, n_eta + 1)

    bins = []
    for ring in rings:
        tth_ring = ring['tth']
        # Select pixels within tth tolerance
        in_ring = np.abs(tth_flat - tth_ring) < tth_tol
        if not np.any(in_ring):
            continue

        ring_eta = eta_flat[in_ring]
        ring_rows = row_flat[in_ring]
        ring_cols = col_flat[in_ring]

        # Subdivide by eta
        eta_idx = np.floor((ring_eta - (-180.0)) / eta_bin_size).astype(int)
        eta_idx = np.clip(eta_idx, 0, n_eta - 1)

        for ie in range(n_eta):
            sel = eta_idx == ie
            if not np.any(sel):
                continue
            bins.append({
                'pixel_rows': ring_rows[sel],
                'pixel_cols': ring_cols[sel],
                'ring_tth': tth_ring,
                'eta_centre': 0.5 * (eta_edges[ie] + eta_edges[ie + 1]),
            })

    return bins


def weighted_mean_positions(image, ring_bins):
    """Compute intensity-weighted mean (col, row) per (ring, eta-bin).

    Parameters
    ----------
    image     : 2-D array, detector image
    ring_bins : list from find_ring_pixels()

    Returns
    -------
    YMean         : 1-D float array, mean column position per bin
    ZMean         : 1-D float array, mean row position per bin
    IdealTtheta   : 1-D float array, ideal 2theta per bin (degrees)
    """
    n = len(ring_bins)
    YMean = np.empty(n)
    ZMean = np.empty(n)
    IdealTtheta = np.empty(n)

    for i, b in enumerate(ring_bins):
        rows = b['pixel_rows']
        cols = b['pixel_cols']
        intensities = image[rows, cols].astype(np.float64)
        total = intensities.sum()
        if total > 0:
            YMean[i] = np.sum(intensities * cols) / total
            ZMean[i] = np.sum(intensities * rows) / total
        else:
            YMean[i] = np.mean(cols.astype(float))
            ZMean[i] = np.mean(rows.astype(float))
        IdealTtheta[i] = b['ring_tth']

    return YMean, ZMean, IdealTtheta


# ── Panel centre computation ─────────────────────────────────────────────

def _panel_centres(panel_map, n_panels):
    """Compute geometric centre (col, row) of each panel.

    Returns dict: panel_id → (centre_col, centre_row).
    """
    centres = {}
    for pid in range(1, n_panels + 1):
        m = panel_map == pid
        if not np.any(m):
            continue
        rows, cols = np.where(m)
        centres[pid] = (np.mean(cols.astype(float)),
                        np.mean(rows.astype(float)))
    return centres


# ── Cost function ─────────────────────────────────────────────────────────

def _unpack_params(x, n_base, n_panels, fix_panel, stride_config):
    """Unpack the flat parameter vector into geometry + panel shifts.

    Returns
    -------
    geom_params : dict with Lsd, bc_y, bc_z, ty, tz, p0..p4
    panel_dicts : dict panel_id → {'dY', 'dZ', 'dTheta', 'dLsd', 'dP2'}
    """
    geom = {
        'Lsd': x[0], 'bc_y': x[1], 'bc_z': x[2],
        'ty': x[3], 'tz': x[4],
        'p0': x[5], 'p1': x[6], 'p2': x[7], 'p3': x[8], 'p4': x[9],
    }

    panel_dicts = {}
    if n_panels > 1:
        has_rotation, has_lsd, has_p2 = stride_config
        p_idx = n_base
        for pid in range(1, n_panels + 1):
            if pid == fix_panel:
                panel_dicts[pid] = {'dY': 0.0, 'dZ': 0.0, 'dTheta': 0.0,
                                    'dLsd': 0.0, 'dP2': 0.0}
                continue
            dY = x[p_idx]; p_idx += 1
            dZ = x[p_idx]; p_idx += 1
            dTheta = 0.0
            if has_rotation:
                dTheta = x[p_idx]; p_idx += 1
            dLsd = 0.0
            if has_lsd:
                dLsd = x[p_idx]; p_idx += 1
            dP2 = 0.0
            if has_p2:
                dP2 = x[p_idx]; p_idx += 1
            panel_dicts[pid] = {'dY': dY, 'dZ': dZ, 'dTheta': dTheta,
                                'dLsd': dLsd, 'dP2': dP2}

    return geom, panel_dicts


def strain_cost(x, YMean, ZMean, IdealTtheta, fixed_geom, panel_map,
                panel_centres, n_panels, fix_panel, stride_config,
                use_l2=False):
    """Vectorized strain cost function for scipy.optimize.

    Parameters
    ----------
    x             : 1-D parameter vector
    YMean, ZMean  : mean pixel positions (col, row) per data point
    IdealTtheta   : ideal 2theta per data point (degrees)
    fixed_geom    : dict with tx, px, rho_d (not optimized)
    panel_map     : int32 array (nrows, ncols)
    panel_centres : dict pid → (centre_col, centre_row)
    n_panels      : total number of panels
    fix_panel     : panel ID fixed as reference
    stride_config : (has_rotation, has_lsd, has_p2)
    use_l2        : if True, use squared strain; else absolute

    Returns
    -------
    total_cost : float
    """
    n_base = 10
    geom, panel_dicts = _unpack_params(x, n_base, n_panels, fix_panel,
                                        stride_config)
    Lsd = geom['Lsd']
    bc_y = geom['bc_y']
    bc_z = geom['bc_z']
    tx = fixed_geom['tx']
    px = fixed_geom['px']
    rho_d = fixed_geom['rho_d']

    TRs = build_tilt_matrix(tx, geom['ty'], geom['tz'])

    # Apply per-point panel corrections
    rawY = YMean.copy()
    rawZ = ZMean.copy()
    dlsd_arr = np.zeros(len(YMean))
    dp2_arr = np.zeros(len(YMean))

    if n_panels > 1:
        # Find panel ID for each data point
        y_int = np.clip(np.round(YMean).astype(int), 0, panel_map.shape[1] - 1)
        z_int = np.clip(np.round(ZMean).astype(int), 0, panel_map.shape[0] - 1)
        point_panels = panel_map[z_int, y_int]

        for pid, shifts in panel_dicts.items():
            sel = point_panels == pid
            if not np.any(sel):
                continue
            dY = shifts['dY']
            dZ = shifts['dZ']
            dTheta = shifts['dTheta']
            dLsd = shifts['dLsd']
            dP2 = shifts['dP2']

            # Apply panel rotation about panel centre
            if abs(dTheta) > 1e-12 and pid in panel_centres:
                cY, cZ = panel_centres[pid]
                cosT = np.cos(np.radians(dTheta))
                sinT = np.sin(np.radians(dTheta))
                dy = rawY[sel] - cY
                dz = rawZ[sel] - cZ
                rawY[sel] = cY + dy * cosT - dz * sinT
                rawZ[sel] = cZ + dy * sinT + dz * cosT

            rawY[sel] += dY
            rawZ[sel] += dZ
            dlsd_arr[sel] = dLsd
            dp2_arr[sel] = dP2

    # Forward model: pixel → R → 2theta
    R_px, _ = pixel_to_r_eta(rawY, rawZ, bc_y, bc_z, TRs, Lsd, rho_d,
                              geom['p0'], geom['p1'], geom['p2'],
                              geom['p3'], geom['p4'], px,
                              dlsd=dlsd_arr, dp2=dp2_arr)

    R_corrected = R_px * px   # microns
    R_ideal = Lsd * np.tan(np.radians(IdealTtheta))

    strain = 1.0 - R_corrected / R_ideal

    if use_l2:
        return np.sum(strain ** 2)
    else:
        return np.sum(np.abs(strain))


# ── Outlier rejection ─────────────────────────────────────────────────────

def _reject_outliers(strains, factor=3.0, max_iter=10):
    """Iterative sigma-clipping.  Returns bool mask, True = keep."""
    keep = np.ones(len(strains), dtype=bool)
    for _ in range(max_iter):
        s = np.abs(strains[keep])
        if len(s) == 0:
            break
        mean_s = np.mean(s)
        threshold = factor * mean_s
        new_keep = np.abs(strains) <= threshold
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return keep


# ── Main calibration entry point ──────────────────────────────────────────

def calibrate(image, mask, geom, panel_map, calibrant_rings,
              fix_panel=None,
              optimize_shifts=True,
              optimize_rotation=False,
              optimize_panel_lsd=False,
              optimize_panel_p2=False,
              tol_lsd=500.0, tol_bc=5.0, tol_tilts=1.0,
              tol_shifts=5.0, tol_rotation=1.0,
              tol_p0=0.0, tol_p1=0.0, tol_p2=0.0,
              tol_p3=0.0, tol_p4=0.0,
              tol_lsd_panel=500.0, tol_p2_panel=0.1,
              tth_tol_factor=3.0,
              eta_bin_size=1.0,
              n_iterations=5,
              outlier_factor=3.0,
              use_l2=False,
              verbose=True,
              progress_cb=None):
    """Calibrate detector geometry and per-panel shifts from a calibrant image.

    Parameters
    ----------
    image           : 2-D array, calibrant diffraction image
    mask            : 2-D bool array, True = bad pixel
    geom            : dict from load_params() — initial geometry
    panel_map       : int32 array (nrows, ncols) from make_panel_id_map()
    calibrant_rings : list of dicts from ring_table() or load_calibrant()
    fix_panel       : panel ID to hold fixed as reference (1-indexed).
                      If None (default), automatically selects the panel
                      with the most calibrant ring coverage (most distinct
                      rings and most ring pixels).
    optimize_shifts : optimize per-panel dY, dZ (default True)
    optimize_rotation : optimize per-panel dTheta (default False)
    optimize_panel_lsd : optimize per-panel dLsd (default False)
    optimize_panel_p2  : optimize per-panel dP2 (default False)
    tol_*           : symmetric tolerance for each parameter
    eta_bin_size    : azimuthal bin width for ring sampling (degrees)
    n_iterations    : number of re-binning iterations
    outlier_factor  : sigma-clipping factor (0 = no rejection)
    use_l2          : use L2 (squared) instead of L1 (absolute) strain
    verbose         : print progress

    Returns
    -------
    dict with keys:
        'geom'            : dict, optimized geometry (same keys as input)
        'panel_shifts'    : list of dicts [{id, dY, dZ, dLsd, dP2, dTheta}, ...]
        'panel_coverage'  : dict {panel_id: {'n_rings': int, 'n_pixels': int}}
                            Coverage of calibrant rings per panel (from initial
                            geometry).  Panels with n_pixels==0 have no ring data;
                            their shifts are fixed at 0 and cannot be calibrated.
        'mean_strain'     : float, mean |strain| (dimensionless)
        'std_strain'      : float
        'n_points'        : int, number of valid data points after outlier rejection
        'ring_strains'    : list of dicts per ring [{tth_ideal, tth_observed,
                            strain, strain_abs, n_points}, ...]
    """
    from .geometry import build_lut

    nrows, ncols = image.shape
    px = geom['px']
    lsd = geom['lsd']
    rho_d = geom['rho_d']

    # Always exclude below-threshold pixels (Pilatus/Eiger store them as -2)
    # User-provided masks (e.g. from Dioptas) may not cover these.
    mask = np.asarray(mask, dtype=bool) | (image < 0)

    # When no panel map is supplied, treat the whole detector as one panel
    if panel_map is None:
        panel_map = np.ones((nrows, ncols), dtype=np.int32)
    elif panel_map.shape != (nrows, ncols):
        raise ValueError(
            f"panel_map shape {panel_map.shape} does not match image shape "
            f"{(nrows, ncols)} — select the correct detector before calibrating."
        )

    # Panel setup
    panel_ids = np.unique(panel_map[panel_map > 0])
    n_panels = len(panel_ids)
    centres = _panel_centres(panel_map, n_panels)

    # Stride config
    has_rotation = optimize_rotation
    has_lsd = optimize_panel_lsd
    has_p2 = optimize_panel_p2
    stride_config = (has_rotation, has_lsd, has_p2)
    stride = 2 + int(has_rotation) + int(has_lsd) + int(has_p2)

    # Fixed geometry (not optimized)
    fixed_geom = {'tx': geom['tx_deg'], 'px': px, 'rho_d': rho_d}

    # Current state
    cur = {
        'Lsd': lsd, 'bc_y': geom['bc_y'], 'bc_z': geom['bc_z'],
        'ty': geom['ty_deg'], 'tz': geom['tz_deg'],
        'p0': geom['p0'], 'p1': geom['p1'], 'p2': geom['p2'],
        'p3': geom['p3'], 'p4': geom['p4'],
    }
    panel_shifts_cur = {pid: {'dY': 0.0, 'dZ': 0.0, 'dTheta': 0.0,
                               'dLsd': 0.0, 'dP2': 0.0}
                        for pid in panel_ids}

    # 2theta tolerance for ring assignment (half of bin width at ring position)
    # Use approximate pixel angular width at median ring position
    median_tth = np.median([r['tth'] for r in calibrant_rings])
    tth_tol_assign = np.degrees(px * np.cos(np.radians(median_tth))**2 / lsd) * tth_tol_factor

    # Compute per-panel ring coverage from initial geometry (always done).
    # Used to: (a) auto-select the reference panel, (b) warn about uncovered panels.
    from .panels import build_lut_with_panels
    _ps0 = [{'id': pid, 'dY': 0, 'dZ': 0, 'dLsd': 0, 'dP2': 0, 'dTheta': 0}
            for pid in panel_ids]
    _tth0, _eta0 = build_lut_with_panels(
        nrows, ncols, cur['bc_y'], cur['bc_z'], cur['Lsd'], px,
        fixed_geom['tx'], cur['ty'], cur['tz'],
        cur['p0'], cur['p1'], cur['p2'], cur['p3'], cur['p4'], rho_d,
        panel_map, _ps0)
    _bins0 = find_ring_pixels(_tth0, _eta0, mask, calibrant_rings,
                               tth_tol_assign, eta_bin_size)
    _panel_n_rings = {pid: set() for pid in panel_ids}
    _panel_n_pixels = {pid: 0 for pid in panel_ids}
    for b in _bins0:
        pids = panel_map[b['pixel_rows'], b['pixel_cols']]
        for pid in np.unique(pids):
            if pid == 0:
                continue
            _panel_n_rings[pid].add(round(b['ring_tth'], 3))
            _panel_n_pixels[pid] += int(np.sum(pids == pid))

    # Auto-select reference panel: the one with the most ring coverage
    if fix_panel is None:
        fix_panel = max(panel_ids,
                        key=lambda p: (len(_panel_n_rings[p]),
                                       _panel_n_pixels[p]))
        if verbose:
            logger.info("  Reference panel: %d (%d rings, %d ring pixels)",
                       fix_panel, len(_panel_n_rings[fix_panel]),
                       _panel_n_pixels[fix_panel])

    # Coverage summary for caller
    panel_coverage = {
        int(pid): {
            'n_rings':  len(_panel_n_rings[pid]),
            'n_pixels': _panel_n_pixels[pid],
        }
        for pid in panel_ids
    }
    uncovered = [pid for pid in panel_ids if _panel_n_pixels[pid] == 0]
    if uncovered and verbose:
        logger.warning("  Panels with no calibrant ring coverage (shifts fixed at 0): %s",
                       uncovered)

    best_mean_strain = 1e30
    best_result = None

    n_base = 10

    for iteration in range(n_iterations):
        if verbose:
            logger.info("--- Iteration %d/%d ---", iteration + 1, n_iterations)

        # 1. Build LUT with current geometry + panel shifts
        panel_shifts_list = [{'id': pid, **panel_shifts_cur[pid]}
                              for pid in panel_ids]
        tth_lut, eta_lut = build_lut_with_panels(
            nrows, ncols, cur['bc_y'], cur['bc_z'], cur['Lsd'], px,
            fixed_geom['tx'], cur['ty'], cur['tz'],
            cur['p0'], cur['p1'], cur['p2'], cur['p3'], cur['p4'], rho_d,
            panel_map, panel_shifts_list)

        # 2. Find ring pixels and compute weighted-mean positions
        ring_bins = find_ring_pixels(tth_lut, eta_lut, mask,
                                     calibrant_rings, tth_tol_assign,
                                     eta_bin_size)
        if len(ring_bins) == 0:
            raise RuntimeError("No calibrant ring pixels found. Check "
                               "geometry, mask, or calibrant ring table.")

        YMean, ZMean, IdealTtheta = weighted_mean_positions(image, ring_bins)

        if verbose:
            logger.info("  Data points: %d", len(YMean))

        # 3. Build parameter vector + bounds
        n_opt_panels = n_panels - 1 if (n_panels > 1 and optimize_shifts) else 0
        n_params = n_base + n_opt_panels * stride

        x0 = np.zeros(n_params)
        lb = np.zeros(n_params)
        ub = np.zeros(n_params)

        # Global params
        x0[0] = cur['Lsd'];  lb[0] = cur['Lsd'] - tol_lsd;  ub[0] = cur['Lsd'] + tol_lsd
        x0[1] = cur['bc_y']; lb[1] = cur['bc_y'] - tol_bc;  ub[1] = cur['bc_y'] + tol_bc
        x0[2] = cur['bc_z']; lb[2] = cur['bc_z'] - tol_bc;  ub[2] = cur['bc_z'] + tol_bc
        x0[3] = cur['ty'];   lb[3] = cur['ty'] - tol_tilts;  ub[3] = cur['ty'] + tol_tilts
        x0[4] = cur['tz'];   lb[4] = cur['tz'] - tol_tilts;  ub[4] = cur['tz'] + tol_tilts
        x0[5] = cur['p0'];   lb[5] = cur['p0'] - tol_p0;    ub[5] = cur['p0'] + tol_p0
        x0[6] = cur['p1'];   lb[6] = cur['p1'] - tol_p1;    ub[6] = cur['p1'] + tol_p1
        x0[7] = cur['p2'];   lb[7] = cur['p2'] - tol_p2;    ub[7] = cur['p2'] + tol_p2
        x0[8] = cur['p3'];   lb[8] = cur['p3'] - tol_p3;    ub[8] = cur['p3'] + tol_p3
        x0[9] = cur['p4'];   lb[9] = cur['p4'] - tol_p4;    ub[9] = cur['p4'] + tol_p4

        # Per-panel params
        if n_panels > 1 and optimize_shifts:
            p_idx = n_base
            for pid in panel_ids:
                if pid == fix_panel:
                    continue
                s = panel_shifts_cur[pid]
                x0[p_idx] = s['dY'];  lb[p_idx] = s['dY'] - tol_shifts;  ub[p_idx] = s['dY'] + tol_shifts
                p_idx += 1
                x0[p_idx] = s['dZ'];  lb[p_idx] = s['dZ'] - tol_shifts;  ub[p_idx] = s['dZ'] + tol_shifts
                p_idx += 1
                if has_rotation:
                    x0[p_idx] = s['dTheta']; lb[p_idx] = s['dTheta'] - tol_rotation; ub[p_idx] = s['dTheta'] + tol_rotation
                    p_idx += 1
                if has_lsd:
                    x0[p_idx] = s['dLsd']; lb[p_idx] = s['dLsd'] - tol_lsd_panel; ub[p_idx] = s['dLsd'] + tol_lsd_panel
                    p_idx += 1
                if has_p2:
                    x0[p_idx] = s['dP2']; lb[p_idx] = s['dP2'] - tol_p2_panel; ub[p_idx] = s['dP2'] + tol_p2_panel
                    p_idx += 1

        bounds = list(zip(lb, ub))

        # 4. Optimize
        n_effective = n_panels if (n_panels > 1 and optimize_shifts) else 1

        cost_args = (YMean, ZMean, IdealTtheta, fixed_geom, panel_map,
                     centres, n_panels if optimize_shifts else 1,
                     fix_panel, stride_config, use_l2)

        # Choose method based on problem dimension
        if n_params > 20:
            method = 'Powell'
        else:
            method = 'Nelder-Mead'

        if method == 'Nelder-Mead':
            opts = {'maxiter': 5000, 'xatol': 1e-6,
                    'fatol': 1e-8, 'adaptive': True}
        else:
            opts = {'maxiter': 5000, 'ftol': 1e-8}

        result = minimize(strain_cost, x0, args=cost_args,
                          method=method, bounds=bounds, options=opts)

        # 5. Unpack result
        opt_geom, opt_panels = _unpack_params(
            result.x, n_base,
            n_panels if optimize_shifts else 1,
            fix_panel, stride_config)

        # Update current state
        cur.update(opt_geom)
        if optimize_shifts and n_panels > 1:
            panel_shifts_cur.update(opt_panels)

        # 6. Compute per-point strains for outlier rejection and reporting
        TRs = build_tilt_matrix(fixed_geom['tx'], cur['ty'], cur['tz'])
        rawY = YMean.copy()
        rawZ = ZMean.copy()
        dlsd_pts = np.zeros(len(YMean))
        dp2_pts = np.zeros(len(YMean))

        if n_panels > 1 and optimize_shifts:
            y_int = np.clip(np.round(YMean).astype(int), 0, ncols - 1)
            z_int = np.clip(np.round(ZMean).astype(int), 0, nrows - 1)
            pt_panels = panel_map[z_int, y_int]
            for pid, shifts in panel_shifts_cur.items():
                sel = pt_panels == pid
                if not np.any(sel):
                    continue
                if abs(shifts['dTheta']) > 1e-12 and pid in centres:
                    cY, cZ = centres[pid]
                    cosT = np.cos(np.radians(shifts['dTheta']))
                    sinT = np.sin(np.radians(shifts['dTheta']))
                    dy = rawY[sel] - cY
                    dz = rawZ[sel] - cZ
                    rawY[sel] = cY + dy * cosT - dz * sinT
                    rawZ[sel] = cZ + dy * sinT + dz * cosT
                rawY[sel] += shifts['dY']
                rawZ[sel] += shifts['dZ']
                dlsd_pts[sel] = shifts['dLsd']
                dp2_pts[sel] = shifts['dP2']

        R_px, _ = pixel_to_r_eta(rawY, rawZ, cur['bc_y'], cur['bc_z'], TRs,
                                  cur['Lsd'], rho_d,
                                  cur['p0'], cur['p1'], cur['p2'],
                                  cur['p3'], cur['p4'], px,
                                  dlsd=dlsd_pts, dp2=dp2_pts)
        R_corr = R_px * px
        R_ideal = cur['Lsd'] * np.tan(np.radians(IdealTtheta))
        strains = 1.0 - R_corr / R_ideal

        # Outlier rejection
        if outlier_factor > 0:
            keep = _reject_outliers(strains, outlier_factor)
        else:
            keep = np.ones(len(strains), dtype=bool)

        mean_strain = np.mean(np.abs(strains[keep]))
        std_strain = np.std(strains[keep])

        if verbose:
            logger.info("  Mean strain: %.1f ppm  Std: %.1f ppm  Points: %d/%d",
                       mean_strain * 1e6, std_strain * 1e6,
                       int(np.sum(keep)), len(keep))

        if progress_cb is not None:
            try:
                progress_cb(iteration + 1, n_iterations,
                            mean_strain * 1e6, int(np.sum(keep)))
            except _CalibrationCancelled:
                raise   # propagate cooperative cancellation out of calibrate()
            except Exception:
                pass

        # Per-ring strain breakdown
        unique_tth = np.unique(IdealTtheta)
        per_ring_data = []
        for tth_ideal in unique_tth:
            sel = (IdealTtheta == tth_ideal) & keep
            if np.any(sel):
                tth_obs = np.degrees(np.arctan(
                    R_corr[sel].mean() / cur['Lsd']))
                per_ring_data.append({
                    'tth_ideal': float(tth_ideal),
                    'tth_observed': float(tth_obs),
                    'strain': float(np.mean(strains[sel])),
                    'strain_abs': float(np.mean(np.abs(strains[sel]))),
                    'n_points': int(np.sum(sel)),
                })

        # Track best
        if mean_strain < best_mean_strain:
            best_mean_strain = mean_strain
            best_result = {
                'geom': _build_output_geom(geom, cur, fixed_geom),
                'panel_shifts': [{'id': int(pid),
                                   'dY': panel_shifts_cur[pid]['dY'],
                                   'dZ': panel_shifts_cur[pid]['dZ'],
                                   'dLsd': panel_shifts_cur[pid]['dLsd'],
                                   'dP2': panel_shifts_cur[pid]['dP2'],
                                   'dTheta': panel_shifts_cur[pid]['dTheta']}
                                  for pid in panel_ids],
                'panel_coverage': panel_coverage,
                'mean_strain': float(mean_strain),
                'std_strain': float(std_strain),
                'n_points': int(np.sum(keep)),
                'ring_strains': per_ring_data,
            }

    if verbose:
        logger.info("Best mean strain: %.1f ppm", best_mean_strain * 1e6)

    return best_result


def _build_output_geom(original_geom, cur, fixed_geom):
    """Build output geometry dict from optimized parameters."""
    out = dict(original_geom)
    out['lsd'] = cur['Lsd']
    out['bc_y'] = cur['bc_y']
    out['bc_z'] = cur['bc_z']
    out['ty_deg'] = cur['ty']
    out['tz_deg'] = cur['tz']
    out['p0'] = cur['p0']
    out['p1'] = cur['p1']
    out['p2'] = cur['p2']
    out['p3'] = cur['p3']
    out['p4'] = cur['p4']
    return out


