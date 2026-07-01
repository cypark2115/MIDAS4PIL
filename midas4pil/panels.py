# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""
Phase 4 — Per-panel module corrections.

Applies per-panel (dY, dZ, dLsd, dP2) offsets to pixel coordinates
before the detector-geometry transform, compensating for module
misalignment in Pilatus/Eiger tiled detectors.

Reference: MIDAS CalibrantPanelShiftsOMP.c, ApplyPanelCorrection().

make_panel_id_map() is fully general: it works for any tiled detector
with uniform panel sizes and arbitrary (possibly non-uniform) inter-panel
gaps.  Convenience presets are provided for common detectors.

Panel numbering: 1-indexed, row-major in the array frame
  panel_id = row_panel * n_panels_y + col_panel + 1
  row_panel ∈ {0 .. n_panels_z−1}  (top→bottom in the array)
  col_panel ∈ {0 .. n_panels_y−1}  (left→right)

Panel-map dtype is int32 to accommodate detectors with many modules
(e.g. Eiger 16M with ~108 panels) without overflow.

Note on pixel bit depth
-----------------------
Pixel values are not touched here; they are handled in io.load_tiff()
(tifffile auto-detects dtype) and converted to float64 in integrate.py
before any arithmetic.
"""

import numpy as np


# ── Panel layout ───────────────────────────────────────────────────────────

def make_panel_id_map(nrows, ncols,
                      n_panels_y, n_panels_z,
                      panel_size_y, panel_size_z,
                      gap_y, gap_z):
    """Return a panel-ID map for a tiled detector.

    Works for any detector whose panels are:
    - Uniform size (all panels the same width and height), and
    - Separated by gaps that may differ between adjacent pairs.

    Parameters
    ----------
    nrows, ncols   : full detector size in pixels (array dimensions)
    n_panels_y     : number of panel columns (fast / horizontal direction)
    n_panels_z     : number of panel rows    (slow / vertical   direction)
    panel_size_y   : width  of each panel in pixels (column direction)
    panel_size_z   : height of each panel in pixels (row    direction)
    gap_y          : (n_panels_y − 1) inter-panel gaps in the column direction
                     (pixels); scalar for uniform gaps or list for variable gaps
    gap_z          : (n_panels_z − 1) inter-panel gaps in the row direction
                     (pixels); scalar for uniform gaps or list for variable gaps

    Returns
    -------
    panel_map : int32 array (nrows, ncols)
        panel_id >= 1  — pixel belongs to that panel (1-indexed, row-major)
        panel_id == 0  — pixel is in a gap (treat as bad)
    """
    # Normalise gap arguments to lists
    if np.isscalar(gap_y):
        gap_y = [int(gap_y)] * (n_panels_y - 1)
    if np.isscalar(gap_z):
        gap_z = [int(gap_z)] * (n_panels_z - 1)

    gap_y = list(gap_y)
    gap_z = list(gap_z)

    if len(gap_y) != n_panels_y - 1:
        raise ValueError(f"gap_y must have {n_panels_y - 1} entries, "
                         f"got {len(gap_y)}")
    if len(gap_z) != n_panels_z - 1:
        raise ValueError(f"gap_z must have {n_panels_z - 1} entries, "
                         f"got {len(gap_z)}")

    panel_map = np.zeros((nrows, ncols), dtype=np.int32)

    # Column-direction (Y) slices
    y_slices, y_pos = [], 0
    for i in range(n_panels_y):
        y_slices.append(slice(y_pos, y_pos + panel_size_y))
        y_pos += panel_size_y
        if i < len(gap_y):
            y_pos += gap_y[i]

    # Row-direction (Z) slices — array frame, row 0 = top
    z_slices, z_pos = [], 0
    for i in range(n_panels_z):
        z_slices.append(slice(z_pos, z_pos + panel_size_z))
        z_pos += panel_size_z
        if i < len(gap_z):
            z_pos += gap_z[i]

    for iz, zs in enumerate(z_slices):
        for iy, ys in enumerate(y_slices):
            panel_id = iz * n_panels_y + iy + 1   # 1-indexed, row-major
            panel_map[zs, ys] = panel_id

    return panel_map



# ── Panel-shift file reader ────────────────────────────────────────────────

def read_panel_shifts(path):
    """Read a MIDAS PanelShiftsFile into a list of dicts.

    Expected format (one panel per line, whitespace-separated):
        panel_id  dY  dZ  [dTheta  [dLsd  [dP2]]]

    Column order matches MIDAS Panel.c SavePanelShifts():
        col 0 = panel_id, 1 = dY (px), 2 = dZ (px),
        3 = dTheta (deg), 4 = dLsd (µm), 5 = dP2 (dimensionless)

    Returns
    -------
    list of dict, one per panel, with keys:
        'id', 'dY', 'dZ', 'dTheta', 'dLsd', 'dP2'
    """
    panels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            panels.append({
                'id':     int(parts[0]),
                'dY':     float(parts[1]),
                'dZ':     float(parts[2]),
                'dTheta': float(parts[3]) if len(parts) > 3 else 0.0,
                'dLsd':   float(parts[4]) if len(parts) > 4 else 0.0,
                'dP2':    float(parts[5]) if len(parts) > 5 else 0.0,
            })
    return panels


def save_panel_shifts(panel_shifts, path):
    """Write panel shifts to a MIDAS-compatible text file.

    Symmetric writer for read_panel_shifts().  Column order matches MIDAS
    Panel.c SavePanelShifts(): panel_id  dY  dZ  dTheta  dLsd  dP2.

    Parameters
    ----------
    panel_shifts : list of dicts with keys 'id', 'dY', 'dZ', 'dTheta', 'dLsd', 'dP2'
    path         : output file path
    """
    with open(path, 'w') as f:
        f.write("# panel_id  dY(px)  dZ(px)  dTheta(deg)  dLsd(um)  dP2\n")
        for p in panel_shifts:
            f.write(f"  {p['id']:3d}  {p.get('dY', 0.0):12.6f}  "
                    f"{p.get('dZ', 0.0):12.6f}  {p.get('dTheta', 0.0):12.6f}  "
                    f"{p.get('dLsd', 0.0):12.6f}  {p.get('dP2', 0.0):12.6f}\n")


# ── Apply corrections ──────────────────────────────────────────────────────

def apply_panel_offsets(nrows, ncols, panel_map, panel_shifts):
    """Compute per-pixel correction arrays from a list of panel shifts.

    Parameters
    ----------
    nrows, ncols   : detector size
    panel_map      : int32 array (nrows, ncols) from make_panel_id_map()
    panel_shifts   : list of dicts from read_panel_shifts()

    Returns
    -------
    dY_map   : float64 (nrows, ncols), column shift (pixels)
    dZ_map   : float64 (nrows, ncols), row    shift (pixels, physical frame)
    dLsd_map : float64 (nrows, ncols), Lsd offset (µm)
    dP2_map  : float64 (nrows, ncols), p2  offset (dimensionless)
    """
    dY_map   = np.zeros((nrows, ncols), dtype=np.float64)
    dZ_map   = np.zeros((nrows, ncols), dtype=np.float64)
    dLsd_map = np.zeros((nrows, ncols), dtype=np.float64)
    dP2_map  = np.zeros((nrows, ncols), dtype=np.float64)

    shifts_by_id = {p['id']: p for p in panel_shifts}
    for pid, shift in shifts_by_id.items():
        m = (panel_map == pid)
        dY_map[m]   = shift['dY']
        dZ_map[m]   = shift['dZ']
        dLsd_map[m] = shift['dLsd']
        dP2_map[m]  = shift['dP2']

    return dY_map, dZ_map, dLsd_map, dP2_map


def build_lut_with_panels(nrows, ncols, bc_y, bc_z, lsd, px,
                          tx_deg, ty_deg, tz_deg,
                          p0, p1, p2, p3, p4, rho_d,
                          panel_map, panel_shifts):
    """Build (2θ, eta) LUTs with per-panel corrections applied.

    Parameters
    ----------
    (same geometry arguments as geometry.build_lut, plus:)
    panel_map    : int32 array (nrows, ncols) from make_panel_id_map()
    panel_shifts : list of dicts from read_panel_shifts()

    Returns
    -------
    tth : shape (nrows, ncols), 2θ in degrees
    eta : shape (nrows, ncols), eta in degrees
    """
    from .geometry import build_tilt_matrix, pixel_to_r_eta, r_to_tth

    dY_map, dZ_map, dLsd_map, dP2_map = apply_panel_offsets(
        nrows, ncols, panel_map, panel_shifts)

    col    = np.arange(ncols, dtype=np.float64)
    Z_phys = np.arange(nrows, dtype=np.float64)[:, None]  # row index from top

    col_corr   = col    + dY_map
    Zphys_corr = Z_phys + dZ_map

    TRs = build_tilt_matrix(tx_deg, ty_deg, tz_deg)
    R, eta = pixel_to_r_eta(col_corr, Zphys_corr, bc_y, bc_z, TRs,
                             lsd, rho_d, p0, p1, p2, p3, p4, px,
                             dlsd=dLsd_map, dp2=dP2_map)
    tth = r_to_tth(R, px, lsd)
    return tth, eta
