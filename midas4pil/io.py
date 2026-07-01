# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""
I/O helpers.

Primary workflow (standalone, no external tools required)
---------------------------------------------------------
1. Build initial geometry:  geom = make_geometry(wavelength, lsd, px, nrows, ncols)
2. Calibrate:               result = calibrate(image, mask, geom, ...)
3. Save result:             save_params(result['geom'], "geometry.toml")
4. Reload:                  geom = load_params("geometry.toml")
5. Use:                     tth, eta = build_lut(**geom)

Native params format
--------------------
TOML with explicit unit suffixes in key names (_um, _px, _deg, _A).
Five sections, always written together in one file:

  [detector]               — pixel size, array dimensions
  [x-ray]                  — wavelength / energy
  [flat_detector_geometry] — Lsd, beam centre, tilts
  [distortion_correction]  — p0–p4 radial distortion model
  [integration]            — 2θ / η range and bin size (soft defaults)
  [[panel_shifts]]         — per-module offsets (array of tables, omitted
                             when no panel correction was applied)

tomllib is stdlib in Python >= 3.11; no extra dependency needed.
"""

import sys
from pathlib import Path

import numpy as np

_MASK_LOAD_THRESHOLD = 0.5

if sys.version_info >= (3, 11):
    import tomllib
else:                           # Python 3.9 / 3.10 fallback
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # pip install tomli


# ── Native TOML params ─────────────────────────────────────────────────────

_REQUIRED = {
    'detector':               ['nrows', 'ncols', 'px_um'],
    'x-ray':                  ['wavelength_A'],
    'flat_detector_geometry': ['lsd_um', 'bc_y_px', 'bc_z_px',
                               'tx_deg', 'ty_deg', 'tz_deg'],
    'distortion_correction':  ['p0', 'p1', 'p2', 'p3_deg', 'p4', 'rho_d_um'],
}

_INTEGRATION_DEFAULTS = dict(
    tth_min_deg=1.0, tth_max_deg=40.0, tth_bin_deg=0.01,
    eta_min_deg=-180.0, eta_max_deg=180.0, eta_bin_deg=1.0,
)


def load_params(path):
    """Load a midas4pil TOML params file.

    Returns
    -------
    dict with keys (all unit-free, ready for build_lut / integrate_1d):
        nrows, ncols, px,
        wavelength,
        lsd, bc_y, bc_z, tx_deg, ty_deg, tz_deg,
        p0, p1, p2, p3, p4, rho_d,
        tth_min, tth_max, tth_bin_size,
        eta_min, eta_max, eta_bin_size,
        panel_shifts  — list of dicts (empty list if no [[panel_shifts]] section)
    """
    with open(path, 'rb') as f:
        data = tomllib.load(f)

    # Legacy format support: old files used [geometry] + [beam] instead of
    # [flat_detector_geometry] + [distortion_correction] + [x-ray].
    if 'geometry' in data and 'flat_detector_geometry' not in data:
        g = data['geometry']
        data['flat_detector_geometry'] = {k: g[k] for k in
            ('lsd_um', 'bc_y_px', 'bc_z_px', 'tx_deg', 'ty_deg', 'tz_deg') if k in g}
        data['distortion_correction'] = {k: g[k] for k in
            ('p0', 'p1', 'p2', 'p3_deg', 'p4', 'rho_d_um') if k in g}
    if 'beam' in data and 'x-ray' not in data:
        data['x-ray'] = data['beam']

    for section, keys in _REQUIRED.items():
        if section not in data:
            raise KeyError(f"Missing TOML section [{section}] in {path}")
        for k in keys:
            if k not in data[section]:
                raise KeyError(f"Missing key '{k}' in [{section}] of {path}")

    det  = data['detector']
    xray = data['x-ray']
    fdg  = data['flat_detector_geometry']
    dist = data['distortion_correction']
    intg = {**_INTEGRATION_DEFAULTS, **data.get('integration', {})}

    # [[panel_shifts]] is an array of tables — list of dicts, may be absent
    raw_ps = data.get('panel_shifts', [])
    panel_shifts = [
        {
            'id':     int(p['id']),
            'dY':     float(p.get('dy_px',      0.0)),
            'dZ':     float(p.get('dz_px',      0.0)),
            'dLsd':   float(p.get('dlsd_um',    0.0)),
            'dP2':    float(p.get('dp2',        0.0)),
            'dTheta': float(p.get('dtheta_deg', 0.0)),
        }
        for p in raw_ps
    ]

    return dict(
        nrows        = int(det['nrows']),
        ncols        = int(det['ncols']),
        px           = float(det['px_um']),
        wavelength   = float(xray['wavelength_A']),
        lsd          = float(fdg['lsd_um']),
        bc_y         = float(fdg['bc_y_px']),
        bc_z         = float(fdg['bc_z_px']),
        tx_deg       = float(fdg['tx_deg']),
        ty_deg       = float(fdg['ty_deg']),
        tz_deg       = float(fdg['tz_deg']),
        p0           = float(dist['p0']),
        p1           = float(dist['p1']),
        p2           = float(dist['p2']),
        p3           = float(dist['p3_deg']),
        p4           = float(dist['p4']),
        rho_d        = float(dist['rho_d_um']),
        mode         = str(intg.get('mode', 'varbin')),
        tth_min      = float(intg['tth_min_deg']),
        tth_max      = float(intg['tth_max_deg']),
        tth_bin_size = float(intg['tth_bin_deg']),
        eta_min      = float(intg['eta_min_deg']),
        eta_max      = float(intg['eta_max_deg']),
        eta_bin_size = float(intg['eta_bin_deg']),
        panel_shifts = panel_shifts,
    )


def save_params(geom, path):
    """Save a geometry dict to a midas4pil TOML file.

    Parameters
    ----------
    geom : dict — same structure as returned by load_params() / make_geometry().
           panel_shifts key (list of dicts) is written as ``[[panel_shifts]]``
           array-of-tables sections appended after all other sections.
           Missing integration keys fall back to defaults.
    path : output path (will be overwritten if it exists)
    """
    g = geom

    def _f(v, fmt='.15g'):
        return format(float(v), fmt)

    intg = {**_INTEGRATION_DEFAULTS}
    for k_toml, k_geom in [('tth_min_deg', 'tth_min'), ('tth_max_deg', 'tth_max'),
                             ('tth_bin_deg', 'tth_bin_size'),
                             ('eta_min_deg', 'eta_min'), ('eta_max_deg', 'eta_max'),
                             ('eta_bin_deg', 'eta_bin_size')]:
        if k_geom in g:
            intg[k_toml] = g[k_geom]

    mode = g.get('mode', 'varbin')

    lines = [
        "# midas4pil geometry parameters",
        "# Units encoded in key names: _um=µm  _px=pixels  _deg=degrees  _A=Angstrom",
        "",
        "[detector]",
        f"nrows  = {int(g['nrows'])}   # detector rows    (NrPixelsZ)",
        f"ncols  = {int(g['ncols'])}   # detector columns (NrPixelsY)",
        f"px_um  = {_f(g['px'])}       # pixel size (µm)",
        "",
        "[x-ray]",
        f"wavelength_A = {_f(g['wavelength'])}   # X-ray wavelength (Å)",
        "",
        "[flat_detector_geometry]",
        f"lsd_um   = {_f(g['lsd'])}     # sample-to-detector distance (µm)",
        f"bc_y_px  = {_f(g['bc_y'])}    # beam-centre column from left   (= Poni2 / px)",
        f"bc_z_px  = {_f(g['bc_z'])}    # beam-centre row    from top    (= Poni1 / px)",
        f"tx_deg   = {_f(g['tx_deg'])}  # tilt about X axis (degrees)",
        f"ty_deg   = {_f(g['ty_deg'])}  # tilt about Y axis (degrees)",
        f"tz_deg   = {_f(g['tz_deg'])}  # tilt about Z axis (degrees)",
        "",
        "[distortion_correction]",
        f"p0       = {_f(g['p0'])}",
        f"p1       = {_f(g['p1'])}",
        f"p2       = {_f(g['p2'])}",
        f"p3_deg   = {_f(g['p3'])}      # distortion phase (degrees)",
        f"p4       = {_f(g['p4'])}",
        f"rho_d_um = {_f(g['rho_d'])}   # distortion reference radius (µm)",
        "",
        "[integration]",
        f'mode         = "{mode}"',
        f"tth_min_deg  = {_f(intg['tth_min_deg'])}   # 2θ lower limit (degrees)",
        f"tth_max_deg  = {_f(intg['tth_max_deg'])}   # 2θ upper limit (degrees)",
        f"tth_bin_deg  = {_f(intg['tth_bin_deg'])}   # 2θ bin size   (degrees)",
        f"eta_min_deg  = {_f(intg['eta_min_deg'])}   # η lower limit (degrees)",
        f"eta_max_deg  = {_f(intg['eta_max_deg'])}   # η upper limit (degrees)",
        f"eta_bin_deg  = {_f(intg['eta_bin_deg'])}   # η bin size   (degrees)",
    ]

    # [[panel_shifts]] array-of-tables — appended after all regular sections
    ps = g.get('panel_shifts', [])
    for p in sorted(ps, key=lambda x: x['id']):
        lines.append("")
        lines.append("[[panel_shifts]]")
        lines.append(f"id         = {int(p['id'])}")
        lines.append(f"dy_px      = {_f(p['dY'])}")
        lines.append(f"dz_px      = {_f(p['dZ'])}")
        lines.append(f"dlsd_um    = {_f(p['dLsd'])}")
        lines.append(f"dp2        = {_f(p['dP2'])}")
        lines.append(f"dtheta_deg = {_f(p['dTheta'])}")

    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ── pyFAI .poni reader ─────────────────────────────────────────────────────

def read_poni(path, px, nrows):
    """Read a pyFAI .poni file and return a partial geometry dict.

    Returns geometry values that pyFAI provides (lsd, bc_y, bc_z, tilts,
    wavelength).  Fields absent from .poni (p0–p4, rho_d, ncols,
    integration limits) must be added before calling save_params() or
    build_lut().

    Parameters
    ----------
    path  : path to the .poni file
    px    : pixel size in µm (required; not reliably stored in .poni)
    nrows : number of detector rows.  Dioptas calibrates from a
            top-bottom-flipped image, so Poni1 is measured from the
            bottom of the original frame.  bc_z = nrows - Poni1/px
            converts to row-from-top in the original image.

    Returns
    -------
    dict with keys: lsd, bc_y, bc_z, tx_deg, ty_deg, tz_deg, wavelength, px
    """
    raw = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                k, _, v = line.partition(':')
                raw[k.strip()] = v.strip()

    px_m = px * 1e-6
    _required = ('Distance', 'Poni1', 'Poni2', 'Rot1', 'Rot2', 'Wavelength')
    missing = [k for k in _required if k not in raw]
    if missing:
        raise ValueError(f".poni file is missing required keys: {missing}")
    return dict(
        lsd        = float(raw['Distance']) * 1e6,
        bc_y       = float(raw['Poni2']) / px_m,
        bc_z       = nrows - float(raw['Poni1']) / px_m,
        tz_deg     = -np.degrees(float(raw['Rot1'])),
        ty_deg     =  np.degrees(float(raw['Rot2'])),
        tx_deg     =  np.degrees(float(raw.get('Rot3', '0'))),
        wavelength = float(raw['Wavelength']) * 1e10,
        px         = float(px),
    )


def write_poni(geom, path):
    """Export a midas4pil geometry dict to pyFAI .poni format.

    Inverse of ``read_poni()``.  The resulting file can be loaded by
    pyFAI or Dioptas for visualization and cross-checking.

    Parameters
    ----------
    geom : dict — must contain at least: lsd, bc_y, bc_z, px, nrows,
           tx_deg, ty_deg, tz_deg, wavelength.
    path : output path (will be overwritten if it exists)
    """
    px_m = geom['px'] * 1e-6
    nrows = geom['nrows']

    distance   = geom['lsd'] * 1e-6
    poni1      = (nrows - geom['bc_z']) * px_m
    poni2      = geom['bc_y'] * px_m
    rot1       = -np.radians(geom.get('tz_deg', 0.0))
    rot2       =  np.radians(geom.get('ty_deg', 0.0))
    rot3       =  np.radians(geom.get('tx_deg', 0.0))
    wavelength = geom['wavelength'] * 1e-10

    from datetime import datetime
    lines = [
        "# Exported by midas4pil",
        f"# {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        "poni_version: 2",
        "Detector: Detector",
        "Detector_config: {}",
        f"Distance: {distance}",
        f"Poni1: {poni1}",
        f"Poni2: {poni2}",
        f"Rot1: {rot1}",
        f"Rot2: {rot2}",
        f"Rot3: {rot3}",
        f"Wavelength: {wavelength}",
    ]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _midas_get(raw, *keys, default=None):
    """Return the first token of the first matching key in a MIDAS raw dict."""
    for k in keys:
        if k in raw:
            return raw[k][0]
    if default is not None:
        return default
    raise KeyError(f"MIDAS params file is missing required key(s): {keys}")


def load_midas_params(path):
    """Load a MIDAS geometry_params.txt and return a midas4pil geometry dict.

    Parses the flat key-value text format used by MIDAS
    (FF_HEDM/src/MIDAS_ParamParser.c).  Key aliases handled:

    - ``Lsd`` / ``Distance`` / ``DetDist``   → lsd (µm)
    - ``px`` / ``PixelSize``                 → px (µm)
    - ``NrPixels`` (square)  or              → ncols, nrows (optional;
      ``NrPixelsY`` + ``NrPixelsZ``           GUI keeps its current values if absent)
    - ``BC Y Z`` (two values on one line)    → bc_y, bc_z
    - ``RhoD`` / ``MaxRingRad``              → rho_d (µm)

    If ``ImTransOpt 2`` is found, BC_Z is converted from the MIDAS
    top-bottom-flipped frame to midas4pil's original-TIFF frame::

        bc_z_midas4pil = nrows − midas_BC_Z

    If a ``PanelShiftsFile`` line is found, the referenced file is loaded
    with :func:`~midas4pil.panels.read_panel_shifts` (relative path is
    resolved against the directory of *path*; missing file is ignored).

    Parameters
    ----------
    path : str or Path — MIDAS geometry_params.txt to read

    Returns
    -------
    dict — same structure as :func:`load_params`, ready for
    :func:`~midas4pil.geometry.build_lut`, :func:`calibrate`, etc.
    """
    raw = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                raw[parts[0]] = parts[1:]   # last occurrence wins

    # Detector dimensions (optional — GUI keeps its current values when absent)
    try:
        nrows = int(_midas_get(raw, 'NrPixelsZ', 'NrPixels'))
        ncols = int(_midas_get(raw, 'NrPixelsY', 'NrPixels'))
    except KeyError:
        nrows = None
        ncols = None
    px    = float(_midas_get(raw, 'px', 'PixelSize'))

    # Geometry
    lsd   = float(_midas_get(raw, 'Lsd', 'Distance', 'DetDist'))

    bc_vals  = raw.get('BC', ['0', '0'])
    bc_y     = float(bc_vals[0])
    bc_z_raw = float(bc_vals[1]) if len(bc_vals) > 1 else 0.0

    # ImTransOpt 2 = flip top-bottom; MIDAS BC_Z is in the flipped frame.
    # Convert to midas4pil frame (original TIFF, row 0 = top):
    #   bc_z_midas4pil = nrows - midas_BC_Z
    # Conversion requires nrows; if absent, bc_z is used as-is.
    im_trans_vals = raw.get('ImTransOpt', ['0'])
    has_flip_z = 2 in [int(v) for v in im_trans_vals]
    bc_z = (nrows - bc_z_raw) if (has_flip_z and nrows is not None) else bc_z_raw

    ty_deg = float(_midas_get(raw, 'ty', default='0'))
    tz_deg = float(_midas_get(raw, 'tz', default='0'))
    tx_deg = float(_midas_get(raw, 'tx', default='0'))

    wavelength = float(_midas_get(raw, 'Wavelength'))

    p0    = float(_midas_get(raw, 'p0', default='0'))
    p1    = float(_midas_get(raw, 'p1', default='0'))
    p2    = float(_midas_get(raw, 'p2', default='0'))
    p3    = float(_midas_get(raw, 'p3', default='0'))
    p4    = float(_midas_get(raw, 'p4', default='0'))
    rho_d = float(_midas_get(raw, 'RhoD', 'MaxRingRad', default='217578.0'))

    # Panel shifts (optional)
    panel_shifts = []
    if 'PanelShiftsFile' in raw:
        psf = Path(raw['PanelShiftsFile'][0])
        if not psf.is_absolute():
            psf = Path(path).parent / psf
        if psf.exists():
            from .panels import read_panel_shifts
            panel_shifts = read_panel_shifts(str(psf))

    if nrows is not None and ncols is not None:
        geom = make_geometry(
            wavelength=wavelength, lsd=lsd, px=px, nrows=nrows, ncols=ncols,
            bc_y=bc_y, bc_z=bc_z,
            tx_deg=tx_deg, ty_deg=ty_deg, tz_deg=tz_deg,
            p0=p0, p1=p1, p2=p2, p3=p3, p4=p4, rho_d=rho_d,
        )
    else:
        # NrPixels keys absent — return partial dict without nrows/ncols.
        # _fill_geom_fields guards on 'nrows' in geom, so the GUI keeps
        # whatever detector dimensions the user has already set.
        geom = dict(
            px=float(px),
            wavelength=float(wavelength),
            lsd=float(lsd), bc_y=float(bc_y), bc_z=float(bc_z),
            tx_deg=float(tx_deg), ty_deg=float(ty_deg), tz_deg=float(tz_deg),
            p0=float(p0), p1=float(p1), p2=float(p2),
            p3=float(p3), p4=float(p4), rho_d=float(rho_d),
            mode='varbin',
            tth_min=2.0, tth_max=29.0, tth_bin_size=0.025,
            eta_min=-180.0, eta_max=180.0, eta_bin_size=1.0,
        )
    geom['panel_shifts'] = panel_shifts
    return geom


def save_midas_params(geom, path, panel_shifts_path=None):
    """Export a midas4pil geometry dict to MIDAS geometry_params.txt format.

    Writes the geometry in the flat key-value format read by MIDAS binary
    tools (CalibrantPanelShiftsOMP, AutoCalibrateZarr, etc.).

    BC_Z is converted to the MIDAS top-bottom-flipped frame
    (``ImTransOpt 2``):  ``midas_BC_Z = nrows - bc_z``.

    Panel topology (NPanelsY, NPanelsZ, PanelSizeY, PanelSizeZ,
    PanelGapsY, PanelGapsZ) is auto-detected from the detector shape
    via the built-in detector preset database; omitted if the shape does
    not match any preset.

    If *panel_shifts_path* is given and ``geom['panel_shifts']`` is
    non-empty, the panel shifts are written to that path and a
    ``PanelShiftsFile`` line is added to the main output file.

    The output file includes commented-out placeholders for the calibrant
    and calibration tolerance keys that MIDAS needs to run a calibration.

    Parameters
    ----------
    geom              : dict — geometry dict from :func:`load_params` /
                        :func:`make_geometry` / :func:`calibrate`.
    path              : str or Path — output file path.
    panel_shifts_path : str or Path, optional — path for the companion
                        panel_shifts.txt.  Written only when the geometry
                        contains panel shifts.
    """
    from datetime import datetime

    g = geom

    def _f(v, fmt='.10g'):
        return format(float(v), fmt)

    nrows = int(g['nrows'])
    ncols = int(g['ncols'])

    # BC_Z: convert midas4pil frame (row from top of original TIFF) to
    # MIDAS ImTransOpt-2 frame (row from top of top-bottom-flipped image)
    midas_bc_z = nrows - g['bc_z']

    # Auto-detect panel topology from detector shape
    try:
        from .gui.detectors import DETECTORS as _DETECTORS
        _topo = None
        for _spec in _DETECTORS.values():
            if _spec['nrows'] == nrows and _spec['ncols'] == ncols:
                if 'panels' in _spec:
                    _topo = _spec['panels']   # (n_y, n_z, sz_y, sz_z, g_y, g_z)
                    break
    except Exception:
        _topo = None

    lines = [
        f"# MIDAS geometry params — exported by midas4pil",
        f"# {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        "#",
        "# ── Detector ────────────────────────────────────────────────────────",
        f"NrPixelsZ  {nrows}",
        f"NrPixelsY  {ncols}",
        f"px         {_f(g['px'])}",
        "",
        "# ── Geometry ─────────────────────────────────────────────────────────",
        f"Lsd        {_f(g['lsd'])}",
        f"BC         {_f(g['bc_y'])} {_f(midas_bc_z)}",
        f"tx         {_f(g['tx_deg'])}",
        f"ty         {_f(g['ty_deg'])}",
        f"tz         {_f(g['tz_deg'])}",
        "Wedge      0",
        "ImTransOpt 2",
        "",
        "# ── X-ray ────────────────────────────────────────────────────────────",
        f"Wavelength {_f(g['wavelength'])}",
        "",
        "# ── Distortion ───────────────────────────────────────────────────────",
        f"p0  {_f(g['p0'])}",
        f"p1  {_f(g['p1'])}",
        f"p2  {_f(g['p2'])}",
        f"p3  {_f(g['p3'])}",
        f"p4  {_f(g['p4'])}",
        f"RhoD {_f(g['rho_d'])}",
    ]

    # Panel topology (auto-detected)
    if _topo is not None:
        n_y, n_z, sz_y, sz_z, gy, gz = _topo
        gy_str = ' '.join(str(gy) for _ in range(n_y - 1)) if np.isscalar(gy) \
                 else ' '.join(str(v) for v in gy)
        gz_str = ' '.join(str(gz) for _ in range(n_z - 1)) if np.isscalar(gz) \
                 else ' '.join(str(v) for v in gz)
        lines += [
            "",
            "# ── Panel topology ───────────────────────────────────────────────────",
            f"NPanelsY   {n_y}",
            f"NPanelsZ   {n_z}",
            f"PanelSizeY {sz_y}",
            f"PanelSizeZ {sz_z}",
            f"PanelGapsY {gy_str}",
            f"PanelGapsZ {gz_str}",
            "FixPanelID 0",
        ]

    # Panel shifts file reference
    ps = g.get('panel_shifts', [])
    if ps and panel_shifts_path is not None:
        from .panels import save_panel_shifts
        save_panel_shifts(ps, str(panel_shifts_path))
        lines += [
            "",
            f"PanelShiftsFile {panel_shifts_path}",
        ]

    lines += [
        "",
        "# ── Calibrant (uncomment and fill in) ────────────────────────────────",
        "# SpaceGroup     225",
        "# LatticeConstant 5.4131 5.4131 5.4131 90 90 90",
        "# RingThresh 1 100",
        "# RingThresh 2 100",
        "",
        "# ── Calibration settings (uncomment and adjust) ──────────────────────",
        "# Width      1500",
        "# EtaBinSize 5.0",
        "# RBinDivisions 4",
        "# tolTilts   3",
        "# tolBC      20",
        "# tolLsd     25000",
        "# tolP       2E-3",
        "# tolP3      45",
        "# tolShifts  1.0",
        "# tolRotation 0.1",
        "# tolLsdPanel 100",
        "# tolP2Panel  0.0001",
    ]

    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def make_geometry(wavelength, lsd, px, nrows, ncols,
                  bc_y=None, bc_z=None,
                  tx_deg=0.0, ty_deg=0.0, tz_deg=0.0,
                  p0=0.0, p1=0.0, p2=0.0, p3=0.0, p4=0.0,
                  rho_d=217578.0,
                  mode='varbin',
                  tth_min=2.0, tth_max=29.0, tth_bin_size=0.025,
                  eta_min=-180.0, eta_max=180.0, eta_bin_size=1.0):
    """Build a complete geometry dict from user-supplied parameters.

    This is the primary entry point for starting a calibration from
    scratch, without a pyFAI .poni file.  Beam centre defaults to the
    detector centre if not provided.

    Parameters
    ----------
    wavelength : X-ray wavelength (angstrom)
    lsd        : sample-to-detector distance (um)
    px         : pixel size (um)
    nrows      : detector rows
    ncols      : detector columns
    bc_y       : beam-centre column from left (pixels); default ncols/2
    bc_z       : beam-centre row from top (pixels); default nrows/2
    (remaining : tilts, distortion, integration limits — sensible defaults)

    Returns
    -------
    dict — same structure as load_params(), ready for save_params() or calibrate()
    """
    if bc_y is None:
        bc_y = ncols / 2.0
    if bc_z is None:
        bc_z = nrows / 2.0

    return dict(
        nrows=int(nrows), ncols=int(ncols), px=float(px),
        wavelength=float(wavelength),
        lsd=float(lsd), bc_y=float(bc_y), bc_z=float(bc_z),
        tx_deg=float(tx_deg), ty_deg=float(ty_deg), tz_deg=float(tz_deg),
        p0=float(p0), p1=float(p1), p2=float(p2),
        p3=float(p3), p4=float(p4), rho_d=float(rho_d),
        mode=str(mode),
        tth_min=float(tth_min), tth_max=float(tth_max),
        tth_bin_size=float(tth_bin_size),
        eta_min=float(eta_min), eta_max=float(eta_max),
        eta_bin_size=float(eta_bin_size),
        panel_shifts=[],
    )


# ── Image and mask loaders ─────────────────────────────────────────────────

def load_tiff(path):
    """Load a TIFF detector image.

    Dtype is preserved as-is (uint16, uint32, int32, float32, …);
    downstream functions cast to float64 as needed.

    Returns
    -------
    numpy array, shape (nrows, ncols)
    """
    import tifffile
    return tifffile.imread(str(path))


def load_mask(path, panel_map=None, image=None):
    """Load a bad-pixel mask TIFF.

    Convention: 1 = bad pixel, 0 = good pixel (pyFAI / Dioptas).
    Threshold at 0.5 so both integer (0/1) and float (0.0/1.0) masks work.

    If *panel_map* (and optionally *image*) is provided, the mask
    orientation is auto-detected and corrected so it matches the
    original image frame.  This removes the need for manual
    ``np.flipud()`` when the mask was created by Dioptas.

    Parameters
    ----------
    path      : path to mask TIFF
    panel_map : int32 array (nrows, ncols) from make_panel_id_map(),
                optional.  Enables automatic orientation detection.
    image     : 2-D array, optional.  Detector image in the original
                TIFF frame.  Provides robust orientation detection via
                dead-pixel correlation.  Required for detectors with
                vertically symmetric gap layouts (e.g. Pilatus 2M).

    Returns
    -------
    bool array, shape (nrows, ncols), True where pixel is bad,
    oriented to match the original image / panel_map frame.
    """
    import tifffile
    raw = tifffile.imread(str(path))
    mask = (raw > _MASK_LOAD_THRESHOLD).astype(bool)

    if panel_map is not None:
        mask, _ = orient_mask(mask, panel_map, image)

    return mask


def orient_mask(mask, panel_map, image=None):
    """Auto-detect and correct mask orientation relative to the image frame.

    Masks created by Dioptas are in a top-bottom-flipped coordinate
    frame.  This function detects whether the mask needs flipping by
    comparing it against the detector's structural features.

    Two detection strategies, in order of preference:

    1. **Image-based** (when *image* is provided): count non-gap dead
       pixels (image <= 0) that overlap with mask bad-pixels.  The
       orientation with higher overlap is correct.  Works for all
       detectors, including those with vertically symmetric gap layouts.

    2. **Gap-row density** (panel_map only): check whether fully-masked
       rows align with known gap positions.  Fails for symmetric
       detectors (e.g. Pilatus 2M) --- issues a warning and returns
       the mask unchanged.

    Parameters
    ----------
    mask      : bool array (nrows, ncols), True = bad pixel
    panel_map : int32 array (nrows, ncols), 0 = gap pixel
    image     : 2-D array, optional.  Original detector image.

    Returns
    -------
    mask_oriented : bool array, corrected to match the panel_map frame
    was_flipped   : bool, True if the input mask was flipped
    """
    import warnings

    if mask.shape != panel_map.shape:
        raise ValueError(
            f"mask shape {mask.shape} != panel_map shape {panel_map.shape}")

    if image is not None:
        if image.shape != mask.shape:
            raise ValueError(
                f"image shape {image.shape} != mask shape {mask.shape}")
        return _orient_by_image(mask, panel_map, image)

    return _orient_by_gaps(mask, panel_map)


def _orient_by_image(mask, panel_map, image):
    """Orient mask using dead-pixel correlation with the image."""
    gap = (panel_map == 0)
    dead = (image <= 0) & ~gap

    n_dead = dead.sum()
    if n_dead < 10:
        return mask, False

    score_orig = (mask & dead).sum()
    score_flip = (np.flipud(mask) & dead).sum()

    if score_flip > score_orig:
        return np.flipud(mask), True
    return mask, False


def _orient_by_gaps(mask, panel_map):
    """Orient mask using gap-row density heuristic."""
    import warnings

    gap_rows = np.where(np.all(panel_map == 0, axis=1))[0]
    if len(gap_rows) == 0:
        return mask, False

    nrows = mask.shape[0]
    gap_rows_flipped = nrows - 1 - gap_rows

    if np.array_equal(np.sort(gap_rows), np.sort(gap_rows_flipped)):
        warnings.warn(
            "orient_mask: gap layout is vertically symmetric; "
            "cannot determine mask orientation from panel_map alone. "
            "Pass image= for reliable detection.",
            stacklevel=3)
        return mask, False

    density_orig = mask[gap_rows, :].mean()
    density_flip = mask[gap_rows_flipped, :].mean()

    if density_flip > density_orig:
        return np.flipud(mask), True
    return mask, False


# ── Auto-mask ─────────────────────────────────────────────────────────────

def auto_mask(image, panel_map=None, sat_val=None):
    """Generate a bad-pixel mask from image data and detector layout.

    Masks three classes of pixels:

    1. **Gap pixels** — where ``panel_map == 0`` (inter-module dead zones).
    2. **Dead / zero pixels** — where ``image <= 0``.
    3. **Saturated pixels** — where ``image >= sat_val``.  Defaults to the
       maximum representable value for integer dtypes (e.g. 2**32-1 for
       uint32).  Pass ``sat_val=None`` to skip saturation masking for
       float images.

    Parameters
    ----------
    image     : 2-D array, detector image
    panel_map : int32 array (nrows, ncols), optional.  Gap pixels
                (panel_map == 0) are masked.
    sat_val   : float, optional.  Pixels >= this value are masked.
                Defaults to dtype max for integer arrays.

    Returns
    -------
    bool array (nrows, ncols), True = bad pixel.
    """
    mask = np.zeros(image.shape, dtype=bool)

    if panel_map is not None:
        mask |= (panel_map == 0)

    mask |= (image <= 0)

    if sat_val is None and np.issubdtype(image.dtype, np.integer):
        sat_val = float(np.iinfo(image.dtype).max)
    if sat_val is not None:
        mask |= (image >= sat_val)

    return mask


# ── Mask save ────────────────────────────────────────────────────────────────

def save_mask(mask, path):
    """Save a boolean mask as a uint8 TIFF (1 = bad pixel, 0 = good).

    Parameters
    ----------
    mask : bool array (nrows, ncols)
    path : str or Path
    """
    import tifffile
    tifffile.imwrite(str(path), mask.astype(np.uint8))
