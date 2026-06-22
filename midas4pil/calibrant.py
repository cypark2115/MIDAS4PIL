"""
Calibrant ring table readers.

Reads crystallographic data from JCPDS or CIF files and computes
expected diffraction ring positions (2theta) for a given wavelength.
This replaces the MIDAS ``GetHKLList`` + ``hkls.csv`` workflow with a
single Python call.

Supported formats
-----------------
- **JCPDS** (version 4 / 5.1 / legacy): d-spacings pre-tabulated.
- **CIF** (via gemmi): unit cell + space group → d-spacings generated
  from symmetry-unique Miller indices.
"""

from pathlib import Path

import numpy as np


# ── JCPDS reader ──────────────────────────────────────────────────────────

def read_jcpds(path):
    """Parse a JCPDS file (.jcpds).

    Handles VERSION 4, 5.1, and legacy (no VERSION header) formats.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict with keys:
        comment    : str
        symmetry   : str or None
        a          : float (lattice parameter, angstrom)
        k0, k0p    : float (bulk modulus and derivative)
        reflections: list of dict, each with keys
                     'd' (angstrom), 'intensity', 'h', 'k', 'l'
    """
    path = Path(path)
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()

    # Detect format: VERSION header present?
    has_version = any(line.strip().upper().startswith('VERSION') for line in lines)

    if has_version:
        return _parse_jcpds_versioned(lines)
    else:
        return _parse_jcpds_legacy(lines)


def _parse_jcpds_versioned(lines):
    """Parse VERSION 4 / 5.1 JCPDS files."""
    info = {
        'comment': '',
        'symmetry': None,
        'a': 0.0,
        'k0': 0.0,
        'k0p': 0.0,
        'reflections': [],
    }

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.upper().startswith('COMMENT:'):
            line_text = stripped.split(':', 1)[1].strip()
            info['comment'] = (info['comment'] + ' ' + line_text).strip()
        elif stripped.upper().startswith('SYMMETRY:'):
            info['symmetry'] = stripped.split(':', 1)[1].strip().upper()
        elif stripped.upper().startswith('A:'):
            info['a'] = float(stripped.split(':', 1)[1].strip())
        elif stripped.upper().startswith('K0:'):
            info['k0'] = float(stripped.split(':', 1)[1].strip())
        elif stripped.upper().startswith('K0P:'):
            info['k0p'] = float(stripped.split(':', 1)[1].strip())
        elif stripped.upper().startswith('DIHKL:'):
            parts = stripped.split(':',  1)[1].split()
            if len(parts) >= 5:
                info['reflections'].append({
                    'd': float(parts[0]),
                    'intensity': float(parts[1]),
                    'h': int(float(parts[2])),
                    'k': int(float(parts[3])),
                    'l': int(float(parts[4])),
                })

    return info


def _parse_jcpds_legacy(lines):
    """Parse legacy JCPDS files (no VERSION header).

    Format:
        Line 1: comment
        Line 2: n_phases  a  k0  k0p  [extra]
        Line 3: column headers (ignored)
        Lines 4+: d  I  h  k  l
    """
    info = {
        'comment': lines[0].strip() if lines else '',
        'symmetry': None,
        'a': 0.0,
        'k0': 0.0,
        'k0p': 0.0,
        'reflections': [],
    }

    if len(lines) >= 2:
        parts = lines[1].split()
        if len(parts) >= 4:
            info['a']   = float(parts[1])
            info['k0']  = float(parts[2])
            info['k0p'] = float(parts[3])

    # Data lines start after the header line (line 3)
    for line in lines[3:]:
        parts = line.split()
        if len(parts) >= 5:
            try:
                info['reflections'].append({
                    'd': float(parts[0]),
                    'intensity': float(parts[1]),
                    'h': int(float(parts[2])),
                    'k': int(float(parts[3])),
                    'l': int(float(parts[4])),
                })
            except ValueError:
                continue

    return info


# ── CIF reader (gemmi) ───────────────────────────────────────────────────

def read_cif(path, d_min=0.5):
    """Parse a CIF file and generate reflections via gemmi.

    Uses the unit cell and space group from the CIF to enumerate
    symmetry-unique Miller indices down to d_min.

    Parameters
    ----------
    path  : str or Path
    d_min : minimum d-spacing (angstrom) for reflection generation.
            Default 0.5 A covers 2theta up to ~50 deg at typical
            synchrotron wavelengths.

    Returns
    -------
    dict with same structure as read_jcpds():
        comment, symmetry, a, k0, k0p, reflections
    """
    import gemmi

    path = Path(path)
    st = gemmi.read_small_structure(str(path))
    cell = st.cell
    sg = gemmi.SpaceGroup(st.spacegroup_hm)

    miller = gemmi.make_miller_array(cell, sg, d_min)

    reflections = []
    for hkl in miller:
        d = cell.calculate_d(hkl)
        reflections.append({
            'd': d,
            'intensity': 100.0,  # CIF typically lacks powder intensities
            'h': int(hkl[0]),
            'k': int(hkl[1]),
            'l': int(hkl[2]),
        })

    # Sort by d-spacing descending (largest d = lowest 2theta first)
    reflections.sort(key=lambda r: r['d'], reverse=True)

    return {
        'comment': st.name,
        'symmetry': st.spacegroup_hm.upper(),
        'a': cell.a,
        'k0': 0.0,
        'k0p': 0.0,
        'reflections': reflections,
    }


# ── Ring table ────────────────────────────────────────────────────────────

def ring_table(calibrant, wavelength, tth_max=None):
    """Compute diffraction ring positions from calibrant data.

    Parameters
    ----------
    calibrant  : dict from read_jcpds() or read_cif()
    wavelength : X-ray wavelength (angstrom)
    tth_max    : upper 2theta limit (degrees); None = no limit

    Returns
    -------
    list of dict, sorted by ascending 2theta, each with keys:
        'd' (angstrom), 'tth' (degrees), 'intensity', 'h', 'k', 'l'
    """
    rings = []
    for ref in calibrant['reflections']:
        d = ref['d']
        arg = wavelength / (2.0 * d)
        if abs(arg) > 1.0:
            continue  # wavelength too long for this reflection
        tth = 2.0 * np.degrees(np.arcsin(arg))
        if tth_max is not None and tth > tth_max:
            continue
        rings.append({
            'd': d,
            'tth': tth,
            'intensity': ref['intensity'],
            'h': ref['h'],
            'k': ref['k'],
            'l': ref['l'],
        })

    rings.sort(key=lambda r: r['tth'])
    return rings


# ── Convenience entry point ───────────────────────────────────────────────

def _resolve_calibrant_path(name):
    """Resolve a calibrant filename to an absolute path.

    If *name* is already an existing file, returns it as-is.
    Otherwise, searches the bundled ``calibrants/`` directory
    (shipped with the package) recursively for a matching filename.

    Parameters
    ----------
    name : str or Path

    Returns
    -------
    Path to the calibrant file.

    Raises
    ------
    FileNotFoundError if the calibrant cannot be found.
    """
    p = Path(name)
    if p.exists():
        return p

    # If no extension given, assume .jcpds
    search_name = p.name if p.suffix else p.name + '.jcpds'

    # Search bundled calibrants directory
    pkg_dir = Path(__file__).resolve().parent
    cal_dir = pkg_dir / 'calibrants'
    if cal_dir.is_dir():
        for hit in cal_dir.rglob(search_name):
            return hit

    raise FileNotFoundError(
        f"Calibrant '{name}' not found.\n"
        f"  Searched: current directory, midas4pil/calibrants/\n"
        f"  Bundled calibrants are in midas4pil/calibrants/ "
        f"(NIST/, metals/, oxides/, halides/, gases/, carbides/, carbon/)."
    )


def load_calibrant(path, wavelength, tth_max=None, d_min=0.5):
    """Read a calibrant file and return the ring table.

    Auto-detects format by file extension (.jcpds or .cif).
    If *path* is a bare filename (e.g. ``"CeO2.jcpds"``), the bundled
    ``calibrants/`` directory is searched automatically.

    Parameters
    ----------
    path       : str or Path to calibrant file (or bare name for bundled)
    wavelength : X-ray wavelength (angstrom)
    tth_max    : upper 2theta limit (degrees); None = no limit
    d_min      : minimum d-spacing for CIF reflection generation (angstrom)

    Returns
    -------
    list of dict — see ring_table()
    """
    path = _resolve_calibrant_path(path)
    ext = path.suffix.lower()

    if ext == '.jcpds':
        cal = read_jcpds(path)
    elif ext == '.cif':
        cal = read_cif(path, d_min=d_min)
    else:
        raise ValueError(f"Unknown calibrant file extension: {ext!r}. "
                         f"Expected .jcpds or .cif")

    return ring_table(cal, wavelength, tth_max=tth_max)
