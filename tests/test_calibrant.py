# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""
Tests for midas4pil.calibrant

Covers:
- read_jcpds: versioned (CeO2) and legacy (fe8si_hcp) formats
- read_cif: gemmi-based CIF parsing
- ring_table: Bragg's law, filtering, sorting
- load_calibrant: auto-detection by extension
"""

from pathlib import Path

import numpy as np
import pytest
from midas4pil.calibrant import read_jcpds, read_cif, ring_table, load_calibrant


# ── Paths to test data ───────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
_PKG  = _ROOT / "midas4pil"
_CAL  = _PKG / "calibrants"

# Bundled calibrants (new curated directory)
CEO2_JCPDS   = _CAL / "NIST" / "CeO2.jcpds"

# Legacy files (preserved at repo root for format-coverage tests)
_LEGACY_JCPDS = _ROOT / "_legacy_JCPDS"
_LEGACY_CIF   = _ROOT / "_legacy_CIF"
TIB2_JCPDS   = _LEGACY_JCPDS / "tib2.jcpds"
FE8SI_JCPDS  = _LEGACY_JCPDS / "fe8si_hcp.jcpds"
ZRC_CIF      = _LEGACY_CIF / "Carbides" / "ZrC.cif"

WAVELENGTH = 0.42460  # angstrom, 29.2 keV


# ── read_jcpds ───────────────────────────────────────────────────────────

def test_read_jcpds_ceo2():
    """Parse CeO2.jcpds — 20 reflections (extended to d=0.72 Å), correct d-spacings."""
    cal = read_jcpds(CEO2_JCPDS)
    assert cal['symmetry'] == 'CUBIC'
    assert len(cal['reflections']) == 20
    d_values = [r['d'] for r in cal['reflections']]
    assert d_values[0] == pytest.approx(3.124, abs=0.001)
    assert d_values[1] == pytest.approx(2.706, abs=0.001)
    # First reflection is (111)
    r0 = cal['reflections'][0]
    assert (r0['h'], r0['k'], r0['l']) == (1, 1, 1)
    assert r0['intensity'] == pytest.approx(100.0)


def test_read_jcpds_hexagonal():
    """Parse tib2.jcpds — hexagonal symmetry."""
    cal = read_jcpds(TIB2_JCPDS)
    assert cal['symmetry'] == 'HEXAGONAL'
    assert cal['a'] == pytest.approx(3.03034, abs=0.001)
    assert len(cal['reflections']) > 10


def test_read_jcpds_legacy():
    """Parse fe8si_hcp.jcpds — legacy format without VERSION header."""
    cal = read_jcpds(FE8SI_JCPDS)
    assert 'iron' in cal['comment'].lower() or 'hcp' in cal['comment'].lower()
    assert len(cal['reflections']) == 6
    # First reflection: d=2.12
    assert cal['reflections'][0]['d'] == pytest.approx(2.12, abs=0.01)
    assert (cal['reflections'][0]['h'], cal['reflections'][0]['k'],
            cal['reflections'][0]['l']) == (1, 0, 0)


def test_read_jcpds_comment():
    cal = read_jcpds(CEO2_JCPDS)
    assert 'CeO2' in cal['comment']


# ── read_cif ─────────────────────────────────────────────────────────────

def test_read_cif_zrc():
    """Parse ZrC.cif — cubic Fm-3m, check reflections generated."""
    cal = read_cif(ZRC_CIF, d_min=0.8)
    assert cal['a'] == pytest.approx(4.691, abs=0.01)
    assert 'F M -3 M' in cal['symmetry'] or 'FM-3M' in cal['symmetry']
    assert len(cal['reflections']) > 5
    # All d-spacings should be >= d_min
    for r in cal['reflections']:
        assert r['d'] >= 0.8 - 1e-6


def test_read_cif_reflections_sorted():
    """CIF reflections should be sorted by d descending."""
    cal = read_cif(ZRC_CIF, d_min=0.8)
    d_values = [r['d'] for r in cal['reflections']]
    assert d_values == sorted(d_values, reverse=True)


# ── ring_table ───────────────────────────────────────────────────────────

def test_ring_table_ceo2():
    """CeO2 at 29.2 keV: first ring near 7.80 deg, 20 total reflections."""
    cal = read_jcpds(CEO2_JCPDS)
    rings = ring_table(cal, WAVELENGTH)
    assert len(rings) == 20
    # First ring: (111) near 7.80 deg
    assert rings[0]['tth'] == pytest.approx(7.80, abs=0.05)
    assert (rings[0]['h'], rings[0]['k'], rings[0]['l']) == (1, 1, 1)


def test_ring_table_sorted():
    """Ring table must be sorted by ascending 2theta."""
    cal = read_jcpds(CEO2_JCPDS)
    rings = ring_table(cal, WAVELENGTH)
    tth_values = [r['tth'] for r in rings]
    assert tth_values == sorted(tth_values)


def test_ring_table_tth_max_filter():
    """tth_max should exclude rings above the limit."""
    cal = read_jcpds(CEO2_JCPDS)
    rings_all = ring_table(cal, WAVELENGTH)
    rings_cut = ring_table(cal, WAVELENGTH, tth_max=15.0)
    assert len(rings_cut) < len(rings_all)
    for r in rings_cut:
        assert r['tth'] <= 15.0


def test_ring_table_wavelength_filter():
    """Very long wavelength should exclude high-d reflections."""
    cal = read_jcpds(CEO2_JCPDS)
    # Wavelength = 10 A — only reflections with d > 5 A would survive
    rings = ring_table(cal, wavelength=10.0)
    for r in rings:
        assert r['d'] >= 5.0


def test_ring_table_bragg_law():
    """Verify 2theta matches Bragg's law: 2theta = 2*arcsin(lambda/2d)."""
    cal = read_jcpds(CEO2_JCPDS)
    rings = ring_table(cal, WAVELENGTH)
    for r in rings:
        expected = 2.0 * np.degrees(np.arcsin(WAVELENGTH / (2.0 * r['d'])))
        assert r['tth'] == pytest.approx(expected, abs=1e-10)


# ── load_calibrant ───────────────────────────────────────────────────────

def test_load_calibrant_jcpds():
    """Auto-detect .jcpds extension."""
    rings = load_calibrant(CEO2_JCPDS, WAVELENGTH, tth_max=29.0)
    assert len(rings) > 0
    assert all('tth' in r for r in rings)


def test_load_calibrant_cif():
    """Auto-detect .cif extension."""
    rings = load_calibrant(ZRC_CIF, WAVELENGTH, tth_max=29.0)
    assert len(rings) > 0
    assert all('tth' in r for r in rings)


def test_load_calibrant_unknown_extension():
    """Unknown extension on a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_calibrant("foo.xyz", WAVELENGTH)


# ── bare-name resolution (bundled calibrants/) ─────────────────────────

def test_load_calibrant_bare_name_ceo2():
    """Bare filename should resolve to bundled calibrants/NIST/CeO2.jcpds."""
    rings = load_calibrant("CeO2.jcpds", WAVELENGTH, tth_max=29.0)
    assert len(rings) == 14  # 14 of 20 reflections visible at 29.2 keV below 29°
    assert rings[0]['tth'] == pytest.approx(7.80, abs=0.05)


def test_load_calibrant_bare_name_lab6():
    """Bare filename should resolve to bundled calibrants/NIST/LaB6.jcpds."""
    rings = load_calibrant("LaB6.jcpds", WAVELENGTH, tth_max=29.0)
    assert len(rings) > 0
    # LaB6 first reflection (100): d=4.1568 → 2θ ≈ 5.86° at 0.42460 Å
    assert rings[0]['tth'] == pytest.approx(5.86, abs=0.1)


def test_load_calibrant_bare_name_not_found():
    """Non-existent bare name should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_calibrant("NonExistent_Foo.jcpds", WAVELENGTH)
