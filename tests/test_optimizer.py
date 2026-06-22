"""
Tests for midas4pil.optimizer

Covers:
- find_ring_pixels: assignment of pixels to calibrant rings
- weighted_mean_positions: intensity-weighted averaging
- strain_cost: zero for perfect geometry, nonzero for shifted BC
- calibrate: recovery of known shifts on synthetic data
- panel_shifts TOML roundtrip: embedded in .toml via save_params/load_params
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from midas4pil.geometry import build_lut, build_tilt_matrix, r_to_tth
from midas4pil.panels import make_panel_id_map
from midas4pil.optimizer import (find_ring_pixels, weighted_mean_positions,
                                  strain_cost, calibrate,
                                  _panel_centres, _unpack_params)


# ── Shared fixtures ───────────────────────────────────────────────────────

# Synthetic detector large enough for rings at 5-12 deg
NROWS, NCOLS = 500, 500
PX = 172.0     # um
LSD = 349510.0 # um
BC_Y = 250.0   # centre of detector
BC_Z = 250.0

GEOM = dict(
    nrows=NROWS, ncols=NCOLS, px=PX, lsd=LSD,
    bc_y=BC_Y, bc_z=BC_Z,
    tx_deg=0.0, ty_deg=0.0, tz_deg=0.0,
    p0=0.0, p1=0.0, p2=0.0, p3=0.0, p4=0.0,
    rho_d=217578.0,
    wavelength=0.42460,
    tth_min=2.0, tth_max=20.0, tth_bin_size=0.025,
    eta_min=-180.0, eta_max=180.0, eta_bin_size=1.0,
)


def _build_lut():
    return build_lut(NROWS, NCOLS, BC_Y, BC_Z, LSD, PX,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 217578.0)


def _build_lut_from_geom(geom):
    return build_lut(geom['nrows'], geom['ncols'], geom['bc_y'], geom['bc_z'],
                     geom['lsd'], geom['px'],
                     geom['tx_deg'], geom['ty_deg'], geom['tz_deg'],
                     geom['p0'], geom['p1'], geom['p2'], geom['p3'],
                     geom['p4'], geom['rho_d'])


def _synthetic_calibrant_image(tth_lut, ring_tth_values, width=0.03):
    """Create a synthetic image with Gaussian rings at known 2theta positions."""
    image = np.ones((NROWS, NCOLS)) * 10.0  # background
    for tth_ring in ring_tth_values:
        ring = 1000.0 * np.exp(-0.5 * ((tth_lut - tth_ring) / width) ** 2)
        image += ring
    return image


def _simple_rings():
    """A few synthetic calibrant rings within the 500x500 detector range (~10 deg)."""
    return [
        {'tth': 3.0, 'd': 1.0, 'intensity': 100.0, 'h': 1, 'k': 1, 'l': 1},
        {'tth': 5.0, 'd': 0.8, 'intensity': 80.0,  'h': 2, 'k': 0, 'l': 0},
        {'tth': 7.0, 'd': 0.6, 'intensity': 60.0,  'h': 2, 'k': 2, 'l': 0},
    ]


# ── find_ring_pixels ─────────────────────────────────────────────────────

def test_find_ring_pixels_basic():
    """Pixels near calibrant rings should be assigned."""
    tth_lut, eta_lut = _build_lut()
    mask = np.zeros((NROWS, NCOLS), dtype=bool)
    rings = _simple_rings()

    bins = find_ring_pixels(tth_lut, eta_lut, mask, rings,
                            tth_tol=0.1, eta_bin_size=10.0)
    assert len(bins) > 0
    # Each bin should have pixel positions
    for b in bins:
        assert len(b['pixel_rows']) > 0
        assert len(b['pixel_cols']) == len(b['pixel_rows'])
        assert b['ring_tth'] in [3.0, 5.0, 7.0]


def test_find_ring_pixels_mask():
    """Masked pixels should be excluded."""
    tth_lut, eta_lut = _build_lut()
    mask = np.ones((NROWS, NCOLS), dtype=bool)  # all bad
    rings = _simple_rings()

    bins = find_ring_pixels(tth_lut, eta_lut, mask, rings, tth_tol=0.1)
    assert len(bins) == 0


# ── weighted_mean_positions ──────────────────────────────────────────────

def test_weighted_mean_positions():
    """Weighted mean should reflect intensity distribution."""
    tth_lut, eta_lut = _build_lut()
    rings = _simple_rings()
    image = _synthetic_calibrant_image(tth_lut, [r['tth'] for r in rings])
    mask = np.zeros((NROWS, NCOLS), dtype=bool)

    bins = find_ring_pixels(tth_lut, eta_lut, mask, rings,
                            tth_tol=0.1, eta_bin_size=10.0)
    YMean, ZMean, IdealTtheta = weighted_mean_positions(image, bins)

    assert len(YMean) == len(bins)
    assert len(IdealTtheta) == len(bins)
    # All positions should be within detector
    assert np.all(YMean >= 0) and np.all(YMean < NCOLS)
    assert np.all(ZMean >= 0) and np.all(ZMean < NROWS)


# ── strain_cost ──────────────────────────────────────────────────────────

def test_strain_cost_perfect():
    """Perfect geometry should give near-zero strain."""
    tth_lut, eta_lut = _build_lut()
    rings = _simple_rings()
    image = _synthetic_calibrant_image(tth_lut, [r['tth'] for r in rings])
    mask = np.zeros((NROWS, NCOLS), dtype=bool)

    bins = find_ring_pixels(tth_lut, eta_lut, mask, rings,
                            tth_tol=0.1, eta_bin_size=10.0)
    YMean, ZMean, IdealTtheta = weighted_mean_positions(image, bins)

    panel_map = np.ones((NROWS, NCOLS), dtype=np.int32)  # single panel
    centres = {1: (NCOLS / 2.0, NROWS / 2.0)}
    fixed_geom = {'tx': 0.0, 'px': PX, 'rho_d': 217578.0}

    x = np.array([LSD, BC_Y, BC_Z, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0])

    cost = strain_cost(x, YMean, ZMean, IdealTtheta, fixed_geom,
                       panel_map, centres, 1, 1, (False, False, False))
    # Cost should be very small for perfect geometry
    mean_strain = cost / len(YMean)
    # Discrete pixel grid limits precision to ~px/Lsd ≈ 500 ppm
    assert mean_strain < 2e-3, f"Mean |strain| = {mean_strain:.6f}"


def test_strain_cost_shifted_bc():
    """Shifted beam centre should increase cost."""
    tth_lut, eta_lut = _build_lut()
    rings = _simple_rings()
    image = _synthetic_calibrant_image(tth_lut, [r['tth'] for r in rings])
    mask = np.zeros((NROWS, NCOLS), dtype=bool)

    bins = find_ring_pixels(tth_lut, eta_lut, mask, rings,
                            tth_tol=0.1, eta_bin_size=10.0)
    YMean, ZMean, IdealTtheta = weighted_mean_positions(image, bins)

    panel_map = np.ones((NROWS, NCOLS), dtype=np.int32)
    centres = {1: (NCOLS / 2.0, NROWS / 2.0)}
    fixed_geom = {'tx': 0.0, 'px': PX, 'rho_d': 217578.0}

    # Perfect geometry
    x_perfect = np.array([LSD, BC_Y, BC_Z, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0])
    cost_perfect = strain_cost(x_perfect, YMean, ZMean, IdealTtheta,
                               fixed_geom, panel_map, centres, 1, 1,
                               (False, False, False))

    # Shifted BC
    x_shifted = np.array([LSD, BC_Y + 5.0, BC_Z, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0])
    cost_shifted = strain_cost(x_shifted, YMean, ZMean, IdealTtheta,
                               fixed_geom, panel_map, centres, 1, 1,
                               (False, False, False))

    assert cost_shifted > cost_perfect * 2.0


# ── calibrate (global geometry only, single panel) ───────────────────────

def test_calibrate_global_only():
    """Optimizer should refine global geometry on synthetic data."""
    tth_lut, eta_lut = _build_lut()
    rings = _simple_rings()
    image = _synthetic_calibrant_image(tth_lut, [r['tth'] for r in rings])
    mask = np.zeros((NROWS, NCOLS), dtype=bool)
    panel_map = np.ones((NROWS, NCOLS), dtype=np.int32)

    # Start with slightly perturbed geometry
    geom_perturbed = dict(GEOM)
    geom_perturbed['bc_y'] = BC_Y + 1.0  # shift by 1 pixel

    result = calibrate(image, mask, geom_perturbed, panel_map, rings,
                       optimize_shifts=False,
                       tol_bc=3.0, tol_lsd=100.0, tol_tilts=0.1,
                       n_iterations=2, verbose=False)

    # BC should be recovered close to true value
    assert abs(result['geom']['bc_y'] - BC_Y) < 0.5, \
        f"bc_y = {result['geom']['bc_y']:.3f}, expected ~{BC_Y}"
    assert result['mean_strain'] < 1e-3


# ── calibrate with panels ────────────────────────────────────────────────

def test_calibrate_with_panels():
    """Optimizer should detect per-panel shift on a 2-panel detector."""
    # Create a 2-panel detector (top half = panel 1, bottom half = panel 2)
    panel_map = np.ones((NROWS, NCOLS), dtype=np.int32)
    panel_map[NROWS // 2:, :] = 2

    tth_lut, eta_lut = _build_lut()
    rings = _simple_rings()
    image = _synthetic_calibrant_image(tth_lut, [r['tth'] for r in rings])
    mask = np.zeros((NROWS, NCOLS), dtype=bool)

    result = calibrate(image, mask, GEOM, panel_map, rings,
                       fix_panel=1,
                       optimize_shifts=True,
                       tol_shifts=3.0, tol_bc=2.0, tol_lsd=100.0,
                       tol_tilts=0.1,
                       n_iterations=1, verbose=False)

    assert 'panel_shifts' in result
    assert len(result['panel_shifts']) == 2
    assert result['n_points'] > 0


# ── panel_shifts TOML roundtrip ───────────────────────────────────────────

def test_panel_shifts_toml_roundtrip(tmp_path):
    """Panel shifts embedded in .toml survive save_params → load_params."""
    from midas4pil.io import make_geometry, save_params, load_params
    shifts = [
        {'id': 1, 'dY': 0.0, 'dZ': 0.0, 'dLsd': 0.0, 'dP2': 0.0, 'dTheta': 0.0},
        {'id': 2, 'dY': 1.5, 'dZ': -0.3, 'dLsd': 10.0, 'dP2': 0.001, 'dTheta': 0.05},
    ]
    g = make_geometry(wavelength=0.42460, lsd=350000, px=172,
                      nrows=1679, ncols=1475)
    g['panel_shifts'] = shifts

    path = tmp_path / "geom.toml"
    save_params(g, path)
    g2 = load_params(path)

    assert len(g2['panel_shifts']) == 2
    ps = {p['id']: p for p in g2['panel_shifts']}
    assert ps[2]['dY']     == pytest.approx(1.5,   abs=1e-9)
    assert ps[2]['dZ']     == pytest.approx(-0.3,  abs=1e-9)
    assert ps[2]['dLsd']   == pytest.approx(10.0,  abs=1e-9)
    assert ps[2]['dTheta'] == pytest.approx(0.05,  abs=1e-9)
    assert ps[1]['dY']     == pytest.approx(0.0,   abs=1e-9)


# ── _panel_centres ───────────────────────────────────────────────────────

def test_panel_centres():
    """Panel centres should be at the geometric middle of each panel."""
    panel_map = np.zeros((100, 100), dtype=np.int32)
    panel_map[:50, :50] = 1   # top-left
    panel_map[:50, 50:] = 2   # top-right
    panel_map[50:, :] = 3     # bottom

    centres = _panel_centres(panel_map, 3)
    assert len(centres) == 3
    assert centres[1] == pytest.approx((24.5, 24.5), abs=0.1)
    assert centres[2] == pytest.approx((74.5, 24.5), abs=0.1)


# ── _unpack_params ───────────────────────────────────────────────────────

def test_unpack_params_global_only():
    """Single-panel case: only global params."""
    x = np.array([350000, 100, 100, 0.1, -0.2,
                   0, 0, 0, 0, 0], dtype=float)
    geom, panels = _unpack_params(x, 10, 1, 1, (False, False, False))
    assert geom['Lsd'] == 350000
    assert geom['ty'] == pytest.approx(0.1)
    assert len(panels) == 0


def test_unpack_params_with_panels():
    """Multi-panel case: global + per-panel dY, dZ."""
    # 3 panels, fix panel 1, stride=2 (dY, dZ only)
    x = np.zeros(10 + 2 * 2, dtype=float)  # 10 global + 2 panels * 2
    x[0] = 350000  # Lsd
    x[10] = 1.5    # panel 2 dY
    x[11] = -0.3   # panel 2 dZ
    x[12] = 0.8    # panel 3 dY
    x[13] = 0.1    # panel 3 dZ

    geom, panels = _unpack_params(x, 10, 3, 1, (False, False, False))
    assert panels[1]['dY'] == 0.0  # fixed panel
    assert panels[2]['dY'] == pytest.approx(1.5)
    assert panels[2]['dZ'] == pytest.approx(-0.3)
    assert panels[3]['dY'] == pytest.approx(0.8)
