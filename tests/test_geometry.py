"""
Tests for midas4pil.geometry

Verification strategy
---------------------
1. Tilt matrix: orthogonality and correct rotation direction for each axis.
2. eta convention: 12/3/6/9 o'clock pixels map to +90/0/−90/±180 degrees (CCW-positive, 0° at 3 o'clock).
3. r_to_tth: round-trip R → 2θ → R.
4. build_lut: beam-centre pixel has R≈0; smoke test on the 16-IDB CeO2
   geometry from refined_MIDAS_params.txt.
5. On-axis pixel (bc_y, bc_z) has eta undefined but R≈0 and tth≈0.
"""

import numpy as np
import pytest
from midas4pil.geometry import (
    build_tilt_matrix, pixel_to_r_eta, r_to_tth, build_lut
)

# ── Reference geometry (16-IDB CeO2, refined_MIDAS_params.txt) ─────────────
GEOM = dict(
    nrows=1679, ncols=1475,
    bc_y=748.03, bc_z=819.95,       # beam-centre row from TOP (= nrows-1 - MIDAS-bc_z)
    lsd=349011.8,                    # µm
    px=172.0,                        # µm
    tx_deg=0.0, ty_deg=-0.000819, tz_deg=0.072625,
    p0=0.001989, p1=0.001963, p2=0.000774, p3=6.369, p4=-0.001778,
    rho_d=217577.7,
)


# ── Tilt matrix ─────────────────────────────────────────────────────────────

def test_tilt_matrix_identity():
    TRs = build_tilt_matrix(0.0, 0.0, 0.0)
    np.testing.assert_allclose(TRs, np.eye(3), atol=1e-15)


def test_tilt_matrix_orthogonal():
    TRs = build_tilt_matrix(1.5, -0.8, 0.5)
    np.testing.assert_allclose(TRs @ TRs.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(TRs), 1.0, atol=1e-12)


def test_tilt_matrix_rz_direction():
    """Rz(90°): [cos -sin 0; sin cos 0; 0 0 1] @ [0,1,0] = [-1,0,0]."""
    TRs = build_tilt_matrix(0.0, 0.0, 90.0)
    v = np.array([0.0, 1.0, 0.0])
    result = TRs @ v
    np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=1e-12)


# ── eta convention ──────────────────────────────────────────────────────────

def _eta_at(row, col, nrows=1679, ncols=1475, bc_y=748.0, bc_z=820.0,
            lsd=349000.0, px=172.0, rho_d=217000.0):
    """Helper: eta for a single pixel with zero tilts and no distortion.

    bc_z is row-from-top (= Poni1/px), consistent with the pyFAI convention.
    The equator (Zc=0) is at array row == bc_z.
    """
    TRs = build_tilt_matrix(0.0, 0.0, 0.0)
    _, eta = pixel_to_r_eta(col, row, bc_y, bc_z, TRs,
                             lsd, rho_d, 0, 0, 0, 0, 0, px)
    return float(eta)


def test_eta_12_oclock():
    """Pixel directly above beam centre → eta = +90° (CCW-positive, 0° at 3 o'clock)."""
    eta = _eta_at(row=200, col=748)   # row < bc_z_array, so above beam centre
    assert abs(eta - 90.0) < 0.1


def test_eta_6_oclock():
    """Pixel directly below beam centre → eta = −90°."""
    eta = _eta_at(row=1600, col=748)  # row > bc_z_array
    assert abs(eta + 90.0) < 0.1


def test_eta_9_oclock():
    """Pixel directly left of beam centre → eta = ±180°.

    bc_z=820 (from top), so the equator is at array row 820 (Zc=0).
    """
    eta = _eta_at(row=820, col=200)   # on horizontal equator (row == bc_z), col < bc_y
    assert abs(abs(eta) - 180.0) < 0.5


def test_eta_3_oclock():
    """Pixel directly right of beam centre → eta = 0° (reference position)."""
    eta = _eta_at(row=820, col=1200)  # on horizontal equator, col > bc_y
    assert abs(eta) < 0.5


# ── r_to_tth round-trip ──────────────────────────────────────────────────────

def test_r_to_tth_roundtrip():
    """Flat-detector round-trip: R → 2θ = atan(R·px/lsd) → R = tan(2θ)·lsd/px."""
    px  = 172.0
    lsd = 349000.0
    R   = np.array([10.0, 100.0, 500.0, 900.0])
    tth = r_to_tth(R, px, lsd)
    R2  = np.tan(np.radians(tth)) * lsd / px
    np.testing.assert_allclose(R2, R, rtol=1e-10)


def test_r_to_tth_zero():
    assert r_to_tth(0.0, 172.0, 349000.0) == 0.0


# ── build_lut smoke tests ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def lut():
    tth, eta = build_lut(**GEOM)
    return tth, eta


def test_lut_shape(lut):
    tth, eta = lut
    assert tth.shape == (GEOM['nrows'], GEOM['ncols'])
    assert eta.shape == (GEOM['nrows'], GEOM['ncols'])


def test_lut_beam_centre_small_tth(lut):
    """Pixel nearest to beam centre should have very small 2θ."""
    tth, _ = lut
    bc_row = int(round(GEOM['bc_z']))    # bc_z is now row-from-top
    bc_col = int(round(GEOM['bc_y']))
    assert tth[bc_row, bc_col] < 0.1


def test_lut_tth_range(lut):
    """All 2θ values should be in [0°, 90°] for a flat detector at this geometry."""
    tth, _ = lut
    assert np.nanmin(tth) >= 0.0
    assert np.nanmax(tth) < 90.0


def test_lut_eta_range(lut):
    """eta should be in (−180°, +180°]."""
    _, eta = lut
    assert np.nanmin(eta) >= -180.0
    assert np.nanmax(eta) <= 180.0


# ── find_beam_center_auto ─────────────────────────────────────────────────

def test_find_beam_centre_synthetic():
    """find_beam_center_auto recovers a known BC offset from synthetic rings."""
    from midas4pil.geometry import find_beam_center_auto

    nrows, ncols = 200, 200
    true_by, true_bz = 110.0, 95.0  # offset from centre

    # Create synthetic concentric rings
    cc, rr = np.meshgrid(np.arange(ncols, dtype=np.float64),
                         np.arange(nrows, dtype=np.float64))
    R = np.sqrt((cc - true_by)**2 + (rr - true_bz)**2)

    image = np.ones((nrows, ncols), dtype=np.float64) * 50.0
    # Add 3 rings at radii 30, 55, 80 pixels
    for r0 in [30, 55, 80]:
        ring = np.exp(-0.5 * ((R - r0) / 1.5)**2) * 500
        image += ring

    mask = np.zeros((nrows, ncols), dtype=bool)

    bc_y, bc_z = find_beam_center_auto(image, mask,
                                       search_range=50, downsample=4)

    assert abs(bc_y - true_by) < 5.0, f"bc_y error: {bc_y - true_by:.1f}"
    assert abs(bc_z - true_bz) < 5.0, f"bc_z error: {bc_z - true_bz:.1f}"
