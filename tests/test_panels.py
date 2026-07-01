# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""
Tests for midas4pil.panels — Phase 4 panel-correction module.
"""

import numpy as np
import pytest

from midas4pil.panels import (
    make_panel_id_map,
    apply_panel_offsets, build_lut_with_panels,
    read_panel_shifts, save_panel_shifts,
)
from midas4pil.geometry import build_lut


# ── make_panel_id_map ──────────────────────────────────────────────────────

class TestMakePanelIdMap:

    def test_shape(self):
        pm = make_panel_id_map(200, 300, n_panels_y=2, n_panels_z=2,
                               panel_size_y=140, panel_size_z=90,
                               gap_y=20, gap_z=20)
        assert pm.shape == (200, 300)

    def test_dtype(self):
        pm = make_panel_id_map(100, 100, 2, 2, 45, 45, 10, 10)
        assert pm.dtype == np.int32

    def test_panel_ids_range(self):
        pm = make_panel_id_map(100, 100, 2, 2, 45, 45, 10, 10)
        # IDs 1..4, gaps are 0
        assert set(np.unique(pm)) == {0, 1, 2, 3, 4}

    def test_gap_pixels_are_zero(self):
        # 1×2 layout: two panels separated by a 5-px gap
        pm = make_panel_id_map(nrows=10, ncols=25,
                               n_panels_y=2, n_panels_z=1,
                               panel_size_y=10, panel_size_z=10,
                               gap_y=5, gap_z=[])
        # columns 10..14 are the gap
        assert (pm[:, 10:15] == 0).all()
        # panel pixels are non-zero
        assert (pm[:, :10] != 0).all()
        assert (pm[:, 15:] != 0).all()

    def test_panel_ids_row_major(self):
        # 2 rows × 3 cols of panels
        pm = make_panel_id_map(nrows=20, ncols=30,
                               n_panels_y=3, n_panels_z=2,
                               panel_size_y=9, panel_size_z=9,
                               gap_y=1, gap_z=2)
        # panel at (row_panel=0, col_panel=0) → id=1  (top-left)
        assert pm[0, 0] == 1
        # panel at (row_panel=0, col_panel=2) → id=3  (last pixel of third panel col)
        # layout: 3×9 px + 2×1 gap = 29 cols; pm[0, 28] is last px of panel 3
        assert pm[0, 28] == 3
        # panel at (row_panel=1, col_panel=0) → id=4  (bottom-left)
        assert pm[19, 0] == 4

    def test_scalar_gap_broadcast(self):
        # scalar gaps should be broadcast to lists internally
        pm = make_panel_id_map(100, 300, 3, 2, 90, 45, 15, 10)
        assert pm.shape == (100, 300)

    def test_non_uniform_gaps(self):
        # Two different column gaps
        pm = make_panel_id_map(nrows=10, ncols=37,
                               n_panels_y=3, n_panels_z=1,
                               panel_size_y=10, panel_size_z=10,
                               gap_y=[3, 4], gap_z=[])
        # First gap: cols 10..12 (3 px)
        assert (pm[:, 10:13] == 0).all()
        # Second gap: cols 23..26 (4 px)
        assert (pm[:, 23:27] == 0).all()

    def test_wrong_gap_length_raises(self):
        with pytest.raises(ValueError):
            make_panel_id_map(100, 100, 3, 1, 30, 100, [5], [])  # need 2 gaps, gave 1

    def test_pixel_coverage(self):
        # All non-gap pixels must be assigned an id >= 1
        pm = make_panel_id_map(200, 300, 2, 2, 140, 90, 20, 20)
        n_panel_pixels = 2 * 2 * 140 * 90
        assert (pm > 0).sum() == n_panel_pixels


# ── make_panel_map / make_panel_map_from_shape ──────────────────────────────

class TestMakePanelMapHelpers:
    """Tests for the detector-aware panel map helpers in gui.detectors."""

    def test_make_panel_map_tiled(self):
        from midas4pil.gui.detectors import make_panel_map
        pm = make_panel_map("Pilatus 1M")
        assert pm.shape == (1043, 981)
        assert pm.max() == 10              # 2 cols × 5 rows

    def test_make_panel_map_monolithic_returns_none(self):
        from midas4pil.gui.detectors import make_panel_map
        pm = make_panel_map("Perkin-Elmer XRD 1621")
        assert pm is None

    def test_make_panel_map_unknown_raises(self):
        from midas4pil.gui.detectors import make_panel_map
        import pytest
        with pytest.raises(ValueError):
            make_panel_map("NoSuchDetector")

    def test_make_panel_map_from_shape_match(self):
        from midas4pil.gui.detectors import make_panel_map_from_shape
        pm = make_panel_map_from_shape(1043, 981)   # Pilatus 1M
        assert pm is not None
        assert pm.shape == (1043, 981)

    def test_make_panel_map_from_shape_no_match(self):
        from midas4pil.gui.detectors import make_panel_map_from_shape
        pm = make_panel_map_from_shape(512, 512)    # no preset
        assert pm is None

    def test_panel_pixel_count(self):
        from midas4pil.gui.detectors import make_panel_map
        pm = make_panel_map("Pilatus 2M CdTe")
        # 24 panels × 487 × 195 pixels each
        assert (pm > 0).sum() == 24 * 487 * 195


# ── apply_panel_offsets ────────────────────────────────────────────────────

class TestApplyPanelOffsets:

    def test_zero_shifts_give_zero_maps(self):
        pm = make_panel_id_map(20, 30, 2, 2, 13, 8, 4, 4)
        shifts = [{'id': i, 'dY': 0, 'dZ': 0, 'dLsd': 0, 'dP2': 0}
                  for i in range(1, 5)]
        dY, dZ, dLsd, dP2 = apply_panel_offsets(20, 30, pm, shifts)
        assert np.all(dY == 0)
        assert np.all(dZ == 0)

    def test_shift_applied_to_correct_panel(self):
        pm = make_panel_id_map(20, 30, 2, 1, 13, 20, 4, [])
        shifts = [
            {'id': 1, 'dY': 2.5, 'dZ':  0.0, 'dLsd': 0, 'dP2': 0},
            {'id': 2, 'dY': 0.0, 'dZ': -1.5, 'dLsd': 0, 'dP2': 0},
        ]
        dY, dZ, _, _ = apply_panel_offsets(20, 30, pm, shifts)
        # Panel 1 occupies cols 0..12
        assert np.all(dY[:, :13][pm[:, :13] == 1] == 2.5)
        # Panel 2 occupies cols 17..29
        assert np.all(dZ[:, 17:][pm[:, 17:] == 2] == -1.5)

    def test_dlsd_and_dp2_applied(self):
        pm = make_panel_id_map(10, 10, 1, 1, 10, 10, [], [])
        shifts = [{'id': 1, 'dY': 0, 'dZ': 0, 'dLsd': 500.0, 'dP2': 0.001}]
        _, _, dLsd, dP2 = apply_panel_offsets(10, 10, pm, shifts)
        assert np.all(dLsd == 500.0)
        assert np.allclose(dP2, 0.001)


# ── build_lut_with_panels ──────────────────────────────────────────────────

class TestBuildLutWithPanels:

    _GEOM = dict(
        nrows=50, ncols=60,
        bc_y=30.0, bc_z=25.0,
        lsd=200000.0, px=172.0,
        tx_deg=0.0, ty_deg=0.0, tz_deg=0.0,
        p0=0.0, p1=0.0, p2=0.0, p3=0.0, p4=0.0,
        rho_d=217578.0,
    )

    def test_zero_shifts_match_build_lut(self):
        g = self._GEOM
        pm = make_panel_id_map(g['nrows'], g['ncols'],
                               n_panels_y=2, n_panels_z=2,
                               panel_size_y=28, panel_size_z=23,
                               gap_y=4, gap_z=4)
        shifts = [{'id': i, 'dY': 0, 'dZ': 0, 'dLsd': 0, 'dP2': 0}
                  for i in range(1, 5)]
        tth_p, eta_p = build_lut_with_panels(
            g['nrows'], g['ncols'], g['bc_y'], g['bc_z'],
            g['lsd'], g['px'],
            g['tx_deg'], g['ty_deg'], g['tz_deg'],
            g['p0'], g['p1'], g['p2'], g['p3'], g['p4'], g['rho_d'],
            pm, shifts,
        )
        tth_ref, eta_ref = build_lut(
            g['nrows'], g['ncols'], g['bc_y'], g['bc_z'],
            g['lsd'], g['px'],
            g['tx_deg'], g['ty_deg'], g['tz_deg'],
            g['p0'], g['p1'], g['p2'], g['p3'], g['p4'], g['rho_d'],
        )
        assert np.allclose(tth_p, tth_ref)
        assert np.allclose(eta_p, eta_ref)

    def test_nonzero_shifts_change_lut(self):
        g = self._GEOM
        pm = make_panel_id_map(g['nrows'], g['ncols'],
                               n_panels_y=1, n_panels_z=1,
                               panel_size_y=60, panel_size_z=50,
                               gap_y=[], gap_z=[])
        shifts_zero = [{'id': 1, 'dY': 0,   'dZ': 0, 'dLsd': 0, 'dP2': 0}]
        shifts_nonz = [{'id': 1, 'dY': 3.0, 'dZ': 0, 'dLsd': 0, 'dP2': 0}]
        tth_z, _  = build_lut_with_panels(
            g['nrows'], g['ncols'], g['bc_y'], g['bc_z'],
            g['lsd'], g['px'],
            g['tx_deg'], g['ty_deg'], g['tz_deg'],
            g['p0'], g['p1'], g['p2'], g['p3'], g['p4'], g['rho_d'],
            pm, shifts_zero,
        )
        tth_n, _ = build_lut_with_panels(
            g['nrows'], g['ncols'], g['bc_y'], g['bc_z'],
            g['lsd'], g['px'],
            g['tx_deg'], g['ty_deg'], g['tz_deg'],
            g['p0'], g['p1'], g['p2'], g['p3'], g['p4'], g['rho_d'],
            pm, shifts_nonz,
        )
        assert not np.allclose(tth_z, tth_n)


# ── read_panel_shifts / save_panel_shifts ──────────────────────────────────

class TestPanelShiftsIO:

    def test_column_order_dtheta_before_dlsd(self, tmp_path):
        """save_panel_shifts writes dTheta (col 3) before dLsd (col 4) — MIDAS order."""
        shifts = [{'id': 0, 'dY': 1.5, 'dZ': -0.3,
                   'dTheta': 0.012, 'dLsd': 45.0, 'dP2': 0.0001}]
        p = tmp_path / "ps.txt"
        save_panel_shifts(shifts, str(p))
        data_lines = [ln for ln in p.read_text().splitlines()
                      if ln.strip() and not ln.startswith('#')]
        parts = data_lines[0].split()
        assert abs(float(parts[3]) - 0.012)  < 1e-9, "col 3 must be dTheta"
        assert abs(float(parts[4]) - 45.0)   < 1e-9, "col 4 must be dLsd"
        assert abs(float(parts[5]) - 0.0001) < 1e-9, "col 5 must be dP2"

    def test_roundtrip_all_fields(self, tmp_path):
        """save → read recovers all six fields exactly."""
        shifts = [
            {'id': i, 'dY': i * 0.1, 'dZ': i * 0.2,
             'dTheta': i * 0.01, 'dLsd': i * 10.0, 'dP2': i * 1e-4}
            for i in range(3)
        ]
        p = tmp_path / "ps.txt"
        save_panel_shifts(shifts, str(p))
        loaded = read_panel_shifts(str(p))
        assert len(loaded) == 3
        for orig, got in zip(shifts, loaded):
            for key in ('dY', 'dZ', 'dTheta', 'dLsd', 'dP2'):
                assert abs(orig[key] - got[key]) < 1e-9, f"{key} mismatch"

    def test_midas_format_compatible(self, tmp_path):
        """A file written in MIDAS column order (id dY dZ dTheta dLsd dP2)
        is read back correctly."""
        p = tmp_path / "midas_ps.txt"
        p.write_text(
            "# panel_id  dY  dZ  dTheta  dLsd  dP2\n"
            "  0   1.200000  -0.400000   0.010000  30.000000  0.000050\n"
        )
        loaded = read_panel_shifts(str(p))
        assert len(loaded) == 1
        s = loaded[0]
        assert abs(s['dY']     - 1.2)    < 1e-6
        assert abs(s['dZ']     - (-0.4)) < 1e-6
        assert abs(s['dTheta'] - 0.01)   < 1e-6
        assert abs(s['dLsd']   - 30.0)   < 1e-6
        assert abs(s['dP2']    - 5e-5)   < 1e-6

    def test_defaults_when_optional_cols_absent(self, tmp_path):
        """Columns 3-5 default to 0.0 when not present."""
        p = tmp_path / "short.txt"
        p.write_text("0  1.0  2.0\n")
        loaded = read_panel_shifts(str(p))
        assert loaded[0]['dTheta'] == 0.0
        assert loaded[0]['dLsd']   == 0.0
        assert loaded[0]['dP2']    == 0.0
