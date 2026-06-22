"""
Tests for midas4pil.io — orient_mask, auto_mask, write_poni, make_geometry.
"""

import numpy as np
import pytest
from midas4pil.io import (orient_mask, auto_mask, write_poni, read_poni,
                          make_geometry, save_params, load_params,
                          load_midas_params, save_midas_params)


def _make_asymmetric_panel_map():
    """3 panels with asymmetric gap positions (not invariant under flipud)."""
    # Panel 1: rows 0-9, Panel 2: rows 12-21, Panel 3: rows 25-39
    # Gaps at rows 10-11 and 22-24 (asymmetric: gap of 2 then gap of 3)
    pm = np.zeros((40, 20), dtype=np.int32)
    pm[0:10, :] = 1
    pm[12:22, :] = 2
    pm[25:40, :] = 3
    return pm


def _make_symmetric_panel_map():
    """2 panels with symmetric gap (invariant under flipud)."""
    pm = np.zeros((21, 20), dtype=np.int32)
    pm[0:10, :] = 1   # rows 0-9
    pm[11:21, :] = 2  # rows 11-20, gap at row 10
    return pm


# ── orient_mask with image (dead-pixel correlation) ──────────────────────

def test_orient_mask_image_correct_orientation():
    """Mask already in correct orientation: no flip."""
    pm = _make_asymmetric_panel_map()
    nrows, ncols = pm.shape
    image = np.ones((nrows, ncols), dtype=np.float64) * 100.0
    # Create dead pixels at specific positions
    image[3, 5] = 0
    image[7, 10] = -1
    image[15, 8] = 0

    # Mask marks the same dead pixels as bad
    mask = np.zeros((nrows, ncols), dtype=bool)
    mask[3, 5] = True
    mask[7, 10] = True
    mask[15, 8] = True

    result, flipped = orient_mask(mask, pm, image)
    assert not flipped
    assert np.array_equal(result, mask)


def test_orient_mask_image_flipped():
    """Mask in flipped orientation: should be auto-corrected."""
    pm = _make_asymmetric_panel_map()
    nrows, ncols = pm.shape
    image = np.ones((nrows, ncols), dtype=np.float64) * 100.0
    # Dead pixels near the top of the image
    image[2, 5] = 0
    image[3, 10] = -1
    image[4, 8] = 0
    image[5, 3] = 0
    image[6, 12] = -1
    image[7, 15] = 0
    image[8, 2] = 0
    image[9, 18] = -1
    image[13, 7] = 0
    image[14, 11] = -1

    # Correct mask (matches image dead pixels)
    mask_correct = np.zeros((nrows, ncols), dtype=bool)
    mask_correct[2, 5] = True
    mask_correct[3, 10] = True
    mask_correct[4, 8] = True
    mask_correct[5, 3] = True
    mask_correct[6, 12] = True
    mask_correct[7, 15] = True
    mask_correct[8, 2] = True
    mask_correct[9, 18] = True
    mask_correct[13, 7] = True
    mask_correct[14, 11] = True

    # Flip the mask (simulating Dioptas output)
    mask_flipped = np.flipud(mask_correct)

    result, flipped = orient_mask(mask_flipped, pm, image)
    assert flipped
    assert np.array_equal(result, mask_correct)


def test_orient_mask_image_symmetric_detector():
    """Image-based detection works even for symmetric gap layouts."""
    pm = _make_symmetric_panel_map()
    nrows, ncols = pm.shape
    image = np.ones((nrows, ncols), dtype=np.float64) * 100.0
    # Dead pixels near the top
    image[1, 5] = 0
    image[2, 10] = -1
    image[3, 8] = 0
    image[4, 3] = 0
    image[5, 12] = -1
    image[6, 15] = 0
    image[7, 2] = 0
    image[8, 18] = -1
    image[12, 7] = 0
    image[13, 11] = -1

    mask_correct = np.zeros((nrows, ncols), dtype=bool)
    for r, c in [(1, 5), (2, 10), (3, 8), (4, 3), (5, 12),
                 (6, 15), (7, 2), (8, 18), (12, 7), (13, 11)]:
        mask_correct[r, c] = True

    mask_flipped = np.flipud(mask_correct)
    result, flipped = orient_mask(mask_flipped, pm, image)
    assert flipped
    assert np.array_equal(result, mask_correct)


# ── orient_mask with panel_map only (gap-row heuristic) ──────────────────

def test_orient_mask_gaps_correct():
    """Asymmetric gaps, mask in correct orientation."""
    pm = _make_asymmetric_panel_map()
    nrows, ncols = pm.shape

    # Mask with gap rows marked as bad (correct orientation)
    mask = np.zeros((nrows, ncols), dtype=bool)
    mask[10:12, :] = True  # gap 1
    mask[22:25, :] = True  # gap 2

    result, flipped = orient_mask(mask, pm)
    assert not flipped


def test_orient_mask_gaps_flipped():
    """Asymmetric gaps, mask in flipped orientation."""
    pm = _make_asymmetric_panel_map()
    nrows, ncols = pm.shape

    # Correct mask
    mask_correct = np.zeros((nrows, ncols), dtype=bool)
    mask_correct[10:12, :] = True
    mask_correct[22:25, :] = True

    mask_flipped = np.flipud(mask_correct)
    result, flipped = orient_mask(mask_flipped, pm)
    assert flipped
    assert np.array_equal(result, mask_correct)


def test_orient_mask_symmetric_warns():
    """Symmetric gaps without image: should warn and return unchanged."""
    pm = _make_symmetric_panel_map()
    mask = np.zeros(pm.shape, dtype=bool)
    mask[10, :] = True  # gap row

    with pytest.warns(UserWarning, match="vertically symmetric"):
        result, flipped = orient_mask(mask, pm)
    assert not flipped
    assert np.array_equal(result, mask)


# ── Shape mismatch ───────────────────────────────────────────────────────

def test_orient_mask_shape_mismatch():
    """Mismatched shapes should raise ValueError."""
    pm = np.ones((10, 10), dtype=np.int32)
    mask = np.zeros((20, 10), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        orient_mask(mask, pm)


# ── Backward compatibility ───────────────────────────────────────────────

def test_load_mask_no_panel_map(tmp_path):
    """load_mask without panel_map returns mask as-is (backward compat)."""
    import tifffile
    from midas4pil.io import load_mask

    mask_data = np.zeros((10, 10), dtype=np.uint8)
    mask_data[3, 5] = 1
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(str(mask_path), mask_data)

    result = load_mask(mask_path)
    assert result[3, 5] == True
    assert result.sum() == 1


# ── auto_mask ──────────────────────────────────────────────────────────────

def test_auto_mask_gaps():
    """Gap pixels (panel_map == 0) are masked."""
    pm = _make_asymmetric_panel_map()
    image = np.ones(pm.shape, dtype=np.float64) * 100.0
    mask = auto_mask(image, panel_map=pm)
    assert mask[10, 5]    # gap row
    assert mask[22, 5]    # gap row
    assert not mask[5, 5] # active panel pixel


def test_auto_mask_dead_pixels():
    """Pixels with value <= 0 are masked."""
    image = np.ones((50, 50), dtype=np.float64) * 100.0
    image[10, 20] = 0
    image[30, 40] = -1
    mask = auto_mask(image)
    assert mask[10, 20]
    assert mask[30, 40]
    assert not mask[25, 25]


def test_auto_mask_sat_val():
    """Pixels at or above sat_val are masked."""
    image = np.ones((100, 100), dtype=np.float64) * 100.0
    image[50, 50] = 1e6
    mask = auto_mask(image, sat_val=1e5)
    assert mask[50, 50]
    assert not mask[25, 25]


def test_auto_mask_normal_not_masked():
    """Normal diffraction ring pixels are not masked."""
    image = np.ones((100, 100), dtype=np.float64) * 100.0
    # Simulate a bright ring: ~10x background
    image[40:45, :] = 1000.0
    mask = auto_mask(image)
    # Positive pixels should not be masked (no sat_val, no panel_map)
    assert not mask[42, 50]


# ── write_poni / read_poni round-trip ──────────────────────────────────────

def test_write_poni_roundtrip(tmp_path):
    """write_poni followed by read_poni recovers original geometry."""
    geom = make_geometry(
        wavelength=0.42460, lsd=349510.0, px=172.0,
        nrows=1679, ncols=1475,
        bc_y=748.845, bc_z=817.516,
        ty_deg=0.189, tz_deg=-0.317,
    )

    poni_path = tmp_path / "test.poni"
    write_poni(geom, poni_path)

    recovered = read_poni(poni_path, px=172.0, nrows=1679)

    assert abs(recovered['lsd'] - geom['lsd']) < 0.01
    assert abs(recovered['bc_y'] - geom['bc_y']) < 0.001
    assert abs(recovered['bc_z'] - geom['bc_z']) < 0.001
    assert abs(recovered['ty_deg'] - geom['ty_deg']) < 1e-6
    assert abs(recovered['tz_deg'] - geom['tz_deg']) < 1e-6
    assert abs(recovered['tx_deg'] - geom['tx_deg']) < 1e-6
    assert abs(recovered['wavelength'] - geom['wavelength']) < 1e-8


# ── make_geometry ──────────────────────────────────────────────────────────

def test_make_geometry_defaults():
    """make_geometry with defaults: bc at detector centre, tilts = 0."""
    g = make_geometry(wavelength=0.42460, lsd=350000, px=172,
                      nrows=1679, ncols=1475)
    assert g['bc_y'] == 1475 / 2.0
    assert g['bc_z'] == 1679 / 2.0
    assert g['tx_deg'] == 0.0
    assert g['ty_deg'] == 0.0
    assert g['tz_deg'] == 0.0
    assert g['p0'] == 0.0
    assert g['wavelength'] == 0.42460
    assert g['nrows'] == 1679


def test_make_geometry_custom_bc():
    """make_geometry with user-specified beam centre."""
    g = make_geometry(wavelength=0.42460, lsd=350000, px=172,
                      nrows=1679, ncols=1475,
                      bc_y=748.8, bc_z=817.5)
    assert g['bc_y'] == 748.8
    assert g['bc_z'] == 817.5


# ── mode roundtrip in save/load ─────────────────────────────────────────

def test_mode_default_varbin():
    """make_geometry defaults mode to 'varbin'."""
    g = make_geometry(wavelength=0.42460, lsd=350000, px=172,
                      nrows=1679, ncols=1475)
    assert g['mode'] == 'varbin'


def test_mode_roundtrip(tmp_path):
    """mode survives save_params → load_params."""
    g = make_geometry(wavelength=0.42460, lsd=350000, px=172,
                      nrows=1679, ncols=1475, mode='unibin')
    assert g['mode'] == 'unibin'

    path = tmp_path / "geom.toml"
    save_params(g, path)
    g2 = load_params(path)
    assert g2['mode'] == 'unibin'


def test_mode_roundtrip_varbin(tmp_path):
    """Default varbin mode survives roundtrip."""
    g = make_geometry(wavelength=0.42460, lsd=350000, px=172,
                      nrows=1679, ncols=1475)
    path = tmp_path / "geom.toml"
    save_params(g, path)
    g2 = load_params(path)
    assert g2['mode'] == 'varbin'


# ── MIDAS geometry_params.txt ─────────────────────────────────────────────

def _write_midas_txt(tmp_path, **overrides):
    """Write a minimal valid MIDAS params file; return path string."""
    fields = {
        'NrPixelsZ': 1679, 'NrPixelsY': 1475, 'px': 172,
        'Lsd': 349253.581, 'BC': '748.048 862.0',
        'ty': 0.1903, 'tz': -0.3005, 'tx': 0.0,
        'Wavelength': 0.42460,
        'p0': 0, 'p1': 0, 'p2': 0, 'p3': 0, 'p4': 0, 'RhoD': 217578,
        'ImTransOpt': 2,
    }
    fields.update(overrides)
    lines = []
    for k, v in fields.items():
        lines.append(f"{k} {v}")
    p = tmp_path / "params.txt"
    p.write_text('\n'.join(lines) + '\n')
    return str(p)


def test_load_midas_params_geometry(tmp_path):
    """Core geometry keys are parsed correctly."""
    path = _write_midas_txt(tmp_path)
    geom = load_midas_params(path)
    assert abs(geom['lsd']        - 349253.581) < 0.001
    assert abs(geom['bc_y']       - 748.048)    < 0.001
    assert abs(geom['bc_z']       - 817.0)      < 0.5    # nrows - 862
    assert abs(geom['ty_deg']     - 0.1903)     < 1e-6
    assert abs(geom['tz_deg']     - (-0.3005))  < 1e-6
    assert abs(geom['wavelength'] - 0.42460)    < 1e-6
    assert geom['nrows'] == 1679
    assert geom['ncols'] == 1475
    assert geom['px']    == 172


def test_load_midas_params_bc_z_no_imtrans(tmp_path):
    """Without ImTransOpt 2, BC_Z is used as-is (no frame conversion)."""
    path = _write_midas_txt(tmp_path, BC='748.0 817.0', ImTransOpt=0)
    geom = load_midas_params(path)
    assert abs(geom['bc_z'] - 817.0) < 0.001


def test_load_midas_params_lsd_alias(tmp_path):
    """Distance is accepted as an alias for Lsd."""
    p = tmp_path / "alias.txt"
    p.write_text(
        "NrPixelsZ 100\nNrPixelsY 100\nPixelSize 172\n"
        "Distance 350000\nBC 50 50\nWavelength 0.42\n"
        "p0 0\np1 0\np2 0\np3 0\np4 0\nRhoD 217578\n"
    )
    geom = load_midas_params(str(p))
    assert abs(geom['lsd'] - 350000) < 0.001
    assert abs(geom['px']  - 172)    < 0.001


def test_load_midas_params_nrpixels_square(tmp_path):
    """NrPixels (square detector) sets both nrows and ncols."""
    p = tmp_path / "sq.txt"
    p.write_text(
        "NrPixels 2048\npx 200\nLsd 500000\nBC 1024 1024\n"
        "Wavelength 0.5\np0 0\np1 0\np2 0\np3 0\np4 0\nRhoD 300000\n"
    )
    geom = load_midas_params(str(p))
    assert geom['nrows'] == 2048
    assert geom['ncols'] == 2048


def test_load_midas_params_no_nrpixels(tmp_path):
    """Missing NrPixels/NrPixelsY/Z returns partial dict without nrows/ncols."""
    p = tmp_path / "no_dim.txt"
    p.write_text(
        "px 172\nLsd 349000\nBC 748 862\nImTransOpt 2\n"
        "Wavelength 0.42460\np0 0\np1 0\np2 0\np3 0\np4 0\nRhoD 217578\n"
    )
    geom = load_midas_params(str(p))
    assert 'nrows' not in geom
    assert 'ncols' not in geom
    assert abs(geom['lsd'] - 349000) < 1
    assert abs(geom['bc_y'] - 748) < 0.1
    # no ImTransOpt conversion without nrows — bc_z kept as-is
    assert abs(geom['bc_z'] - 862) < 0.1


def test_midas_params_roundtrip(tmp_path):
    """save_midas_params → load_midas_params recovers all geometry values."""
    g = make_geometry(wavelength=0.42460, lsd=349253.0, px=172,
                      nrows=1679, ncols=1475,
                      bc_y=748.0, bc_z=817.0,
                      ty_deg=0.19, tz_deg=-0.30, tx_deg=0.0,
                      p0=0.0, p1=0.0, p2=0.0, p3=0.0, p4=0.0, rho_d=217578.0)
    out = tmp_path / "geom.txt"
    save_midas_params(g, str(out))
    g2 = load_midas_params(str(out))
    for key in ('lsd', 'bc_y', 'bc_z', 'ty_deg', 'tz_deg', 'tx_deg',
                'wavelength', 'p0', 'p1', 'p2', 'p3', 'p4', 'rho_d'):
        assert abs(g[key] - g2[key]) < 0.01, f"{key}: {g[key]} != {g2[key]}"


def test_midas_params_bc_z_frame_roundtrip(tmp_path):
    """BC_Z flip is correctly applied on write (ImTransOpt 2) and undone on read."""
    g = make_geometry(wavelength=0.42460, lsd=349000, px=172,
                      nrows=1679, ncols=1475, bc_y=748.0, bc_z=817.0)
    out = tmp_path / "geom.txt"
    save_midas_params(g, str(out))
    # The file should contain BC ... ~862 (= 1679 - 817)
    content = out.read_text()
    bc_line = next(ln for ln in content.splitlines() if ln.startswith('BC '))
    midas_bc_z = float(bc_line.split()[2])
    assert abs(midas_bc_z - (1679 - 817)) < 0.5
    # Round-trip must recover bc_z
    g2 = load_midas_params(str(out))
    assert abs(g2['bc_z'] - 817.0) < 0.5


def test_midas_params_panel_shifts_file(tmp_path):
    """Panel shifts are written and re-read via PanelShiftsFile."""
    from midas4pil.panels import read_panel_shifts
    shifts = [{'id': 0, 'dY': 1.2, 'dZ': -0.4,
               'dTheta': 0.01, 'dLsd': 30.0, 'dP2': 5e-5}]
    g = make_geometry(0.42460, 349000, 172, 1679, 1475)
    g['panel_shifts'] = shifts
    out    = tmp_path / "geom.txt"
    ps_out = tmp_path / "geom_panel_shifts.txt"
    save_midas_params(g, str(out), panel_shifts_path=str(ps_out))
    # PanelShiftsFile line must be present
    assert 'PanelShiftsFile' in out.read_text()
    # Panel shifts file must exist and be readable
    assert ps_out.exists()
    loaded_ps = read_panel_shifts(str(ps_out))
    assert len(loaded_ps) == 1
    assert abs(loaded_ps[0]['dY']     - 1.2)    < 1e-6
    assert abs(loaded_ps[0]['dLsd']   - 30.0)   < 1e-6
    assert abs(loaded_ps[0]['dTheta'] - 0.01)   < 1e-6
    # load_midas_params must pick up the panel shifts automatically
    g2 = load_midas_params(str(out))
    assert len(g2['panel_shifts']) == 1
    assert abs(g2['panel_shifts'][0]['dY']   - 1.2)  < 1e-6
    assert abs(g2['panel_shifts'][0]['dLsd'] - 30.0) < 1e-6
