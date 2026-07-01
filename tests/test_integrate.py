# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""
Tests for midas4pil.integrate and midas4pil.cake

Covers:
- snip_background: flat signal, pure background, positivity.
- integrate_1d: uniform image gives flat lineout; mask exclusion; eta wedge.
- cake: shape and NaN coverage.
"""

import numpy as np
import pytest
from midas4pil.integrate import snip_background, integrate_1d, integrate_1d_varbin, reduce_frame
from midas4pil.geometry import varbin_tth_edges, pixel_resolution
from midas4pil.cake import cake, cake_varbin


# ── SNIP background ──────────────────────────────────────────────────────────

def test_snip_flat():
    """SNIP on a flat signal should return the signal itself (no peaks to clip)."""
    y = np.ones(200) * 1000.0
    bg = snip_background(y, n_iter=50)
    np.testing.assert_allclose(bg, y, rtol=1e-3)


def test_snip_nonnegative():
    """Background must always be >= 0."""
    rng = np.random.default_rng(42)
    y = rng.exponential(500, 500)
    bg = snip_background(y, n_iter=50)
    assert np.all(bg >= 0.0)


def test_snip_below_signal():
    """Background must not exceed the raw signal (after clipping)."""
    rng = np.random.default_rng(0)
    y = np.abs(rng.normal(100, 20, 300)) + 5.0
    bg = snip_background(y, n_iter=50)
    # Background should be <= signal (with small numerical tolerance)
    assert np.all(bg <= y + 1e-6)


def test_snip_peak_suppression():
    """A sharp Gaussian peak should be mostly removed by SNIP."""
    x = np.linspace(0, 10, 500)
    base = 100.0
    peak = 5000.0 * np.exp(-0.5 * ((x - 5.0) / 0.1) ** 2)
    y = base + peak
    bg = snip_background(y, n_iter=50)
    # Background at peak centre should be close to the base level
    peak_idx = np.argmax(peak)
    assert bg[peak_idx] < base * 2.0


# ── integrate_1d ─────────────────────────────────────────────────────────────

def _make_uniform_luts(nrows=100, ncols=100, tth_min=1.0, tth_max=11.0):
    """Synthetic LUTs: linear tth gradient, uniform eta=0."""
    tth_lut = np.linspace(tth_min, tth_max, ncols)[None, :] * np.ones(nrows)[:, None]
    eta_lut = np.zeros((nrows, ncols))
    return tth_lut, eta_lut


def test_integrate_1d_uniform_image():
    """Uniform image with no mask should give a flat lineout."""
    nrows, ncols = 100, 100
    tth_lut, eta_lut = _make_uniform_luts(nrows, ncols)
    image = np.ones((nrows, ncols)) * 42.0
    mask  = np.zeros((nrows, ncols), dtype=bool)

    tth, I, bg, I_sub, sigma, px_cnt = integrate_1d(
        image, mask, tth_lut,
        tth_min=1.5, tth_max=10.5, tth_bin_size=0.1,
        snip_iter=0,
    )
    # All populated bins should have mean = 42
    populated = ~np.isnan(I)
    assert populated.sum() > 0
    np.testing.assert_allclose(I[populated], 42.0, rtol=1e-6)


def test_integrate_1d_mask_excludes_pixels():
    """Masking all pixels in a 2θ sub-range should give NaN there."""
    nrows, ncols = 50, 200
    tth_lut, _ = _make_uniform_luts(nrows, ncols, tth_min=0.0, tth_max=20.0)
    image = np.ones((nrows, ncols))
    mask  = np.zeros((nrows, ncols), dtype=bool)
    # Mask a strip covering tth ~ [5, 10]
    mask[:, 50:100] = True

    tth, I, _, _, _, _ = integrate_1d(
        image, mask, tth_lut,
        tth_min=0.0, tth_max=20.0, tth_bin_size=0.5,
        snip_iter=0,
    )
    masked_bins = (tth >= 5.0) & (tth <= 10.0)
    assert np.all(np.isnan(I[masked_bins]))


def test_integrate_1d_eta_wedge():
    """Eta wedge should restrict which pixels contribute."""
    nrows, ncols = 100, 100
    tth_lut = np.ones((nrows, ncols)) * 5.0   # all pixels at 5°
    # Left half: eta=+90, right half: eta=−90
    eta_lut = np.ones((nrows, ncols)) * 90.0
    eta_lut[:, 50:] = -90.0

    image = np.ones((nrows, ncols))
    image[:, 50:] = 999.0   # right half has large intensity
    mask = np.zeros((nrows, ncols), dtype=bool)

    # Only left half (eta > 0) should contribute
    tth, I, _, _, _, _ = integrate_1d(
        image, mask, tth_lut,
        tth_min=4.0, tth_max=6.0, tth_bin_size=0.5,
        eta_lut=eta_lut, eta_min=0.0, eta_max=180.0,
        snip_iter=0,
    )
    populated = ~np.isnan(I)
    assert populated.sum() > 0
    np.testing.assert_allclose(I[populated], 1.0, rtol=1e-6)


def test_integrate_1d_returns_six_columns():
    nrows, ncols = 20, 20
    tth_lut, _ = _make_uniform_luts(nrows, ncols)
    image = np.ones((nrows, ncols))
    mask  = np.zeros((nrows, ncols), dtype=bool)
    result = integrate_1d(image, mask, tth_lut,
                          tth_min=2.0, tth_max=10.0, tth_bin_size=0.5)
    assert len(result) == 6
    tth, I, bg, I_sub, sigma, px_cnt = result
    assert tth.shape == I.shape == bg.shape == I_sub.shape == sigma.shape == px_cnt.shape


# ── cake ──────────────────────────────────────────────────────────────────────

def test_cake_shape():
    nrows, ncols = 100, 100
    tth_lut = np.linspace(1, 10, ncols)[None, :] * np.ones(nrows)[:, None]
    eta_lut = np.linspace(-90, 90, nrows)[:, None] * np.ones(ncols)[None, :]
    image = np.ones((nrows, ncols))
    mask  = np.zeros((nrows, ncols), dtype=bool)

    cake_img, tth_c, eta_c, cnt_map = cake(
        image, mask, tth_lut, eta_lut,
        tth_min=1.0, tth_max=10.0, tth_bin_size=1.0,
        eta_min=-90.0, eta_max=90.0, eta_bin_size=10.0,
    )
    assert cake_img.shape == (len(tth_c), len(eta_c))
    assert len(tth_c) == 9
    assert len(eta_c) == 18


def test_cake_uniform_image():
    """Uniform image → all populated bins equal to 1."""
    nrows, ncols = 50, 50
    tth_lut = np.linspace(2, 8, ncols)[None, :] * np.ones(nrows)[:, None]
    eta_lut = np.linspace(-45, 45, nrows)[:, None] * np.ones(ncols)[None, :]
    image = np.ones((nrows, ncols)) * 7.0
    mask  = np.zeros((nrows, ncols), dtype=bool)

    cake_img, _, _, _ = cake(
        image, mask, tth_lut, eta_lut,
        tth_min=2.0, tth_max=8.0, tth_bin_size=1.0,
        eta_min=-45.0, eta_max=45.0, eta_bin_size=5.0,
    )
    populated = ~np.isnan(cake_img)
    assert populated.sum() > 0
    np.testing.assert_allclose(cake_img[populated], 7.0, rtol=1e-6)


# ── varbin_tth_edges ─────────────────────────────────────────────────────────

PX  = 172.0      # µm
LSD = 349510.0   # µm


def test_varbin_tth_edges_boundaries():
    """Edges snap to R-grid: first edge >= tth_min, last edge <= tth_max."""
    edges = varbin_tth_edges(2.0, 29.0, PX, LSD)
    assert edges[0] >= 2.0
    assert edges[-1] <= 29.0
    # First edge should be close to tth_min (within one pixel's angular width)
    delta_tth_at_min = np.degrees(PX * np.cos(np.radians(2.0))**2 / LSD)
    assert edges[0] - 2.0 < delta_tth_at_min


def test_varbin_tth_edges_monotonic():
    edges = varbin_tth_edges(2.0, 29.0, PX, LSD)
    assert np.all(np.diff(edges) > 0)


def test_varbin_tth_edges_pixel_matched():
    """With dR=1, bin widths should match px*cos^2(tth)/lsd."""
    edges = varbin_tth_edges(2.0, 29.0, PX, LSD)
    widths = np.diff(edges)
    mids = 0.5 * (edges[:-1] + edges[1:])
    expected = np.degrees(PX * np.cos(np.radians(mids))**2 / LSD)
    np.testing.assert_allclose(widths, expected, rtol=0.02)


def test_varbin_tth_edges_widths_decrease():
    """Bin widths should decrease with increasing 2theta (cos^2 compression)."""
    edges = varbin_tth_edges(2.0, 29.0, PX, LSD)
    widths = np.diff(edges)
    assert widths[0] > widths[-1]


def test_varbin_tth_edges_dR_oversampled():
    """dR=0.5 should give about twice as many bins as dR=1.0."""
    edges_1 = varbin_tth_edges(2.0, 29.0, PX, LSD, dR=1.0)
    edges_h = varbin_tth_edges(2.0, 29.0, PX, LSD, dR=0.5)
    ratio = (len(edges_h) - 1) / (len(edges_1) - 1)
    assert 1.8 < ratio < 2.2


# ── pixel_resolution ─────────────────────────────────────────────────────────

def test_pixel_resolution_at_zero():
    """At 2theta=0, delta_tth = px/lsd, delta_eta = inf."""
    dt, de = pixel_resolution(0.0, PX, LSD)
    np.testing.assert_allclose(dt, np.degrees(PX / LSD), rtol=1e-10)
    assert np.isinf(de)


def test_pixel_resolution_decreases():
    """delta_tth should decrease with increasing 2theta."""
    dt_lo, _ = pixel_resolution(5.0, PX, LSD)
    dt_hi, _ = pixel_resolution(25.0, PX, LSD)
    assert dt_lo > dt_hi


def test_pixel_resolution_eta_decreases():
    """delta_eta should decrease with increasing 2theta."""
    _, de_lo = pixel_resolution(5.0, PX, LSD)
    _, de_hi = pixel_resolution(25.0, PX, LSD)
    assert de_lo > de_hi


# ── integrate_1d_varbin ──────────────────────────────────────────────────────

def test_varbin_uniform_image():
    """Uniform image with varbin should give a flat lineout."""
    nrows, ncols = 100, 100
    tth_lut, _ = _make_uniform_luts(nrows, ncols, tth_min=2.0, tth_max=12.0)
    image = np.ones((nrows, ncols)) * 42.0
    mask  = np.zeros((nrows, ncols), dtype=bool)

    tth, I, bg, I_sub, sigma, px_cnt = integrate_1d_varbin(
        image, mask, tth_lut,
        tth_min=2.5, tth_max=11.5, px=PX, lsd=LSD,
        snip_iter=0,
    )
    populated = ~np.isnan(I)
    assert populated.sum() > 0
    np.testing.assert_allclose(I[populated], 42.0, rtol=1e-6)


def test_varbin_returns_six_columns():
    nrows, ncols = 20, 20
    tth_lut, _ = _make_uniform_luts(nrows, ncols)
    image = np.ones((nrows, ncols))
    mask  = np.zeros((nrows, ncols), dtype=bool)
    result = integrate_1d_varbin(image, mask, tth_lut,
                                 tth_min=2.0, tth_max=10.0, px=PX, lsd=LSD)
    assert len(result) == 6
    tth, I, bg, I_sub, sigma, px_cnt = result
    assert tth.shape == I.shape == bg.shape == I_sub.shape == sigma.shape == px_cnt.shape


def test_varbin_custom_edges():
    """Pre-computed tth_edges should be used as-is."""
    nrows, ncols = 50, 50
    tth_lut, _ = _make_uniform_luts(nrows, ncols, tth_min=1.0, tth_max=11.0)
    image = np.ones((nrows, ncols)) * 10.0
    mask  = np.zeros((nrows, ncols), dtype=bool)

    custom_edges = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    tth, I, _, _, _, _ = integrate_1d_varbin(
        image, mask, tth_lut,
        tth_min=2.0, tth_max=10.0, px=PX, lsd=LSD,
        tth_edges=custom_edges, snip_iter=0,
    )
    assert len(tth) == 4  # 4 bins from 5 edges
    np.testing.assert_allclose(tth, [3.0, 5.0, 7.0, 9.0])


def test_varbin_nonuniform_centres():
    """Varbin bin centres should be non-uniformly spaced."""
    edges = varbin_tth_edges(2.0, 29.0, PX, LSD)
    centres = 0.5 * (edges[:-1] + edges[1:])
    spacings = np.diff(centres)
    # Not all spacings equal
    assert np.std(spacings) / np.mean(spacings) > 0.01


# ── cake_varbin ──────────────────────────────────────────────────────────────

def test_cake_varbin_shape():
    nrows, ncols = 100, 100
    tth_lut = np.linspace(2, 10, ncols)[None, :] * np.ones(nrows)[:, None]
    eta_lut = np.linspace(-90, 90, nrows)[:, None] * np.ones(ncols)[None, :]
    image = np.ones((nrows, ncols))
    mask  = np.zeros((nrows, ncols), dtype=bool)

    result = cake_varbin(
        image, mask, tth_lut, eta_lut,
        tth_min=2.0, tth_max=10.0, px=PX, lsd=LSD,
        eta_min=-90.0, eta_max=90.0,
    )
    assert len(result) == 5  # returns 5 values
    cake_img, tth_c, eta_c, eta_bin, cnt_map = result
    assert cake_img.shape == (len(tth_c), len(eta_c))


def test_cake_varbin_uniform_image():
    """Uniform image with varbin cake should give constant populated bins."""
    nrows, ncols = 50, 50
    tth_lut = np.linspace(3, 8, ncols)[None, :] * np.ones(nrows)[:, None]
    eta_lut = np.linspace(-45, 45, nrows)[:, None] * np.ones(ncols)[None, :]
    image = np.ones((nrows, ncols)) * 7.0
    mask  = np.zeros((nrows, ncols), dtype=bool)

    cake_img, _, _, _, _ = cake_varbin(
        image, mask, tth_lut, eta_lut,
        tth_min=3.0, tth_max=8.0, px=PX, lsd=LSD,
        eta_min=-45.0, eta_max=45.0,
    )
    populated = ~np.isnan(cake_img)
    assert populated.sum() > 0
    np.testing.assert_allclose(cake_img[populated], 7.0, rtol=1e-6)


def test_cake_varbin_eta_bin_auto():
    """Auto-computed eta_bin_size should match pixel resolution at tth_min."""
    nrows, ncols = 50, 50
    tth_lut = np.linspace(5, 20, ncols)[None, :] * np.ones(nrows)[:, None]
    eta_lut = np.zeros((nrows, ncols))
    image = np.ones((nrows, ncols))
    mask  = np.zeros((nrows, ncols), dtype=bool)

    _, _, _, eta_bin, _ = cake_varbin(
        image, mask, tth_lut, eta_lut,
        tth_min=5.0, tth_max=20.0, px=PX, lsd=LSD,
    )
    expected = np.degrees(PX / (LSD * np.tan(np.radians(5.0))))
    np.testing.assert_allclose(eta_bin, expected, rtol=1e-10)


# ── reduce_frame ────────────────────────────────────────────────────────────

def _make_geom(mode='varbin'):
    """Minimal geom dict for reduce_frame."""
    return {
        'tth_min': 2.5, 'tth_max': 11.5,
        'px': PX, 'lsd': LSD,
        'tth_bin_size': 0.1,
        'mode': mode,
    }


def test_reduce_frame_varbin_lineout_and_cake():
    nrows, ncols = 100, 100
    tth_lut = np.linspace(2, 12, ncols)[None, :] * np.ones(nrows)[:, None]
    eta_lut = np.linspace(-45, 45, nrows)[:, None] * np.ones(ncols)[None, :]
    image = np.ones((nrows, ncols)) * 42.0
    mask  = np.zeros((nrows, ncols), dtype=bool)

    res = reduce_frame(image, mask, tth_lut, eta_lut, _make_geom('varbin'))

    # Lineout keys
    assert 'tth' in res and 'I' in res and 'bg' in res and 'I_sub' in res
    populated = ~np.isnan(res['I'])
    np.testing.assert_allclose(res['I'][populated], 42.0, rtol=1e-6)

    # Cake keys
    assert 'cake_img' in res and 'tth_cake' in res and 'eta_cake' in res
    assert 'eta_bin_size' in res  # varbin-specific


def test_reduce_frame_unibin():
    nrows, ncols = 100, 100
    tth_lut = np.linspace(2, 12, ncols)[None, :] * np.ones(nrows)[:, None]
    eta_lut = np.linspace(-45, 45, nrows)[:, None] * np.ones(ncols)[None, :]
    image = np.ones((nrows, ncols)) * 7.0
    mask  = np.zeros((nrows, ncols), dtype=bool)

    res = reduce_frame(image, mask, tth_lut, eta_lut, _make_geom('unibin'))
    populated = ~np.isnan(res['I'])
    np.testing.assert_allclose(res['I'][populated], 7.0, rtol=1e-6)
    assert 'cake_img' in res
    assert 'eta_bin_size' not in res  # unibin does not return eta_bin_size


def test_reduce_frame_lineout_only():
    nrows, ncols = 50, 50
    tth_lut, eta_lut = _make_uniform_luts(nrows, ncols, tth_min=2.0, tth_max=12.0)
    image = np.ones((nrows, ncols))
    mask  = np.zeros((nrows, ncols), dtype=bool)

    res = reduce_frame(image, mask, tth_lut, eta_lut, _make_geom(), cake_out=False)
    assert 'tth' in res
    assert 'cake_img' not in res


def test_reduce_frame_cake_only():
    nrows, ncols = 50, 50
    tth_lut = np.linspace(2, 12, ncols)[None, :] * np.ones(nrows)[:, None]
    eta_lut = np.linspace(-45, 45, nrows)[:, None] * np.ones(ncols)[None, :]
    image = np.ones((nrows, ncols))
    mask  = np.zeros((nrows, ncols), dtype=bool)

    res = reduce_frame(image, mask, tth_lut, eta_lut, _make_geom(), lineout=False)
    assert 'cake_img' in res
    assert 'tth' not in res
