"""
End-to-end calibration and reduction example for midas4pil.
Covers every public API in midas4pil.__all__.

Usage:
    python run_calibration.py                   # uses ./geometry_init.toml
    python run_calibration.py my_config.toml    # uses custom config file

The user edits geometry_init.toml (or a copy) with their experimental
parameters.  This script reads that file, runs calibration, and produces:

    <BASE>_geometry.toml          -- calibrated geometry (TOML, for midas4pil)
    <BASE>_geometry.poni          -- .poni format export (for cross-checking)
    <BASE>_geometry_midas.txt     -- MIDAS geometry_params.txt export
    <BASE>_panel_shifts.txt       -- per-panel corrections (MIDAS text format)
    <BASE>_auto_mask.tif          -- auto-generated mask (if no mask provided)
    <BASE>_lineout_conventional.xy   -- 6-column lineout (Stage 2, no panels)
    <BASE>_lineout_panels.xy         -- 6-column lineout (panel-corrected)
    <BASE>_lineout_global.xy         -- 6-column lineout (informational)
    <BASE>_lineout_rebinned.xy       -- rebinned lineout (factor 4, varbin)
    <BASE>_cake_conventional.tif     -- caked image (Stage 2)
    <BASE>_cake_panels.tif           -- caked image (panel-corrected)
    <BASE>_cake_global.tif           -- caked image (informational)
    <BASE>_comparison.png            -- 3-way comparison figure

Alternative: start from an existing MIDAS geometry_params.txt instead of
geometry_init.toml — see the "MIDAS params import" note in STEP 3.

No external calibration tool required.

Columns in 6-column .xy lineout files:
    2theta_deg  intensity  snip_bg  I_sub  sigma  px_cnt
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

# ── All public API imports from midas4pil ────────────────────────────────
from midas4pil import (
    # geometry
    build_tilt_matrix, pixel_to_r_eta, r_to_tth, build_lut,
    pixel_resolution, varbin_tth_edges, lut_tth_range,
    find_beam_center, find_beam_center_auto, fit_circle, lsd_from_ring,
    # integration
    snip_background, integrate_1d, integrate_1d_varbin,
    reduce_frame, precompute_bin_maps, rebin_lineout,
    # caking
    cake, cake_varbin,
    # panel corrections
    make_panel_id_map, read_panel_shifts, save_panel_shifts,
    apply_panel_offsets, build_lut_with_panels,
    # I/O
    read_poni, write_poni, make_geometry,
    save_params, load_params,
    load_midas_params, save_midas_params,
    load_tiff, load_mask, orient_mask, auto_mask, save_mask,
    # calibrant
    read_jcpds, read_cif, ring_table, load_calibrant,
    # optimizer
    calibrate,
)
# find_ring_pixels / weighted_mean_positions are used internally by calibrate();
# import here only for the manual strain calculation in the comparison step.
from midas4pil.optimizer import find_ring_pixels, weighted_mean_positions


# ── Load configuration ───────────────────────────────────────────────────

config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('geometry_init.toml')
if not config_path.exists():
    print(f"Error: config file not found: {config_path}")
    print("Usage:  python run_calibration.py [geometry_init.toml]")
    sys.exit(1)

print(f"Reading configuration from: {config_path}")
with open(config_path, 'rb') as f:
    cfg = tomllib.load(f)

for section in ('input', 'detector', 'beam', 'geometry'):
    if section not in cfg:
        print(f"Error: missing [{section}] section in {config_path}")
        sys.exit(1)

config_dir = config_path.resolve().parent

# Input files
IMAGE_FILE     = config_dir / cfg['input']['image']
mask_val       = cfg['input'].get('mask', '')
MASK_FILE      = config_dir / mask_val if mask_val else None
CALIBRANT_NAME = cfg['input']['calibrant']

# Detector
det_cfg    = cfg['detector']
NROWS      = int(det_cfg['nrows'])
NCOLS      = int(det_cfg['ncols'])
PIXEL_SIZE = float(det_cfg['px_um'])

# Beam
WAVELENGTH = float(cfg['beam']['wavelength_A'])

# Geometry starting values
geo_cfg    = cfg['geometry']
LSD_APPROX = float(geo_cfg['lsd_um'])
BC_Y_INIT  = float(geo_cfg['bc_y_px'])
BC_Z_INIT  = float(geo_cfg['bc_z_px'])
TX_DEG     = float(geo_cfg['tx_deg'])
TY_DEG     = float(geo_cfg['ty_deg'])
TZ_DEG     = float(geo_cfg['tz_deg'])
P0         = float(geo_cfg['p0'])
P1         = float(geo_cfg['p1'])
P2         = float(geo_cfg['p2'])
P3         = float(geo_cfg['p3_deg'])
P4         = float(geo_cfg['p4'])
RHO_D      = float(geo_cfg['rho_d_um'])

# Integration limits
intg    = cfg.get('integration', {})
MODE    = intg.get('mode', 'varbin')
TTH_MIN = float(intg.get('tth_min_deg', 2.0))
TTH_MAX = float(intg.get('tth_max_deg', 29.0))
TTH_BIN = float(intg.get('tth_bin_deg', 0.025))
ETA_MIN = float(intg.get('eta_min_deg', -180.0))
ETA_MAX = float(intg.get('eta_max_deg', 180.0))
ETA_BIN = float(intg.get('eta_bin_deg', 1.0))

if MODE not in ('varbin', 'unibin'):
    print(f"Error: mode must be 'varbin' or 'unibin', got '{MODE}'")
    sys.exit(1)

# Calibration control
cal_cfg       = cfg.get('calibration', {})
TTH_ROUGH_MAX = float(cal_cfg.get('tth_rough_max_deg', 15.0))

# Output filenames
BASE = IMAGE_FILE.stem
HERE = IMAGE_FILE.parent

_cal_rel       = config_dir / CALIBRANT_NAME
CALIBRANT_FILE = str(_cal_rel) if _cal_rel.exists() else CALIBRANT_NAME
print(f"  Calibrant: {CALIBRANT_NAME}")


# ======================================================================
# STEP 1: Load image and create mask
# ======================================================================
print("="*70)
print("  STEP 1: Load image and create mask")
print("="*70)

image = load_tiff(IMAGE_FILE).astype(np.float64)
print(f"  Image: {IMAGE_FILE.name}  shape={image.shape}  max={image.max():.0f}")

nrows, ncols = image.shape

# Build panel map.  For a tiled detector, specify panel layout in the TOML
# [detector] section (n_panels_y, n_panels_z, panel_size_y, panel_size_z,
# gap_y, gap_z).  If absent, every pixel belongs to one panel (single-panel
# or unknown detector).
#
# Example for Pilatus 2M CdTe (3×8 panels, 487×195 px, 7/17 px gaps):
#   panel_map = make_panel_id_map(1679, 1475, 3, 8, 487, 195, gap_y=7, gap_z=17)
#
# Example for Pilatus 300K (3×1 panels, 487×195 px, 7 px gap):
#   panel_map = make_panel_id_map(195, 1475, 3, 1, 487, 195, gap_y=7, gap_z=0)
#
if 'n_panels_y' in det_cfg:
    panel_map = make_panel_id_map(
        nrows, ncols,
        int(det_cfg['n_panels_y']), int(det_cfg['n_panels_z']),
        int(det_cfg['panel_size_y']), int(det_cfg['panel_size_z']),
        gap_y=int(det_cfg.get('gap_y', 0)),
        gap_z=int(det_cfg.get('gap_z', 0)),
    )
    print(f"  Panel map: {panel_map.max()} panels  "
          f"({(panel_map == 0).sum()} gap pixels)")
else:
    panel_map = np.ones((nrows, ncols), dtype=np.int32)
    print(f"  Panel map: single panel (no layout specified in [detector])")

# Load user mask or auto-generate.
# orient_mask corrects the orientation of a mask that may have been saved in
# a different frame (e.g. pyFAI flips top-bottom); it reads the panel map and
# optionally an image to detect the required orientation automatically.
if MASK_FILE is not None and MASK_FILE.exists():
    # load_mask with panel_map + image calls orient_mask internally.
    mask = load_mask(MASK_FILE, panel_map=panel_map, image=image)
    print(f"  User mask:  {mask.sum()} bad pixels ({100*mask.mean():.2f} %)")
else:
    mask = auto_mask(image, panel_map=panel_map)
    print(f"  Auto-mask:  {mask.sum()} bad pixels ({100*mask.mean():.2f} %)")
    # Save auto-generated mask so it can be reused and edited.
    AUTO_MASK_FILE = HERE / f'{BASE}_auto_mask.tif'
    save_mask(mask, AUTO_MASK_FILE)
    print(f"  Saved {AUTO_MASK_FILE.name}  (reuse with mask = \"{AUTO_MASK_FILE.name}\")")

# Always mask gap pixels (panel_map == 0).
mask = mask | (panel_map == 0)


# ======================================================================
# STEP 2: Find beam centre
# ======================================================================
print(f"\n{'='*70}")
print("  STEP 2: Find beam centre from diffraction pattern")
print("="*70)

bc_y, bc_z = find_beam_center_auto(image, mask,
                                   bc_y_init=BC_Y_INIT, bc_z_init=BC_Z_INIT)
print(f"  Initial (from TOML): bc_y = {BC_Y_INIT:.1f} px,  bc_z = {BC_Z_INIT:.1f} px")
print(f"  Auto-detected:       bc_y = {bc_y:.2f} px,  bc_z = {bc_z:.2f} px")

# ── Manual alternative: fit_circle + lsd_from_ring ───────────────────
# If find_beam_center_auto fails (very off-centre, few rings visible),
# fit a circle to manually selected ring pixels and derive Lsd.
#
#   # Select pixels on the first CeO2 ring (example: ring 0)
#   first_ring_tth = rings_all[0]['tth']     # 2theta in degrees
#   # ... interactively select pixel positions (y, z) on that ring ...
#   ring_yz = np.array([[y1,z1], [y2,z2], ...])   # shape (N, 2)
#   cy, cz, radius_px = fit_circle(ring_yz)        # centre + radius in pixels
#   lsd_manual = lsd_from_ring(radius_px, PIXEL_SIZE, first_ring_tth)
#   print(f"  Manual: bc_y={cy:.2f}  bc_z={cz:.2f}  lsd={lsd_manual:.1f} um")
#
# fit_circle uses an algebraic least-squares algorithm (Kåsa method).
# lsd_from_ring: Lsd = radius_px * px / tan(tth_deg * pi/180).
# ─────────────────────────────────────────────────────────────────────


# ======================================================================
# STEP 3: Build initial geometry
# ======================================================================
print(f"\n{'='*70}")
print("  STEP 3: Build initial geometry")
print("="*70)

# ── MIDAS params import alternative ──────────────────────────────────
# If a MIDAS geometry_params.txt exists, load it instead of building from
# TOML values.  The function handles BC_Z frame conversion (ImTransOpt 2)
# and auto-loads PanelShiftsFile if present.
#
#   MIDAS_INIT = Path("ceo2_params.txt")
#   if MIDAS_INIT.exists():
#       geom = load_midas_params(str(MIDAS_INIT))
#       bc_y, bc_z = geom['bc_y'], geom['bc_z']
#       print(f"  Loaded MIDAS params: lsd={geom['lsd']:.1f}  "
#             f"bc=({bc_y:.2f}, {bc_z:.2f})")
#
# ─────────────────────────────────────────────────────────────────────

geom = make_geometry(
    wavelength=WAVELENGTH, lsd=LSD_APPROX, px=PIXEL_SIZE,
    nrows=NROWS, ncols=NCOLS,
    bc_y=bc_y, bc_z=bc_z,
    tx_deg=TX_DEG, ty_deg=TY_DEG, tz_deg=TZ_DEG,
    p0=P0, p1=P1, p2=P2, p3=P3, p4=P4,
    rho_d=RHO_D,
    tth_min=TTH_MIN, tth_max=TTH_MAX, tth_bin_size=TTH_BIN,
    eta_min=ETA_MIN, eta_max=ETA_MAX, eta_bin_size=ETA_BIN,
)

print(f"  lsd  = {geom['lsd']:.1f} um (approximate)")
print(f"  bc_y = {geom['bc_y']:.2f} px   bc_z = {geom['bc_z']:.2f} px")

# ── Geometry utilities ────────────────────────────────────────────────
# pixel_resolution: returns (delta_tth, delta_eta) in degrees per pixel.
delta_tth, delta_eta = pixel_resolution(TTH_MIN, geom['px'], geom['lsd'])
print(f"  Pixel resolution at {TTH_MIN:.1f} deg:  "
      f"radial {delta_tth*1000:.3f} mdeg/px,  azimuthal {delta_eta*1000:.3f} mdeg/px")

# varbin_tth_edges: the bin-edge array that integrate_1d_varbin uses internally.
edges = varbin_tth_edges(TTH_MIN, TTH_MAX, geom['px'], geom['lsd'])
print(f"  Varbin: {len(edges)-1} bins  ({edges[0]:.3f} – {edges[-1]:.3f} deg)")
# ─────────────────────────────────────────────────────────────────────


# ======================================================================
# STEP 4: Load calibrant rings
# ======================================================================
print(f"\n{'='*70}")
print("  STEP 4: Load calibrant rings")
print("="*70)

rings_rough = load_calibrant(CALIBRANT_FILE, WAVELENGTH, tth_max=TTH_ROUGH_MAX)
rings_all   = load_calibrant(CALIBRANT_FILE, WAVELENGTH, tth_max=TTH_MAX)
print(f"  {len(rings_rough)} low-angle rings (< {TTH_ROUGH_MAX} deg)")
print(f"  {len(rings_all)} rings up to {TTH_MAX} deg")

# ── Lower-level calibrant API ─────────────────────────────────────────
# load_calibrant calls read_jcpds or read_cif + ring_table internally.
# Use these directly if you need the full crystallographic data dict
# or want to inspect all rings without a 2theta cut-off.
#
#   cal_data = read_jcpds(CALIBRANT_FILE)   # or read_cif(CALIBRANT_FILE)
#   # cal_data: dict with 'a','b','c','alpha','beta','gamma','v','reflections'
#   all_rings = ring_table(cal_data, WAVELENGTH)   # array of dicts: tth, d, hkl
#   print(f"  {len(all_rings)} total reflections in calibrant file")
#   for r in all_rings[:5]:
#       print(f"    {r['hkl']}  d={r['d']:.4f} A  2theta={r['tth']:.4f} deg")
# ─────────────────────────────────────────────────────────────────────


# ======================================================================
# STEP 5: Stage 1 -- Rough calibration (Lsd + BC, low-angle rings)
# ======================================================================
print(f"\n{'='*70}")
print(f"  STEP 5: Stage 1 -- rough calibration (rings < {TTH_ROUGH_MAX} deg)")
print("="*70)

result_rough = calibrate(
    image, mask, geom, panel_map, rings_rough,
    optimize_shifts=False,
    tth_tol_factor=10.0,
    tol_lsd=LSD_APPROX * 0.05,
    tol_bc=10.0,
    tol_tilts=0.01,
    n_iterations=3,
    verbose=True,
)

g1 = result_rough['geom']
print(f"\n  Rough result: lsd={g1['lsd']:.1f}  bc=({g1['bc_y']:.2f}, {g1['bc_z']:.2f})")
print(f"  Mean strain: {result_rough['mean_strain']*1e6:.0f} ppm")


# ======================================================================
# STEP 6: Stage 2 -- Fine calibration (all rings, full optimization)
# ======================================================================
print(f"\n{'='*70}")
print("  STEP 6: Stage 2 -- fine calibration (all rings)")
print("="*70)

result_fine = calibrate(
    image, mask, result_rough['geom'], panel_map, rings_all,
    optimize_shifts=False,
    tol_lsd=1000.0,
    tol_bc=5.0,
    tol_tilts=1.0,
    tol_p3=45.0,
    n_iterations=5,
    verbose=True,
)

g2 = result_fine['geom']
print(f"\n  Fine result: lsd={g2['lsd']:.1f}  bc=({g2['bc_y']:.2f}, {g2['bc_z']:.2f})")
print(f"  Mean strain: {result_fine['mean_strain']*1e6:.0f} ppm")


# ======================================================================
# STEP 7: Save calibrated geometry (fine, no panel corrections)
# ======================================================================
print(f"\n{'='*70}")
print("  STEP 7: Save Stage 2 geometry")
print("="*70)

GEOM_TOML      = HERE / f'{BASE}_geometry.toml'
PONI_FILE      = HERE / f'{BASE}_geometry.poni'
MIDAS_FILE     = HERE / f'{BASE}_geometry_midas.txt'

save_params(g2, GEOM_TOML)
print(f"  Saved {GEOM_TOML.name}  (midas4pil TOML format)")

write_poni(g2, PONI_FILE)
print(f"  Saved {PONI_FILE.name}  (.poni format for cross-checking)")

# Export MIDAS geometry_params.txt.  BC_Z is converted to the MIDAS flipped
# frame (ImTransOpt 2 convention).  Panel topology is auto-detected from
# detector shape.  Commented calibrant and tolerance placeholders are included.
save_midas_params(g2, MIDAS_FILE)
print(f"  Saved {MIDAS_FILE.name}  (MIDAS geometry_params.txt format)")

# ── read_poni: import geometry from an existing .poni file ────────────
# geom_from_poni = read_poni(PONI_FILE, px=PIXEL_SIZE, nrows=NROWS)
# print(f"  Poni round-trip: bc_z = {geom_from_poni['bc_z']:.3f}")
# ─────────────────────────────────────────────────────────────────────


# ======================================================================
# STEP 8: Per-panel calibration (one-time per detector)
# ======================================================================
print(f"\n{'='*70}")
print("  STEP 8: Stage 3 -- per-panel calibration")
print("="*70)

result_panels = calibrate(
    image, mask, result_fine['geom'], panel_map, rings_all,
    optimize_shifts=True,
    tol_lsd=1000.0,
    tol_bc=5.0,
    tol_tilts=1.0,
    tol_shifts=5.0,
    tol_p3=45.0,
    n_iterations=5,
    verbose=True,
)

g3 = result_panels['geom']
print(f"\n  Panel result: lsd={g3['lsd']:.1f}  bc=({g3['bc_y']:.2f}, {g3['bc_z']:.2f})")
print(f"  Mean strain: {result_panels['mean_strain']*1e6:.0f} ppm")

g3['panel_shifts'] = result_panels['panel_shifts']

# save_params embeds panel shifts as [[panel_shifts]] TOML tables at end of file.
save_params(g3, GEOM_TOML)
print(f"  Updated {GEOM_TOML.name}  (includes panel_shifts)")

# Also write a MIDAS-format panel_shifts.txt (for MIDAS binary tools).
SHIFTS_FILE = HERE / f'{BASE}_panel_shifts.txt'
save_panel_shifts(result_panels['panel_shifts'], SHIFTS_FILE)
print(f"  Saved {SHIFTS_FILE.name}  (MIDAS text format: id dY dZ dTheta dLsd dP2)")

# Export MIDAS params with panel shifts embedded via PanelShiftsFile.
MIDAS_FILE_P  = HERE / f'{BASE}_geometry_midas_panels.txt'
save_midas_params(g3, MIDAS_FILE_P, panel_shifts_path=str(SHIFTS_FILE))
print(f"  Saved {MIDAS_FILE_P.name}  (references {SHIFTS_FILE.name})")

print("\n  Panel shifts (non-zero):")
for ps in result_panels['panel_shifts']:
    if abs(ps['dY']) > 0.001 or abs(ps['dZ']) > 0.001:
        print(f"    panel {ps['id']:2d}:  dY={ps['dY']:+.4f} px  dZ={ps['dZ']:+.4f} px  "
              f"dLsd={ps['dLsd']:+.1f} um")


# ======================================================================
# STEP 9: Data reduction -- build LUTs and reduce calibrant frame
# ======================================================================
print(f"\n{'='*70}")
print(f"  STEP 9: Data reduction ({MODE} mode)")
print("="*70)

lut_keys = ['nrows','ncols','bc_y','bc_z','lsd','px',
            'tx_deg','ty_deg','tz_deg','p0','p1','p2','p3','p4','rho_d']

panel_shifts = g3['panel_shifts']

_LO_HDR = '2theta_deg  intensity  snip_bg  I_sub  sigma  px_cnt'
_LO_FMT = '%.8f'

# --- 9a. Conventional: Stage 2 geometry, no panel corrections -----------
print("  [conventional — Stage 2 geometry, no panels]")

tth_lut_c, eta_lut_c = build_lut(**{k: g2[k] for k in lut_keys})

# lut_tth_range: useful sanity check — actual 2theta range of unmasked pixels.
tth_lo, tth_hi = lut_tth_range(tth_lut_c, mask=mask)
print(f"    Data 2theta range: {tth_lo:.3f} – {tth_hi:.3f} deg")

# Batch reduction pattern: precompute_bin_maps builds the bin-index arrays
# once, then reduce_frame re-uses them for every subsequent frame.
# Requires numba.  Falls back to per-frame assignment if not available.
try:
    bin_maps_c = precompute_bin_maps(mask, tth_lut_c, eta_lut_c, g2)
    res_c = reduce_frame(image, mask, tth_lut_c, eta_lut_c, g2,
                         bin_maps=bin_maps_c)
    print(f"    precompute_bin_maps: {bin_maps_c['n_bins_1d']} 1D bins  "
          f"{bin_maps_c.get('n_tth_2d', '?')} x {bin_maps_c.get('n_eta', '?')} cake bins")
except RuntimeError:
    res_c = reduce_frame(image, mask, tth_lut_c, eta_lut_c, g2)

# 6-column lineout: 2theta, I, SNIP background, I_sub, sigma, px_cnt
LINEOUT_C = HERE / f'{BASE}_lineout_conventional.xy'
np.savetxt(LINEOUT_C,
           np.column_stack([res_c['tth'], res_c['I'], res_c['bg'],
                            res_c['I_sub'], res_c['sigma'], res_c['px_cnt']]),
           header=_LO_HDR, fmt=_LO_FMT)
print(f"    Saved {LINEOUT_C.name}  ({len(res_c['tth'])} bins)")

CAKE_C = HERE / f'{BASE}_cake_conventional.tif'
tifffile.imwrite(str(CAKE_C), res_c['cake_img'].T.astype(np.float32))
print(f"    Saved {CAKE_C.name}")

# ── Direct integration API (equivalent to what reduce_frame calls) ────
# integrate_1d_varbin: pixel-matched variable bins (the recommended mode).
# Use this directly when you need only the lineout without the cake, or
# when you want to pass pre-computed tth_edges for multiple frames.
tth_v, I_v, bg_v, I_sub_v, sigma_v, px_cnt_v = integrate_1d_varbin(
    image, mask, tth_lut_c,
    g2['tth_min'], g2['tth_max'], g2['px'], g2['lsd'],
)

# integrate_1d: uniform bin width (unibin mode, e.g. for PDF analysis).
tth_u, I_u, bg_u, I_sub_u, sigma_u, px_cnt_u = integrate_1d(
    image, mask, tth_lut_c,
    g2['tth_min'], g2['tth_max'], tth_bin_size=TTH_BIN,
    eta_lut=eta_lut_c, eta_min=ETA_MIN, eta_max=ETA_MAX,
)
print(f"    integrate_1d_varbin: {len(tth_v)} bins  |  "
      f"integrate_1d (unibin): {len(tth_u)} bins")

# rebin_lineout: merge factor consecutive bins.  Exact Poisson propagation.
# Useful when you want a coarser pattern for display or peak fitting.
REBIN_FACTOR = 4
tth_r, I_r, px_cnt_r, sigma_r = rebin_lineout(tth_v, I_v, px_cnt_v,
                                               factor=REBIN_FACTOR)
LINEOUT_R = HERE / f'{BASE}_lineout_rebinned.xy'
np.savetxt(LINEOUT_R,
           np.column_stack([tth_r, I_r, snip_background(I_r),
                            I_r - snip_background(I_r), sigma_r, px_cnt_r]),
           header=_LO_HDR, fmt=_LO_FMT)
print(f"    rebin_lineout (×{REBIN_FACTOR}): {len(tth_r)} bins → saved {LINEOUT_R.name}")

# snip_background: direct SNIP call (also called internally by integrate_1d*).
# Useful to re-estimate background after rebinning or manual editing.
bg_direct = snip_background(I_v)
assert np.allclose(bg_direct, bg_v, equal_nan=True), "SNIP mismatch"

# cake / cake_varbin: direct caking API (equivalent to the cake part of
# reduce_frame).  Use when you need the cake without a lineout.
#
#   cake_img, tth_cake, eta_cake = cake_varbin(
#       image, mask, tth_lut_c, eta_lut_c,
#       tth_min=g2['tth_min'], tth_max=g2['tth_max'],
#       px=g2['px'], lsd=g2['lsd'],
#       eta_min=ETA_MIN, eta_max=ETA_MAX,
#   )
# ─────────────────────────────────────────────────────────────────────

# --- 9b. Panel-corrected: Stage 3 geometry + panel shifts ---------------
print("  [panel-corrected — Stage 3 geometry + panel shifts]")
tth_lut_p, eta_lut_p = build_lut_with_panels(
    **{k: g3[k] for k in lut_keys},
    panel_map=panel_map,
    panel_shifts=panel_shifts,
)
res_p = reduce_frame(image, mask, tth_lut_p, eta_lut_p, g3)

LINEOUT_P = HERE / f'{BASE}_lineout_panels.xy'
np.savetxt(LINEOUT_P,
           np.column_stack([res_p['tth'], res_p['I'], res_p['bg'],
                            res_p['I_sub'], res_p['sigma'], res_p['px_cnt']]),
           header=_LO_HDR, fmt=_LO_FMT)
print(f"    Saved {LINEOUT_P.name}")

CAKE_P = HERE / f'{BASE}_cake_panels.tif'
tifffile.imwrite(str(CAKE_P), res_p['cake_img'].T.astype(np.float32))
print(f"    Saved {CAKE_P.name}")

# --- 9c. Global-only: Stage 3 geometry WITHOUT panel shifts -------------
print("  [global-only — Stage 3 geometry, no panels (informational)]")
tth_lut_g, eta_lut_g = build_lut(**{k: g3[k] for k in lut_keys})
res_g = reduce_frame(image, mask, tth_lut_g, eta_lut_g, g3)

LINEOUT_G = HERE / f'{BASE}_lineout_global.xy'
np.savetxt(LINEOUT_G,
           np.column_stack([res_g['tth'], res_g['I'], res_g['bg'],
                            res_g['I_sub'], res_g['sigma'], res_g['px_cnt']]),
           header=_LO_HDR, fmt=_LO_FMT)
print(f"    Saved {LINEOUT_G.name}")

CAKE_G = HERE / f'{BASE}_cake_global.tif'
tifffile.imwrite(str(CAKE_G), res_g['cake_img'].T.astype(np.float32))
print(f"    Saved {CAKE_G.name}")

# ── apply_panel_offsets: lower-level per-pixel correction maps ────────
# build_lut_with_panels calls this internally.  Use directly only if you
# need the raw dY/dZ/dLsd/dP2 arrays, e.g. for custom LUT construction.
#
#   dY_map, dZ_map, dLsd_map, dP2_map = apply_panel_offsets(
#       nrows, ncols, panel_map, panel_shifts)
# ─────────────────────────────────────────────────────────────────────


# ======================================================================
# STEP 10: Round-trip — reload geometry from TOML
# ======================================================================
print(f"\n{'='*70}")
print("  STEP 10: Geometry round-trip via load_params")
print("="*70)

g_reload = load_params(GEOM_TOML)
for key in ('lsd', 'bc_y', 'bc_z', 'ty_deg', 'tz_deg', 'wavelength'):
    assert abs(g3[key] - g_reload[key]) < 0.01, f"Round-trip mismatch: {key}"
print(f"  load_params round-trip OK  ({len(g_reload.get('panel_shifts', []))} panel shifts)")

# ── MIDAS params round-trip ───────────────────────────────────────────
g_midas_rt = load_midas_params(str(MIDAS_FILE))
for key in ('lsd', 'bc_y', 'bc_z', 'wavelength'):
    assert abs(g2[key] - g_midas_rt[key]) < 0.01, f"MIDAS round-trip mismatch: {key}"
print(f"  load_midas_params round-trip OK")
# ─────────────────────────────────────────────────────────────────────


# ======================================================================
# STEP 11: 3-way comparison figure
# ======================================================================
print(f"\n{'='*70}")
print("  STEP 11: 3-way comparison")
print("="*70)

# Compute mean strain for global-only (Stage 3 geometry, no panel shifts)
# using the lower-level ring-pixel API.
tth_tol = np.degrees(g3['px'] / g3['lsd'] * np.cos(
    np.radians(np.mean([r['tth'] for r in rings_all])))**2) * 3.0
ring_bins_g = find_ring_pixels(tth_lut_g, eta_lut_g, mask, rings_all,
                                tth_tol, ETA_BIN)
YMean_g, ZMean_g, IdealTth_g = weighted_mean_positions(image, ring_bins_g)
TRs_g = build_tilt_matrix(g3['tx_deg'], g3['ty_deg'], g3['tz_deg'])
R_px_g, _ = pixel_to_r_eta(YMean_g, ZMean_g, g3['bc_y'], g3['bc_z'],
                             TRs_g, g3['lsd'], g3['rho_d'],
                             g3['p0'], g3['p1'], g3['p2'], g3['p3'], g3['p4'],
                             g3['px'])
strain_g = float(np.mean(np.abs(
    1.0 - R_px_g * g3['px'] / (g3['lsd'] * np.tan(np.radians(IdealTth_g))))))

strain_conv  = result_fine['mean_strain']
strain_panel = result_panels['mean_strain']

print(f"  Conventional (Stage 2):     {strain_conv*1e6:.0f} ppm")
print(f"  Panel-corrected (Stage 3):  {strain_panel*1e6:.0f} ppm")
print(f"  Global-only (Stage 3 geom): {strain_g*1e6:.0f} ppm  (informational)")
print(f"  Panel improvement:          {strain_conv*1e6:.0f} -> {strain_panel*1e6:.0f} ppm "
      f"({(1 - strain_panel/strain_conv)*100:.0f}% reduction)")

fig, axes = plt.subplots(2, 3, figsize=(20, 10))

columns = [
    ('Conventional\n(best geometry, no panels)',
     f'{strain_conv*1e6:.0f} ppm', 'steelblue', res_c),
    ('Panel-corrected\n(geometry + panel shifts)',
     f'{strain_panel*1e6:.0f} ppm', 'orangered', res_p),
    ('Global-only (informational)\n(panel geom w/o shifts)',
     f'{strain_g*1e6:.0f} ppm', '0.5', res_g),
]

for j, (label, strain_str, color, res) in enumerate(columns):
    ax = axes[0, j]
    ax.plot(res['tth'], res['I'], lw=0.8, color=color)
    ax.set_title(f'{label}\n{strain_str}', fontsize=10)
    ax.set_xlabel('2theta (deg)')
    if j == 0:
        ax.set_ylabel('Intensity (counts/pixel)')

vlo, vhi = np.nanpercentile(res_p['cake_img'], [2, 98])
for j, (label, strain_str, color, res) in enumerate(columns):
    ax = axes[1, j]
    ax.imshow(res['cake_img'].T, aspect='auto', origin='lower',
              extent=[res['tth_cake'][0], res['tth_cake'][-1],
                      res['eta_cake'][0], res['eta_cake'][-1]],
              vmin=vlo, vmax=vhi, cmap='turbo')
    ax.set_xlabel('2theta (deg)')
    if j == 0:
        ax.set_ylabel('eta (deg)')

plt.suptitle(f'midas4pil calibration comparison — {BASE}', fontsize=13, y=1.01)
plt.tight_layout()
FIG_FILE = HERE / f'{BASE}_comparison.png'
plt.savefig(FIG_FILE, dpi=150, bbox_inches='tight')
print(f"  Saved {FIG_FILE.name}")


# ======================================================================
# SUMMARY
# ======================================================================
print(f"\n{'='*70}")
print("  SUMMARY")
print("="*70)
print(f"  Config:  {config_path}")
print(f"  Image:   {IMAGE_FILE.name}")
print(f"  Mode:    {MODE}")
print(f"  Initial:                  lsd={geom['lsd']:.0f}  bc=({geom['bc_y']:.2f}, {geom['bc_z']:.2f})")
print(f"  Stage 1 (rough):          lsd={g1['lsd']:.0f}  strain={result_rough['mean_strain']*1e6:.0f} ppm")
print(f"  Stage 2 (conventional):   lsd={g2['lsd']:.0f}  strain={strain_conv*1e6:.0f} ppm")
print(f"  Stage 3 (panel-corrected):lsd={g3['lsd']:.0f}  strain={strain_panel*1e6:.0f} ppm")
print(f"  Improvement:              {strain_conv*1e6:.0f} -> {strain_panel*1e6:.0f} ppm "
      f"({(1 - strain_panel/strain_conv)*100:.0f}% strain reduction)")
print()
print("  Output files:")
_files = [
    (GEOM_TOML,    "calibrated geometry (Stage 3, includes panel shifts)"),
    (PONI_FILE,    ".poni format (for cross-checking)"),
    (MIDAS_FILE,   "MIDAS geometry_params.txt (Stage 2)"),
    (MIDAS_FILE_P, "MIDAS geometry_params.txt (Stage 3, with PanelShiftsFile)"),
    (SHIFTS_FILE,  "per-panel shifts (MIDAS text format)"),
    (LINEOUT_C,    "conventional lineout (Stage 2, 6 columns)"),
    (LINEOUT_P,    "panel-corrected lineout (6 columns)"),
    (LINEOUT_G,    "global-only lineout (informational)"),
    (LINEOUT_R,    f"rebinned lineout (factor {REBIN_FACTOR})"),
    (CAKE_C,       "conventional caked image"),
    (CAKE_P,       "panel-corrected caked image"),
    (CAKE_G,       "global-only caked image (informational)"),
    (FIG_FILE,     "3-way comparison figure"),
]
for fpath, desc in _files:
    print(f"    {fpath.name:48s} {desc}")
print("\nDone.")
