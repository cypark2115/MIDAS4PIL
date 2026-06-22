# midas4pil

**Author:**    Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory (cypark@anl.gov)
**Co-author:** Claude Code (claude-sonnet-4-6, Anthropic AI)
**Version:**   1.1.3 (2026-04-27)

Lightweight powder / high-pressure diffraction reduction for
Pilatus and Eiger detectors at HPCAT, APS.

**Credit:** Hemant Sharma, X-ray Science Division, Argonne National Laboratory

## Detector-native integration: the working principle

A flat area detector has finite angular resolution that varies across
its face. At any pixel, the smallest angular step the detector can
physically distinguish is set by the pixel size, the sample-to-detector
distance, and the projection geometry:

```
delta(2theta) = px * cos^2(2theta) / Lsd       (radians)
delta(eta)    = px / (Lsd * tan(2theta))        (radians)
```

Both quantities _decrease_ at high 2theta --- the detector resolves
finer angles there --- but the point is that they are **finite and
position-dependent**. Any integration bin narrower than these limits
contains no new information. Neighbouring bins become correlated, peaks
appear artificially smooth, and the profile is oversampled beyond what
the physics allows.

midas4pil's pixel-matched "varbin" mode sets every bin width to exactly
the detector's angular resolution at that position. The result is the
true diffraction profile at the resolution the detector actually
delivers.

## What the output looks like

- **Pixel-matched 2theta step size.** Each bin spans exactly one pixel's angular
  footprint. The step size is wider at low 2theta (large pixel footprint) and
  narrower at high 2theta. No sub-pixel interpolation is performed.

- **Peaks appear at their true instrumental width.** The peak width you see
  is the genuine angular resolution of the detector at that angle, not a
  smoothed or interpolated approximation.

- **Non-uniform 2theta spacing.** Bin centres are not equally spaced
  because pixel angular resolution varies with 2theta. This is the
  physically correct representation.

- **Eta bin size in caked images is user-controlled.** The default eta
  bin width is **1 deg**.  The bin size and the lower 2theta cutoff (2th_min)
  are coupled — see [Pixel-matched binning: the math](#pixel-matched-binning-the-math)
  below for the formula and a worked example.  The user can override via
  `eta_bin_size` in the API or via the **Eta bin** spinbox in the GUI.

The raw image preserves all the angular information the detector captured.
midas4pil reports each bin at the angular resolution the detector physically
delivers, without interpolation or resampling.

## Pixel-matched binning: the math

midas4pil bins uniformly in R-space (radial distance from beam centre
in pixels), then converts to 2theta:

```
R = Lsd * tan(2theta) / px          (pixels)
```

With a bin width of dR = 1 pixel, each 2theta bin spans exactly one
pixel's angular footprint. This is equivalent to using RBinSize = 1 in the MIDAS integration
framework (which uses RBinSize in the same R-space parameterisation).

The bin edges snap to integer pixel boundaries in R-space rather than
to user-specified 2theta limits. This means the actual 2theta range of
a varbin integration will differ slightly from the requested range (by
less than one pixel's angular width at each end).

For the eta direction (caked images only), the bin size is set by the
user (default 1 deg) and is linked to the lower 2theta cutoff via:

```
delta(eta) [rad] = px / (Lsd * tan(2th_min))
```

This ensures that no eta bin is finer than one pixel anywhere in the
range.  Choosing a finer eta bin raises 2th_min; choosing a coarser bin
lowers 2th_min.  Both the cake and the lineout use the same 2th_min.
With 172 µm pixels (Pilatus) at Lsd ≈ 350 mm:

| Eta bin (deg) | 2th_min (deg) | Eta bins (360 deg) |
|:---:|:---:|:---:|
| 2.0 | ~0.8 | ~180 |
| 1.0 (default) | ~1.6 | ~360 |
| 0.5 | ~3.2 | ~720 |
| 0.2 | ~8.1 | ~1800 |

## Integration approach

| Feature | midas4pil |
|---------|-----------|
| 2th bin width | pixel-matched (dR = 1 px), non-uniform spacing |
| Pixel splitting | histogram — no sub-pixel interpolation |
| Eta bin width | user-set (default 1 deg); no sub-pixel bins anywhere |
| Panel corrections | full optimizer (per-panel dY, dZ, dTheta, dLsd, dP2) |
| Oversampling | none — 1 bin = 1 pixel |

## Features

- **Detector geometry correction** --- three correction layers, calibrated
  and applied in this order:
  1. **Flat detector geometry** (always applied) --- beam centre
     (BC_Y, BC_Z), sample-to-detector distance (Lsd), and detector
     tilts (ty, tz). This is the foundational model.
  2. **Per-panel module correction** (optional) --- per-module rigid-body
     shifts (dY, dZ, dLsd, dP2, dTheta) that account for physical
     misalignment of individual modules in tiled detectors (Pilatus/Eiger).
     Panel shifts are embedded in the geometry TOML towards the end of
     file. A separate MIDAS-format text file can be read from or written
     to via ``load_midas_params()`` / ``save_midas_params()`` for
     interoperability with the MIDAS binary tools.
  3. **Radial distortion** (optional) --- polynomial correction (p0--p4)
     for smooth deviations of the detector surface from the ideal flat
     plane (e.g. slight curvature of the sensor board).

  Panel shifts must be calibrated before distortion: the smooth polynomial
  in layer 3 is defined on the corrected coordinates from layer 2, and
  fitting them together without this ordering causes cross-contamination.
- **Bad-pixel mask** --- convention: 1 = bad, 0 = good.
- **1-D lineout** --- 4-column output: `2θ, I, σ, px_cnt`
  (2θ in degrees, I = mean intensity, σ = Poisson standard error of the
  mean per bin, px_cnt = number of active pixels in the bin). Column
  names are written explicitly in the file header.
  Background-subtracted 6-column output `[2θ, I, SNIP_bg, I_sub, σ, px_cnt]`
  saved to `lineouts_sub/` via Export I_sub…
- **Caked image** --- (2theta, eta) polar map saved as `{base}.tif`. Axis arrays (2θ, η bin centres in degrees) saved alongside as `{base}_axes.npz`.
- **SNIP background** --- Statistics-sensitive Non-linear Iterative
  Peak-clipping (Morhac et al., NIM A 401, 1997).
- **Dual mode** --- pixel-matched variable-bin (`varbin`, default) and
  uniform-bin (`unibin`) integration.  Varbin is the physically correct
  mode; unibin is available when uniform spacing is strictly required
  (e.g. FFT for pair distribution function).  Even then, bin width
  must not be finer than the pixel angular resolution.
- **.poni file import/export** --- load an existing `.poni` geometry
  file as a starting point (File > Load .poni), and export calibration
  results to `.poni` format.  The coordinate conversion
  (Poni1/Poni2 ↔ BC\_Z/BC\_Y with top-bottom flip) is implemented
  internally; no external library is required.

## Geometry model and calibration parameters

midas4pil uses a three-layer correction model, exactly matching MIDAS
`DetectorGeometry.c` and `CalibrantPanelShiftsOMP.c`. Each layer is
independently calibrated and stored separately.

### 1. Flat detector geometry (PONI)

Five parameters define the position and orientation of the ideal flat
detector plane with respect to the incident beam and sample position.
PONI stands for Point Of Normal Incidence.

| Parameter | Unit | Description |
|-----------|------|-------------|
| Lsd | µm | Sample-to-detector distance along the beam axis |
| BC\_Y | px | Beam centre column (from left, i.e. Poni2/px) |
| BC\_Z | px | Beam centre row (from top, i.e. Poni1/px) |
| ty | ° | Detector tilt around the Y axis (horizontal transverse) |
| tz | ° | Detector tilt around the Z axis (vertical transverse) |
| tx | ° | Detector rotation around the X axis (beam direction) — **fixed at 0** |

tx is fixed because rotating the detector around the beam axis (X)
only shifts the azimuthal reference (eta = 0 direction) and cannot be
determined from azimuthally uniform powder rings.
In `.poni` file convention: rot1 → tz, rot2 → ty.

The tilt matrices (angles in radians internally) are:

```
         ┌ 1    0       0    ┐          ┌  cos ty   0   sin ty ┐
Rx(tx) = │ 0   cos tx  -sin tx│   Ry(ty)= │  0        1   0     │
         └ 0   sin tx   cos tx┘          └ -sin ty   0   cos ty ┘

         ┌  cos tz  -sin tz   0 ┐
Rz(tz) = │  sin tz   cos tz   0 │        TRs = Rx(tx) · Ry(ty) · Rz(tz)
         └  0        0        1 ┘
```

The geometry maps each pixel (col, row) to scattering angles via:

```
Yc  = (BC_Y − col) · px              [µm, positive toward left]
Zc  = (BC_Z − row) · px              [µm, positive toward top]

[XYZ_x, XYZ_y, XYZ_z]ᵀ = TRs · [Lsd, Yc, Zc]ᵀ

Rad = (Lsd / XYZ_x) · √(XYZ_y² + XYZ_z²)    [µm, in-plane distance]
η   = atan2(XYZ_z, −XYZ_y)                    [degrees, CCW from 3 o'clock]
2θ  = arctan(R_corrected · px / Lsd)          [degrees]
```

where R_corrected is Rad after distortion correction (see Section 3).
This follows MIDAS `dg_pixel_to_REta()` exactly.

### 2. Panel shift correction

Tiled pixel detectors (Pilatus, Eiger) have small physical
misalignments between modules — each module is a separate circuit board
mounted independently. These shifts are a **hardware property of the
detector**. They do not change with X-ray energy, detector distance, or
calibrant. Calibrate once and reuse indefinitely.

Per-panel rigid-body corrections are applied to each pixel's (col, row)
coordinates **before** the global geometry transform.

| Parameter | Unit | Description |
|-----------|------|-------------|
| dY | µm | In-plane horizontal shift (column direction) |
| dZ | µm | In-plane vertical shift (row direction) |
| dLsd | µm | Axial offset along the beam (panel closer/farther) |
| dTheta | ° | In-plane rotation about the panel's geometric centre |
| dP2 | — | Per-panel additive offset to the global p2 distortion coefficient |

For panel i with centre (c_col, c_row) in image coordinates, the corrected
pixel position is:

```
θ = dTheta_i  [radians]

col' = (col − c_col) · cos θ + (row − c_row) · sin θ + c_col + dY_i / px
row' = (row − c_row) · cos θ − (col − c_col) · sin θ + c_row + dZ_i / px
Lsd' = Lsd + dLsd_i
p2'  = p2  + dP2_i
```

The corrected (col', row', Lsd', p2') are then passed to the global
pixel-to-angle transform. This matches `CalibrantPanelShiftsOMP.c`.

### 3. Distortion correction (p0–p4)

The distortion model corrects smooth deviations of the detector surface
from the ideal flat plane — for example, a slight spherical curvature of
the sensor board, a small indentation from a mounting bracket, or
asymmetric warping. It is applied as a multiplicative correction to the
effective pixel-to-sample distance **after** panel shifts have been
applied:

```
ρ_norm = Rad / rho_d                  [dimensionless; rho_d ≈ 217 578 µm for Pilatus 2M]
ηT     = 180° − η                     [MIDAS internal azimuth; makes ηT = 0 at beam right]

distort = 1 + p0 · ρ_norm² · cos(2 ηT)
            + p1 · ρ_norm⁴ · cos(4 ηT + p3)
            + p2 · ρ_norm²
            + p4 · ρ_norm⁶

R_corrected = (Rad · distort / px) · (Lsd / Lsd_panel)   [pixels]
```

where `rho_d` is a normalisation radius — the half-diagonal of the detector in µm
(default 217 578 µm for a Pilatus 2M CdTe). Note that ηT = 180° − η transforms
the panel convention so that p0–p4 coefficients are directly interchangeable with
those fitted by the MIDAS `CalibrantPanelShiftsOMP` binary.

| Parameter | Description |
|-----------|-------------|
| p0 | 2-fold azimuthal distortion (cos 2η): asymmetric left–right or top–bottom bowing |
| p1 | 4-fold azimuthal distortion (cos 4η): 4-fold symmetric warping |
| p2 | Isotropic radial distortion: uniform curvature (spherical bulge or depression) |
| p3 | Phase of the p1 term (°): rotates the 4-fold pattern; not an independent shape |
| p4 | 6th-order isotropic correction: higher-order radial curvature |

For most synchrotron flat-panel detectors p0–p4 are very small
(≪ 0.001). Enable "Fit distortion" only if per-ring strain residuals
show a systematic radial or azimuthal trend after panel-corrected
flat-geometry optimization. For Pilatus/Eiger sensors this is rarely
needed.

### Calibration optimization order

The three layers must be fitted in the order listed above. The reason
is structural: panel shifts alter pixel coordinates **before** the
geometry transform; the distortion polynomial is defined on the
already-transformed (R, η) coordinates that come **after** panel
correction. Fitting distortion before panels causes the smooth
polynomial to absorb the discrete per-module offsets, giving wrong
values for both.

The GUI calibration stages follow this order automatically:

| Scenario | Stage 1 | Stage 2 | Stage 3 |
|----------|---------|---------|---------|
| PONI only | rough alignment | full PONI | — |
| PONI + panels | rough alignment | full PONI | PONI + panels |
| PONI + distortion | rough alignment | PONI + distortion | — |
| PONI + panels + distortion | rough alignment | PONI + panels | PONI + panels + distortion |

Panel shifts are a one-time hardware calibration. Once determined on a
good calibrant image, load them via `build_lut_with_panels()` for all
subsequent experiments on the same detector — there is no need to
re-fit them when energy or distance changes.

### Calibration fit quality

After fitting, the optimizer reports a **geometry residual strain** for
each calibrant ring spot. This is NOT physical strain in the calibrant
material — it is the fractional d-spacing residual that remains after
the geometry optimisation:

For each pixel assigned to ring k, the fractional radius residual is:

```
strain_k = (R_obs − R_pred) / R_pred

where  R_pred = Lsd · tan(2θ_k) / px          [pixels, predicted ring radius]
       R_obs                                    [pixels, measured centroid]
       2θ_k  from calibrant d-spacing via Bragg's law: λ = 2 d sin θ
```

The optimizer minimises the mean absolute strain over all assigned pixels:

```
J = (1/N) · Σ |strain_k|        [dimensionless; report in ppm = × 10⁶]
```

In terms of d-spacing (first-order approximation for small strain):

```
strain ≈ (d_obs − d_ideal) / d_ideal  ≈  (sin θ_ideal − sin θ_obs) / sin θ_obs
```

where θ_ideal comes from the known calibrant crystal structure (e.g. CeO₂ JCPDS).
Note: the optimizer minimises the R-space residual; the d-spacing form is shown for
physical intuition only. The two expressions agree to first order in strain.

**Mean |strain|** is the primary figure of merit. Lower is better:

| Mean |strain| | Calibration quality |
|----------------|---------------------|
| < 200 ppm | Excellent |
| 200–500 ppm | Good for most HP diffraction work |
| 500–1000 ppm | Acceptable; consider wider tolerances or checking the mask |
| > 1000 ppm | Poor; likely a bad starting geometry or aggressive mask |

**Per-ring strain table** breaks down the residual by ring:

- Monotonically increasing or decreasing strain vs. 2θ → wrong Lsd or
  beam centre (the ring positions shift coherently with distance).
- Large strain on one ring only → bad ring assignment or a masked-out
  arc reducing the effective ring coverage on that reflection.
- All rings close to zero → geometry is well-determined.

**Points used** — number of ring pixels contributing to the fit. Low
values (< 5 000) may indicate an aggressive mask or a missed ring.

## Quick start

For a complete end-to-end example, see [`calib_example/`](calib_example/).
It demonstrates the full standalone workflow: auto-detect beam centre,
two-stage geometry optimization, per-panel calibration, and data
reduction. Fully standalone.

### Using the calibration script

1. Edit `calib_example/geometry_init.toml` with your experimental
   parameters (image path, wavelength, approximate distance, pixel size).
2. Run the script:

```bash
cd calib_example
python run_calibration.py                   # uses ./geometry_init.toml
python run_calibration.py my_config.toml    # uses a custom config file
```

Users only edit the TOML file -- the script does not need to be modified.

### Calibration from Python (programmatic)

```python
from midas4pil.io import make_geometry, save_params, load_tiff, auto_mask
from midas4pil.geometry import find_beam_center_auto
from midas4pil.calibrant import load_calibrant
from midas4pil.panels import make_panel_id_map
from midas4pil.optimizer import calibrate

image = load_tiff("calibrant.tif").astype("float64")
panel_map = make_panel_id_map(1679, 1475, 3, 8, 487, 195, 7, 17)  # Pilatus 2M CdTe
mask = auto_mask(image, panel_map)

bc_y, bc_z = find_beam_center_auto(image, mask)   # sharpness-based; ~10 px off on tilted detectors
geom = make_geometry(wavelength=0.42460, lsd=350000, px=172,
                     nrows=1679, ncols=1475, bc_y=bc_y, bc_z=bc_z)

rings = load_calibrant("CeO2.jcpds", geom['wavelength'], tth_max=29.0)
result = calibrate(image, mask, geom, panel_map, rings,
                   optimize_shifts=False, n_iterations=5)
save_params(result['geom'], "geometry.toml")
```

### Data reduction

```python
from midas4pil.io import load_params, load_tiff, load_mask
from midas4pil.geometry import build_lut
from midas4pil.integrate import integrate_1d, integrate_1d_varbin

geom  = load_params("geometry.toml")
tth_lut, eta_lut = build_lut(**{k: geom[k] for k in
    ['nrows','ncols','bc_y','bc_z','lsd','px',
     'tx_deg','ty_deg','tz_deg','p0','p1','p2','p3','p4','rho_d']})

image = load_tiff("frame.tif")
mask  = load_mask("mask.tif")

# Pixel-matched integration (default, physically correct resolution)
tth, I, bg, I_sub, sigma, px_cnt = integrate_1d_varbin(
    image, mask, tth_lut,
    geom['tth_min'], geom['tth_max'],
    px=geom['px'], lsd=geom['lsd'])

# Uniform-bin integration (only when downstream tools require equal spacing)
tth_u, I_u, bg_u, I_sub_u, sigma_u, px_cnt_u = integrate_1d(
    image, mask, tth_lut,
    geom['tth_min'], geom['tth_max'], geom['tth_bin_size'])
```

## Module reference

midas4pil is organized into focused modules. This section shows how to
use each one independently.

### `midas4pil.io` -- file I/O and geometry construction

```python
from midas4pil.io import (make_geometry, save_params, load_params,
                           load_midas_params, save_midas_params,
                           load_tiff, load_mask, auto_mask,
                           read_poni, write_poni)
```

**Build geometry from scratch** (no external calibration tool needed):

```python
geom = make_geometry(
    wavelength=0.42460,    # angstrom
    lsd=350000.0,          # um (approximate)
    px=172.0,              # um
    nrows=1679, ncols=1475,
    bc_y=748.8, bc_z=861.5,          # beam centre (pixels); None = detector centre
    tx_deg=0.0, ty_deg=0.0, tz_deg=0.0,   # tilts (degrees)
    p0=0, p1=0, p2=0, p3=0, p4=0,         # radial distortion
    rho_d=217578.0,                        # distortion reference radius (um)
    tth_min=2.0, tth_max=29.0, tth_bin_size=0.025,   # 2theta range/bin (degrees)
    eta_min=-180.0, eta_max=180.0, eta_bin_size=1.0,  # eta range/bin (degrees)
)
save_params(geom, "geometry.toml")
```

**Load/save geometry:**

```python
geom = load_params("geometry.toml")    # returns dict with all geometry keys
save_params(geom, "geometry.toml")     # write dict to TOML
```

**Import/export MIDAS geometry_params.txt:**

```python
# Import from MIDAS params file (full geometry, including panel shifts if PanelShiftsFile present)
geom = load_midas_params("geometry_params.txt")

# Export to MIDAS format (panel shifts written to companion file if present)
save_midas_params(geom, "geometry_params.txt",
                  panel_shifts_path="panel_shifts.txt")
```

The exported file contains `ImTransOpt 2` (standard for TIFF images) and
commented-out placeholders for calibrant and calibration tolerance keys.

**Import/export .poni geometry file** (optional):

```python
# Import from .poni (partial -- add ncols, distortion, integration limits)
geom = read_poni("calibration.poni", px=172.0, nrows=1679)
geom.update(ncols=1475, p0=0, p1=0, p2=0, p3=0, p4=0, rho_d=217578.,
            tth_min=2, tth_max=29, tth_bin_size=0.025,
            eta_min=-180, eta_max=180, eta_bin_size=1)

# Export to .poni
write_poni(geom, "geometry.poni")
```

**Image and mask loading:**

```python
image = load_tiff("frame.tif")           # preserves dtype; cast to float64 for reduction
mask  = load_mask("mask.tif",             # 1=bad, 0=good
                  panel_map=panel_map,    # optional: auto-detect orientation
                  image=image)            # optional: used for orientation check

# Or generate mask automatically (no mask file needed):
mask = auto_mask(image, panel_map=panel_map, hot_threshold=10.0)
# Identifies: gap pixels (panel_map==0), dead pixels (<=0), hot outliers (>10 MAD sigma)
```

### `midas4pil.geometry` -- detector geometry and lookup tables

```python
from midas4pil.geometry import (build_lut, find_beam_center_auto,
                                 pixel_resolution, varbin_tth_edges)
```

**Build lookup tables** (required once per geometry for all integration):

```python
lut_keys = ['nrows','ncols','bc_y','bc_z','lsd','px',
            'tx_deg','ty_deg','tz_deg','p0','p1','p2','p3','p4','rho_d']
tth_lut, eta_lut = build_lut(**{k: geom[k] for k in lut_keys})
# tth_lut: 2D array (nrows x ncols), 2theta in degrees for each pixel
# eta_lut: 2D array (nrows x ncols), eta (azimuth) in degrees for each pixel
```

**Auto-detect beam centre** from a powder diffraction image:

```python
bc_y, bc_z = find_beam_center_auto(image, mask,
                               bc_y_init=None, bc_z_init=None,  # None = detector centre
                               search_range=100,  # pixels to search around init
                               downsample=8)       # speedup factor for coarse pass
```

**Query pixel angular resolution** at a given 2theta:

```python
delta_tth = pixel_resolution(tth_deg=10.0, px=172.0, lsd=350000.0)
# Returns angular width of one pixel (degrees) at that 2theta
```

**Compute pixel-matched bin edges** (used internally by varbin functions):

```python
edges = varbin_tth_edges(tth_min=2.0, tth_max=29.0, px=172.0, lsd=350000.0, dR=1.0)
# Non-uniform 2theta edges where each bin spans exactly dR pixels in R-space
```

### `midas4pil.integrate` -- 1-D powder pattern lineouts

```python
from midas4pil.integrate import integrate_1d_varbin, integrate_1d, snip_background
```

**Pixel-matched integration** (default, bin width = detector angular
resolution):

```python
tth_centres, I, bg, I_sub, sigma, px_cnt = integrate_1d_varbin(
    image, mask, tth_lut,
    tth_min=2.0, tth_max=29.0,
    px=172.0, lsd=350000.0,      # needed to compute pixel resolution
    dR=1.0,                       # bins per pixel (1.0 = pixel-matched)
    eta_lut=eta_lut, eta_min=-180.0, eta_max=180.0,
    snip_iter=50,
)
# Returns: tth_centres, I, snip_bg, I_sub, sigma (Poisson s.e.m.), px_cnt (pixel count)
# Non-uniform 2theta spacing; each bin spans exactly one pixel's angular width
```

**Uniform-bin integration** (only when uniform spacing is strictly
required -- e.g. FFT for pair distribution function).  Bin width must
not be finer than the pixel angular resolution:

```python
tth_centres, I, bg, I_sub, sigma, px_cnt = integrate_1d(
    image, mask, tth_lut,
    tth_min=2.0, tth_max=29.0, tth_bin_size=0.025,   # degrees
    eta_lut=eta_lut, eta_min=-180.0, eta_max=180.0,   # optional eta wedge
    snip_iter=50,     # SNIP background iterations (0 to disable)
)
# Returns: tth_centres, I, snip_bg, I_sub, sigma (Poisson s.e.m.), px_cnt (pixel count)
```

**SNIP background** (standalone, for custom workflows):

```python
bg = snip_background(intensities, n_iter=50)
# Returns smoothed baseline estimate via iterative peak-clipping
```

The SNIP algorithm (Morhač et al., NIM A 401, 1997) applies a triple-log
compression before iteration to linearise Poisson statistics:

```
y(k)  = log( log( √(x(k) + 1) + 1 ) + 1 )          [forward compression]

S⁽ᵖ⁾(k) = min( S⁽ᵖ⁻¹⁾(k),  [S⁽ᵖ⁻¹⁾(k−p) + S⁽ᵖ⁻¹⁾(k+p)] / 2 )
                                                       [clipping iteration, p = 1..W]

x̂(k) = (exp(exp(Ŝ(k)) − 1) − 1)² − 1               [back-transform]
```

where W = n_iter is the half-width of the clipping window.  Peaks narrower
than 2W bins are removed; the background estimate x̂ tracks only broad,
slowly varying features.

### `midas4pil.cake` -- 2-D polar (2theta, eta) images

```python
from midas4pil.cake import cake_varbin, cake
```

**Pixel-matched cake** (default, resolution-matched in both directions):

```python
cake_img, tth_centres, eta_centres, eta_bin_size, px_cnt_map = cake_varbin(
    image, mask, tth_lut, eta_lut,
    tth_min=2.0, tth_max=29.0,
    px=172.0, lsd=350000.0,
    dR=1.0,                                            # pixel-matched 2theta bins
    eta_min=-180.0, eta_max=180.0,                     # eta bins auto-computed
)
# eta_bin_size: auto-computed eta bin width (degrees) from pixel resolution at tth_min
# px_cnt_map: 2-D int32 array of pixel counts per bin (same shape as cake_img)
# Save as TIFF:
import tifffile
tifffile.imwrite("cake.tif", cake_img.T.astype(np.float32))
```

**Uniform-bin cake** (for equally-spaced output):

```python
cake_img, tth_centres, eta_centres, px_cnt_map = cake(
    image, mask, tth_lut, eta_lut,
    tth_min=2.0, tth_max=29.0, tth_bin_size=0.05,    # degrees
    eta_min=-180.0, eta_max=180.0, eta_bin_size=1.0,  # degrees
)
# cake_img shape: (n_tth_bins, n_eta_bins)
# px_cnt_map: 2-D int32 array of pixel counts per bin (same shape as cake_img)
```

### `midas4pil.panels` -- tiled detector panel corrections

```python
from midas4pil.panels import (make_panel_id_map,
                               read_panel_shifts, build_lut_with_panels)
```

**Build panel map** for your detector:

```python
# Pilatus 2M CdTe (24 panels: 3 col × 8 row, 487×195 px, 7/17 px gaps)
panel_map = make_panel_id_map(1679, 1475, 3, 8, 487, 195, 7, 17)

# General form (keyword args):
panel_map = make_panel_id_map(
    nrows=1679, ncols=1475,
    n_panels_y=3, n_panels_z=8,         # number of modules (cols × rows)
    panel_size_y=487, panel_size_z=195, # module size in pixels
    gap_y=7, gap_z=17,                  # gap size in pixels (scalar or list)
)
# Returns: int32 array (nrows, ncols), panel ID per pixel (≥1), 0 = gap
```

**Build LUT with panel corrections** (replaces `build_lut` when panel
shifts are available):

```python
# Primary path: panel shifts embedded in the geometry TOML
geom = load_params("geometry.toml")          # includes panel_shifts if present
tth_lut, eta_lut = build_lut_with_panels(
    **{k: geom[k] for k in lut_keys},
    panel_map=panel_map,
    panel_shifts=geom['panel_shifts'],
)
# Use tth_lut, eta_lut with integrate_1d() or cake() as usual
```

**MIDAS interoperability** --- read/write the MIDAS whitespace-delimited
text format used by the MIDAS binary tools:

```python
from midas4pil.panels import read_panel_shifts, save_panel_shifts

panel_shifts = read_panel_shifts("panel_shifts.txt")   # read MIDAS text file
save_panel_shifts(panel_shifts, "panel_shifts.txt")    # write MIDAS text file
```

### `midas4pil.calibrant` -- calibrant ring positions

```python
from midas4pil.calibrant import load_calibrant
```

**Load calibrant rings** for calibration:

```python
rings = load_calibrant(
    "CeO2.jcpds",     # path to .jcpds or .cif file
    wavelength=0.42460, # angstrom
    tth_max=29.0,       # include rings up to this 2theta (degrees)
)
# Returns list of dicts: [{'tth': ..., 'h': ..., 'k': ..., 'l': ..., 'd': ...}, ...]
```

Bundled calibrants are in `midas4pil/_JCPDS/` (Oxides, Elements,
Fluorides, etc.) and `midas4pil/_CIF/`.

### `midas4pil.optimizer` -- geometry calibration

```python
from midas4pil.optimizer import calibrate
```

**Calibrate detector geometry** from a calibrant image:

```python
result = calibrate(
    image, mask, geom, panel_map, rings,
    optimize_shifts=False,       # True to refine per-panel dY, dZ
    fix_panel=1,                 # reference panel (held fixed) when optimize_shifts=True
    tth_tol_factor=3.0,          # ring assignment tolerance (multiples of pixel resolution)
    tol_lsd=500.0,               # Lsd search range (um)
    tol_bc=5.0,                  # beam centre search range (pixels)
    tol_tilts=1.0,               # tilt search range (degrees)
    tol_shifts=5.0,              # panel shift search range (pixels)
    tol_p3=45.0,                 # p3 distortion phase search range (degrees)
    n_iterations=5,              # optimizer iterations
    verbose=True,
)

# result keys:
#   'geom'          -- refined geometry dict
#   'mean_strain'   -- mean residual strain (lower is better; < 200 ppm is good)
#   'panel_shifts'  -- list of per-panel shift dicts (when optimize_shifts=True)

# When optimize_shifts=True, merge panel shifts into geom before saving:
geom = result['geom']
geom['panel_shifts'] = result['panel_shifts']   # embed shifts in the geometry dict
save_params(geom, "geometry.toml")              # one TOML file holds everything
```

See [Panel correction workflow](#panel-correction-workflow) for full details.

## Geometry input: TOML parameter file

midas4pil reads detector geometry from a TOML file. Units are encoded
in key names (`_um` = micrometres, `_px` = pixels, `_deg` = degrees,
`_A` = angstroms). A complete template:

```toml
# midas4pil geometry parameters

[detector]
nrows  = 1679              # detector rows    (NrPixelsZ)
ncols  = 1475              # detector columns (NrPixelsY)
px_um  = 172               # pixel size (um)

[geometry]
lsd_um    = 349510.0       # sample-to-detector distance (um)
bc_y_px   = 748.8          # beam-centre column from left
bc_z_px   = 861.5          # beam-centre row from top
tx_deg    = 0.0            # tilt about X axis (degrees)
ty_deg    = 0.189          # tilt about Y axis (degrees)
tz_deg    = -0.317         # tilt about Z axis (degrees)
p0        = 0              # radial distortion coefficients
p1        = 0
p2        = 0
p3_deg    = 0.0            # distortion phase (degrees)
p4        = 0
rho_d_um  = 217578         # distortion reference radius (um)

[beam]
wavelength_A = 0.42460     # X-ray wavelength (angstrom) = 12398.4193 / E(eV)

[integration]
mode         = "varbin"    # "varbin" (default) or "unibin"
tth_min_deg  = 2.0         # 2theta lower limit (degrees)
tth_max_deg  = 29.0        # 2theta upper limit (degrees)
tth_bin_deg  = 0.025       # 2theta bin size (degrees) -- unibin only, ignored in varbin
eta_min_deg  = -180        # eta lower limit (degrees)
eta_max_deg  = 180         # eta upper limit (degrees)
eta_bin_deg  = 1.0         # eta bin size (degrees) -- unibin only, ignored in varbin
```

**varbin** (default): bin widths match the detector's angular resolution
at each 2theta. This is the physically correct representation -- no
artificial oversampling. `tth_bin_deg` and `eta_bin_deg` are ignored.

**unibin**: user-specified uniform bin widths. Use only when uniform
spacing is strictly required (e.g. FFT for pair distribution function).
Even then, set `tth_bin_deg` no smaller than the detector's pixel angular
resolution -- bins finer than the pixel footprint create information the
detector cannot physically deliver.

To create this file programmatically:

```python
from midas4pil.io import make_geometry, save_params

# From scratch (no external tools needed):
geom = make_geometry(wavelength=0.42460, lsd=350000, px=172,
                     nrows=1679, ncols=1475)
save_params(geom, "geometry.toml")

# Or from an existing .poni geometry file:
from midas4pil.io import read_poni
geom = read_poni("calibration.poni", px=172.0, nrows=1679)
geom.update(ncols=1475, p0=0, p1=0, p2=0, p3=0, p4=0, rho_d=217578.)
save_params(geom, "geometry.toml")
```

Then edit the TOML to adjust integration limits, tilts, or distortion
parameters as needed.

## Panel correction workflow

Tiled pixel detectors (Pilatus, Eiger) have small physical
misalignments between modules. midas4pil can calibrate and correct
these per-panel shifts.

**Key concept:** panel shifts are a hardware property of the detector.
They do not change when you change X-ray energy, detector distance, or
calibrant. Calibrate them once and reuse for all experiments on that
detector.

**GUI workflow:** The GUI embeds panel shifts inside the geometry TOML
file (`[[panel_shifts]]` tables). After a panel-corrected calibration,
save the TOML — the shifts are included automatically. Load that TOML
in a new session and the GUI applies the corrections without a separate
file. There is no standalone "Load Panel Shifts…" button in the GUI.

**Python API / script workflow:** Use the TOML-embedded path shown
below unless you need to exchange data with the MIDAS binary tools.

### Storage options

Panel shifts can be stored in two ways:

- **TOML-embedded (primary, recommended):** `save_params()` writes panel
  shifts as `[[panel_shifts]]` tables inside the geometry TOML.
  `load_params()` reads them back automatically. One file holds
  everything. This is what the GUI does — no separate panel shifts
  file is needed.

- **Text file (for MIDAS interoperability):** `save_panel_shifts()` /
  `read_panel_shifts()` in `midas4pil.panels` read and write the MIDAS
  whitespace-delimited format. Use this only when exchanging data with
  the MIDAS binary tools.

### Step 1: One-time panel calibration

Run the optimizer on a calibrant image (CeO2, LaB6, etc.) with panel
shifts enabled:

```python
from midas4pil.io import load_params, save_params, load_tiff, load_mask
from midas4pil.calibrant import load_calibrant
from midas4pil.panels import make_panel_id_map
from midas4pil.optimizer import calibrate

geom  = load_params("geometry.toml")
image = load_tiff("calibrant.tif").astype("float64")
panel_map = make_panel_id_map(1679, 1475, 3, 8, 487, 195, 7, 17)  # Pilatus 2M CdTe

mask = load_mask("mask.tif", panel_map=panel_map, image=image)
mask = mask | (panel_map == 0)  # mask gap pixels

rings = load_calibrant("CeO2.jcpds", geom['wavelength'], tth_max=29.0)

result = calibrate(
    image, mask, geom, panel_map, rings,
    fix_panel=1,              # hold panel 1 as reference
    optimize_shifts=True,     # enable per-panel dY, dZ
    tol_lsd=1000.0,
    tol_bc=5.0,
    tol_tilts=1.0,
    tol_shifts=5.0,
    n_iterations=5,
)

# Merge panel shifts into the geometry dict before saving
geom = result['geom']
geom['panel_shifts'] = result['panel_shifts']

# Save as a single TOML (panel shifts are embedded as [[panel_shifts]] tables)
save_params(geom, "hpcat_16idb_pilatus2m.toml")

# Optional: also export as a text file for MIDAS interoperability
from midas4pil.panels import save_panel_shifts
save_panel_shifts(result['panel_shifts'],
                  "hpcat_16idb_pilatus2m_panel_shifts.txt")
```

### Step 2: Per-experiment data reduction with panel corrections

Load the single TOML that contains both global geometry and panel shifts:

```python
from midas4pil.io import load_params, load_tiff, load_mask
from midas4pil.integrate import reduce_frame
from midas4pil.panels import make_panel_id_map, build_lut_with_panels

geom  = load_params("hpcat_16idb_pilatus2m.toml")  # includes panel_shifts
image = load_tiff("data.tif").astype("float64")
panel_map = make_panel_id_map(1679, 1475, 3, 8, 487, 195, 7, 17)
mask  = load_mask("mask.tif", panel_map=panel_map, image=image)
mask  = mask | (panel_map == 0)

lut_keys = ['nrows','ncols','bc_y','bc_z','lsd','px',
            'tx_deg','ty_deg','tz_deg','p0','p1','p2','p3','p4','rho_d']

tth_lut, eta_lut = build_lut_with_panels(
    **{k: geom[k] for k in lut_keys},
    panel_map=panel_map,
    panel_shifts=geom['panel_shifts'],  # from the same TOML
)

result = reduce_frame(image, mask, tth_lut, eta_lut, geom)
# result keys: tth, I, bg, I_sub, sigma, px_cnt, cake_img, tth_cake, eta_cake, px_cnt_cake
```

If you are working from a text file (MIDAS interop), load shifts separately:

```python
from midas4pil.panels import read_panel_shifts

panel_shifts = read_panel_shifts("hpcat_16idb_pilatus2m_panel_shifts.txt")
tth_lut, eta_lut = build_lut_with_panels(
    **{k: geom[k] for k in lut_keys},
    panel_map=panel_map,
    panel_shifts=panel_shifts,
)
```

### Panel shifts text file format

Plain text, one row per panel. Shifts are in pixels (dY, dZ), microns
(dLsd), or degrees (dTheta):

```
# panel_id  dY(px)  dZ(px)  dLsd(um)  dP2  dTheta(deg)
   1      0.000000      0.000000      0.000000      0.000000      0.000000
   2      0.208000     -0.133300      0.000000      0.000000      0.000000
  ...
```

### Custom detectors

For detectors other than Pilatus 2M, create a panel map with
`make_panel_id_map()`:

```python
from midas4pil.panels import make_panel_id_map

panel_map = make_panel_id_map(
    nrows=1679, ncols=1475,
    n_panels_y=3, n_panels_z=8,
    panel_size_y=487, panel_size_z=195,
    gap_y=7, gap_z=17,
)
```

The optimizer and panel correction functions work with any panel map ---
not limited to Pilatus.

## Installation

```bash
pip install numpy tifffile
pip install .
```

Requires Python >= 3.9.

For development (editable install with test dependencies):

```bash
pip install -e ".[dev]"
```

## Testing

```bash
pytest tests/ -v
```

## Coordinate conventions

| Convention | Value |
|---|---|
| TIFF (0,0) | top-left (standard numpy raster) |
| Mask | 1 = bad pixel, 0 = good |
| eta | 0 at 3 o'clock, CCW positive, +90 at 12 o'clock (IUCr/MIDAS) |
| BC_Y | beam-centre column from left (Poni2 / px) |
| BC_Z | beam-centre row from top (Poni1 / px) |

## Credits

The table below identifies every component by origin.
MIDAS is developed by Hemant Sharma, X-ray Science Division,
Argonne National Laboratory (https://github.com/marinerhemant/MIDAS).

| Component | Module | Origin | MIDAS source | midas4pil contribution |
|-----------|--------|--------|--------------|------------------------|
| Pixel-to-angle transform | `geometry.py` | **Inherited** | `dg_pixel_to_REta()` in `DetectorGeometry.c` | Vectorized numpy port; same math exactly |
| Tilt matrices (Rx, Ry, Rz) | `geometry.py` | **Inherited** | `dg_build_tilt_matrix()` in `DetectorGeometry.c` | Direct reimplementation |
| Distortion model (p0–p4, ηT) | `geometry.py` | **Inherited** | Distortion polynomial in `DetectorGeometry.c` | Same coefficients and ηT = 180° − η convention |
| SNIP background | `integrate.py` | **Inherited** | `utils/extract_lineouts.py` | Python port; triple-log transform + iterative clipping |
| Per-panel shift application (dY, dZ, dLsd, dP2, dTheta) | `panels.py` | **Inherited** | `ApplyPanelCorrection()` in `CalibrantPanelShiftsOMP.c` | Direct reimplementation |
| Panel shifts file format | `panels.py` | **Mixed** | MIDAS PanelShiftsFile format | Compatible reader/writer; Python implementation new |
| Calibration cost function (J = mean\|strain\|) | `optimizer.py` | **Mixed** | Same problem as `CalibrantPanelShiftsOMP` | Concept analogous; Python implementation (scipy) entirely new |
| LUT construction | `geometry.py` | **Mixed** | Uses inherited pixel_to_r_eta math | Vectorization, precomputed caching, and panel-corrected variant are new |
| Pixel-matched varbin bins (dR = 1) | `geometry.py` | **Original** | MIDAS RBinSize defaults to 0.25 (4× oversampled) | New working principle: bin width = detector angular resolution |
| 1-D integration engine | `integrate.py` | **Original** | — | New Python histogram; σ (Poisson SEM) and px\_cnt output not in MIDAS |
| Pixel-matched varbin integration | `integrate.py` | **Original** | — | New concept and implementation |
| Caked image (2-D polar map) | `cake.py` | **Original** | MIDAS has caking but not ported | New Python implementation with pixel-matched eta bins |
| Calibrant reader (JCPDS / CIF) | `calibrant.py` | **Original** | Replaces MIDAS `GetHKLList` + `hkls.csv` | JCPDS v4/5.1 + CIF via gemmi; single Python call |
| General panel map builder | `panels.py` | **Original** | — | Works for any tiled detector geometry |
| I/O layer (TOML, poni, TIFF, mask) | `io.py` | **Original** | — | New format; .poni import/export built in |
| Auto beam-centre finding | `geometry.py` | **Original** | — | 2-step sharpness + first-ring profile algorithm |
| Poisson SEM (σ = √ΣI / N) | `integrate.py` | **Original** | — | Per-bin uncertainty propagation; not in MIDAS output |
| Rebin lineout | `integrate.py` | **Original** | — | Exact Poisson error propagation across rebinned groups |
| Eta sector integration | `integrate.py` | **Original** | — | Azimuthal wedge selection with fast path |
| Numba JIT kernels | `_jit.py` | **Original** | — | Single-pass histogram; 17× speedup over numpy |
| PySide6 GUI | `gui/` | **Original** | — | Full 2-tab GUI: Calibration + Integration + Mask Editor |

SNIP algorithm reference: Morhac et al., NIM A 401 (1997) 113-132.

## License

BSD 3-Clause License. Copyright (c) 2026 Changyong Park, HPCAT, X-ray Science Division,
Argonne National Laboratory. See LICENSE for the full text.
