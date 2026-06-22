# Calibration example

Complete end-to-end example: from a calibrant image to panel-corrected
data reduction. No external calibration tool required.

## What you need

1. **Calibrant diffraction image** (`calibrant.tif`) -- a TIFF image
   of a known calibrant (CeO2, LaB6, etc.).

2. **Bad-pixel mask** (`mask.tif`, optional) -- any TIFF mask file
   (1=bad, 0=good). If not provided, an auto-mask is generated from
   the image (gap pixels, dead pixels, hot pixel outliers). Mask
   orientation is auto-detected; no manual flipping required.

3. **`geometry_init.toml`** -- configuration file containing your
   experimental parameters. Edit this file before running the script.
   See [How to edit geometry_init.toml](#how-to-edit-geometry_inittoml)
   below.

## How to run

```bash
cd calib_example

# Edit geometry_init.toml with your parameters, then:
python run_calibration.py                   # uses ./geometry_init.toml
python run_calibration.py my_config.toml    # uses a custom config file
```

The script runs in about 3 minutes and produces all output files in
the same directory as the input image.

## How to edit geometry_init.toml

The `geometry_init.toml` file is the **only file you edit**. It
contains all experimental parameters grouped into sections. Open it
in any text editor; each parameter has a comment explaining what it
means.

### `[input]` -- file paths

```toml
[input]
image     = "CeO2_350mm_29p2keV.tif"       # calibrant image (TIFF)
mask      = "CeO2_350mm_29p2keV_mask.tif"   # bad-pixel mask (optional, "" for auto)
calibrant = "CeO2.jcpds"                    # calibrant file (.jcpds or .cif)
```

- **image**: Path to your calibrant TIFF, relative to the TOML file's
  directory. All output filenames are derived from this filename.
- **mask**: Path to a bad-pixel mask. Set to `""` (empty string) to
  auto-generate a mask from the image. Mask orientation is auto-detected.
- **calibrant**: Name of the calibrant file. The script searches
  `midas4pil/_JCPDS/` and `midas4pil/_CIF/` automatically, so you
  only need the filename (e.g. `"CeO2.jcpds"`, `"LaB6.jcpds"`,
  `"Si.jcpds"`). You can also give a full or relative path.

### `[detector]` -- known from detector model

These come from the detector datasheet. They do **not** change
between experiments.

| Key | Description | Pilatus 2M | Eiger 4M |
|-----|-------------|-----------|----------|
| `nrows` | Detector rows (pixels) | 1679 | 2167 |
| `ncols` | Detector columns (pixels) | 1475 | 2070 |
| `px_um` | Pixel size (um) | 172 | 75 |

### `[beam]` -- known from beamline setup

| Key | Unit | Description |
|-----|------|-------------|
| `wavelength_A` | angstrom | X-ray wavelength = 12398.4193 / E(eV) |

Common energies: 29.2 keV = 0.42460 A, 20.0 keV = 0.61992 A.

### `[geometry]` -- approximate starting values

These are refined by calibration. Only `lsd_um` and `bc_y_px`/`bc_z_px`
need to be approximately correct; tilts and distortion can start at zero.

| Key | Unit | Description |
|-----|------|-------------|
| `lsd_um` | um | Approximate sample-to-detector distance. Does not need to be precise -- the two-stage calibration handles errors up to ~5%. |
| `bc_y_px` | pixels | Beam centre column from left. If unknown, set to `ncols / 2`. |
| `bc_z_px` | pixels | Beam centre row from top. If unknown, set to `nrows / 2`. |
| `tx_deg` | degrees | Detector tilt about X axis. Set to 0 if unknown. |
| `ty_deg` | degrees | Detector tilt about Y axis. Set to 0 if unknown. |
| `tz_deg` | degrees | Detector tilt about Z axis. Set to 0 if unknown. |
| `p0`..`p4` | -- | Radial distortion coefficients. Set all to 0 if unknown. |
| `p3_deg` | degrees | Distortion phase angle. Set to 0 if unknown. |
| `rho_d_um` | um | Distortion reference radius. 217578 is the default for Pilatus 2M. |

### `[integration]` -- data reduction limits

| Key | Unit | Default | Description |
|-----|------|---------|-------------|
| `mode` | -- | `"varbin"` | `"varbin"` or `"unibin"` (see below) |
| `tth_min_deg` | degrees | 2.0 | 2theta lower limit |
| `tth_max_deg` | degrees | 29.0 | 2theta upper limit. Set to cover all visible calibrant rings. |
| `tth_bin_deg` | degrees | 0.025 | 2theta bin width (**unibin only**, ignored in varbin) |
| `eta_min_deg` | degrees | -180 | Eta lower limit. -180 for full ring. |
| `eta_max_deg` | degrees | 180 | Eta upper limit. +180 for full ring. |
| `eta_bin_deg` | degrees | 1.0 | Eta bin width (**unibin only**, ignored in varbin) |

**varbin** (default, recommended): each 2theta bin spans exactly one
pixel's angular footprint. Bin widths are computed from the detector
geometry -- no user tuning needed. This is the physically correct
representation of the detector's angular resolution.

**unibin**: user-specified uniform bin width. Use only when uniform
spacing is strictly required (e.g. FFT for pair distribution function).
Even then, set `tth_bin_deg` no smaller than the detector's pixel
angular resolution -- bins finer than the pixel footprint create
information the detector cannot physically deliver.

### `[calibration]` -- advanced (optional)

| Key | Default | Description |
|-----|---------|-------------|
| `tth_rough_max_deg` | 15.0 | Stage 1 uses only rings below this 2theta. Must be below the first pair of closely-spaced rings in your calibrant. |

## What happens

The calibration uses a two-stage approach to handle rough initial
estimates:

1. **Load image and mask** -- auto-mask if no mask file provided.
   Masks gap pixels, dead pixels (value <= 0), and hot pixel outliers.

2. **Find beam centre** -- two steps:
   (a) BC from azimuthal sharpness: maximises the variance of the
   radially-averaged intensity profile, starting from the TOML values.
   On a tilted detector the sharpness metric finds the ring's image
   centre, which may differ from the true geometric BC by ~10 px.
   (b) Lsd from the first Bragg ring radial profile using prominence-
   based peak detection. Accuracy: ~0.2 %.
   Both values are used as seeds for Stage 1.

3. **Build initial geometry** -- assembles an initial geometry in memory
   from the TOML parameters and the auto-detected beam centre.

4. **Stage 1: rough calibration** -- uses only low-angle rings
   (< `tth_rough_max_deg`) with a wide 2theta tolerance to handle the
   approximate Lsd. Refines Lsd and beam centre to within ~0.1%.

5. **Stage 2: fine calibration** -- uses all rings with normal
   tolerance. Full optimization of Lsd, beam centre, tilts, and
   distortion. Saves `<BASE>_geometry.toml`.

6. **Export .poni** -- creates `<BASE>_geometry.poni` for cross-checking
   (optional).

7. **Per-panel calibration** -- refines per-panel dY, dZ shifts for
   all detector modules. Saves the panel shifts embedded in
   `<BASE>_geometry.toml` (primary) and also exports
   `<BASE>_panel_shifts.txt` (MIDAS binary interoperability). Updates
   `<BASE>_geometry.toml` with the panel-refined global geometry.

8. **Data reduction -- 3-way comparison** -- integrates the calibrant
   image three ways for comparison:
   - *Conventional*: Stage 2 geometry without panel shifts — best result
     without panel corrections. Saves `<BASE>_lineout_conventional.xy`
     and `<BASE>_cake_conventional.tif`.
   - *Panel-corrected*: Stage 3 geometry with panel shifts applied —
     best overall result. Saves `<BASE>_lineout_panels.xy` and
     `<BASE>_cake_panels.tif`.
   - *Global-only (informational)*: Stage 3 geometry without panel
     shifts — shows what the Stage 3 geometry looks like alone. Saves
     `<BASE>_lineout_global.xy` and `<BASE>_cake_global.tif`.

9. **3-way comparison figure** -- plots all three lineouts and caked
   images side by side. Saves `<BASE>_comparison.png`.

## Output files

All output filenames are derived from the image base name (`<BASE>`).
For example, with `CeO2_350mm_29p2keV.tif`:

| File | Description | Scope |
|------|-------------|-------|
| `<BASE>_geometry.toml` | Final optimized geometry (Stage 3) | Per-experiment |
| `<BASE>_geometry.poni` | `.poni` format export | Cross-checking |
| `<BASE>_panel_shifts.txt` | Per-panel corrections (MIDAS text format) | MIDAS interoperability |
| `<BASE>_lineout_conventional.xy` | 1-D pattern, conventional (Stage 2, no panels) | -- |
| `<BASE>_lineout_panels.xy` | 1-D pattern, panel-corrected (Stage 3 + shifts) | -- |
| `<BASE>_lineout_global.xy` | 1-D pattern, global-only (Stage 3 geom, no shifts) | -- |
| `<BASE>_cake_conventional.tif` | Caked image, conventional | -- |
| `<BASE>_cake_panels.tif` | Caked image, panel-corrected | -- |
| `<BASE>_cake_global.tif` | Caked image, global-only | -- |
| `<BASE>_comparison.png` | 3-way comparison figure | -- |

## Interpreting the comparison figure

The comparison figure (`<BASE>_comparison.png`) shows three 1-D lineouts
and three caked images side by side:

- **Conventional** (steelblue) -- Stage 2 global geometry, no panel
  shifts. Best result achievable without panel corrections.
- **Panel-corrected** (orangered) -- Stage 3 global geometry *plus*
  per-panel dY, dZ shifts. Best overall result.
- **Global-only** (gray, informational) -- Stage 3 global geometry
  *without* panel shifts. Shows the effect of panel corrections alone.

The mean |strain| (ppm) for each is shown above its lineout column.

### How to read the comparison

**If conventional and panel-corrected lineouts are nearly identical**
(peaks overlap, similar height and width), panel misalignment on your
detector is small. The global geometry alone provides an adequate
calibration, and you may choose to skip panel corrections for routine
data reduction. This is a good sign -- it means the detector modules
are well aligned.

**If the panel-corrected column shows taller, sharper peaks**
(especially at high 2theta), the detector has significant module
misalignment. At high scattering angles, the pixel angular resolution
is finer, so even sub-pixel panel offsets produce measurable peak
broadening. In this case, you should always apply panel corrections
when reducing data from this detector.

**General guidelines:**

- The effect of panel misalignment grows with 2theta. Check the
  highest-angle peaks first -- that is where differences are most
  visible.
- The mean |strain| (ppm) shown above each column is a single-number
  summary: lower is better. Below ~200 ppm the global calibration is
  good. Panel correction typically brings it below ~150 ppm.
- If the conventional column looks sharper than panel-corrected at any
  angle, something is wrong -- the panel shifts may need to be
  re-calibrated, or the optimizer may not have converged. Try
  increasing `n_iterations` or `tol_shifts`.
- The caked images should show straight, horizontal bands at each ring
  position. Wavy bands at high 2theta indicate residual panel
  misalignment or tilt errors.

## Adapting for your detector

Edit `geometry_init.toml`:

- **Different pixel size or dimensions**: change `[detector]` values.
- **Different wavelength or distance**: change `[beam]` and
  `[geometry]` values.
- **Different calibrant**: change `calibrant` in `[input]`. Bundled
  calibrants are in `midas4pil/_JCPDS/` (organized by category) and
  `midas4pil/_CIF/`.
- **Different detector layout**: the script currently calls
  `make_panel_id_map(...)` for the Pilatus 2M CdTe. For other tiled detectors,
  modify the script to supply your module geometry:
  ```python
  from midas4pil.panels import make_panel_id_map
  panel_map = make_panel_id_map(
      nrows=..., ncols=...,
      n_panels_y=..., n_panels_z=...,
      panel_size_y=..., panel_size_z=...,
      gap_y=..., gap_z=...,
  )
  ```

## For subsequent experiments

Once you have `<BASE>_panel_shifts.txt`, you do not need to re-run the
panel calibration. For each new experiment:

1. Copy `geometry_init.toml` and edit with your new experiment
   parameters (wavelength, approximate Lsd, new image filename).
2. Run `python run_calibration.py my_new_config.toml` to produce a
   new `<BASE>_geometry.toml`.
3. Load data with panel corrections in your own scripts:

```python
from midas4pil.io import load_params, load_tiff, load_mask
from midas4pil.panels import (make_panel_id_map, read_panel_shifts,
                               build_lut_with_panels)
from midas4pil.integrate import reduce_frame

geom = load_params("experiment_geometry.toml")
image = load_tiff("data.tif").astype("float64")
panel_map = make_panel_id_map(1679, 1475, 3, 8, 487, 195, gap_y=7, gap_z=17)
mask = load_mask("mask.tif", panel_map=panel_map, image=image)
mask = mask | (panel_map == 0)

panel_shifts = read_panel_shifts("CeO2_350mm_29p2keV_panel_shifts.txt")   # reuse!

lut_keys = ['nrows','ncols','bc_y','bc_z','lsd','px',
            'tx_deg','ty_deg','tz_deg','p0','p1','p2','p3','p4','rho_d']
tth_lut, eta_lut = build_lut_with_panels(
    **{k: geom[k] for k in lut_keys},
    panel_map=panel_map,
    panel_shifts=panel_shifts,
)

result = reduce_frame(image, mask, tth_lut, eta_lut, geom)
# result['tth'], result['I'], result['bg'], result['I_sub'], result['sigma']
```

## Finding the example after installation

After `pip install midas4pil`, this example directory is bundled inside
the package. To find it:

```bash
python -c "import midas4pil, pathlib; print(pathlib.Path(midas4pil.__file__).parent / 'calib_example')"
```

Copy the full directory to your working directory before running:

```bash
# Replace the path below with the output of the command above
cp -r /path/to/site-packages/midas4pil/calib_example .
cd calib_example
python run_calibration.py
```
