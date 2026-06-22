# midas4pil — Usage Guide

**Author:**    Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory (cypark@anl.gov)
**Co-author:** Claude Code (claude-sonnet-4-6, Anthropic AI)
**Version:**   1.1.3 (2026-04-27)

midas4pil is a lightweight diffraction data reduction tool for synchrotron
powder and high-pressure experiments (Pilatus / Eiger detectors).
It has two tabs: **Calibration** and **Integration**.

---

## Calibration Tab

Use this tab to determine the detector geometry from a calibrant image
(e.g. CeO2, LaB6).

1. Select your detector from the dropdown (default: Pilatus 2M CdTe).
2. Enter beam energy (keV) or wavelength (Å) — the other field updates automatically.
3. Select a calibrant from the dropdown (CeO2 is the default).
4. Click **Load Image** to load your calibrant diffraction image (`.tif`).
   You can also load a previously saved `geometry.toml` via **File > Load Geometry**,
   or from an existing `.poni` geometry file via **File > Load .poni**.

### Step 5 — Find the beam centre

#### Auto Find Center (easy case)

Use when at least one complete Debye-Scherrer ring is visible.
Enter an approximate Lsd first.

- Step 1 — beam centre from ring sharpness (no calibrant needed).
- Step 2 — Lsd from the first Bragg ring profile; the search is bounded by the
  **Rough <=** field (default 8°).
- Tilts (ty, tz) are not changed by Auto Find Center.

> **Note:** On a tilted detector the sharpness BC may differ from the true geometric
> BC by ~10–15 px. This is expected — Run Calibration refines the geometric BC and
> tilts simultaneously.

#### Manual Find Center (difficult case)

Use when the detector is heavily off-axis, only partial arcs are visible, the visible
ring is not the first Bragg reflection, or Auto Find Center gave a poor result.

a. Click **Manual Find Center** — a ring table appears.
   Select the ring that corresponds to the arc you can see.
b. Click 3 or more points along that arc on the image.
c. Click **Done**. A tilt-aware optimizer (Nelder-Mead on `pixel_to_r_eta`) fits
   BC_Y, BC_Z, and Lsd simultaneously using the current ty/tz values from the
   geometry fields. The result is correct even when the ring appears as an ellipse.

#### Combining steps (recommended for challenging cases)

1. Load `.toml` → initial geometry from a previous session
2. Manual Find Center → update BC and Lsd from visible arc
3. Auto Find Center → if a complete ring is now in range
4. Run Calibration → full refinement (BC, Lsd, tilts, panels)

Each step uses the current geometry fields as its starting point.

### Step 6 — Mask

Load or edit the mask if needed (see [Mask Editor](#mask-editor) below).

### Step 7 — Run Calibration

Click **Run Calibration** to refine all parameters. The optimizer runs in stages;
strain is reported after each stage.

#### Calibration checkboxes

- **Optimize panels** — refine per-module rigid-body shifts (dY, dZ, dLsd, dTheta)
  for tiled detectors. Panel shifts are a hardware property; calibrate once and reuse
  for all experiments on that detector. Requires a calibrant with many rings.
- **Fit distortion (p2, p3)** — refine smooth surface warping. Leave unchecked unless
  per-ring strain shows a systematic radial trend after flat-geometry calibration.
  Rarely needed for Pilatus/Eiger.

### Step 8 — Review results

Click **Show/Save Result…** to see the calibration output:

| Section | Contents |
|---------|----------|
| 1 | Flat Detector Geometry — Lsd, BC, tilts |
| 2 | Panel Shift Correction — per-module dY/dZ/dLsd/dTheta |
| 3 | Distortion Correction — p0–p4 polynomial |
| 4 | Fit Quality — mean \|strain\| (< 200 ppm is excellent) |
| 5 | Per-Ring Strains — residual per calibrant ring |

- Systematic strain trend vs 2θ → wrong Lsd or beam centre.
- Single outlier ring → bad ring ID or masked arc.

Save buttons in this dialog:

| Button | Output |
|--------|--------|
| Save Params… | Full geometry as `.toml` (recommended — includes panel shifts and distortion) |
| Export .poni… | `.poni` file for cross-checking (flat geometry only; no panel shifts or distortion) |
| Save Lineout… | Current 1-D lineout as `.xye`; columns: `2θ, I[, SNIP_bg][, I_sub][, σ][, px_cnt]` (variable, depending on available data) |
| Save Cake… | Caked image as TIFF |

### Step 9 — Send to Integration

Save `geometry.toml`, then click **Send to Integration**.

### Ring overlay

After any parameter change the rings are redrawn on the image. For tilted detectors
the overlay traces the true elliptical shape computed by ray-tracing through the
MIDAS geometry model.

### Strain colour-map overlay

After calibration, a colour overlay shows the per-pixel (raw image) or per-bin-cell
(cake image) fractional 2θ residual in parts-per-million (ppm):

```
strain_ppm = (2θ_measured − 2θ_ring) / 2θ_ring × 10⁶
```

Colour convention (blue–green–red):

| Colour | Meaning |
|--------|---------|
| Blue | Measured 2θ below ideal → d-spacing expanded (lattice larger than reference) |
| Green | Zero residual — perfect agreement with the calibrant |
| Red | Measured 2θ above ideal → d-spacing compressed (lattice smaller than reference) |
| Transparent | Masked pixel / cell, or outside every ring's tolerance band |

On the **cake image**, bin cells that contain no valid (unmasked) pixels are fully
transparent, so masked regions propagate correctly from the raw image.

The per-ring strain values in the result dialog are **intensity-weighted centroids**:

```
centroid_2θ = Σ(I_i · 2θ_i) / Σ(I_i)     [sum over unmasked in-band pixels]
strain_ppm  = (centroid_2θ − 2θ_ring) / 2θ_ring × 10⁶
```

Weighting by intensity emphasises the Bragg peak over the background wings,
giving a robust estimate of where the ring truly is in the data.
A mean |strain| below 200 ppm indicates excellent calibration.

### Mouse / intensity display

- Raw/cake image — status bar shows 2θ, η, pixel intensity I, and `px_cnt = nnn`
  (active pixel count for the hovered cake cell, from live reduction).
- Lineout plot — status bar shows 2θ, the intensity value (I, I_sub, or SNIP)
  of the current lineout at cursor (priority follows toggle button state),
  and `px_cnt = nnn` (pixel count for the nearest bin).

---

## Mask Editor

Open from either tab via the **Mask…** button. Red overlay = masked (bad).

**Convention:** mask = 1 = bad pixel, 0 = good pixel.
Masked pixels are excluded from both calibration and integration.

### Drawing tools (toolbar, top-left)

| Tool | Behaviour |
|------|-----------|
| Circle | Click (brush-size circle) or drag (centre-to-edge radius) |
| Rect | Click (single pixel) or drag (corner to corner) |
| Polygon | Sequential clicks; double-click or click first vertex to close. Vertices outside the image snap to the nearest border. |

Click an active Circle or Rect button again to deactivate it.
In deactivated state, left-drag draws a zoom box (rubber-band zoom).
Scroll wheel zooms in/out centred on cursor. Right-click resets zoom.

- **Mode toggle:** Mask (add bad pixels) / Unmask (remove bad pixels)
- **Brush size:** spinbox controls the circle radius for click (not drag)

### Action buttons

| Button | Action |
|--------|--------|
| Load | Load mask from a TIFF file (must match image dimensions) |
| Save | Save mask to TIFF (uint8: 1 = bad, 0 = good) |
| Auto | Generate mask from image: gaps, dead pixels, saturated pixels |
| Despeckle | Remove isolated masked pixels with no 4-connected neighbours |
| Reverse | Invert the entire mask (good ↔ bad); useful when a loaded mask uses the opposite convention |
| Mask >= | Mask all pixels at or above the threshold value |
| Mask <= | Mask all pixels at or below the threshold value (default 0) |
| Reset to Auto | Discard all edits and regenerate from image data |

**Undo / Redo:** up to 30 steps for all drawing and action operations.

---

## Integration Tab

Use this tab to reduce one or many diffraction images to 1D lineouts and 2D caked
images. Send geometry from the Calibration tab first.

### Modes

- **Batch** — click **Start** to process all `.tif` files in the selected folder.
  Results (lineouts, caked images) are saved to `lineouts/` and `cakes/` subfolders.
  Enable **Watch for new files** to keep processing incoming data automatically.
- **Manual** — click any file in the list to process and display it immediately.
  Results are saved to `lineouts/` (mask on) or `lineouts_nomask/` (mask off).
  Use for single-frame inspection.

### Integration method

- **varbin** (default) — pixel-matched variable-width bins; the physically correct
  choice. Each bin spans the angular footprint of one detector pixel. Recommended for
  Rietveld refinement (GSAS-II, FullProf, TOPAS).
- **unibin** — uniform angular spacing. Use only for PDF analysis (FFT requires a
  uniform grid) or when uniform spacing is specifically needed.

### File list

- Click a file to display its cake and lineout.
- Check the checkbox next to a file to include its lineout in the overlay.
- Multiple checked files are overlaid in the lineout plot simultaneously.
- The cake image always shows the currently selected (clicked) file.

**Shortcuts:**

| Key / action | Effect |
|---|---|
| Space | Toggle check state of all selected files |
| Shift+scroll | Extend selection up / down |
| Drag | Drag over items to check/uncheck in one stroke |
| Right-click | Context menu: check/uncheck selected or all files |

### Lineout display

Toggle buttons **I** / **I_sub** / **SNIP** control which curves are shown:

| Button | Curve |
|--------|-------|
| I | Raw integrated intensity |
| I_sub | Background-subtracted (I minus SNIP); shown with a ±σ shaded band (Poisson standard error of the mean per bin) |
| SNIP | Estimated background (Statistics-sensitive Non-linear Iterative Peak-clipping algorithm) |

In multi-overlay mode: checked files show I only (thin, cycling colours);
the currently selected file shows I/I_sub/SNIP per the toggle buttons.
When the active η range or SNIP 2θ range changes, all checked overlay files
(including the current file) are recomputed with the new range automatically.

### SNIP 2θ range

Two orange dashed vertical lines on the lineout plot define the 2θ range for SNIP
background computation. Outside this range bg = NaN and I_sub = NaN.

- Drag either line, or edit the **SNIP 2θ:** spinboxes in the lineout header.
- The range resets to full detector coverage when a new geometry is loaded.
- Batch processing always uses full-range SNIP; this restriction applies only to
  the interactive display and Export I_sub.

### Export I_sub

**Export I_sub…** opens a scope dialog before saving:

| Scope | Which files |
|-------|-------------|
| Current file | Only the frame currently displayed (no disk I/O needed) |
| Checked overlay files | All files checked in the file list |
| All files in folder | Every `.tif` in the data folder |

Each file is re-integrated with the active SNIP 2θ range and η sector and saved
as a 6-column file `[2θ, I, SNIP_bg, I_sub, σ, px_cnt]` to `lineouts_sub/`
(mask on) or `lineouts_sub_nomask/` (mask off). The file header records all
column names plus wavelength, energy, 2θ range, and η range. Progress is
shown in the status bar while multiple files are being exported.

### Saved file formats

| Output | Columns | Header |
|--------|---------|--------|
| Batch / Manual (I-export) | `2θ, I, σ, px_cnt` | `col1=2theta_deg  col2=I  col3=sigma_I  col4=px_cnt  [wavelength=…  eta=…]` |
| Export I\_sub | `2θ, I, SNIP_bg, I_sub, σ, px_cnt` | `col1=2theta_deg  col2=I  col3=SNIP_bg  col4=I_sub  col5=sigma_I  col6=px_cnt  [wavelength=…  2th=…  eta=…]` |

Older 3-column and 5-column files written by previous versions are still loaded
correctly (backward-compatible).

### Mouse / intensity display

- Image panel (raw/cake) — shows 2θ, η, pixel intensity I, and `px_cnt = nnn`
  (active pixel count for the hovered cake cell; available after live reduction).
- Lineout plot — shows 2θ, intensity value at cursor (I > I_sub > SNIP priority
  following toggle button state), and `px_cnt = nnn` (pixel count for the nearest bin).

### Mask

- **Mask O/●** button in the display header applies the mask during integration.
- **Mask…** opens the mask editor (see [Mask Editor](#mask-editor) above).
- If no mask is loaded, Auto mask is generated from the first image.

---

## Technical Reference

### Variable-bin integration (varbin)

The default integration mode uses pixel-matched variable-width 2θ bins.
Each bin spans the angular width of one detector pixel at that 2θ:

```
delta(2theta) = px * cos²(2theta) / Lsd
```

This ensures no artificial oversampling: each bin contains exactly the angular
information that one pixel can physically deliver. Bins are wider at low 2θ (where
the pixel subtends a larger angle) and narrower at high 2θ.

Rietveld refinement codes (GSAS-II, FullProf, TOPAS) accept non-uniform bin spacing.
Varbin is the appropriate choice whenever the goal is to faithfully represent the
angular resolution of the detector.

### Uniform-bin integration (unibin)

Uniform bins are available for cases that strictly require equal spacing:

- FFT-based pair distribution function (PDF) analysis
- Real-time monitoring when speed matters more than bin resolution

When using unibin, the bin width should never be set smaller than the detector's
angular resolution at the lowest 2θ in the range.

### SNIP background

The SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) algorithm removes
the amorphous/thermal diffuse scattering background from 1D patterns.

Reference: Morháč et al., *NIM A* 401 (1997) 113–132.
Default: 50 iterations. Set to 0 to disable.

### Coordinate conventions

| Symbol | Definition |
|--------|-----------|
| BC_Y | Beam centre column from left edge (pixels) |
| BC_Z | Beam centre row from top edge (pixels) |
| η (eta) | Azimuthal angle: 0° at 3 o'clock, positive CCW from sample view (12 o'clock = +90°, 6 o'clock = −90°, 9 o'clock = ±180°) |
| Mask | 1 = bad pixel, 0 = good pixel |

### Detector geometry model

Three correction layers, calibrated and applied in this order
(geometry model from MIDAS `DetectorGeometry.c`):

**Layer 1 — Flat detector geometry (PONI)**

The foundational model. Five fitted parameters; tx (rotation around the
beam axis X) is fixed at zero because it only shifts the azimuthal
reference (eta = 0 direction) and cannot be determined from azimuthally
uniform powder rings.

| Parameter | Description |
|-----------|-------------|
| Lsd | Sample-to-detector distance (µm) |
| BC_Y | Beam centre column from left (pixels) |
| BC_Z | Beam centre row from top (pixels) |
| ty | Detector tilt around Y axis (horizontal transverse, degrees) |
| tz | Detector tilt around Z axis (vertical transverse, degrees) |

**Layer 2 — Per-panel shifts** (tiled detectors: Pilatus, Eiger)

Per-module rigid-body corrections for physical misalignment of individual
sensor boards relative to the flat detector model. Hardware constant:
calibrate once per detector, reuse for all experiments.

| Parameter | Description |
|-----------|-------------|
| dY, dZ | In-plane module shifts (pixels) |
| dLsd | Distance offset along beam axis (µm) |
| dTheta | In-plane module rotation (degrees) |

**Layer 3 — Distortion polynomial** (non-flat detector surface)

Smooth correction for deviations of the detector surface from the ideal
flat plane. Applied to the effective radial distance after the global
transform.

```
rho_norm = Rad / rho_d           (Rad in µm; rho_d ≈ 217578 µm for Pilatus 2M)
etaT     = 180 - eta             (MIDAS internal azimuth variable)
distort  = 1 + p0*rho_norm²*cos(2*etaT)
             + p1*rho_norm⁴*cos(4*etaT + p3)
             + p2*rho_norm²
             + p4*rho_norm⁶
R_corr = (Rad * distort / px) * (Lsd / Lsd_panel)   [pixels]
```

For Pilatus/Eiger CMOS sensors this correction is typically negligible.

### Calibration stage order

| Scenario | Stage 1 | Stage 2 | Stage 3 |
|----------|---------|---------|---------|
| PONI only | rough PONI | full PONI | — |
| PONI + panels | rough PONI | full PONI | PONI + panels |
| PONI + distortion | rough PONI | PONI + distortion | — |
| PONI + panels + distortion | rough PONI | PONI + panels | PONI + panels + distortion |

Panels are always calibrated after the flat geometry and before distortion.
Fitting distortion before panels causes the smooth polynomial to absorb
discrete per-module offsets, giving wrong values for both corrections.

### Auto Find Center — algorithm

Two steps; no calibrant ring knowledge required for step 1:

1. **BC from azimuthal sharpness** — maximises var(I̅(R)), the variance of the
   radially-averaged intensity profile, over a coarse grid ±200 px from the detector
   centre, then Nelder-Mead. On a tilted detector (ty, tz ≠ 0) the sharpness peak is
   at the ring's *image* centre, which can differ from the true geometric BC by
   ~10–15 px. This is expected physics, not a bug. Run Calibration to recover the
   geometric BC and tilts simultaneously.

2. **Lsd from the first Bragg ring** — builds I̅(R) from the step-1 BC; finds the
   innermost peak using prominence-based detection (`scipy.signal.find_peaks`). The
   **Rough <=** field bounds the upper search radius. Accuracy: ~0.2 %.

If the current geometry is already good (e.g. loaded from `.toml`), Auto Find Center
preserves it: the coarse scan uses the init geometry score as a floor and cannot
degrade a good starting point.

Tilts are not estimated by Auto Find Center — they are refined by Run Calibration.

### Manual Find Center — algorithm

For off-centre geometries with only partial Debye-Scherrer arcs:

1. Click **Manual Find Center** — a table of calibrant rings appears. Select the ring
   whose arc you can see on the detector.
2. Click 3+ points along the arc on the raw image.
3. Click **Done**.
4. A tilt-aware Nelder-Mead optimizer minimises

```
sum[ (tth_computed(xi, yi) - tth_ring)² ]
```

over (BC_Y, BC_Z, Lsd), using the current ty and tz values from the geometry fields
and the full `pixel_to_r_eta()` geometry model. The result is accurate even when the
ring appears as an ellipse (tilted detector, large tz). No approximation is made.

This gives the correct geometric beam centre directly from clicked points, regardless
of detector tilt or partial arc visibility.

### MIDAS geometry_params.txt interoperability

`load_midas_params(path)` reads a MIDAS flat key-value geometry file and
returns a midas4pil geometry dict. Key aliases are supported: `Distance` /
`DetDist` for `Lsd`; `PixelSize` for `px`; `NrPixels` for a square detector.
`BC Y Z` are two values on one line. If `ImTransOpt 2` is present, BC_Z is
converted from the MIDAS top-bottom-flipped frame to midas4pil's
original-TIFF frame (`bc_z = nrows − midas_BC_Z`). If a `PanelShiftsFile`
line is found, the referenced file is loaded automatically and the panel
shifts are stored in `geom['panel_shifts']`, ready to be used as initial
values for the next calibration.

`save_midas_params(geom, path, panel_shifts_path=None)` writes the geometry
in MIDAS format with `ImTransOpt 2`. Panel topology (NPanelsY, NPanelsZ,
PanelSizeY, PanelSizeZ, PanelGapsY, PanelGapsZ) is auto-detected from the
detector shape. If `panel_shifts_path` is provided and the geometry contains
panel shifts, a companion `panel_shifts.txt` is written in MIDAS column order
(`id dY dZ dTheta dLsd dP2`) and a `PanelShiftsFile` line is added to the
main file. The output includes commented-out placeholders for calibrant and
calibration tolerance keys (SpaceGroup, RingThresh, tolBC, etc.) so the file
can be used directly with MIDAS binary calibration tools after uncommenting.

**Panel shifts file column order (MIDAS-compatible):**

| Column | Name | Units |
|--------|------|-------|
| 0 | panel_id | — |
| 1 | dY | pixels |
| 2 | dZ | pixels |
| 3 | dTheta | degrees |
| 4 | dLsd | µm |
| 5 | dP2 | dimensionless |

**GUI access:** File > Load MIDAS params… loads a geometry_params.txt and
fills all geometry fields (including panel shifts). Export MIDAS… in the
calibration result dialog exports the calibrated geometry to MIDAS format.

### Performance

With numba JIT and precomputed bin maps:

| Operation | Time |
|-----------|------|
| Per-frame reduction (`reduce_frame`) | ~13 ms (79 Hz) for Pilatus 2M |
| One-time precompute (`precompute_bin_maps`) | ~1.2 s (done once when geometry is set) |
| Without numba (pure numpy fallback) | ~160 ms (6 Hz) per frame |

Precomputed bin maps are used in both Batch and Manual modes whenever the geometry is
unchanged between frames. numba JIT is always active once geometry is loaded, even
when no mask file is provided (a zero-valued null mask is used internally).
