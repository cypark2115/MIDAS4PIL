# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""Help content for the midas4pil GUI."""

USAGE_GUIDE = """\
midas4pil -- Quick Start Guide
================================

midas4pil is a lightweight diffraction data reduction tool for synchrotron
powder and high-pressure experiments (Pilatus / Eiger detectors, HPCAT APS).
It has two tabs: Calibration and Integration.


CALIBRATION TAB
---------------
Use this tab to determine the detector geometry from a calibrant image
(e.g. CeO2, LaB6).

1. Select your detector from the dropdown (default: Pilatus 2M CdTe).
2. Enter beam energy (keV) or wavelength (angstrom) -- the other field
   updates automatically.
3. Select a calibrant from the dropdown (CeO2 is the default).
4. Click "Load Image" to load your calibrant diffraction image (.tif).
   You can also load a previously saved geometry.toml via File > Load Geometry,
   from an existing .poni geometry file via File > Load .poni,
   or from a MIDAS geometry_params.txt via File > Load MIDAS params.

5. Find the beam centre:

   AUTO FIND CENTER (easy case)
     Use when at least one complete Debye-Scherrer ring is visible.
     Enter an approximate Lsd first.
     Step 1 -- beam centre from ring sharpness (no calibrant needed).
     Step 2 -- Lsd from the first Bragg ring profile; the search is
               bounded by the "Rough <=" field (default 8 deg).
     Tilts (ty, tz) are not changed by Auto Find Center.
     Note: on a tilted detector the sharpness BC may differ from the
     true geometric BC by ~10-15 px.  This is expected -- Run Calibration
     refines the geometric BC and tilts simultaneously.

   MANUAL FIND CENTER (difficult case)
     Use when the detector is heavily off-axis, only partial arcs are
     visible, the visible ring is not the first Bragg reflection, or Auto
     Find Center gave a poor result.
       a. Click "Manual Find Center" -- a ring table appears.
          Select the ring that corresponds to the arc you can see.
       b. Click 3 or more points along that arc on the image.
       c. Click "Done".  A tilt-aware optimizer (Nelder-Mead on
          pixel_to_r_eta) fits BC_Y, BC_Z, and Lsd simultaneously
          using the current ty/tz values from the geometry fields.
          The result is correct even when the ring appears as an ellipse.

   COMBINING STEPS (recommended workflow for challenging cases):
     1. Load .toml  →  initial geometry from a previous session
     2. Manual Find Center  →  update BC and Lsd from visible arc
     3. Auto Find Center  →  if a complete ring is now in range
     4. Run Calibration  →  full refinement (BC, Lsd, tilts, panels)
   Each step uses the current geometry fields as its starting point.

6. Load or edit the mask if needed (see MASK EDITOR below).

7. Click "Run Calibration" to refine all parameters.  The optimizer
   runs in stages; strain is reported after each stage.

   CALIBRATION CHECKBOXES:
     "Optimize panels" -- refine per-module rigid-body shifts (dY, dZ,
       dLsd, dTheta) for tiled detectors.  Panel shifts are a hardware
       property; calibrate once and reuse for all experiments on that
       detector.  Requires a calibrant with many rings.
     "Fit distortion (p2, p3)" -- refine smooth surface warping.  Leave
       unchecked unless per-ring strain shows a systematic radial trend
       after flat-geometry calibration.  Rarely needed for Pilatus/Eiger.

8. Review the calibration result (Show/Save Result...):
     Section 1: Flat Detector Geometry -- Lsd, BC, tilts
     Section 2: Panel Shift Correction -- per-module dY/dZ/dLsd/dTheta
     Section 3: Distortion Correction -- p0-p4 polynomial
     Section 4: Fit Quality -- mean |strain| (< 200 ppm is excellent)
     Section 5: Per-Ring Strains -- residual per calibrant ring

   Systematic strain trend vs 2theta  ->  wrong Lsd or beam centre.
   Single outlier ring                ->  bad ring ID or masked arc.

   Save buttons in this dialog:
     "Save Params..."  -- save full geometry as .toml (recommended)
     "Export .poni..." -- export geometry as a
                          .poni file (for cross-checking; no panel
                          shifts or distortion coefficients)
     "Export MIDAS..." -- export geometry as a MIDAS geometry_params.txt
                          (ImTransOpt 2 frame; panel shifts written to a
                          companion _panel_shifts.txt if present; includes
                          commented-out calibrant and tolerance placeholders)
     "Save Lineout..." -- save the current 1-D lineout as .xye
                          columns: 2theta, I [, SNIP_bg] [, I_sub] [, sigma] [, px_cnt]
                          (variable — only columns with data are written)
     "Save Cake..."    -- save the caked image as TIFF

9. Save geometry.toml, then click "Send to Integration".

RING OVERLAY:
  After any parameter change the rings are redrawn on the image.
  For tilted detectors the overlay traces the true elliptical shape
  computed by ray-tracing through the MIDAS geometry model.

STRAIN COLOR MAP OVERLAY:
  The strain overlay visualises the residual 2-theta error at each position
  on the detector, expressed as a fractional lattice-parameter deviation in
  parts-per-million (ppm).

  How the colour is computed (raw image):
    For each unmasked detector pixel at (row, col), the geometry model maps
    it to a 2theta value from the look-up table.  If that 2theta falls within
    the tolerance band of a calibrant ring, the strain is:

        strain_ppm = (2theta_pixel - 2theta_ring) / 2theta_ring * 1e6

    Blue  = smaller 2theta than the ring predicts = d-spacing larger
            than the calibrant reference (expanded lattice).
    Green = zero residual (perfect agreement with the calibrant).
    Red   = larger 2theta than the ring predicts  = d-spacing smaller
            than the calibrant reference (compressed lattice).
    Transparent = masked pixel or outside every ring's tolerance band.

  How the colour is computed (cake image):
    The cake image is a 2D histogram in (2theta, eta).  Each bin cell is
    coloured by the same fractional offset of its 2theta bin centre relative
    to the nearest calibrant ring.  Cells that contain no valid (unmasked)
    pixels are transparent, so the masked regions in the raw image propagate
    correctly into the cake strain overlay.

  What "centroid" means in the per-ring strain table:
    The reported strain per ring is NOT the single-pixel value at the ring
    centre.  It is the intensity-weighted mean radial position of all
    unmasked pixels within tol_deg of that ring, compared to the theoretical
    ring position.  Formally:

        centroid_tth = sum(I_i * tth_i) / sum(I_i)     [over in-band pixels]
        strain_ppm   = (centroid_tth - tth_ring) / tth_ring * 1e6

    Weighting by intensity emphasises the peak of the Bragg reflection rather
    than the wings or background, giving a more robust estimate of where the
    ring truly is in the data.  A strain near zero means the geometry model
    places the ring within ~1 pixel of where the data say it is.  Values
    above ~200 ppm usually indicate a systematic error in Lsd, BC, or tilts.

MOUSE / INTENSITY DISPLAY:
  Raw/cake image -- status bar shows 2theta, eta, and pixel intensity I.
  Lineout plot   -- status bar shows 2theta and the intensity value
                    (I, I_sub, or SNIP) of the current lineout at cursor,
                    following the priority of the toggle buttons.


MASK EDITOR
-----------
Open from either tab via "Mask..." button.  Red overlay = masked (bad).
Convention: mask = 1 = bad pixel, 0 = good pixel.
Masked pixels are excluded from both calibration and integration.

Drawing tools (toolbar, top-left):
  Circle  : click (brush-size circle) or drag (center-to-edge radius)
  Rect    : click (single pixel) or drag (corner to corner)
  Polygon : sequential clicks; double-click or click first vertex to close
            Vertices outside the image snap to the nearest border.

  Click an active Circle or Rect button again to deactivate it.
  In deactivated state, left-drag draws a zoom box (rubber-band zoom).
  Scroll wheel zooms in/out centred on cursor.  Right-click resets zoom.

Mode toggle:  Mask (add bad pixels) / Unmask (remove bad pixels)
Brush size:   spinbox controls the circle radius for click (not drag)

Action buttons (second row):
  Load       -- load mask from a TIFF file (must match image dimensions)
  Save       -- save mask to TIFF (uint8: 1=bad, 0=good)
  Auto       -- generate mask from image: gaps, dead pixels, saturated pixels
  Despeckle  -- remove isolated masked pixels with no 4-connected neighbours
  Reverse    -- invert the entire mask (good <-> bad for every pixel);
                useful when a loaded mask uses the opposite convention
  Mask >=    -- mask all pixels at or above the threshold value
  Mask <=    -- mask all pixels at or below the threshold value (default 0)
  Reset to Auto -- discard all edits and regenerate from image data

Undo / Redo:  up to 30 steps for all drawing and action operations.


INTEGRATION TAB
---------------
Use this tab to reduce one or many diffraction images to 1D lineouts
and 2D caked images.  Send geometry from the Calibration tab first.

MODES:
  Batch  -- click "Start" to process all .tif files in the selected folder.
             Results (lineouts, caked images) are saved to lineouts/ and
             cakes/ subfolders.  Enable "Watch for new files" to keep
             processing incoming data automatically.
  Manual -- click any file in the list to process and display it immediately.
             Results are saved to lineouts/ (mask on) or lineouts_nomask/
             (mask off).  Use for single-frame inspection.

INTEGRATION METHOD:
  varbin (default) -- pixel-matched variable-width bins; the physically
    correct choice.  Each bin spans the angular footprint of one detector
    pixel.  Recommended for Rietveld refinement (GSAS-II, FullProf, TOPAS).
  unibin -- uniform angular spacing.  Use only for PDF analysis (FFT
    requires a uniform grid) or when uniform spacing is specifically needed.

ETA BIN SIZE (varbin cake):
  The "Eta bin (deg)" spinbox sets the azimuthal bin width for the varbin
  cake image.  Default is 1.00 deg.  The eta bin size and the lower 2theta
  cutoff (2th_min) are coupled:

      delta_eta [rad] = px / (Lsd * tan(2th_min))

  The "2th_min" readout next to the spinbox shows the resulting cutoff.
  Data below 2th_min is excluded from both the cake image and the lineout.
  A finer eta bin gives better azimuthal resolution at the cost of a higher
  2th_min.  A warning appears when the value is finer than the default.
  Changing the spinbox triggers a full bin map recompute.

FILE LIST:
  Click a file to display its cake and lineout.
  Check the checkbox next to a file to include its lineout in the overlay.
  Multiple checked files are overlaid in the lineout plot simultaneously.
  The cake image always shows the currently selected (clicked) file.

  File list shortcuts:
    Space        -- toggle check state of all currently selected files
    Shift+scroll -- extend selection up / down
    Drag         -- drag over items to check/uncheck them in one stroke
    Right-click  -- context menu: check/uncheck selected or all files

LINEOUT DISPLAY:
  Toggle buttons I / I_sub / SNIP control which curves are shown:
    I       -- raw integrated intensity
    I_sub   -- background-subtracted (I minus SNIP); shown with a ±sigma
               shaded band (Poisson standard error of the mean per bin)
    SNIP    -- estimated background (Statistics-sensitive Non-linear
               Iterative Peak-clipping algorithm)
  In multi-overlay mode: checked files show I only (thin, cycling colours);
  the currently selected file shows I/I_sub/SNIP per the toggle buttons.

SNIP 2theta RANGE:
  Two orange dashed vertical lines on the lineout plot define the 2theta
  range for SNIP background computation.  Outside this range bg = NaN and
  I_sub = NaN.  Drag either line or edit the "SNIP 2theta:" spinboxes in
  the lineout header.  The range resets to full detector coverage when a new
  geometry is loaded.  Batch processing always uses full-range SNIP; this
  restriction applies only to the interactive display and Export I_sub.

EXPORT I_sub:
  "Export I_sub..." opens a scope dialog:
    Current file       -- re-integrate the frame currently displayed
    Checked files      -- re-integrate all files checked in the file list
    All files in folder-- re-integrate every .tif in the data folder
  Each file is saved as a 6-column file [2theta, I, SNIP_bg, I_sub, sigma,
  px_cnt] to lineouts_sub/ (mask on) or lineouts_sub_nomask/ (mask off).
  The header records all column names plus wavelength, energy, 2theta range,
  and eta range.

SAVED FILE FORMATS:
  Batch / Manual (I-export):  [2theta, I, sigma, px_cnt]  -- 4 columns
  Export I_sub:               [2theta, I, SNIP_bg, I_sub, sigma, px_cnt] -- 6 columns
  Both include explicit column names in the file header.
  Older 3-column and 5-column files from previous versions are still loaded.

MOUSE / INTENSITY DISPLAY:
  Image panel (raw/cake) -- shows 2theta, eta, pixel intensity I, and
                            px_cnt=nnn (active pixels in the hovered cake cell;
                            available after live reduction).
  Lineout plot           -- shows 2theta, intensity value at cursor
                            (I > I_sub > SNIP priority following toggle
                            button state), and px_cnt=nnn (pixel count for
                            the nearest 2theta bin).

MASK:
  "Mask O/●" button in the display header applies the mask during integration.
  "Mask..." opens the mask editor (see MASK EDITOR above).
  If no mask is loaded, Auto mask is generated from the first image.
"""


TECHNICAL_README = """\
midas4pil -- Technical Reference
==================================

VARIABLE-BIN INTEGRATION (varbin)
---------------------------------
The default integration mode uses pixel-matched variable-width 2theta bins.
Each bin spans the angular width of one detector pixel at that 2theta:

    delta(2theta) = px * cos^2(2theta) / Lsd

Each bin contains exactly the angular information that one pixel can
physically deliver.  Bins are wider at low 2theta (where the pixel
subtends a larger angle) and narrower at high 2theta.

For the eta direction (cake image), the bin size is user-controlled via the
"Eta bin" spinbox (default 1.0 deg).  It is linked to the lower 2theta
cutoff by:

    delta_eta [rad] = px / (Lsd * tan(2th_min))

A finer eta bin raises 2th_min; a coarser bin lowers it.  Both cake and
lineout use the same 2th_min.

Rietveld refinement codes (GSAS-II, FullProf, TOPAS) accept non-uniform
bin spacing.  Varbin is the appropriate choice whenever the goal is to
faithfully represent the angular resolution of the detector.

UNIFORM-BIN INTEGRATION (unibin)
---------------------------------
Uniform bins are available for cases that strictly require equal spacing:
  - FFT-based pair distribution function (PDF) analysis
  - Real-time monitoring when speed matters more than bin resolution

When using unibin, the bin width should never be set smaller than the
detector's angular resolution at the lowest 2theta in the range.


SNIP BACKGROUND
---------------
The SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) algorithm
removes the amorphous/thermal diffuse scattering background from 1D patterns.

Reference: Morhac et al., NIM A 401 (1997) 113-132.
Default: 50 iterations.  Set to 0 to disable.


COORDINATE CONVENTIONS
----------------------
  BC_Y:  beam centre column from left edge (pixels)
  BC_Z:  beam centre row from top edge (pixels)
  eta:   azimuthal angle, 0 at 3 o'clock, positive CCW from sample view
         (12 o'clock = +90, 6 o'clock = -90, 9 o'clock = +/-180)
  Mask:  1 = bad pixel, 0 = good pixel


DETECTOR GEOMETRY MODEL
-----------------------
Three correction layers, calibrated and applied in this order
(geometry model from MIDAS DetectorGeometry.c):

  Layer 1 -- Flat detector geometry (PONI)
    The foundational model.  Five fitted parameters; tx (rotation around
    the beam axis X) is fixed at zero because it only shifts the azimuthal
    reference (eta=0 direction) and cannot be determined from azimuthally
    uniform powder rings.
      Lsd      : sample-to-detector distance (um)
      BC_Y     : beam centre column from left (pixels)
      BC_Z     : beam centre row from top (pixels)
      ty       : detector tilt around Y axis (horizontal transverse, degrees)
      tz       : detector tilt around Z axis (vertical transverse, degrees)

  Layer 2 -- Per-panel shifts (tiled detectors: Pilatus, Eiger)
    Per-module rigid-body corrections for physical misalignment of individual
    sensor boards relative to the flat detector model.  Hardware constant:
    calibrate once per detector, reuse for all experiments.
      dY, dZ   : in-plane module shifts (pixels)
      dLsd     : distance offset along beam axis (um)
      dTheta   : in-plane module rotation (degrees)

  Layer 3 -- Distortion polynomial (non-flat detector surface)
    Smooth correction for deviations of the detector surface from the ideal
    flat plane.  Applied to the effective radial distance after the global
    transform.
    rho_norm = Rad / rho_d  (Rad in um; rho_d ~ 217 578 um for Pilatus 2M)
    etaT     = 180 - eta    (MIDAS internal azimuth variable)
    distort  = 1 + p0*rho_norm^2*cos(2*etaT)
                 + p1*rho_norm^4*cos(4*etaT + p3)
                 + p2*rho_norm^2
                 + p4*rho_norm^6
    R_corr = (Rad * distort / px) * (Lsd / Lsd_panel)  [pixels]
    For Pilatus/Eiger CMOS sensors this correction is typically negligible.

CALIBRATION STAGE ORDER
------------------------
  Only panels    : Stage1=rough PONI  Stage2=PONI  Stage3=PONI+panels
  Only distortion: Stage1=rough PONI  Stage2=PONI+distortion
  Both           : Stage1=rough PONI  Stage2=PONI+panels  Stage3=PONI+panels+distortion

Panels are always fitted before distortion.  Fitting distortion first
causes the smooth polynomial to absorb discrete per-module offsets,
giving wrong values for both corrections.


STRAIN RESIDUAL -- DEFINITION
------------------------------
The strain reported per ring and shown in the colour-map overlays is the
fractional 2-theta deviation from the calibrant's expected ring position:

    strain_ppm = (2theta_measured - 2theta_theory) / 2theta_theory * 1e6

"2theta_measured" is the intensity-weighted centroid of the ring in the
data -- the mean 2theta of all unmasked pixels within the tolerance band,
weighted by their diffracted intensity:

    centroid_tth = sum_i( I_i * tth_i ) / sum_i( I_i )

Intensity weighting emphasises the peak of the Bragg reflection rather
than the flat-intensity wings inside the band, so the centroid closely
tracks the true peak position even when tol_deg spans several pixels.

"2theta_theory" is computed from the calibrant's known lattice parameters
and wavelength via Bragg's law:   2theta = 2 * arcsin(lambda / (2 * d_hkl))

Sign convention:
  strain > 0  →  2theta_measured > 2theta_theory
               →  d-spacing smaller than reference  (compressed)
  strain < 0  →  2theta_measured < 2theta_theory
               →  d-spacing larger than reference   (expanded)

Colour map (blue–green–red):
  Blue  = expanded    (strain < 0, 2theta below ideal)
  Green = zero residual
  Red   = compressed  (strain > 0, 2theta above ideal)

Interpreting the result:
  < 200 ppm (mean |strain|) : excellent calibration
  Systematic trend vs 2theta : Lsd or BC error; re-run calibration
  Single-ring outlier        : ring misidentified or arc heavily masked


BEAM CENTRE -- AUTO FIND CENTER
--------------------------------
Two steps; no calibrant ring knowledge required for step 1:

  1. BC from azimuthal sharpness: maximises var(I_bar(R)), the
     variance of the radially-averaged intensity profile, over a
     coarse grid ±200 px from the detector centre, then Nelder-Mead.
     On a tilted detector (ty, tz != 0) the sharpness peak is at the
     ring's IMAGE centre, which can differ from the true geometric BC
     by ~10-15 px.  This is expected physics, not a bug.  Run
     Calibration to recover the geometric BC and tilts simultaneously.

  2. Lsd from the first Bragg ring: builds I_bar(R) from the step-1
     BC; finds the innermost peak using prominence-based detection
     (scipy.signal.find_peaks).  The "Rough <= (deg)" field bounds
     the upper search radius.  Accuracy: ~0.2 %.

If the current geometry is already good (e.g. loaded from .toml), Auto
Find Center preserves it: the coarse scan uses the init geometry score
as a floor and cannot degrade a good starting point.

These two values seed Run Calibration.  Tilts are not estimated by
Auto Find Center -- they are refined by Calibration.


BEAM CENTRE -- MANUAL FIND CENTER
----------------------------------
For off-center geometries with only partial Debye-Scherrer arcs:
  1. Click "Manual Find Center" -- a table of calibrant rings appears.
     Select the ring whose arc you can see on the detector.
  2. Click 3+ points along the arc on the raw image.
  3. Click "Done".
  4. A tilt-aware Nelder-Mead optimizer minimizes
       sum[ (tth_computed(xi, yi) - tth_ring)^2 ]
     over (BC_Y, BC_Z, Lsd), using the current ty and tz values from
     the geometry fields and the full pixel_to_r_eta() geometry model.
     The result is accurate even when the ring appears as an ellipse
     (tilted detector, large tz).  No approximation is made.

This gives the correct geometric beam centre directly from clicked points,
regardless of detector tilt or partial arc visibility.


PERFORMANCE
-----------
With numba JIT and precomputed bin maps:
  - Per-frame reduction:  ~13 ms (79 Hz) for Pilatus 2M
  - One-time precompute:  ~1.2 s (done once when geometry is set)
  - Without numba:        ~160 ms (6 Hz) per frame

Precomputed bin maps are used in both batch and manual modes whenever
the geometry is unchanged between frames.
"""
