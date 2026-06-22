# midas4pil v1.1.3 — Reference Report

**Date:** 2026-04-27
**Author:**    Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory (cypark@anl.gov)
**Co-author:** Claude Code (claude-sonnet-4-6, Anthropic AI)

---

## 1. Package Summary

| Item | Value |
|------|-------|
| Package | midas4pil |
| Version | 1.1.3 |
| Release date | 2026-04-27 |
| Python | >= 3.9 |
| Core dependencies | numpy, tifffile, gemmi, scipy, numba, tomli (Python < 3.11) |
| GUI dependencies | PySide6 >= 6.4, matplotlib >= 3.5 |
| Install | `pip install --user midas4pil-1.1.3-py3-none-any.whl` |
| Install with GUI | `pip install --user "midas4pil-1.1.3-py3-none-any.whl[gui]"` |
| Core modules | 7 (geometry, integrate, cake, panels, optimizer, io, calibrant) |
| GUI modules | 13 (in `midas4pil/gui/`) |
| Tests | 121 passing (~6 s) |
| Public API | 41 functions (in `__all__`) |
| Bundled calibrants | 45 JCPDS files (7 subdirectories) |
| Wheel contents | Core + GUI + calibrants + docs/ + calib_example/ |

---

## 2. Architecture

### Core package

```
midas4pil/
  __init__.py        -- 41 public symbols; version 1.1.3
  geometry.py        -- build_tilt_matrix, pixel_to_r_eta, r_to_tth, build_lut,
                        pixel_resolution, varbin_tth_edges, lut_tth_range,
                        find_beam_center, find_beam_center_auto,
                        fit_circle, lsd_from_ring
  integrate.py       -- snip_background, integrate_1d, integrate_1d_varbin,
                        reduce_frame, precompute_bin_maps, rebin_lineout
  cake.py            -- cake, cake_varbin
  panels.py          -- make_panel_id_map, read_panel_shifts,
                        apply_panel_offsets, build_lut_with_panels
  optimizer.py       -- calibrate (logging-based diagnostics)
  io.py              -- read_poni, write_poni, make_geometry, save_params,
                        load_params, load_tiff, load_mask, orient_mask,
                        auto_mask, save_mask
  calibrant.py       -- read_jcpds, read_cif, ring_table, load_calibrant
  _jit.py            -- numba JIT compilation helpers
  calibrants/        -- 45 curated JCPDS files (7 subdirectories)
  gui/               -- 13 GUI modules (PySide6, optional)
  docs/              -- README.md, USAGE_GUIDE.md, REFERENCE_REPORT.md (in wheel)
  calib_example/     -- run_calibration.py + input data + README (in wheel)
```

### GUI package (`midas4pil/gui/`)

```
  __init__.py        -- PySide6 dep check, launch_gui()
  __main__.py        -- python -m midas4pil.gui entry point
  main_window.py     -- dark QPalette + stylesheet, menu bar, tab host
  calib_tab.py       -- scrollable left panel, ring overlay, Auto/Manual
                        Find Center, Run Calibration, Mask Editor launch
  integ_tab.py       -- folder browse, batch/manual mode, eta sector,
                        rebin, Export I_sub, Watch mode, BatchWorker
  display_widget.py  -- TIFF/Cake toggle, green crosshair, histogram +
                        RangeSlider, tth sync, SNIP range vlines,
                        I_sub error band, lineout intensity hover
  stack_widget.py    -- sequence map (X=2theta, Y=frame), grayscale,
                        histogram + RangeSlider, tth sync line
  mask_editor.py     -- Circle/Rect/Polygon draw, Mask/Unmask, Undo/Redo,
                        Auto, Despeckle, Reverse, Mask >=, Mask <=
  crosshair.py       -- SyncBus(QObject): shared Signal bus for tth sync
  worker.py          -- BatchWorker (QThread), ManualWorker
  detectors.py       -- detector registry, make_panel_map_from_shape
  hist_clim.py       -- histogram + RangeSlider side panel (reusable widget)
  logging_setup.py   -- rotating log file handler
  help_text.py       -- USAGE_GUIDE and TECHNICAL_README plain-text content
```

### Key API return values (current)

| Function | Returns |
|----------|---------|
| `integrate_1d` | `(tth, I, bg, I_sub, sigma, px_cnt)` — 6-tuple |
| `integrate_1d_varbin` | `(tth, I, bg, I_sub, sigma, px_cnt)` — 6-tuple |
| `cake` | `(cake_img, tth_c, eta_c, px_cnt_map)` — 4-tuple |
| `cake_varbin` | `(cake_img, tth_c, eta_c, eta_bin_size, px_cnt_map)` — 5-tuple |
| `reduce_frame` | dict with keys: `tth, I, bg, I_sub, sigma, px_cnt, cake_img, tth_cake, eta_cake, px_cnt_cake, eta_bin_size`. Optional `geom['eta_bin_size']` sets varbin azimuthal bin width (default auto from tth_min). |
| `rebin_lineout` | `(tth_r, I_r, px_cnt_r, sigma_r)` — 4-tuple, co-adds N native bins |
| `precompute_bin_maps` | dict — keys: `mode`, `bin_map_1d`, `n_bins_1d`, `tth_centres`; plus 2-D cake entries |

### Data flow

```
geometry_init.toml  -->  make_geometry()
                              |
image.tif  -->  load_tiff()   |
mask.tif   -->  load_mask()   v
                         build_lut()  -->  (tth_lut, eta_lut)
                              |
                    precompute_bin_maps()  -->  bin_maps  [one-time, ~1.2 s]
                              |
calibrant.jcpds  -->  load_calibrant()  -->  rings
                              |
                         calibrate()  -->  geom, panel_shifts  [one-time]
                              |
                    build_lut_with_panels()  -->  (tth_lut, eta_lut)
                              |
                       reduce_frame()  -->  {tth, I, bg, I_sub, sigma,
                    (per frame, ~13 ms)       px_cnt, cake_img, ...}
                              |
                       rebin_lineout()  [optional: co-add N bins]
```

---

## 3. Development History

### v0.2.0 — Core API (2026-04-10)

| # | Item | Files |
|---|------|-------|
| 0 | Curate calibrant data into `calibrants/` | 45 JCPDS files |
| 1 | Package calibrant data in wheels | `pyproject.toml` |
| 2 | Preserve `mode` through save/load | `io.py` |
| 3 | Add `tomli` fallback dependency | `pyproject.toml` |
| 4 | Delete dead MIDAS compat code | `io.py` |
| 5 | Move calibrant resolver into package | `calibrant.py` |
| 6 | Add `reduce_frame()` facade | `integrate.py`, `__init__.py` |
| 7 | Replace print() with logging | `optimizer.py` |
| 8 | Level 2 performance: precompute + numba JIT | `integrate.py`, `_jit.py` |

### v1.0.0 — GUI + Full Feature Set (2026-04-16 to 2026-04-21)

| Feature | Description |
|---------|-------------|
| PySide6 GUI | Complete 2-tab GUI (Calibration + Integration), dark theme |
| Auto Find Center | 2-step: azimuthal sharpness BC + first-ring Lsd profile |
| Manual Find Center | Tilt-aware Nelder-Mead on clicked arc points |
| Ring overlay | True ellipses via ray-tracing through TRs tilt model |
| Run Calibration | 3-stage optimizer with collapsible result dialog |
| Mask Editor | Circle/Rect/Polygon, Undo/Redo (30 steps), Reverse, Despeckle |
| Batch integration | Folder watcher, BatachWorker, Watch mode |
| Eta sector | Spinboxes + draggable orange dashed lines on cake |
| Rebin ×N | Co-add N native bins; σ = √(ΣI\_sum)/Σpx\_cnt; re-runs SNIP |
| Counts display | px\_cnt\_cake shown in viridis on demand |
| SNIP 2θ range | Draggable orange vlines + spinboxes; limits SNIP computation |
| Export I\_sub | Scope dialog (current / checked / all files); saves 6-col [2θ, I, SNIP\_bg, I\_sub, σ, px\_cnt] |
| I\_sub error band | ±σ fill\_between on lineout plot |
| Lineout hover | I/I\_sub/SNIP value at cursor (priority order) |
| tth sync | SyncBus.tth\_changed propagates green dashed line to all canvases |
| `load_params` compat | Reads old `[geometry]`+`[beam]` AND new TOML section names |
| `rebin_lineout` | New public API: `(tth, I, cnt, factor)` → 4-tuple `(tth_r, I_r, cnt_r, sigma_r)` |
| `precompute_bin_maps` | New public API: one-time bin maps for fast per-frame reduction |
| Panel shifts in TOML | `[[panel_shifts]]` tables embedded in geometry TOML |
| 6-tuple lineout | `integrate_1d`/`integrate_1d_varbin` return `(tth, I, bg, I_sub, sigma, px_cnt)` |
| 4/5-tuple cake | `cake` returns 4-tuple, `cake_varbin` returns 5-tuple (added px\_cnt\_map) |
| calib_example in wheel | Bundled in `midas4pil/calib_example/`; decluttered to 6 files |
| docs in wheel | `midas4pil/docs/`: README.md, USAGE_GUIDE.md, REFERENCE_REPORT.md |

### Bugs fixed (2026-04-21 audit, 36 cases)

| ID | Bug | Fix location |
|----|-----|-------------|
| F1 | 3-col .xye not parsed in `_on_item_check_changed` | `integ_tab.py` |
| F2 | 3-col .xye not parsed in `show_pair()` | `display_widget.py` |
| F3 | `load_params()` rejected old-format TOML files | `io.py` |
| F4 | `calib_example` TOML in old format | `calib_example/` TOML |
| F5 | `_on_file_selected` hardcoded `lineouts/` (missed nomask path) | `integ_tab.py` |
| F6 | `show_pair()` hardcoded `lineouts/` (missed nomask path) | `display_widget.py` |
| F7 | ✓ checkmark not updating in Manual mode (mask-off) | `integ_tab.py` |
| F8 | Mask overlay invisible in TIFF/Image view | `display_widget.py` |

### v1.1.3 — Bug fixes: NrPixels parsing, geom dict mutation, numba path (2026-04-27)

| Item | Description | Files |
|------|-------------|-------|
| `load_midas_params()` NrPixels square detector | When a MIDAS params file contains only `NrPixels` (square detector) instead of separate `NrPixelsY` + `NrPixelsZ`, both `nrows` and `ncols` are now correctly set to that value. Previously the `NrPixels` key was not recognised by the alias lookup and the function silently returned a partial dict without detector dimensions. | `io.py`, `tests/test_io.py` |
| `load_geometry()` geom dict mutation | When called with a partial geom dict (MIDAS params file without `NrPixels` lines), `load_geometry()` was adding `nrows`/`ncols` fallback values directly into the caller's dict via `setdefault`, corrupting state for any code that held a reference to the same dict. Fix: `geom = dict(geom)` creates a shallow copy before the `setdefault` calls. | `gui/integ_tab.py` |
| numba path always active without mask | `_recompute_bin_maps()` returned early when `self.mask is None`, leaving `bin_maps = None` and forcing the pure-numpy fallback (~6 Hz) even after geometry was loaded. Fix: a `null_mask = np.zeros(self.tth_lut.shape, dtype=bool)` is used in place of the real mask when no mask is loaded, so `precompute_bin_maps()` always runs and numba JIT is always active. | `gui/integ_tab.py` |
| Tests | 1 new test: `test_load_midas_params_nrpixels_square` in `test_io.py`. Total: 121 tests. | `tests/test_io.py` |

### v1.1.2 — Cake strain overlay masking fix + strain color map documentation (2026-04-26)

| Item | Description | Files |
|------|-------------|-------|
| Cake strain overlay masking | `_compute_cake` now returns the per-cell pixel-count array (`counts.T`) as a 4th return value. `show_cake_strain_overlay` accepts `px_cnt_cake` and builds a full 2D RGBA array; cells with zero valid pixels are transparent, matching the per-pixel masking already applied by the raw-image strain overlay. Previously all ring-band cells were filled uniformly regardless of mask. | `gui/calib_tab.py` |
| Strain color map documentation | Added "STRAIN COLOR MAP OVERLAY" and "STRAIN RESIDUAL — DEFINITION" sections to the in-app help browser. Describes the blue–green–red convention (blue = expanded lattice, green = zero residual, red = compressed), the intensity-weighted centroid definition, and how to interpret per-ring strain values. | `gui/help_text.py` |
| GUI_SCENARIOS C11 | New scenario: "Read the strain colour-map overlay" with correct blue = expanded, red = compressed description. | `GUI_SCENARIOS.md` |

### v1.1.1 — MIDAS geometry_params.txt interoperability + panel_shifts column order fix (2026-04-26)

| Item | Description | Files |
|------|-------------|-------|
| `load_midas_params()` | New function in `io.py`. Reads a MIDAS flat key-value geometry_params.txt. Handles key aliases (`Distance`/`DetDist`→`Lsd`, `PixelSize`→`px`, `NrPixels` for square detectors). If `ImTransOpt 2` is present, BC_Z is converted from the MIDAS top-bottom-flipped frame to midas4pil's original-TIFF frame. Loads `PanelShiftsFile` automatically (relative path resolved against params file directory). Returns a complete midas4pil geometry dict. | `io.py` |
| `save_midas_params()` | New function in `io.py`. Exports a midas4pil geometry dict as a MIDAS geometry_params.txt with `ImTransOpt 2` (BC_Z converted to flipped frame). Auto-detects panel topology (NPanelsY, NPanelsZ, etc.) from the detector shape via the preset database. Writes a companion `panel_shifts.txt` if panel shifts are present and `panel_shifts_path` is given. Includes commented-out placeholders for calibrant (SpaceGroup, LatticeConstant, RingThresh) and calibration tolerance keys. | `io.py` |
| `panel_shifts` column order fix | `read_panel_shifts` / `save_panel_shifts` in `panels.py` used wrong column order (`dLsd` before `dTheta`). Corrected to MIDAS Panel.c order: `id dY dZ dTheta dLsd dP2`. | `panels.py` |
| GUI: Load MIDAS params | New File menu action "Load MIDAS params…". Calls `load_midas_params()` and fills all geometry fields (Lsd, BC, tilts, wavelength, distortion, panel shifts). | `gui/main_window.py`, `gui/calib_tab.py` |
| GUI: Export MIDAS… button | New button in the calibration result dialog (next to "Export .poni…"). Calls `save_midas_params()` with auto-derived panel shifts path. | `gui/calib_tab.py` |
| Tests | 11 new tests: 4 in `test_panels.py` (column order, roundtrip, MIDAS-format input, optional-cols defaults) and 7 in `test_io.py` (geometry parsing, BC_Z frame, key aliases, square detector, roundtrip, BC_Z frame roundtrip, PanelShiftsFile). Total: 120 tests. | `tests/` |

### v1.1.0 — median removal + README/Credits cleanup + eta bin control + calib tab px_cnt (2026-04-25)

| Item | Description | Files |
|------|-------------|-------|
| Calibration tab `px_cnt` | `_recompute_lineout` now captures `px_cnt` (was discarded with `_`). `_lineout` tuple extended from 5 to 6 elements: `(tth, I, bg, I_sub, sigma, px_cnt)`. Hover readout shows `px_cnt=nnn`. `_on_save_lineout` appends `px_cnt` column after `sigma` when available, making the max column format `[2θ, I, SNIP_bg, I_sub, σ, px_cnt]` — consistent with Integration tab's 6-col I\_sub export. | `gui/calib_tab.py` |

### v1.1.0 — median removal + README/Credits cleanup + eta bin control (2026-04-24)

| Item | Description | Files |
|------|-------------|-------|
| `method` parameter removed | `integrate_1d`, `integrate_1d_varbin`, `cake`, `cake_varbin` no longer accept `method=`. Mean is hardcoded. Median cannot be combined with correct Poisson SEM propagation. | `integrate.py`, `cake.py` |
| `_median_per_bin` removed | Helper function and `_SNIP_MEDIAN_CORRECTION` constant deleted | `integrate.py` |
| README Credits rewritten | Three MIDAS-derived components (geometry, panel correction, SNIP) credited explicitly to Hemant Sharma; new components identified as original work | `README.md` |
| `optimizer.py` docstring | "Equivalent to MIDAS CalibrantPanelShiftsOMP" → "independent Python implementation" | `optimizer.py` |
| Comparison table replaced | "How it compares" multi-tool table replaced by standalone "Integration approach" table describing midas4pil features only | `README.md` |
| `eta_bin_size` parameter added to `cake_varbin` | New optional parameter (default None = auto from tth_min). When supplied, sets the azimuthal bin width directly, overriding the auto-computed value. `reduce_frame` reads `geom.get('eta_bin_size')` and passes it through. | `cake.py`, `integrate.py` |
| Eta bin spinbox in Integration tab GUI | New **Eta bin (deg)** spinbox (default 1.0, range 0.1–10.0). Coupled read-only **2th_min** label updates live. Warning label shown when eta bin < 1.0 deg (finer than default). Changing the spinbox re-calls `lut_tth_range` with the new `max_eta_bin_deg`, updates `geom['tth_min']` and `geom['eta_bin_size']`, and triggers bin map recompute. | `gui/integ_tab.py` |

### v1.0.3 — px_cnt unification + 4/6-col xye format + Export I_sub redesign (2026-04-23)

| Item | Description | Files |
|------|-------------|-------|
| `px_cnt` naming unification | All pixel-count quantities renamed `cnt` → `px_cnt`, `cnt_map` → `px_cnt_map`, `cnt_cake` → `px_cnt_cake` project-wide (API, JIT kernels, GUI, tests, docs) | `_jit.py`, `cake.py`, `integrate.py`, `__init__.py`, `gui/integ_tab.py`, `gui/worker.py`, `tests/test_integrate.py` |
| 4-column I-export | Main batch `.xye` changed from 3 to 4 columns: `[2θ, I, σ, px_cnt]`; all 4 column names written explicitly in the file header | `gui/worker.py`, `gui/integ_tab.py` |
| 6-column I\_sub export | Export I\_sub saves `[2θ, I, SNIP_bg, I_sub, σ, px_cnt]` (6 cols) with full column header | `gui/integ_tab.py` |
| Export I\_sub scope dialog | "Export I\_sub…" opens a scope dialog: **Current file** / **Checked overlay files** / **All files in folder** / Cancel; multi-file loop updates progress label | `gui/integ_tab.py` |
| px\_cnt in hover labels | Cake view hover shows `px = nnn`; Lineout hover shows `px = nnn` next to intensity (from live reduction only for cake; from file for lineout if 4-col or 6-col format) | `gui/display_widget.py` |
| Backward-compatible file reading | Column detection updated: ≥6 cols → new I\_sub; 5 cols → legacy I\_sub; 4 cols → new I-export; 3 cols → legacy I-export; all load into 6-tuples internally | `gui/integ_tab.py`, `gui/display_widget.py` |

### v1.0.2 — GUI usability fixes + theme color corrections (2026-04-23)

| Item | Description | Files |
|------|-------------|-------|
| Resizable panels | Horizontal `QSplitter` between left control panel and display area; vertical splitter handle width set to 4 px (was invisible on dark backgrounds) | `integ_tab.py`, `display_widget.py` |
| Toolbar cleanup | `_NavBar` subclass removes non-functional "Subplots" / "Customize" buttons — matplotlib subplot params dialog has no effect on `add_axes()` axes | `display_widget.py` |
| Dark theme — menus invisible | Added explicit `QMenuBar` / `QMenu` stylesheet rules (dark bg, light text, selection highlight) | `main_window.py` |
| Light/dark theme — widgets invisible | Added explicit `color:` to all `QPushButton`, `QLabel`, `QRadioButton`, `QCheckBox`, `QGroupBox`, `QLineEdit`, `QComboBox`, `QListWidget`, `QTabBar::tab` rules; prevents native platform style bleeding through | `main_window.py` |
| Consistency audit | Fixed `cake()` / `cake_varbin()` return-variable names in README code examples (`tth_edges` → `tth_centres`, missing `cnt_map`); renamed two misnamed test functions (`five_columns` → `six_columns`) | `README.md`, `midas4pil/docs/README.md`, `tests/test_integrate.py` |

### v1.0.1 — GUI fixes + Export I_sub format (2026-04-22)

| Item | Description | Files |
|------|-------------|-------|
| Overlay stale data | Current file's overlay now always uses the active η and SNIP 2θ range; `show_results()` syncs `_lineout_store` when the title is already in the overlay | `display_widget.py` |
| Lineout label colour | `_lo_pos_label` colour is now theme-aware (white on dark, near-black on light); was hardcoded `#cccccc` — invisible on the light theme | `display_widget.py` |
| Export I\_sub format | 5-column output: `[2θ, I, SNIP, I_sub, σ]`; header comment records 2θ and η ranges used; was 3-column `[2θ, I_sub, σ]` | `integ_tab.py` |
| Help browser buttons | "Usage Guide" and "Technical README" both show "Open in Browser" button pointing to bundled `.md` files; HTML generation removed | `main_window.py` |
| Lineout zoom preserved | Toggle I/I\_sub/SNIP/σ no longer resets zoomed x/y view; `_line_x_locked` flag prevents zoom reset when recomputing same file | `display_widget.py` |

---

## 4. Curated Calibrant Library

### NIST SRM Standards (7 files)

| SRM | Material | Space Group | a (Å) | Notes |
|-----|----------|-------------|-------|-------|
| 640f | Si | Fd-3m | 5.431144 | Line position standard |
| 660c | LaB6 | Pm-3m | 4.156826 | Instrument profile function |
| 674b | CeO2 | Fm-3m | 5.41129 | Quantitative analysis |
| 674b | TiO2 (rutile) | P4\_2/mnm | a=4.5933, c=2.9592 | Quantitative analysis |
| 674b | Cr2O3 | R-3c | a=4.9590, c=13.5940 | Quantitative analysis |
| 674b | ZnO (wurtzite) | P6\_3mc | a=3.2501, c=5.2066 | Quantitative analysis |
| 676a | Al2O3 (corundum) | R-3c | a=4.7589, c=12.9917 | Quantitative analysis |

### HP-XRD Pressure Markers and Media (38 files)

| Category | Materials |
|----------|-----------|
| metals/ (13) | Au, Pt, Cu, Mo, Ta, Re, Fe\_bcc, Fe\_hcp, Nb, V, Ag, Pb, W |
| oxides/ (6) | MgO, CaO, FeO, Fe2O3, Fe3O4, EuO |
| halides/ (7) | NaCl\_B1, NaCl\_B2, KBr\_B1, KBr\_B2, KCl\_B1, LiF, CsI |
| gases/ (5) | Ne, Ar, Xe, Kr, H2 |
| carbides/ (5) | WC, TiC, ZrC, B4C, SiC\_3C |
| carbon/ (2) | diamond, graphite |

---

## 5. Performance Benchmarks (Pilatus 2M, 1679 × 1475)

### Achieved (Level 1 + Level 2 — precompute + numba JIT)

| Operation | Time | Notes |
|-----------|------|-------|
| `build_lut` | ~270 ms | one-time per geometry |
| `precompute_bin_maps` | ~210 ms | one-time; reused across all frames |
| `reduce_frame` (varbin, lineout + cake) | **~15 ms (65–79 Hz)** | with precomputed maps + numba JIT |
| `reduce_frame` (unibin) | ~10 ms (~100 Hz) | with precomputed maps + numba |
| GUI pipeline (compute + matplotlib render) | **~117 ms (~9 Hz)** | cake imshow 70 ms + lineout 29 ms |
| GUI display cap (QTimer) | 200 ms (5 Hz) | intentional throttle; render is the ceiling |
| `find_beam_center_auto` | ~10 s | one-time, calibration only |
| `calibrate` (3-stage) | ~45 s | one-time, calibration only |
| Peak memory | ~141 MB | for one Pilatus 2M frame |

### Baseline (Level 0 — pure numpy, no precompute)

| Operation | Time | Rate |
|-----------|------|------|
| `reduce_frame` (varbin) | ~173 ms | 6 Hz |
| `reduce_frame` (unibin) | ~93 ms | 11 Hz |

### Performance roadmap

| Level | Method | Compute rate | GUI pipeline | Status |
|-------|--------|-------------|--------------|--------|
| 0 | pure numpy | 6 Hz | — | baseline |
| 1 | precompute bin index maps | 8–10 Hz | — | **DONE** (2026-04-10) |
| 2 | numba JIT on hot loops | 65–79 Hz | **~9 Hz** | **DONE** (2026-04-10) |
| 3 | CuPy (GPU bincount/reduce) | 500+ Hz | still ~9 Hz† | future (Eiger upgrade) |
| 4 | custom CUDA + ring buffer | kHz+ | still ~9 Hz† | future (PVA streaming) |

† The GUI pipeline ceiling at Levels 3–4 is **matplotlib rendering (~99 ms/frame)**,
not compute. Even at 500 Hz compute, the display is limited to ~10 Hz by
`matplotlib.imshow` rasterisation (~70 ms) and lineout plot (~29 ms).
Reaching 30+ Hz display requires matplotlib blitting or replacing the image
canvas with a pyqtgraph/OpenGL widget.

Levels 1 and 2 are implemented together: `precompute_bin_maps` eliminates the
dominant `searchsorted` cost (~40 ms); numba JIT eliminates the bincount loop
overhead. The combined effect gives 65–79 Hz on a Pilatus 2M (≥10× speedup).

CPU-only at Levels 0–2. GPU (A6000 available) for Levels 3–4.

---

## 6. Confirmed Calibration Result

**Dataset:** CeO2 at 29.2 keV (0.42460 Å), Pilatus 2M CdTe, ~350 mm

| Stage | Lsd (µm) | BC (y, z) | Strain (ppm) |
|-------|----------|-----------|--------------|
| Initial (TOML) | 350000 | (736.82, 810.21) | — |
| Stage 1 (rough, 4 rings) | 349472 | (737.85, 811.22) | 366 |
| Stage 2 (fine, 8 rings) | 349444 | (747.74, 817.04) | 175 |
| Stage 3 (panels, 24 panels) | 349254 | (748.05, 817.03) | **117** |
| Global-only (Stage 3 geom, no panels) | — | — | 614 |

Best result: **117 ppm** (with panel corrections), **175 ppm** (without).
Both are in the "Excellent" band (< 200 ppm).

---

## 7. Test Coverage Summary

| Module | Tests | Coverage areas |
|--------|-------|----------------|
| test_geometry.py | 14 | tilt matrix, eta convention, LUT shape/range, find_beam_centre (synthetic) |
| test_integrate.py | 29 | SNIP, integrate_1d (uniform/mask/eta/6-tuple), cake (4-tuple), cake_varbin (5-tuple/eta_bin_auto), varbin_tth_edges, pixel_resolution, reduce_frame (varbin/unibin/lineout-only/cake-only) |
| test_io.py | 26 | orient_mask, auto_mask, poni roundtrip, make_geometry, mode roundtrip, load_midas_params (geometry/BC_Z/aliases/NrPixels-square/no-NrPixels), midas_params roundtrip, BC_Z frame roundtrip, PanelShiftsFile |
| test_calibrant.py | 17 | JCPDS/CIF parsing, ring_table, load_calibrant, bare-name resolution |
| test_optimizer.py | 11 | ring pixels, strain cost, calibrate (global/panels), panel centres |
| test_panels.py | 24 | make_panel_id_map, panel map helpers, apply_panel_offsets, build_lut_with_panels, shifts IO (column order/roundtrip/MIDAS-format/optional-cols) |
| **Total** | **121** | all passing as of 2026-04-27 |

---

## 8. GUI Feature Status

| Feature | Status |
|---------|--------|
| Framework | PySide6 >= 6.4 (LGPL), dark QPalette + QSS stylesheet |
| Calibration tab | **DONE** — full workflow: load, find center, run calibration, rings |
| Integration tab | **DONE** — batch, manual, eta sector, rebin, export I\_sub |
| Mask editor | **DONE** — draw tools, undo/redo, auto, despeckle, reverse |
| Auto Find Center | **DONE** — 2-step: sharpness BC + first-ring Lsd |
| Manual Find Center | **DONE** — tilt-aware Nelder-Mead on clicked arc points |
| Ring overlay | **DONE** — true ellipses via ray-tracing through TRs |
| SNIP 2θ range | **DONE** — draggable vlines + spinboxes; limits SNIP computation |
| Export I\_sub | **DONE** — scope dialog (current/checked/all files); 6-col output |
| I\_sub error band | **DONE** — ±σ fill\_between on lineout plot |
| tth sync | **DONE** — SyncBus.tth\_changed across all canvases |
| Lineout hover | **DONE** — I/I\_sub/SNIP value + `px = nnn` at cursor |
| Cake hover | **DONE** — 2θ, η, I, and `px = nnn` (live reduction only) |
| 79 Hz reduction | **DONE** — precompute\_bin\_maps + numba, ~13 ms/frame |
| Panel shifts | via TOML embedded tables; no separate "Load Panel Shifts…" button |
| PVA streaming | future (Levels 3–4 required) |

### GUI launch

```bash
python -m midas4pil.gui
```

```python
from midas4pil.gui import launch_gui
launch_gui()
```

---

## 9. Distribution

### Finding bundled docs and example after installation

```python
import midas4pil, pathlib
pkg = pathlib.Path(midas4pil.__file__).parent
print(pkg / 'docs')           # README.md, USAGE_GUIDE.md, REFERENCE_REPORT.md
print(pkg / 'calib_example')  # example script + input data
```

---

## 10. References

1. Sharma, H. *MIDAS* — Microstructural Imaging using Diffraction Analysis Software.
   https://github.com/marinerhemant/MIDAS
2. Morháč, M. et al. "Background elimination methods for multidimensional
   coincidence gamma-ray spectra." *NIM A* 401 (1997) 113–132. (SNIP algorithm)
3. NIST Powder Diffraction Standard Reference Materials.
   https://www.nist.gov/programs-projects/powder-diffraction-srms
4. Ashiotis, G. et al. "The fast azimuthal integration Python library: pyFAI."
   *J. Appl. Cryst.* 48 (2015) 510–519.
5. Decker, D.L. et al. "High-pressure calibration: A critical review."
   *J. Phys. Chem. Ref. Data* 1 (1972) 773–836.
