# Changelog

All notable changes to midas4pil are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.1.3] — 2026-04-27

Initial public release under ANL open-source approval (SF-26-094, BSD 3-Clause).

### Core library

- Detector geometry correction — Lsd, beam centre (BC_Y, BC_Z), tilts (ty, tz, tx),
  radial distortion (p0–p4); matches MIDAS `DetectorGeometry.c` exactly.
- Per-panel rigid-body shift correction for Pilatus/Eiger tiled detectors
  (dY, dZ, dLsd, dP2, dTheta); matches MIDAS `CalibrantPanelShiftsOMP.c`.
- Pixel-matched varbin integration — bin width equals detector angular resolution
  at each 2θ; no artificial oversampling.
- Uniform-bin (unibin) integration for downstream FFT/PDF workflows.
- SNIP background subtraction (Morhač et al., NIM A 401, 1997).
- Caked image output — (2θ, η) polar map with pixel-matched or uniform bins.
- 6-column lineout: 2θ, I, SNIP_bg, I_sub, σ (Poisson SEM), px_cnt.
- Three-stage calibration optimizer: rough → PONI → panels.
- Auto beam-centre finding: 2-step sharpness + first-ring profile algorithm.
- .poni import/export (pyFAI/Dioptas compatible, no external library needed).
- MIDAS geometry_params.txt import/export for interoperability with MIDAS binaries.
- JCPDS v4/5.1 and CIF calibrant readers (bundled calibrants included).
- Numba JIT kernels (`_jit.py`) — 17× speedup; 79 Hz sustained reduction rate.

### GUI (PySide6)

- Calibration tab — Load .poni, Auto Find Center, Run Calibration, panel optimizer.
- Integration tab — varbin/unibin lineout, caked image, mask editor, Export.
- Dark theme, minimalist layout; tested on 16-IDB 2M CdTe Pilatus.

### Legal

- ANL open-source release approved (SF-26-094).
  Copyright (c) 2026 UChicago Argonne, LLC. All Rights Reserved.
  BSD 3-Clause license headers added to all 30 Python source files (2026-07-01).

### Reference result (16-IDB, 2026-03-31, CeO2, E = 29.200 keV)

| Mode | Mean strain |
|------|------------|
| PONI only | 175 ppm |
| PONI + panel correction | 117 ppm |
